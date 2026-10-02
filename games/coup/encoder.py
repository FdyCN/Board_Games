"""
政变疑云（Coup）状态编码器（相对视角 + 事件时间线）。

观察向量分两段，供 GRU 序列编码器使用（与情书同款 hybrid 结构）：

【静态段】—— 当前状态快照
    1. 自己 2 张手牌（2 × 5 one-hot）
    2. 自己金币（/10）
    3. 牌堆剩余（/15）
    4. 当前阶段（6 one-hot）
    5. 每个相对对手：金币 /10、存活、已公开弃牌数 /2
    6. 已公开弃牌计数（5 角色，/3）—— 信念先验
    7. 当前玩家 ID（绝对 one-hot）
    8. pending 行动类型（7 one-hot）
    9. pending 目标（相对 one-hot，dim0=无）
    10. pending 宣称/反制角色（5 one-hot）

【序列段】—— 最近 seq_len 个事件（谁 + 做了什么 + 涉及哪个角色）
    每个事件 = 玩家 one-hot(n) + 事件类型 one-hot(9) + 角色 one-hot(5)

信念（对手是什么身份）主要由序列段隐式学习：宣称/质疑/反制/翻开/换牌这些事件
告诉模型「谁可能是什么」，与情书用事件时间线学私密知识是同一思路。
"""

from __future__ import annotations

import numpy as np

from games.coup.constants import (
    NUM_ROLES,
    NUM_PHASES,
    NUM_EVENT_TYPES,
    NUM_ACTION_TYPES,
)
from games.coup.state import CoupState


class CoupEncoder:
    """Coup 状态编码器（相对视角 + 事件时间线）。"""

    def __init__(self, num_players: int):
        self.num_players = num_players
        n = num_players
        self.seq_len = 64
        self.evt_dim = n + NUM_EVENT_TYPES + NUM_ROLES

        self.hand_dim = 2 * NUM_ROLES          # 10
        self.coin_dim = 1
        self.deck_dim = 1
        self.phase_dim = NUM_PHASES            # 6
        self.opp_dim = 3 * (n - 1)             # 金币/存活/弃牌数
        self.revealed_dim = NUM_ROLES          # 5
        self.belief_dim = NUM_ROLES * (n - 1)  # 每对手 × 持有各角色概率（显式信念）
        self.cur_player_dim = n
        self.pending_kind_dim = NUM_ACTION_TYPES  # 7
        self.pending_target_dim = n
        self.pending_role_dim = NUM_ROLES      # 5

        self.static_dim = (
            self.hand_dim + self.coin_dim + self.deck_dim + self.phase_dim
            + self.opp_dim + self.revealed_dim + self.belief_dim + self.cur_player_dim
            + self.pending_kind_dim + self.pending_target_dim + self.pending_role_dim
        )
        self.observation_dim = self.static_dim + self.seq_len * self.evt_dim

    @property
    def model_encoder_params(self) -> dict:
        """GRU 编码器所需的拆分参数。"""
        return {
            "static_dim": self.static_dim,
            "seq_len": self.seq_len,
            "evt_dim": self.evt_dim,
        }

    def encode(self, state: CoupState, player_id: int) -> np.ndarray:
        n = state.num_players
        obs = np.zeros(self.observation_dim, dtype=np.float32)
        idx = 0

        # 1. 自己手牌
        hand = state.hands[player_id]
        for i in range(2):
            if i < len(hand):
                obs[idx + hand[i]] = 1.0
            idx += NUM_ROLES

        # 2. 自己金币
        obs[idx] = state.coins[player_id] / 10.0
        idx += 1

        # 3. 牌堆剩余
        obs[idx] = len(state.deck) / 15.0
        idx += 1

        # 4. 阶段
        if 0 <= state.phase < NUM_PHASES:
            obs[idx + state.phase] = 1.0
        idx += NUM_PHASES

        # 5. 相对对手：金币/存活/弃牌数
        for r in range(1, n):
            other = (player_id + r) % n
            obs[idx] = state.coins[other] / 10.0
            obs[idx + 1] = 1.0 if state.alive[other] else 0.0
            obs[idx + 2] = min(len(state.revealed[other]), 2) / 2.0
            idx += 3

        # 6. 已公开弃牌计数（信念先验）
        seen = [0] * NUM_ROLES
        for p in range(n):
            for role in state.revealed[p]:
                seen[role] += 1
        for role in range(NUM_ROLES):
            obs[idx] = min(seen[role], 3) / 3.0
            idx += 1

        # 6.5 显式信念状态：每个相对对手持有各角色的概率（算牌先验）
        #     remaining[x] = 我还没看到的角色 x 张数；P(对手 q 持有 x) = 1 - (1 - 密度)^手牌数
        seen_belief = list(seen)
        for v in state.hands[player_id]:
            seen_belief[v] += 1
        remaining = [max(0, 3 - seen_belief[role]) for role in range(NUM_ROLES)]
        total = sum(remaining)
        for r in range(1, n):
            q = (player_id + r) % n
            k = len(state.hands[q]) if state.alive[q] else 0
            for x in range(NUM_ROLES):
                if total > 0 and k > 0:
                    obs[idx] = 1.0 - (1.0 - remaining[x] / total) ** k
                else:
                    obs[idx] = 0.0
                idx += 1

        # 7. 当前玩家（绝对 one-hot，对齐时间线）
        if 0 <= state.current_player < n:
            obs[idx + state.current_player] = 1.0
        idx += n

        # 8. pending 行动类型
        kind_map = {
            "income": 0, "foreign_aid": 1, "coup": 2, "tax": 3,
            "assassinate": 4, "steal": 5, "exchange": 6,
        }
        k = kind_map.get(state.pending_kind)
        if k is not None:
            obs[idx + k] = 1.0
        idx += NUM_ACTION_TYPES

        # 9. pending 目标（相对，dim0=无）
        if state.pending_target >= 0:
            rel = (state.pending_target - player_id) % n
            if 1 <= rel < n:
                obs[idx + rel] = 1.0
            else:
                obs[idx] = 1.0
        else:
            obs[idx] = 1.0
        idx += n

        # 10. pending 宣称/反制角色
        if state.pending_role >= 0:
            obs[idx + state.pending_role] = 1.0
        idx += NUM_ROLES

        assert idx == self.static_dim, f"静态段维度不一致: {idx} != {self.static_dim}"

        # ===== 序列段（最近 seq_len 个事件）=====
        recent = state.events[-self.seq_len:] if state.events else []
        for i in range(self.seq_len):
            if i < len(recent):
                p, ev_type, role = recent[i]
                obs[idx + p] = 1.0
                obs[idx + n + ev_type] = 1.0
                if role >= 0:
                    obs[idx + n + NUM_EVENT_TYPES + role] = 1.0
            idx += self.evt_dim

        assert idx == self.observation_dim, f"编码维度不一致: {idx} != {self.observation_dim}"
        return obs


__all__ = ["CoupEncoder"]
