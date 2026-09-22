"""
情书（Love Letter）状态编码器（相对视角 + 私密知识 + 有序事件时间线）。

观察向量分两段，供 GRU 序列编码器使用：

【静态段 static（25 + 5n 维）】—— 当前状态的快照
    1. 自己手牌 (2 × 8 = 16 维)
    2. 爱心标记 (n 维，相对顺序)
    3. 私密知识 (n-1 维)：我私底下知道每个相对对手是哪张牌（0=未知）
    4. 未露面卡计数 (8 维)：信念先验
    5. 牌堆剩余 (1 维)
    6. 全局行动计数 (1 维)
    7. 出局标记 (n 维，相对顺序)
    8. 侍女保护 (n 维，相对顺序)
    9. 当前玩家 ID (n 维，绝对 one-hot，用于对齐绝对时间线)

【序列段 seq（seq_len × (n+8) 维）】—— 本轮按出牌顺序的事件时间线
    每个事件 = 谁打的（n 维 one-hot）+ 打了什么牌（8 维 one-hot）
    填充到固定 seq_len，空槽全 0。

总计 = (25 + 5n) + seq_len * (n + 8) 维。
"""

from __future__ import annotations

import numpy as np

from games.love_letter.constants import CARD_COPIES
from games.love_letter.state import LoveLetterState


class LoveLetterEncoder:
    """情书状态编码器（相对视角 + 私密知识 + 有序事件时间线）。"""

    def __init__(self, num_players: int):
        self.num_players = num_players
        self.seq_len = 24
        self.evt_dim = num_players + 8
        self.static_dim = 25 + 5 * num_players
        self.observation_dim = self.static_dim + self.seq_len * self.evt_dim

    @property
    def model_encoder_params(self) -> dict:
        """GRU 编码器所需的拆分参数。"""
        return {
            "static_dim": self.static_dim,
            "seq_len": self.seq_len,
            "evt_dim": self.evt_dim,
        }

    def encode(self, state: LoveLetterState, player_id: int) -> np.ndarray:
        n = state.num_players
        obs = np.zeros(self.observation_dim, dtype=np.float32)
        idx = 0

        # ===== 静态段 =====
        # 1. 自己手牌（2 × 8）
        hand = state.hands[player_id]
        for i in range(2):
            if i < len(hand):
                obs[idx + (hand[i] - 1)] = 1.0
            idx += 8

        # 2. 爱心标记（相对顺序）
        idx = self._encode_tokens(obs, idx, state, player_id)

        # 3. 私密知识（相对顺序）
        idx = self._encode_known(obs, idx, state, player_id)

        # 4. 未露面卡计数（信念先验）
        idx = self._encode_unseen(obs, idx, state, player_id)

        # 5. 牌堆剩余
        obs[idx] = len(state.deck) / 16.0
        idx += 1

        # 6. 全局行动计数
        obs[idx] = min(state.turn_number / 100.0, 1.0)
        idx += 1

        # 7. 出局标记（相对顺序）
        for r in range(n):
            obs[idx] = 1.0 if state.eliminated[(player_id + r) % n] else 0.0
            idx += 1

        # 8. 侍女保护（相对顺序）
        for r in range(n):
            obs[idx] = 1.0 if state.protected[(player_id + r) % n] else 0.0
            idx += 1

        # 9. 当前玩家 ID（绝对 one-hot，用于对齐绝对时间线）
        if 0 <= state.current_player < n:
            obs[idx + state.current_player] = 1.0
        idx += n

        assert idx == self.static_dim, f"静态段维度不一致: {idx} != {self.static_dim}"

        # ===== 序列段（有序事件时间线）=====
        for i in range(self.seq_len):
            if i < len(state.events):
                p, card = state.events[i]
                obs[idx + p] = 1.0                     # 谁打的（绝对 one-hot）
                obs[idx + n + (card - 1)] = 1.0        # 打了什么牌（one-hot）
            idx += self.evt_dim

        assert idx == self.observation_dim, f"编码维度不一致: {idx} != {self.observation_dim}"
        return obs

    def _encode_tokens(
        self, obs: np.ndarray, idx: int, state: LoveLetterState, player_id: int
    ) -> int:
        n = state.num_players
        for r in range(n):
            other = (player_id + r) % n
            obs[idx] = state.tokens[other] / max(state.target_tokens, 1)
            idx += 1
        return idx

    def _encode_known(
        self, obs: np.ndarray, idx: int, state: LoveLetterState, player_id: int
    ) -> int:
        n = state.num_players
        for r in range(1, n):
            other = (player_id + r) % n
            known = state.known[player_id][other]
            obs[idx] = 0.0 if known is None else known / 8.0
            idx += 1
        return idx

    def _encode_unseen(
        self, obs: np.ndarray, idx: int, state: LoveLetterState, player_id: int
    ) -> int:
        n = state.num_players
        seen = [0] * 8
        for v in state.hands[player_id]:
            seen[v - 1] += 1
        for p in range(n):
            for v in state.discards[p]:
                seen[v - 1] += 1
        for v in range(1, 9):
            unseen = CARD_COPIES[v] - seen[v - 1]
            obs[idx] = max(unseen, 0) / max(CARD_COPIES[v], 1)
            idx += 1
        return idx


__all__ = ["LoveLetterEncoder"]
