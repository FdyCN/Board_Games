"""
情书（Love Letter）状态编码器。

观察向量按「当前玩家相对视角」排列（减少先手/座位偏差）：
- 所有玩家信息按相对顺序排列：槽位 0 = 自己，槽位 r = 顺时针第 r 个玩家。

布局（固定维度，取决于玩家数 n）：

    1. 自己手牌 (2 × 8 = 16 维)：每张卡按值 one-hot（1-8 → 索引 0-7）
    2. 自己公开信息 (10 维)：弃牌堆各值数量(8) + 出局(1) + 受保护(1)
    3. 其他玩家公开信息 ((n-1) × 10 维)：相对顺序（下家、下下家…）
    4. 爱心标记 (n 维)：自己 + 各对手，按相对顺序
    5. 牌堆剩余 (1 维，/16 归一化)
    6. 全局行动计数 (1 维，/100 归一化)

    总计 = 18 + 11n 维。

隐藏信息约定：只编码自己手牌；其他玩家手牌不编码。
"""

from __future__ import annotations

import numpy as np

from games.love_letter.state import LoveLetterState


class LoveLetterEncoder:
    """情书状态编码器（相对玩家视角）。"""

    def __init__(self, num_players: int):
        self.num_players = num_players
        self.observation_dim = 18 + 11 * num_players

    def encode(self, state: LoveLetterState, player_id: int) -> np.ndarray:
        n = state.num_players
        obs = np.zeros(self.observation_dim, dtype=np.float32)
        idx = 0

        # 1. 自己手牌（2 × 8）
        hand = state.hands[player_id]
        for i in range(2):
            if i < len(hand):
                v = hand[i]
                obs[idx + (v - 1)] = 1.0  # 值 1-8 → 索引 0-7
            idx += 8

        # 2. 自己公开信息（10 维）
        idx = self._encode_player_public(obs, idx, state, player_id)

        # 3. 其他玩家公开信息（相对顺序，下家在前）
        for r in range(1, n):
            other = (player_id + r) % n
            idx = self._encode_player_public(obs, idx, state, other)

        # 4. 爱心标记（相对顺序）
        idx = self._encode_tokens(obs, idx, state, player_id)

        # 5. 牌堆剩余（1 维）
        obs[idx] = len(state.deck) / 16.0
        idx += 1

        # 6. 全局行动计数（1 维）
        obs[idx] = min(state.turn_number / 100.0, 1.0)
        idx += 1

        assert idx == self.observation_dim, f"编码维度不一致: {idx} != {self.observation_dim}"
        return obs

    def _encode_player_public(
        self, obs: np.ndarray, idx: int, state: LoveLetterState, p: int
    ) -> int:
        """编码单个玩家的公开信息（10 维）：弃牌堆各值数量 + 出局 + 受保护。"""
        counts = [0] * 8
        for v in state.discards[p]:
            counts[v - 1] += 1
        for j in range(8):
            obs[idx + j] = min(counts[j], 3) / 3.0
        idx += 8
        obs[idx] = 1.0 if state.eliminated[p] else 0.0
        idx += 1
        obs[idx] = 1.0 if state.protected[p] else 0.0
        idx += 1
        return idx

    def _encode_tokens(
        self, obs: np.ndarray, idx: int, state: LoveLetterState, player_id: int
    ) -> int:
        """编码爱心标记（相对顺序：自己 + 各对手）。"""
        n = state.num_players
        obs[idx] = state.tokens[player_id] / max(state.target_tokens, 1)
        idx += 1
        for r in range(1, n):
            other = (player_id + r) % n
            obs[idx] = state.tokens[other] / max(state.target_tokens, 1)
            idx += 1
        return idx


__all__ = ["LoveLetterEncoder"]
