"""
情书（Love Letter）状态编码器。

观察向量（固定维度，取决于玩家数 n）：

    1. 当前玩家手牌 (2 × 8 = 16 维)：每张卡按值 one-hot（1-8 → 索引 0-7）
    2. 每个玩家公开信息 (n × 10 维)：
       - 弃牌堆中各值的数量 (8 维，封顶 3 后归一化)
       - 是否出局 (1 维)
       - 是否受侍女保护 (1 维)
    3. 每个玩家爱心标记数 (n 维，按 target_tokens 归一化)
    4. 牌堆剩余数量 (1 维，/16 归一化)
    5. 当前玩家 ID (n 维 one-hot)
    6. 全局行动计数 (1 维，/100 归一化)

    总计 = 18 + 12n 维。

隐藏信息约定：只编码当前玩家自己的手牌；其他玩家的手牌不编码。
"""

from __future__ import annotations

import numpy as np

from games.love_letter.state import LoveLetterState


class LoveLetterEncoder:
    """情书状态编码器。"""

    def __init__(self, num_players: int):
        self.num_players = num_players
        self.observation_dim = 18 + 12 * num_players

    def encode(self, state: LoveLetterState, player_id: int) -> np.ndarray:
        n = state.num_players
        obs = np.zeros(self.observation_dim, dtype=np.float32)
        idx = 0

        # 1. 当前玩家手牌（2 × 8）
        hand = state.hands[player_id]
        for i in range(2):
            if i < len(hand):
                v = hand[i]
                obs[idx + (v - 1)] = 1.0  # 值 1-8 → 索引 0-7
            idx += 8

        # 2. 每个玩家公开信息（n × 10）
        for p in range(n):
            # 弃牌堆各值数量（8 维，封顶 3）
            counts = [0] * 8
            for v in state.discards[p]:
                counts[v - 1] += 1
            for j in range(8):
                obs[idx + j] = min(counts[j], 3) / 3.0
            idx += 8
            # 出局 / 受保护
            obs[idx] = 1.0 if state.eliminated[p] else 0.0
            idx += 1
            obs[idx] = 1.0 if state.protected[p] else 0.0
            idx += 1

        # 3. 爱心标记（n 维）
        for p in range(n):
            obs[idx] = state.tokens[p] / max(state.target_tokens, 1)
            idx += 1

        # 4. 牌堆剩余（1 维）
        obs[idx] = len(state.deck) / 16.0
        idx += 1

        # 5. 当前玩家 ID（n 维 one-hot）
        if 0 <= state.current_player < n:
            obs[idx + state.current_player] = 1.0
        idx += n

        # 6. 全局行动计数（1 维）
        obs[idx] = min(state.turn_number / 100.0, 1.0)
        idx += 1

        assert idx == self.observation_dim, f"编码维度不一致: {idx} != {self.observation_dim}"
        return obs


__all__ = ["LoveLetterEncoder"]
