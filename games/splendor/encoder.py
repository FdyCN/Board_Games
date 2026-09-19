"""
Splendor 状态编码器

将游戏状态编码为固定长度的观察向量，用于神经网络输入。
"""

import numpy as np

from games.splendor.constants import CardTier, GemColor, NUM_GEM_COLORS, WINNING_SCORE
from games.splendor.state import SplendorState

# ===== 观察向量中"逐卡特征"区段的索引（供模型抽取，实现共享卡评估器） =====
# 观察布局详见下方 docstring：
#   保留卡（当前玩家 3 张）在 [15:60]，每张 15 维
#   公开卡牌（12 张明牌）在 [105:285]，每张 15 维
CARD_FEAT_DIM = 15
RESERVED_CARDS_START = 15
RESERVED_CARDS_END = 60
OPEN_CARDS_START = 105
OPEN_CARDS_END = 285


class SplendorEncoder:
    """
    Splendor 状态编码器

    观察向量结构（总计 384 维，实际使用 326 维）：
    1. 当前玩家状态 (60 维)
       - 宝石 (6): 红绿蓝白黑金，归一化到 [0,1]
       - 卡牌加成 (5): 每种颜色的加成数量，归一化
       - 分数 (1): 归一化分数
       - 保留卡数 (1)
       - 已购卡数 (1)
       - 贵族数 (1)
       - 保留卡详情 (45): 3张保留卡 × 15维
         每张卡: [tier(3), points(1), bonus_color(5), cost(5), valid(1)]

    2. 其他玩家状态 (3 × 15 = 45 维)
       每个玩家: [gems(6), bonuses(5), score(1), cards(1), reserved(1), nobles(1)]

    3. 公开卡牌 (12 × 15 = 180 维)
       每张卡: [tier(3), points(1), bonus_color(5), cost(5), valid(1)]

    4. 贵族 (5 × 6 = 30 维)
       每张贵族: [requirements(5), valid(1)]

    5. 游戏信息 (11 维)
       - 宝石堆 (6): 每种颜色宝石数量
       - 当前玩家 ID (4): one-hot
       - 回合数 (1): 归一化

    总计: 60 + 45 + 180 + 30 + 11 = 326 维 (分配 384 维空间)
    """

    def __init__(self):
        self.observation_dim = 384  # 增加到 384 以容纳所有信息

    def encode(self, state: SplendorState, player_id: int) -> np.ndarray:
        """
        编码游戏状态为观察向量

        Args:
            state: 游戏状态
            player_id: 观察者玩家 ID

        Returns:
            观察向量 (384,)
        """
        obs = np.zeros(self.observation_dim, dtype=np.float32)
        idx = 0

        # 1. 当前玩家状态 (60 维)
        player = state.players[player_id]

        # 宝石 (6)
        for i in range(6):
            obs[idx] = player.gems[i] / 10.0  # 归一化到 [0,1]
            idx += 1

        # 卡牌加成 (5)
        bonuses = player.get_total_bonuses()
        for i in range(NUM_GEM_COLORS):
            obs[idx] = bonuses[i] / 10.0
            idx += 1

        # 分数 (1)
        obs[idx] = player.get_score() / WINNING_SCORE
        idx += 1

        # 保留卡数 (1)
        obs[idx] = len(player.reserved_cards) / 3.0
        idx += 1

        # 已购卡数 (1)
        obs[idx] = len(player.cards) / 20.0
        idx += 1

        # 贵族数 (1)
        obs[idx] = len(player.nobles) / 5.0
        idx += 1

        # 保留卡详情 (45 = 3 × 15)
        for i in range(3):
            if i < len(player.reserved_cards):
                card = player.reserved_cards[i]
                idx = self._encode_card(obs, idx, card, valid=True)
            else:
                idx = self._encode_card(obs, idx, None, valid=False)

        # 2. 其他玩家状态 (45 = 3 × 15)
        for i in range(state.num_players):
            if i == player_id:
                continue
            other = state.players[i]

            # 宝石 (6)
            for j in range(6):
                obs[idx] = other.gems[j] / 10.0
                idx += 1

            # 加成 (5)
            other_bonuses = other.get_total_bonuses()
            for j in range(NUM_GEM_COLORS):
                obs[idx] = other_bonuses[j] / 10.0
                idx += 1

            # 其他信息 (4)
            obs[idx] = other.get_score() / WINNING_SCORE
            idx += 1
            obs[idx] = len(other.cards) / 20.0
            idx += 1
            obs[idx] = len(other.reserved_cards) / 3.0
            idx += 1
            obs[idx] = len(other.nobles) / 5.0
            idx += 1

        # 填充未使用的玩家槽位（如果玩家数 < 4）
        for _ in range(3 - (state.num_players - 1)):
            obs[idx : idx + 15] = 0
            idx += 15

        # 3. 公开卡牌 (180 = 12 × 15)
        all_open_cards = []
        for tier in [CardTier.TIER_1, CardTier.TIER_2, CardTier.TIER_3]:
            all_open_cards.extend(state.open_cards[tier])

        for i in range(12):
            if i < len(all_open_cards):
                card = all_open_cards[i]
                idx = self._encode_card(obs, idx, card, valid=True)
            else:
                idx = self._encode_card(obs, idx, None, valid=False)

        # 4. 贵族 (30 = 5 × 6)
        for i in range(5):
            if i < len(state.nobles):
                noble = state.nobles[i]
                # 需求 (5)
                for j in range(NUM_GEM_COLORS):
                    obs[idx] = noble.requirements[j] / 5.0
                    idx += 1
                # valid (1)
                obs[idx] = 1.0
                idx += 1
            else:
                obs[idx : idx + 6] = 0
                idx += 6

        # 5. 游戏信息 (11)
        # 宝石堆 (6)
        for i in range(6):
            obs[idx] = state.gem_bank[i] / 7.0  # 最多 7 个
            idx += 1

        # 当前玩家 ID (4) - one-hot
        obs[idx : idx + 4] = 0
        obs[idx + state.current_player] = 1.0
        idx += 4

        # 回合数 (1)
        obs[idx] = min(state.turn_number / 100.0, 1.0)
        idx += 1

        # 确保维度正确
        assert idx <= self.observation_dim, f"编码超出预期维度: {idx} > {self.observation_dim}"

        return obs

    def _encode_card(self, obs: np.ndarray, idx: int, card, valid: bool) -> int:
        """
        编码单张卡牌 (15 维)

        Args:
            obs: 观察数组
            idx: 当前索引
            card: 卡牌对象（可以是 DevelopmentCard 或 None）
            valid: 是否有效

        Returns:
            新的索引位置
        """
        if not valid or card is None:
            obs[idx : idx + 15] = 0
            return idx + 15

        # tier (3) - one-hot
        obs[idx : idx + 3] = 0
        obs[idx + card.tier - 1] = 1.0
        idx += 3

        # points (1)
        obs[idx] = card.points / 5.0
        idx += 1

        # bonus_color (5) - one-hot
        obs[idx : idx + 5] = 0
        obs[idx + card.bonus_color] = 1.0
        idx += 5

        # cost (5)
        for i in range(NUM_GEM_COLORS):
            obs[idx] = card.cost[i] / 7.0
            idx += 1

        # valid (1)
        obs[idx] = 1.0
        idx += 1

        return idx


# ===== 导出 =====
__all__ = ["SplendorEncoder"]
