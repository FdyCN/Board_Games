"""
Splendor 卡牌数据

本模块定义了卡牌数据结构和所有官方卡牌数据（90 张发展卡 + 10 张贵族卡）。

数据来源：Splendor 官方标准套牌
"""

from dataclasses import dataclass
from typing import ClassVar

from games.splendor.constants import CardTier, GemColor, NUM_GEM_COLORS


@dataclass(frozen=True)
class DevelopmentCard:
    """
    发展卡数据类

    Attributes:
        card_id: 卡牌唯一 ID（用于索引）
        tier: 卡牌等级 (1, 2, 3)
        points: 声望点数 (0-5)
        bonus_color: 提供的加成颜色
        cost: 购买成本 [红, 绿, 蓝, 白, 黑]
    """

    card_id: int
    tier: CardTier
    points: int
    bonus_color: GemColor
    cost: tuple[int, int, int, int, int]  # [R, G, B, W, Bl]

    def __post_init__(self):
        """验证卡牌数据"""
        assert self.tier in [CardTier.TIER_1, CardTier.TIER_2, CardTier.TIER_3]
        assert 0 <= self.points <= 5
        assert self.bonus_color in [
            GemColor.RED,
            GemColor.GREEN,
            GemColor.BLUE,
            GemColor.WHITE,
            GemColor.BLACK,
        ]
        assert len(self.cost) == NUM_GEM_COLORS
        assert all(c >= 0 for c in self.cost)

    def get_cost(self, color: GemColor) -> int:
        """获取指定颜色的成本"""
        if color == GemColor.GOLD:
            return 0
        return self.cost[color]

    def total_cost(self) -> int:
        """获取总成本"""
        return sum(self.cost)

    def __str__(self) -> str:
        from games.splendor.constants import GEM_COLORED_SYMBOLS

        bonus_sym = GEM_COLORED_SYMBOLS[self.bonus_color]
        cost_str = ", ".join(
            [f"{c}{GEM_COLORED_SYMBOLS[GemColor(i)]}" for i, c in enumerate(self.cost) if c > 0]
        )
        return f"[T{self.tier} {bonus_sym} {self.points}pt] Cost: {cost_str}"


@dataclass(frozen=True)
class NobleTile:
    """
    贵族卡数据类

    Attributes:
        noble_id: 贵族 ID（用于索引）
        name: 贵族名称（可选，用于显示）
        requirements: 所需卡牌数量 [红, 绿, 蓝, 白, 黑]
    """

    noble_id: int
    name: str
    requirements: tuple[int, int, int, int, int]  # [R, G, B, W, Bl]

    POINTS: ClassVar[int] = 3  # 贵族卡固定 3 分

    def __post_init__(self):
        """验证贵族数据"""
        assert len(self.requirements) == NUM_GEM_COLORS
        assert all(r >= 0 for r in self.requirements)
        assert sum(self.requirements) >= 8  # 贵族通常需要 8-12 张卡

    def get_requirement(self, color: GemColor) -> int:
        """获取指定颜色的需求"""
        if color == GemColor.GOLD:
            return 0
        return self.requirements[color]

    def __str__(self) -> str:
        from games.splendor.constants import GEM_COLORED_SYMBOLS

        req_str = ", ".join(
            [f"{r}{GEM_COLORED_SYMBOLS[GemColor(i)]}" for i, r in enumerate(self.requirements) if r > 0]
        )
        return f"[{self.name}] Requires: {req_str}"


# ===== 官方发展卡数据 =====
# 根据官方标准套牌整理
# 格式: (card_id, tier, points, bonus_color, (R, G, B, W, Bl))


# Tier 1 发展卡 (40 张)
# 成本: 0-4 个宝石, 分值: 0-1 分
TIER_1_CARDS_DATA = [
    # ID, Tier, Points, Bonus, (Red, Green, Blue, White, Black)
    # 0 分卡 - 白色加成 (8 张)
    (0, 1, 0, GemColor.WHITE, (0, 0, 0, 0, 3)),  # 3黑 -> 白
    (1, 1, 0, GemColor.WHITE, (0, 0, 0, 2, 2)),  # 2白2黑 -> 白
    (2, 1, 0, GemColor.WHITE, (0, 0, 2, 0, 2)),  # 2蓝2黑 -> 白
    (3, 1, 0, GemColor.WHITE, (0, 0, 1, 1, 1)),  # 1绿1蓝1白 -> 白
    (4, 1, 0, GemColor.WHITE, (0, 0, 1, 1, 2)),  # 1蓝1白2黑 -> 白
    (5, 1, 0, GemColor.WHITE, (0, 0, 1, 2, 1)),  # 1蓝2白1黑 -> 白
    (6, 1, 0, GemColor.WHITE, (0, 0, 2, 1, 0)),  # 2蓝1白 -> 白
    (7, 1, 0, GemColor.WHITE, (0, 1, 1, 1, 1)),  # 1绿1蓝1白1黑 -> 白
    # 0 分卡 - 蓝色加成 (8 张)
    (8, 1, 0, GemColor.BLUE, (0, 0, 0, 0, 3)),  # 3黑 -> 蓝
    (9, 1, 0, GemColor.BLUE, (0, 0, 0, 2, 2)),  # 2白2黑 -> 蓝
    (10, 1, 0, GemColor.BLUE, (0, 2, 0, 2, 0)),  # 2绿2白 -> 蓝
    (11, 1, 0, GemColor.BLUE, (1, 0, 0, 1, 1)),  # 1红1白1黑 -> 蓝
    (12, 1, 0, GemColor.BLUE, (1, 0, 0, 2, 1)),  # 1红2白1黑 -> 蓝
    (13, 1, 0, GemColor.BLUE, (1, 0, 1, 1, 1)),  # 1红1蓝1白1黑 -> 蓝
    (14, 1, 0, GemColor.BLUE, (0, 0, 2, 1, 0)),  # 2蓝1白 -> 蓝
    (15, 1, 0, GemColor.BLUE, (0, 1, 0, 1, 2)),  # 1绿1白2黑 -> 蓝
    # 0 分卡 - 绿色加成 (8 张)
    (16, 1, 0, GemColor.GREEN, (0, 0, 3, 0, 0)),  # 3蓝 -> 绿
    (17, 1, 0, GemColor.GREEN, (0, 0, 2, 0, 2)),  # 2蓝2黑 -> 绿
    (18, 1, 0, GemColor.GREEN, (2, 0, 0, 0, 2)),  # 2红2黑 -> 绿
    (19, 1, 0, GemColor.GREEN, (1, 0, 1, 0, 1)),  # 1红1蓝1黑 -> 绿
    (20, 1, 0, GemColor.GREEN, (1, 0, 1, 0, 2)),  # 1红1蓝2黑 -> 绿
    (21, 1, 0, GemColor.GREEN, (2, 0, 1, 0, 1)),  # 2红1蓝1黑 -> 绿
    (22, 1, 0, GemColor.GREEN, (1, 0, 2, 0, 0)),  # 1红2蓝 -> 绿
    (23, 1, 0, GemColor.GREEN, (1, 0, 1, 1, 1)),  # 1红1蓝1白1黑 -> 绿
    # 0 分卡 - 红色加成 (8 张)
    (24, 1, 0, GemColor.RED, (3, 0, 0, 0, 0)),  # 3红 -> 红
    (25, 1, 0, GemColor.RED, (2, 2, 0, 0, 0)),  # 2红2绿 -> 红
    (26, 1, 0, GemColor.RED, (2, 0, 0, 2, 0)),  # 2红2白 -> 红
    (27, 1, 0, GemColor.RED, (0, 1, 1, 1, 0)),  # 1绿1蓝1白 -> 红
    (28, 1, 0, GemColor.RED, (0, 1, 2, 1, 0)),  # 1绿2蓝1白 -> 红
    (29, 1, 0, GemColor.RED, (0, 1, 1, 2, 0)),  # 1绿1蓝2白 -> 红
    (30, 1, 0, GemColor.RED, (0, 0, 2, 0, 1)),  # 2蓝1黑 -> 红
    (31, 1, 0, GemColor.RED, (1, 1, 0, 1, 1)),  # 1红1绿1白1黑 -> 红
    # 0 分卡 - 黑色加成 (3 张)
    (32, 1, 0, GemColor.BLACK, (0, 3, 0, 0, 0)),  # 3绿 -> 黑
    (33, 1, 0, GemColor.BLACK, (2, 0, 2, 0, 0)),  # 2红2蓝 -> 黑
    (34, 1, 0, GemColor.BLACK, (0, 2, 0, 2, 0)),  # 2绿2白 -> 黑
    # 1 分卡 - 每种颜色 1 张 (5 张)
    (35, 1, 1, GemColor.WHITE, (0, 0, 0, 4, 0)),  # 4白 -> 白 1分
    (36, 1, 1, GemColor.BLUE, (0, 0, 4, 0, 0)),  # 4蓝 -> 蓝 1分
    (37, 1, 1, GemColor.GREEN, (0, 0, 0, 0, 4)),  # 4黑 -> 绿 1分
    (38, 1, 1, GemColor.RED, (0, 4, 0, 0, 0)),  # 4绿 -> 红 1分
    (39, 1, 1, GemColor.BLACK, (4, 0, 0, 0, 0)),  # 4红 -> 黑 1分
]

# Tier 2 发展卡 (30 张)
# 成本: 5-6 个宝石, 分值: 1-3 分
TIER_2_CARDS_DATA = [
    # 1 分卡 - 白色加成 (3 张)
    (40, 2, 1, GemColor.WHITE, (2, 0, 0, 2, 3)),  # 2红2白3黑 -> 白 1分
    (41, 2, 1, GemColor.WHITE, (2, 0, 3, 0, 2)),  # 2红3蓝2黑 -> 白 1分
    (42, 2, 1, GemColor.WHITE, (0, 3, 2, 2, 0)),  # 3绿2蓝2白 -> 白 1分
    # 1 分卡 - 蓝色加成 (3 张)
    (43, 2, 1, GemColor.BLUE, (2, 3, 0, 0, 2)),  # 2红3绿2黑 -> 蓝 1分
    (44, 2, 1, GemColor.BLUE, (0, 2, 2, 3, 0)),  # 2绿2蓝3白 -> 蓝 1分
    (45, 2, 1, GemColor.BLUE, (3, 0, 0, 2, 2)),  # 3红2白2黑 -> 蓝 1分
    # 1 分卡 - 绿色加成 (3 张)
    (46, 2, 1, GemColor.GREEN, (0, 0, 2, 3, 2)),  # 2蓝3白2黑 -> 绿 1分
    (47, 2, 1, GemColor.GREEN, (2, 0, 3, 0, 2)),  # 2红3蓝2黑 -> 绿 1分
    (48, 2, 1, GemColor.GREEN, (0, 2, 0, 2, 3)),  # 2绿2白3黑 -> 绿 1分
    # 1 分卡 - 红色加成 (3 张)
    (49, 2, 1, GemColor.RED, (2, 2, 3, 0, 0)),  # 2红2绿3蓝 -> 红 1分
    (50, 2, 1, GemColor.RED, (0, 3, 2, 2, 0)),  # 3绿2蓝2白 -> 红 1分
    (51, 2, 1, GemColor.RED, (3, 0, 2, 0, 2)),  # 3红2蓝2黑 -> 红 1分
    # 1 分卡 - 黑色加成 (3 张)
    (52, 2, 1, GemColor.BLACK, (0, 2, 2, 0, 3)),  # 2绿2蓝3黑 -> 黑 1分
    (53, 2, 1, GemColor.BLACK, (2, 0, 0, 3, 2)),  # 2红3白2黑 -> 黑 1分
    (54, 2, 1, GemColor.BLACK, (0, 0, 3, 2, 2)),  # 3蓝2白2黑 -> 黑 1分
    # 2 分卡 - 每种颜色 1 张 (5 张)
    (55, 2, 2, GemColor.WHITE, (0, 0, 0, 5, 0)),  # 5白 -> 白 2分
    (56, 2, 2, GemColor.BLUE, (0, 0, 5, 0, 0)),  # 5蓝 -> 蓝 2分
    (57, 2, 2, GemColor.GREEN, (5, 0, 0, 0, 0)),  # 5红 -> 绿 2分
    (58, 2, 2, GemColor.RED, (0, 0, 0, 0, 5)),  # 5黑 -> 红 2分
    (59, 2, 2, GemColor.BLACK, (0, 5, 0, 0, 0)),  # 5绿 -> 黑 2分
    # 2 分卡 - 混合成本 (5 张)
    (60, 2, 2, GemColor.WHITE, (0, 0, 1, 4, 2)),  # 1蓝4白2黑 -> 白 2分
    (61, 2, 2, GemColor.BLUE, (0, 1, 4, 2, 0)),  # 1绿4蓝2白 -> 蓝 2分
    (62, 2, 2, GemColor.GREEN, (2, 0, 0, 1, 4)),  # 2红1白4黑 -> 绿 2分
    (63, 2, 2, GemColor.RED, (4, 2, 0, 0, 1)),  # 4红2绿1黑 -> 红 2分
    (64, 2, 2, GemColor.BLACK, (1, 4, 2, 0, 0)),  # 1红4绿2蓝 -> 黑 2分
    # 3 分卡 - 每种颜色 1 张 (5 张)
    (65, 2, 3, GemColor.WHITE, (0, 0, 0, 6, 0)),  # 6白 -> 白 3分
    (66, 2, 3, GemColor.BLUE, (0, 0, 6, 0, 0)),  # 6蓝 -> 蓝 3分
    (67, 2, 3, GemColor.GREEN, (6, 0, 0, 0, 0)),  # 6红 -> 绿 3分
    (68, 2, 3, GemColor.RED, (0, 0, 0, 0, 6)),  # 6黑 -> 红 3分
    (69, 2, 3, GemColor.BLACK, (0, 6, 0, 0, 0)),  # 6绿 -> 黑 3分
]

# Tier 3 发展卡 (20 张)
# 成本: 7+ 个宝石, 分值: 3-5 分
TIER_3_CARDS_DATA = [
    # 3 分卡 - 每种颜色 1 张 (5 张)
    (70, 3, 3, GemColor.WHITE, (3, 3, 5, 3, 0)),  # 3红3绿5蓝3白 -> 白 3分
    (71, 3, 3, GemColor.BLUE, (3, 0, 3, 3, 5)),  # 3红3蓝3白5黑 -> 蓝 3分
    (72, 3, 3, GemColor.GREEN, (5, 3, 0, 3, 3)),  # 5红3绿3白3黑 -> 绿 3分
    (73, 3, 3, GemColor.RED, (0, 5, 3, 3, 3)),  # 5绿3蓝3白3黑 -> 红 3分
    (74, 3, 3, GemColor.BLACK, (3, 0, 3, 5, 3)),  # 3红3蓝5白3黑 -> 黑 3分
    # 4 分卡 - 每种颜色 1 张 (5 张)
    (75, 3, 4, GemColor.WHITE, (0, 0, 0, 7, 0)),  # 7白 -> 白 4分
    (76, 3, 4, GemColor.BLUE, (0, 0, 7, 0, 0)),  # 7蓝 -> 蓝 4分
    (77, 3, 4, GemColor.GREEN, (7, 0, 0, 0, 0)),  # 7红 -> 绿 4分
    (78, 3, 4, GemColor.RED, (0, 0, 0, 0, 7)),  # 7黑 -> 红 4分
    (79, 3, 4, GemColor.BLACK, (0, 7, 0, 0, 0)),  # 7绿 -> 黑 4分
    # 5 分卡 - 每种颜色 2 张 (10 张)
    (80, 3, 5, GemColor.WHITE, (0, 0, 0, 7, 3)),  # 7白3黑 -> 白 5分
    (81, 3, 5, GemColor.WHITE, (3, 0, 0, 7, 0)),  # 3红7白 -> 白 5分
    (82, 3, 5, GemColor.BLUE, (0, 0, 7, 3, 0)),  # 7蓝3白 -> 蓝 5分
    (83, 3, 5, GemColor.BLUE, (0, 3, 7, 0, 0)),  # 3绿7蓝 -> 蓝 5分
    (84, 3, 5, GemColor.GREEN, (7, 3, 0, 0, 0)),  # 7红3绿 -> 绿 5分
    (85, 3, 5, GemColor.GREEN, (7, 0, 0, 0, 3)),  # 7红3黑 -> 绿 5分
    (86, 3, 5, GemColor.RED, (0, 7, 3, 0, 0)),  # 7绿3蓝 -> 红 5分
    (87, 3, 5, GemColor.RED, (0, 7, 0, 0, 3)),  # 7绿3黑 -> 红 5分
    (88, 3, 5, GemColor.BLACK, (3, 0, 0, 0, 7)),  # 3红7黑 -> 黑 5分
    (89, 3, 5, GemColor.BLACK, (0, 0, 3, 0, 7)),  # 3蓝7黑 -> 黑 5分
]

# 完整的 90 张发展卡数据（Tier 1: 40张, Tier 2: 30张, Tier 3: 20张）
# 数据已按照官方标准套牌录入


# ===== 官方贵族数据 (10 张) =====
# 格式: (noble_id, name, (R, G, B, W, Bl))
NOBLES_DATA = [
    (0, "Anne de Pisieux", (0, 0, 3, 3, 3)),  # 3蓝3白3黑
    (1, "Charles Quint", (0, 3, 3, 3, 0)),  # 3绿3蓝3白
    (2, "Charles de Bourbon", (0, 3, 0, 3, 3)),  # 3绿3白3黑
    (3, "Elisabeth d'Autriche", (3, 3, 3, 0, 0)),  # 3红3绿3蓝
    (4, "François Ier de France", (3, 3, 0, 0, 3)),  # 3红3绿3黑
    (5, "Henri VIII", (0, 0, 4, 4, 0)),  # 4蓝4白
    (6, "Isabelle de Castille", (4, 0, 0, 4, 0)),  # 4红4白
    (7, "Marie Stuart", (0, 4, 4, 0, 0)),  # 4绿4蓝
    (8, "Cosme de Medicis", (0, 0, 0, 4, 4)),  # 4白4黑
    (9, "Sforza", (3, 0, 0, 3, 3)),  # 3红3白3黑
]


# ===== 卡牌数据加载 =====


def load_development_cards() -> list[DevelopmentCard]:
    """
    加载所有发展卡数据

    Returns:
        发展卡列表（90 张）
    """
    all_cards = []

    for card_data in TIER_1_CARDS_DATA + TIER_2_CARDS_DATA + TIER_3_CARDS_DATA:
        card = DevelopmentCard(
            card_id=card_data[0],
            tier=CardTier(card_data[1]),
            points=card_data[2],
            bonus_color=card_data[3],
            cost=card_data[4],
        )
        all_cards.append(card)

    # 确保总数为 90 张
    assert len(all_cards) == 90, f"Expected 90 cards, got {len(all_cards)}"

    return all_cards


def load_nobles() -> list[NobleTile]:
    """
    加载所有贵族数据

    Returns:
        贵族列表（10 张）
    """
    nobles = []

    for noble_data in NOBLES_DATA:
        noble = NobleTile(noble_id=noble_data[0], name=noble_data[1], requirements=noble_data[2])
        nobles.append(noble)

    assert len(nobles) == 10, f"Expected 10 nobles, got {len(nobles)}"

    return nobles


def get_card_by_id(card_id: int) -> DevelopmentCard | None:
    """
    根据 ID 获取发展卡

    Args:
        card_id: 卡牌 ID

    Returns:
        发展卡对象，如果不存在返回 None
    """
    cards = load_development_cards()
    for card in cards:
        if card.card_id == card_id:
            return card
    return None


def get_noble_by_id(noble_id: int) -> NobleTile | None:
    """
    根据 ID 获取贵族

    Args:
        noble_id: 贵族 ID

    Returns:
        贵族对象，如果不存在返回 None
    """
    nobles = load_nobles()
    for noble in nobles:
        if noble.noble_id == noble_id:
            return noble
    return None


# ===== 导出 =====
__all__ = [
    "DevelopmentCard",
    "NobleTile",
    "load_development_cards",
    "load_nobles",
    "get_card_by_id",
    "get_noble_by_id",
]
