"""
Splendor 游戏动作定义

本模块定义了 Splendor 中的所有玩家动作类型。
"""

from dataclasses import dataclass
from typing import ClassVar

from games.splendor.constants import ActionType, CardTier, GemColor


@dataclass(frozen=True)
class SplendorAction:
    """动作基类"""

    action_type: ActionType

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


@dataclass(frozen=True)
class TakeGemsAction(SplendorAction):
    """
    拿宝石动作

    支持两种模式：
    1. 拿 3 个不同颜色的宝石
    2. 拿 2 个相同颜色的宝石
    """

    gems: tuple[int, int, int, int, int]  # [R, G, B, W, Bl]

    def __post_init__(self):
        """验证动作有效性"""
        # 推断动作类型
        non_zero_colors = sum(1 for g in self.gems if g > 0)
        total_gems = sum(self.gems)

        if non_zero_colors == 3 and total_gems == 3:
            # 拿 3 个不同颜色
            object.__setattr__(self, "action_type", ActionType.TAKE_3_DIFFERENT)
        elif non_zero_colors == 1 and total_gems == 2:
            # 拿 2 个相同颜色
            object.__setattr__(self, "action_type", ActionType.TAKE_2_SAME)
        else:
            raise ValueError(f"Invalid gems combination: {self.gems}")

    def is_take_three_different(self) -> bool:
        """是否是拿 3 个不同颜色"""
        return self.action_type == ActionType.TAKE_3_DIFFERENT

    def is_take_two_same(self) -> bool:
        """是否是拿 2 个相同颜色"""
        return self.action_type == ActionType.TAKE_2_SAME

    def get_gem_count(self, color: GemColor) -> int:
        """获取指定颜色的宝石数量"""
        if color == GemColor.GOLD:
            return 0
        return self.gems[color]

    def __str__(self) -> str:
        from games.splendor.constants import GEM_SYMBOLS

        gems_str = ", ".join([f"{g}{GEM_SYMBOLS[GemColor(i)]}" for i, g in enumerate(self.gems) if g > 0])
        return f"TakeGems({gems_str})"


@dataclass(frozen=True)
class ReserveCardAction(SplendorAction):
    """
    保留卡牌动作

    玩家可以从明牌或牌堆顶保留 1 张卡牌，并获得 1 个金宝石
    """

    tier: CardTier  # 卡牌等级
    card_id: int | None  # 卡牌 ID（None 表示从牌堆顶拿）

    def __post_init__(self):
        object.__setattr__(self, "action_type", ActionType.RESERVE_CARD)

    def is_from_deck(self) -> bool:
        """是否从牌堆顶保留"""
        return self.card_id is None

    def __str__(self) -> str:
        if self.is_from_deck():
            return f"ReserveCard(T{self.tier} from deck)"
        else:
            return f"ReserveCard(T{self.tier} #{self.card_id})"


@dataclass(frozen=True)
class BuyCardAction(SplendorAction):
    """
    购买卡牌动作

    玩家可以购买明牌或已保留的卡牌
    """

    card_id: int  # 卡牌 ID
    from_reserved: bool  # 是否从保留区购买
    payment: tuple[int, int, int, int, int, int] | None = None  # [R, G, B, W, Bl, Gold]

    def __post_init__(self):
        object.__setattr__(self, "action_type", ActionType.BUY_CARD)

        # 如果没有指定支付方式，设为 None（游戏引擎会自动计算）
        if self.payment is None:
            object.__setattr__(self, "payment", None)

    def is_from_hand(self) -> bool:
        """是否从保留区购买"""
        return self.from_reserved

    def __str__(self) -> str:
        source = "reserved" if self.from_reserved else "open"
        return f"BuyCard(#{self.card_id} from {source})"


# ===== 辅助函数 =====


def create_take_three_different(colors: list[GemColor]) -> TakeGemsAction:
    """
    创建"拿 3 个不同颜色"动作

    Args:
        colors: 3 种不同颜色的列表

    Returns:
        TakeGemsAction

    Example:
        >>> action = create_take_three_different([GemColor.RED, GemColor.GREEN, GemColor.BLUE])
    """
    if len(colors) != 3 or len(set(colors)) != 3:
        raise ValueError("Must provide exactly 3 different colors")

    if GemColor.GOLD in colors:
        raise ValueError("Cannot take gold gems directly")

    gems = [0, 0, 0, 0, 0]
    for color in colors:
        gems[color] = 1

    return TakeGemsAction(action_type=ActionType.TAKE_3_DIFFERENT, gems=tuple(gems))


def create_take_two_same(color: GemColor) -> TakeGemsAction:
    """
    创建"拿 2 个相同颜色"动作

    Args:
        color: 宝石颜色

    Returns:
        TakeGemsAction

    Example:
        >>> action = create_take_two_same(GemColor.RED)
    """
    if color == GemColor.GOLD:
        raise ValueError("Cannot take 2 gold gems")

    gems = [0, 0, 0, 0, 0]
    gems[color] = 2

    return TakeGemsAction(action_type=ActionType.TAKE_2_SAME, gems=tuple(gems))


def create_reserve_card(tier: CardTier, card_id: int | None = None) -> ReserveCardAction:
    """
    创建"保留卡牌"动作

    Args:
        tier: 卡牌等级
        card_id: 卡牌 ID（None 表示从牌堆顶拿）

    Returns:
        ReserveCardAction

    Example:
        >>> action = create_reserve_card(CardTier.TIER_1, 5)  # 保留明牌
        >>> action = create_reserve_card(CardTier.TIER_2)  # 从牌堆顶保留
    """
    return ReserveCardAction(action_type=ActionType.RESERVE_CARD, tier=tier, card_id=card_id)


def create_buy_card(card_id: int, from_reserved: bool = False) -> BuyCardAction:
    """
    创建"购买卡牌"动作

    Args:
        card_id: 卡牌 ID
        from_reserved: 是否从保留区购买

    Returns:
        BuyCardAction

    Example:
        >>> action = create_buy_card(10, from_reserved=False)  # 购买明牌
        >>> action = create_buy_card(15, from_reserved=True)   # 购买保留卡
    """
    return BuyCardAction(
        action_type=ActionType.BUY_CARD, card_id=card_id, from_reserved=from_reserved
    )


# ===== 导出 =====
__all__ = [
    "SplendorAction",
    "TakeGemsAction",
    "ReserveCardAction",
    "BuyCardAction",
    "create_take_three_different",
    "create_take_two_same",
    "create_reserve_card",
    "create_buy_card",
]
