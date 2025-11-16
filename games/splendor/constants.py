"""
Splendor 游戏常量定义

本模块定义了 Splendor 游戏中的所有常量，包括宝石颜色、游戏规则参数等。
"""

from enum import IntEnum


# ===== 宝石颜色 =====


class GemColor(IntEnum):
    """
    宝石颜色枚举

    5 种基础颜色 + 1 种万能颜色（金）
    """

    RED = 0  # 红宝石 (Ruby)
    GREEN = 1  # 绿宝石 (Emerald)
    BLUE = 2  # 蓝宝石 (Sapphire)
    WHITE = 3  # 白宝石 (Diamond)
    BLACK = 4  # 黑宝石 (Onyx)
    GOLD = 5  # 金宝石 (Gold, 万能)


# 颜色名称映射
GEM_NAMES = {
    GemColor.RED: "Red",
    GemColor.GREEN: "Green",
    GemColor.BLUE: "Blue",
    GemColor.WHITE: "White",
    GemColor.BLACK: "Black",
    GemColor.GOLD: "Gold",
}

# 颜色符号（用于显示）
GEM_SYMBOLS = {
    GemColor.RED: "♦",
    GemColor.GREEN: "♣",
    GemColor.BLUE: "♠",
    GemColor.WHITE: "○",
    GemColor.BLACK: "●",
    GemColor.GOLD: "★",
}

# 基础颜色（不包括金色）
BASE_GEM_COLORS = [GemColor.RED, GemColor.GREEN, GemColor.BLUE, GemColor.WHITE, GemColor.BLACK]

# 颜色数量
NUM_GEM_COLORS = 5  # 不包括金色
NUM_TOTAL_COLORS = 6  # 包括金色


# ===== 游戏规则参数 =====


# 玩家数量
MIN_PLAYERS = 2
MAX_PLAYERS = 4

# 宝石数量配置 {玩家数: 每种颜色宝石数}
GEMS_BY_PLAYER_COUNT = {
    2: 4,
    3: 5,
    4: 7,
}

# 金宝石数量（固定）
NUM_GOLD_GEMS = 5

# 玩家持有上限
MAX_GEMS_IN_HAND = 10  # 玩家最多持有 10 个宝石
MAX_RESERVED_CARDS = 3  # 玩家最多保留 3 张卡

# 公开卡牌数量
NUM_OPEN_CARDS_PER_TIER = 4  # 每个等级展示 4 张明牌

# 贵族数量 {玩家数: 贵族卡数}
NOBLES_BY_PLAYER_COUNT = {
    2: 3,
    3: 4,
    4: 5,
}

# 胜利条件
WINNING_SCORE = 15  # 达到 15 分触发游戏结束

# 贵族分值
NOBLE_POINTS = 3  # 每张贵族卡 3 分


# ===== 卡牌等级 =====


class CardTier(IntEnum):
    """卡牌等级"""

    TIER_1 = 1  # 低级卡（绿色背景）
    TIER_2 = 2  # 中级卡（黄色背景）
    TIER_3 = 3  # 高级卡（蓝色背景）


# 每个等级的卡牌数量
CARDS_PER_TIER = {
    CardTier.TIER_1: 40,
    CardTier.TIER_2: 30,
    CardTier.TIER_3: 20,
}

# 总发展卡数量
TOTAL_DEVELOPMENT_CARDS = sum(CARDS_PER_TIER.values())  # 90 张

# 总贵族数量
TOTAL_NOBLES = 10


# ===== 动作类型 =====


class ActionType(IntEnum):
    """玩家动作类型"""

    TAKE_3_DIFFERENT = 0  # 拿 3 个不同颜色的宝石
    TAKE_2_SAME = 1  # 拿 2 个相同颜色的宝石
    RESERVE_CARD = 2  # 保留 1 张卡牌
    BUY_CARD = 3  # 购买 1 张卡牌


# ===== 游戏阶段 =====


class GamePhase(IntEnum):
    """游戏阶段"""

    SETUP = 0  # 设置阶段
    PLAYING = 1  # 游戏进行中
    FINAL_ROUND = 2  # 最后一轮（有人达到 15 分）
    ENDED = 3  # 游戏结束


# ===== 辅助函数 =====


def get_gems_count(num_players: int) -> int:
    """
    获取指定玩家数对应的每种颜色宝石数量

    Args:
        num_players: 玩家数量 (2-4)

    Returns:
        每种颜色宝石数量

    Raises:
        ValueError: 如果玩家数量无效
    """
    if num_players not in GEMS_BY_PLAYER_COUNT:
        raise ValueError(f"Invalid player count: {num_players}, must be 2-4")
    return GEMS_BY_PLAYER_COUNT[num_players]


def get_nobles_count(num_players: int) -> int:
    """
    获取指定玩家数对应的贵族卡数量

    Args:
        num_players: 玩家数量 (2-4)

    Returns:
        贵族卡数量

    Raises:
        ValueError: 如果玩家数量无效
    """
    if num_players not in NOBLES_BY_PLAYER_COUNT:
        raise ValueError(f"Invalid player count: {num_players}, must be 2-4")
    return NOBLES_BY_PLAYER_COUNT[num_players]


def gem_color_to_str(color: GemColor) -> str:
    """
    将宝石颜色转换为字符串表示

    Args:
        color: 宝石颜色

    Returns:
        颜色名称和符号，例如 "Red ♦"
    """
    name = GEM_NAMES.get(color, "Unknown")
    symbol = GEM_SYMBOLS.get(color, "?")
    return f"{name} {symbol}"
