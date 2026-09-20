"""
情书（Love Letter）游戏常量。

卡牌值（1-8）既标识卡牌类型，也用于终局比大小。
"""

from __future__ import annotations

# ===== 卡牌值 =====
GUARD = 1       # 卫兵：猜对手手牌
PRIEST = 2      # 神父：看对手手牌
BARON = 3       # 男爵：比较手牌
HANDMAID = 4    # 侍女：直到下回合免疫他人效果
PRINCE = 5      # 王子：指定玩家弃牌重抽
KING = 6        # 国王：交换手牌
COUNTESS = 7    # 女伯爵：手上有国王/王子时必须打出
PRINCESS = 8    # 公主：打出即出局

# ===== 每张卡在牌堆中的数量 =====
CARD_COPIES: dict[int, int] = {
    GUARD: 5,
    PRIEST: 2,
    BARON: 2,
    HANDMAID: 2,
    PRINCE: 2,
    KING: 1,
    COUNTESS: 1,
    PRINCESS: 1,
}

DECK_SIZE = 16

CARD_NAMES: dict[int, str] = {
    GUARD: "卫兵",
    PRIEST: "神父",
    BARON: "男爵",
    HANDMAID: "侍女",
    PRINCE: "王子",
    KING: "国王",
    COUNTESS: "女伯爵",
    PRINCESS: "公主",
}

# ===== 获胜所需爱心标记数 =====
TARGET_TOKENS: dict[int, int] = {
    2: 7,
    3: 5,
    4: 4,
}


def make_deck() -> list[int]:
    """构造一副完整的 16 张牌（未洗牌，按值排列）。"""
    deck: list[int] = []
    for value, copies in CARD_COPIES.items():
        deck.extend([value] * copies)
    return deck


__all__ = [
    "GUARD",
    "PRIEST",
    "BARON",
    "HANDMAID",
    "PRINCE",
    "KING",
    "COUNTESS",
    "PRINCESS",
    "CARD_COPIES",
    "DECK_SIZE",
    "CARD_NAMES",
    "TARGET_TOKENS",
    "make_deck",
]
