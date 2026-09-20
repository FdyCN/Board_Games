"""
情书（Love Letter）动作定义。

动作 = 「打出某张卡」+ 可选的「目标玩家」+（卫兵）可选的「猜测牌值」。

使用固定动作空间：动作被映射到语义稳定的槽位，槽位布局见
LoveLetterGame._build_action_slots 的注释。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlayCardAction:
    """打出一张卡牌的动作。

    Attributes:
        card: 要打出的卡牌值（1-8）。
        target: 目标玩家 ID（-1 表示无目标）。
        guess: 卫兵的猜测卡牌值（2-8；-1 表示非卫兵或未指定）。
    """

    card: int
    target: int = -1
    guess: int = -1

    def __str__(self) -> str:
        if self.card == 1:
            return f"卫兵(猜{self.guess}) -> 玩家{self.target}"
        if self.card in (2, 3, 6):
            return f"{_NAME[self.card]} -> 玩家{self.target}"
        if self.card == 5:
            return f"王子 -> 玩家{self.target}"
        return _NAME[self.card]


_NAME = {
    1: "卫兵",
    2: "神父",
    3: "男爵",
    4: "侍女",
    5: "王子",
    6: "国王",
    7: "女伯爵",
    8: "公主",
}


__all__ = ["PlayCardAction"]
