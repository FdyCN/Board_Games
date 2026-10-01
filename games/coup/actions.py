"""
政变疑云（Coup）动作定义。

动作 = 固定槽位上的「语义标签」，槽位布局见 game.py 与 constants.py。
不同 phase 下合法动作集合不同，`action_to_index` 依据 action.kind 映射到稳定槽位。

kind 全集：
    income, foreign_aid, coup, tax, assassinate, steal, exchange   （ACTION 阶段）
    pass, challenge                                                 （CHALLENGE / BLOCK_CHALLENGE）
    pass_block, block                                               （BLOCK）
    reveal                                                          （REVEAL，掉血时选翻哪张）
    exchange_keep                                                   （EXCHANGE，抽 2 后选保留哪 2 张）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoupAction:
    """Coup 动作对象。

    Attributes:
        kind: 动作类别（见模块 docstring）。
        target: 目标玩家的「相对编号」（1..n-1；-1 表示无目标）。
        role: 反制/宣称的角色（block 时有效；其余为 -1）。
        reveal_slot: 扣血时翻开哪张手牌（0/1；其余为 -1）。
        keep: 换牌时「保留的 2 张」在原 4 张中的索引组合（元组；其余为空）。
    """

    kind: str
    target: int = -1
    role: int = -1
    reveal_slot: int = -1
    keep: tuple = ()

    def __str__(self) -> str:
        if self.kind == "block":
            from games.coup.constants import ROLE_NAMES
            return f"反制(宣{ROLE_NAMES.get(self.role, self.role)})"
        if self.kind == "reveal":
            return f"翻开手牌槽{self.reveal_slot}"
        if self.kind == "exchange_keep":
            return f"保留{self.keep}"
        if self.target >= 0:
            return f"{self.kind} -> 相对{self.target}"
        return self.kind


__all__ = ["CoupAction"]
