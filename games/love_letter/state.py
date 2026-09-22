"""
情书（Love Letter）游戏状态。

状态对象不可变语义：step() 基于 clone() 后修改，不原地修改传入状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoveLetterState:
    """一局完整情书游戏的状态（包含多轮）。

    Attributes:
        num_players: 玩家数量
        tokens: 每个玩家已获得的爱心标记数
        target_tokens: 获胜所需爱心标记数
        deck: 当前轮的抽牌堆（列表尾部为牌堆顶）
        hands: 每个玩家的手牌（列表，长度 0/1/2；当前玩家决策时为 2 张）
        discards: 每个玩家的弃牌堆（面朝上，公开）
        eliminated: 每个玩家是否在本轮出局
        protected: 每个玩家是否受侍女保护
        current_player: 当前行动玩家
        round_number: 当前轮次（从 1 开始）
        turn_number: 全局行动计数（用于回合上限）
        round_over: 本轮是否结束
        game_over: 整局是否结束
        winner: 整局胜者（-1 表示尚未结束）
    """

    num_players: int
    tokens: list[int]
    target_tokens: int
    deck: list[int]
    hands: list[list[int]]
    discards: list[list[int]]
    eliminated: list[bool]
    protected: list[bool]
    known: list[list[int | None]] = field(default_factory=list)  # known[p][q] = p 已知 q 的手牌值，None=未知
    events: list[tuple[int, int]] = field(default_factory=list)  # 本轮按出牌顺序的 (玩家, 牌值) 时间线
    current_player: int = 0
    round_number: int = 1
    turn_number: int = 0
    round_over: bool = False
    game_over: bool = False
    winner: int = -1

    def clone(self) -> "LoveLetterState":
        """深拷贝（卡牌是 int，列表浅拷贝即可）。"""
        return LoveLetterState(
            num_players=self.num_players,
            tokens=list(self.tokens),
            target_tokens=self.target_tokens,
            deck=list(self.deck),
            hands=[list(h) for h in self.hands],
            discards=[list(d) for d in self.discards],
            eliminated=list(self.eliminated),
            protected=list(self.protected),
            known=[list(k) for k in self.known],
            events=list(self.events),
            current_player=self.current_player,
            round_number=self.round_number,
            turn_number=self.turn_number,
            round_over=self.round_over,
            game_over=self.game_over,
            winner=self.winner,
        )

    def alive_players(self) -> list[int]:
        """本轮尚未出局的玩家 ID 列表。"""
        return [i for i in range(self.num_players) if not self.eliminated[i]]


__all__ = ["LoveLetterState"]
