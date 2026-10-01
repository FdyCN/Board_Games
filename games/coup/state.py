"""
政变疑云（Coup）游戏状态。

step() 基于 clone() 后修改，不原地修改传入状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from games.coup.constants import PHASE_ACTION


@dataclass
class CoupState:
    """一局完整 Coup 游戏的状态。

    Attributes:
        num_players: 玩家数量
        hands: 每个玩家的隐藏手牌（角色 id，0-2 张）
        coins: 每个玩家的金币数
        revealed: 每个玩家「扣血翻开弃掉」的公开牌（角色 id）
        deck: 宫廷牌堆（列表尾部 = 牌堆顶）
        alive: 每个玩家是否存活（有 >=1 张手牌）
        phase: 当前阶段（见 constants.PHASE_*）
        current_player: 当前决策者
        turn_player: 本回合的回合玩家（发起行动者）
        pending_kind: 待决行动 kind（"" 表示无）
        pending_target: 待决行动的绝对目标（-1 无）
        pending_role: 待决行动的宣称角色（-1 无）
        challenge_seat: 下一个待质询的玩家（-1 无）
        block_seat: 下一个待反制决策的玩家（-1 无）
        block_role: 已宣告的反制角色（-1 无）
        reveal_resume: 扣血结算后要恢复的流程（"" / "end_turn" / "block" / "execute"）
        exchange_hand: 换牌时临时 4 张（含抽到的），结算后清空
        events: 事件时间线 [(player, event_type, role), ...]，供 GRU 序列编码
        log: 全对局富事件日志（前端展示）
        turn_number: 全局决策步数（回合上限）
        game_over: 是否结束
        winner: 胜者（-1 未定）
    """

    num_players: int
    hands: list[list[int]]
    coins: list[int]
    revealed: list[list[int]]
    deck: list[int]
    alive: list[bool]
    phase: int = PHASE_ACTION
    current_player: int = 0
    turn_player: int = 0
    pending_kind: str = ""
    pending_target: int = -1
    pending_role: int = -1
    challenge_seat: int = -1
    challenge_start: int = -1   # 质疑迭代起点（用于绕回检测）
    block_seat: int = -1
    block_start: int = -1       # 反制迭代起点（用于绕回检测）
    block_role: int = -1
    reveal_resume: str = ""
    exchange_hand: list[int] = field(default_factory=list)
    events: list[tuple[int, int, int]] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    turn_number: int = 0
    game_over: bool = False
    winner: int = -1

    def clone(self) -> "CoupState":
        return CoupState(
            num_players=self.num_players,
            hands=[list(h) for h in self.hands],
            coins=list(self.coins),
            revealed=[list(r) for r in self.revealed],
            deck=list(self.deck),
            alive=list(self.alive),
            phase=self.phase,
            current_player=self.current_player,
            turn_player=self.turn_player,
            pending_kind=self.pending_kind,
            pending_target=self.pending_target,
            pending_role=self.pending_role,
            challenge_seat=self.challenge_seat,
            challenge_start=self.challenge_start,
            block_seat=self.block_seat,
            block_start=self.block_start,
            block_role=self.block_role,
            reveal_resume=self.reveal_resume,
            exchange_hand=list(self.exchange_hand),
            events=list(self.events),
            log=[dict(e) for e in self.log],
            turn_number=self.turn_number,
            game_over=self.game_over,
            winner=self.winner,
        )

    def alive_players(self) -> list[int]:
        """存活玩家 ID 列表。"""
        return [i for i in range(self.num_players) if self.alive[i]]


__all__ = ["CoupState"]
