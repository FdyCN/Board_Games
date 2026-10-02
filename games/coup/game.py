"""
政变疑云（Coup）游戏引擎。

隐藏身份 + 说谎游戏：一「回合」由多条决策链组成（宣告行动 → 质疑 → 反制 →
质疑反制 → 扣血翻牌 → 换牌），这里把每个决策点映射成一次 `step()`，当前玩家
随 phase 在决策者之间跳转。设计详见 docs/coup_design.md。

动作空间（固定槽位，共 3n + 16 个）：
    0           income（收入 +1 币）
    1           foreign_aid（外援 +2 币，可被公爵反制）
    2           tax（征税，宣称公爵，+3 币）
    3           exchange（换牌，宣称大使，抽 2 留 2）
    4 ..        coup 目标（n-1 个，付 7 币令目标掉血）
    ...         assassinate 目标（n-1 个，付 3 币令目标掉血，被女伯爵反制）
    ...         steal 目标（n-1 个，偷 2 币，被队长/大使反制）
    base+0      pass（质疑阶段放弃）
    base+1      challenge（质疑）
    base+2      pass_block（不反制）
    base+3..6   block（反制，宣称女伯爵/公爵/队长/大使）
    base+7..8   reveal（扣血时翻开手牌槽 0/1）
    base+9..14  exchange_keep（换牌保留 2 张的 6 组合）

规则约定（简化，详见 docs/coup_design.md §1.7 与 RULES.md）：
- 刺杀/政变的费用在「行动真正结算」时扣除（被质疑/反制取消则不扣）。
- 扣血 = 翻开并弃掉一张手牌；「质疑失败后洗回重抽」是证明后换牌，不掉血。
"""

from __future__ import annotations

import random
from typing import Any

from core.exceptions import IllegalActionError
from core.game_interface import GameInterface
from games.registry import register_game

from games.coup.actions import CoupAction
from games.coup.constants import (
    DUKE, ASSASSIN, CAPTAIN, AMBASSADOR, CONTESSA,
    NUM_ROLES,
    INITIAL_COINS, INCOME_GAIN, FOREIGN_AID_GAIN, TAX_GAIN, STEAL_GAIN,
    ASSASSINATE_COST, COUP_COST, FORCED_COUP_COINS,
    PHASE_ACTION, PHASE_CHALLENGE, PHASE_BLOCK, PHASE_BLOCK_CHALLENGE,
    PHASE_REVEAL, PHASE_EXCHANGE,
    EV_INCOME, EV_FOREIGN_AID, EV_COUP, EV_CLAIM, EV_CHALLENGE,
    EV_BLOCK, EV_REVEAL, EV_SWAP, EV_EXCHANGE,
    CLAIM_ROLE_OF_ACTION, BLOCKABLE, EXCHANGE_KEEP_COMBOS, make_deck, ROLE_NAMES,
)
from games.coup.encoder import CoupEncoder
from games.coup.state import CoupState

# block 槽位里 4 个反制角色的顺序（用于 action_to_index/index_to_action）
BLOCK_ROLE_ORDER = [CONTESSA, DUKE, CAPTAIN, AMBASSADOR]

_EVENT_TYPE_OF_KIND = {
    "income": EV_INCOME,
    "foreign_aid": EV_FOREIGN_AID,
    "coup": EV_COUP,
    "tax": EV_CLAIM,
    "assassinate": EV_CLAIM,
    "steal": EV_CLAIM,
    "exchange": EV_CLAIM,
}


@register_game("coup")
class CoupGame(GameInterface):
    """政变疑云（Coup）游戏（2-6 人，建议 3-4 人）。"""

    DEFAULT_REWARDS = {
        "win": 1.0,
        "lose_influence": -0.4,   # 掉血（自己扣）
        "challenge_success": 0.3,  # 揭穿谎言
        "challenge_fail": -0.2,    # 质疑失败
        "decl_income": 0.01,
        "decl_foreign_aid": 0.02,
        "decl_tax": 0.03,
        "decl_steal": 0.02,
        "decl_assassinate": 0.2,
        "decl_coup": 0.15,
        "step_penalty": -0.01,
    }

    def __init__(
        self,
        num_players: int = 3,
        seed: int | None = None,
        reward_config: dict | None = None,
    ):
        if not 2 <= num_players <= 6:
            raise ValueError(f"玩家数量必须在 2-6 之间，当前: {num_players}")
        self._num_players = num_players
        self._seed = seed
        self._rng = random.Random(seed)
        self._rewards = dict(self.DEFAULT_REWARDS)
        if reward_config:
            self._rewards.update(reward_config)
        self._encoder = CoupEncoder(num_players)
        self._max_turns = 400
        self._state: CoupState | None = None

    # ===== 动作空间 =====

    @property
    def _base(self) -> int:
        """基础槽位之后的偏移起点。"""
        return 3 * self._num_players + 1

    def reset(self) -> CoupState:
        n = self._num_players
        deck = make_deck()
        self._rng.shuffle(deck)
        hands = [[deck.pop(), deck.pop()] for _ in range(n)]
        state = CoupState(
            num_players=n,
            hands=hands,
            coins=[INITIAL_COINS] * n,
            revealed=[[] for _ in range(n)],
            deck=deck,
            alive=[True] * n,
            phase=PHASE_ACTION,
            current_player=0,
            turn_player=0,
        )
        self._state = state
        return state

    def step(self, action: CoupAction) -> tuple[CoupState, list[float], bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("游戏未初始化，请先调用 reset()")
        if self._state.game_over:
            raise RuntimeError("游戏已结束")

        legal = self.get_legal_actions(self._state)
        if action not in legal:
            raise IllegalActionError(action)

        state = self._state.clone()
        n = state.num_players
        p = state.current_player
        dense = self._rewards.get("step_penalty", 0.0)
        info: dict[str, Any] = {"player_id": p, "phase": state.phase, "action": action.kind}

        if state.phase == PHASE_ACTION:
            dense += self._do_action_phase(state, action, p)
        elif state.phase == PHASE_CHALLENGE:
            dense += self._do_challenge_phase(state, action, p)
        elif state.phase == PHASE_BLOCK:
            dense += self._do_block_phase(state, action, p)
        elif state.phase == PHASE_BLOCK_CHALLENGE:
            dense += self._do_block_challenge_phase(state, action, p)
        elif state.phase == PHASE_REVEAL:
            dense += self._do_reveal_phase(state, action, p)
        elif state.phase == PHASE_EXCHANGE:
            dense += self._do_exchange_phase(state, action, p)
        else:
            raise ValueError(f"未知阶段: {state.phase}")

        state.turn_number += 1
        if not state.game_over and state.turn_number >= self._max_turns:
            state.game_over = True
            state.winner = self._winner_by_influence(state)

        rewards = [0.0] * n
        rewards[p] = dense
        done = state.game_over
        info["dense_reward"] = dense
        info["winner"] = state.winner if done else -1

        self._state = state
        return state, rewards, done, info

    # ===== 各阶段处理 =====

    def _do_action_phase(self, state: CoupState, action: CoupAction, p: int) -> float:
        kind = action.kind
        abs_target = (p + action.target) % state.num_players if action.target >= 0 else -1
        claim_role = CLAIM_ROLE_OF_ACTION.get(kind, -1)
        self._record_event(state, p, _EVENT_TYPE_OF_KIND[kind], claim_role)
        state.log.append({
            "type": "claim", "player": p, "action": kind,
            "target": abs_target, "role": claim_role,
        })

        state.pending_kind = kind
        state.pending_target = abs_target
        state.pending_role = claim_role

        decl_reward = self._rewards.get(f"decl_{kind}", 0.0)

        if kind in ("tax", "assassinate", "steal", "exchange"):
            # 宣称角色 → 进入质疑阶段
            state.phase = PHASE_CHALLENGE
            state.challenge_seat = self._next_candidate(state, p, p)
            state.challenge_start = state.challenge_seat
            state.current_player = state.challenge_seat
            return decl_reward
        if kind == "foreign_aid":
            # 可被反制 → 进入反制阶段
            state.phase = PHASE_BLOCK
            state.block_seat = self._next_candidate(state, p, p)
            state.block_start = state.block_seat
            state.current_player = state.block_seat
            return decl_reward
        # income / coup：直接结算
        self._execute_action(state, kind, abs_target)
        return decl_reward

    def _do_challenge_phase(self, state: CoupState, action: CoupAction, p: int) -> float:
        if action.kind == "pass":
            nxt = self._next_candidate(state, p, state.turn_player)
            if nxt < 0 or nxt == state.challenge_start:
                self._proceed_action(state)
                return 0.0
            state.challenge_seat = nxt
            state.current_player = nxt
            return 0.0

        # challenge
        self._record_event(state, p, EV_CHALLENGE)
        state.log.append({"type": "challenge", "player": p, "challenged": state.turn_player})
        claimed = state.pending_role
        if claimed in state.hands[state.turn_player]:
            # 质疑失败：宣称者真有该角色 → 洗回重抽，质疑者掉血，行动继续
            dense = self._rewards.get("challenge_fail", 0.0)
            self._swap_card(state, state.turn_player, claimed)
            self._lose_influence(state, p, "block")
            return dense
        # 质疑成功：宣称者说谎 → 宣称者掉血，行动取消
        dense = self._rewards.get("challenge_success", 0.0)
        self._lose_influence(state, state.turn_player, "end_turn")
        return dense

    def _do_block_phase(self, state: CoupState, action: CoupAction, p: int) -> float:
        if action.kind == "pass_block":
            if state.pending_kind != "foreign_aid":
                # 刺杀/偷窃只有目标能反制，目标已 pass → 无反制
                self._execute_action(state, state.pending_kind, state.pending_target)
                return 0.0
            nxt = self._next_candidate(state, p, state.turn_player)
            if nxt < 0 or nxt == state.block_start:
                self._execute_action(state, state.pending_kind, state.pending_target)
                return 0.0
            state.block_seat = nxt
            state.current_player = nxt
            return 0.0

        # block（宣称反制角色，可说谎）
        role = action.role
        self._record_event(state, p, EV_BLOCK, role)
        state.log.append({"type": "block", "player": p, "role": role})
        state.block_role = role
        state.block_seat = p
        state.phase = PHASE_BLOCK_CHALLENGE
        state.challenge_seat = self._next_candidate(state, p, p)
        state.challenge_start = state.challenge_seat
        state.current_player = state.challenge_seat
        return 0.0

    def _do_block_challenge_phase(self, state: CoupState, action: CoupAction, p: int) -> float:
        if action.kind == "pass":
            nxt = self._next_candidate(state, p, state.block_seat)
            if nxt < 0 or nxt == state.challenge_start:
                # 无人质疑反制 → 反制成立，行动取消
                self._end_turn(state)
                return 0.0
            state.challenge_seat = nxt
            state.current_player = nxt
            return 0.0

        self._record_event(state, p, EV_CHALLENGE)
        state.log.append({"type": "block_challenge", "player": p, "blocker": state.block_seat})
        role = state.block_role
        if role in state.hands[state.block_seat]:
            # 反制者真有该角色 → 质疑失败，质疑者掉血，反制成立 → 行动取消
            dense = self._rewards.get("challenge_fail", 0.0)
            self._swap_card(state, state.block_seat, role)
            self._lose_influence(state, p, "end_turn")
            return dense
        # 反制者说谎 → 质疑成功，反制者掉血，反制取消 → 行动执行
        dense = self._rewards.get("challenge_success", 0.0)
        self._lose_influence(state, state.block_seat, "execute")
        return dense

    def _do_reveal_phase(self, state: CoupState, action: CoupAction, p: int) -> float:
        hand = state.hands[p]
        slot = action.reveal_slot if 0 <= action.reveal_slot < len(hand) else 0
        card = hand.pop(slot)
        state.revealed[p].append(card)
        self._record_event(state, p, EV_REVEAL, card)
        state.log.append({"type": "reveal", "player": p, "role": card})
        if not hand:
            state.alive[p] = False
            self._check_game_over(state)
        if not state.game_over:
            self._resume_after_reveal(state)
        return self._rewards.get("lose_influence", 0.0)

    def _do_exchange_phase(self, state: CoupState, action: CoupAction, p: int) -> float:
        hand4 = state.exchange_hand
        keep = action.keep
        kept = [hand4[i] for i in keep]
        returned = [hand4[i] for i in range(len(hand4)) if i not in keep and hand4[i] >= 0]
        state.hands[p] = kept
        state.deck = returned + state.deck
        self._rng.shuffle(state.deck)
        state.exchange_hand = []
        self._record_event(state, p, EV_EXCHANGE)
        state.log.append({"type": "exchange", "player": p})
        self._end_turn(state)
        return 0.0

    # ===== 结算辅助 =====

    def _proceed_action(self, state: CoupState) -> None:
        """行动未被质疑（或质疑失败）后，继续：可被反制则进 BLOCK，否则结算。"""
        if state.pending_kind in BLOCKABLE:
            state.phase = PHASE_BLOCK
            if state.pending_kind == "foreign_aid":
                state.block_seat = self._next_candidate(state, state.turn_player, state.turn_player)
            else:
                state.block_seat = state.pending_target
            state.current_player = state.block_seat
        else:
            self._execute_action(state, state.pending_kind, state.pending_target)

    def _execute_action(self, state: CoupState, kind: str, target: int) -> None:
        src = state.turn_player
        if kind == "income":
            state.coins[src] += INCOME_GAIN
            self._end_turn(state)
        elif kind == "foreign_aid":
            state.coins[src] += FOREIGN_AID_GAIN
            self._end_turn(state)
        elif kind == "tax":
            state.coins[src] += TAX_GAIN
            self._end_turn(state)
        elif kind == "steal":
            steal = min(STEAL_GAIN, state.coins[target])
            state.coins[src] += steal
            state.coins[target] -= steal
            self._end_turn(state)
        elif kind == "assassinate":
            state.coins[src] -= ASSASSINATE_COST
            self._lose_influence(state, target, "end_turn")
        elif kind == "coup":
            state.coins[src] -= COUP_COST
            self._lose_influence(state, target, "end_turn")
        elif kind == "exchange":
            self._start_exchange(state, src)

    def _lose_influence(self, state: CoupState, loser: int, resume: str) -> None:
        """令 loser 掉 1 影响力（扣血翻开一张）。若两张手牌则暂停等待选择，否则直接结算。"""
        state.reveal_resume = resume
        if len(state.hands[loser]) <= 1:
            if state.hands[loser]:
                card = state.hands[loser].pop()
                state.revealed[loser].append(card)
                self._record_event(state, loser, EV_REVEAL, card)
                state.log.append({"type": "reveal", "player": loser, "role": card})
            state.alive[loser] = False
            self._check_game_over(state)
            if not state.game_over:
                self._resume_after_reveal(state)
        else:
            state.phase = PHASE_REVEAL
            state.current_player = loser

    def _resume_after_reveal(self, state: CoupState) -> None:
        resume = state.reveal_resume
        state.reveal_resume = ""
        if resume == "block":
            self._proceed_action(state)
        elif resume == "execute":
            self._execute_action(state, state.pending_kind, state.pending_target)
        else:  # "end_turn" 或兜底
            self._end_turn(state)

    def _end_turn(self, state: CoupState) -> None:
        state.pending_kind = ""
        state.pending_target = -1
        state.pending_role = -1
        state.challenge_seat = -1
        state.challenge_start = -1
        state.block_seat = -1
        state.block_start = -1
        state.block_role = -1
        state.reveal_resume = ""
        state.exchange_hand = []
        nxt = self._next_candidate(state, state.turn_player, state.turn_player)
        if nxt < 0:
            self._check_game_over(state)
            return
        state.turn_player = nxt
        state.phase = PHASE_ACTION
        state.current_player = nxt

    def _start_exchange(self, state: CoupState, p: int) -> None:
        hand = list(state.hands[p])
        drawn = []
        while len(drawn) < 2 and state.deck:
            card = state.deck.pop()
            if card >= 0:  # 防御：跳过可能的 -1 占位牌
                drawn.append(card)
        real = hand + drawn
        if len(real) < 2:
            # 手牌不足 2 张且牌堆抽不够 → 换牌退化为无操作（保留现有牌）
            state.hands[p] = real
            self._record_event(state, p, EV_EXCHANGE)
            state.log.append({"type": "exchange", "player": p})
            self._end_turn(state)
            return
        hand4 = real + [-1] * (4 - len(real))  # 占位（牌堆不足）
        state.exchange_hand = hand4
        state.phase = PHASE_EXCHANGE
        state.current_player = p

    def _swap_card(self, state: CoupState, player: int, role: int) -> None:
        """质疑失败后：证明持有 role，洗回牌堆并重抽一张（不掉血）。"""
        hand = state.hands[player]
        if role in hand:
            hand.remove(role)
            state.deck.append(role)
            self._rng.shuffle(state.deck)
            if state.deck:
                hand.append(state.deck.pop())
        self._record_event(state, player, EV_SWAP, role)
        state.log.append({"type": "swap", "player": player, "role": role})

    def _check_game_over(self, state: CoupState) -> None:
        alive = state.alive_players()
        if len(alive) <= 1:
            state.game_over = True
            state.winner = alive[0] if alive else -1

    def _winner_by_influence(self, state: CoupState) -> int:
        alive = [i for i in range(state.num_players) if state.alive[i]]
        if not alive:
            return -1
        return max(alive, key=lambda i: (len(state.hands[i]), state.coins[i], -i))

    def _next_candidate(self, state: CoupState, start_after: int, skip: int) -> int:
        """从 start_after 顺时针找下一个存活的、且不等于 skip 的玩家；找不到返回 -1。"""
        n = state.num_players
        for step in range(1, n + 1):
            cand = (start_after + step) % n
            if cand == skip or not state.alive[cand]:
                continue
            return cand
        return -1

    def _record_event(self, state: CoupState, player: int, ev_type: int, role: int = -1) -> None:
        state.events.append((player, ev_type, role))

    # ===== 合法动作 =====

    def get_legal_actions(self, state: CoupState | None = None) -> list[CoupAction]:
        state = state or self._state
        if state is None or state.game_over:
            return []
        p = state.current_player
        phase = state.phase
        if phase == PHASE_ACTION:
            legal = self._legal_actions_action(state, p)
        elif phase in (PHASE_CHALLENGE, PHASE_BLOCK_CHALLENGE):
            legal = [CoupAction("pass"), CoupAction("challenge")]
        elif phase == PHASE_BLOCK:
            legal = self._legal_actions_block(state)
        elif phase == PHASE_REVEAL:
            legal = [CoupAction("reveal", reveal_slot=i) for i in range(len(state.hands[p]))]
        elif phase == PHASE_EXCHANGE:
            legal = self._legal_actions_exchange(state)
        else:
            legal = []
        if not legal:
            raise RuntimeError(
                f"非终局无合法动作: phase={phase} cur={p} turn={state.turn_player} "
                f"hands={state.hands} coins={state.coins} alive={state.alive} "
                f"pending={state.pending_kind}/{state.pending_target} "
                f"exchange_hand={state.exchange_hand} deck={len(state.deck)} "
                f"reveal_resume={state.reveal_resume}"
            )
        return legal

    def _legal_actions_action(self, state: CoupState, p: int) -> list[CoupAction]:
        n = state.num_players
        opponents = [r for r in range(1, n) if state.alive[(p + r) % n]]

        if state.coins[p] >= FORCED_COUP_COINS:
            return [CoupAction("coup", target=r) for r in opponents]

        actions = [CoupAction("income"), CoupAction("foreign_aid")]
        actions.append(CoupAction("tax"))
        actions.append(CoupAction("exchange"))
        if state.coins[p] >= COUP_COST:
            actions.extend(CoupAction("coup", target=r) for r in opponents)
        if state.coins[p] >= ASSASSINATE_COST:
            actions.extend(CoupAction("assassinate", target=r) for r in opponents)
        actions.extend(CoupAction("steal", target=r) for r in opponents)
        return actions

    def _legal_actions_block(self, state: CoupState) -> list[CoupAction]:
        actions = [CoupAction("pass_block")]
        for role in BLOCKABLE.get(state.pending_kind, []):
            actions.append(CoupAction("block", role=role))
        return actions

    def _legal_actions_exchange(self, state: CoupState) -> list[CoupAction]:
        actions = []
        hand = state.exchange_hand
        for combo in EXCHANGE_KEEP_COMBOS:
            if all(hand[i] >= 0 for i in combo):
                actions.append(CoupAction("exchange_keep", keep=combo))
        return actions

    # ===== 状态查询 =====

    def get_current_player(self, state: CoupState | None = None) -> int:
        state = state or self._state
        if state is None:
            return 0
        return state.current_player

    def is_terminal(self, state: CoupState | None = None) -> bool:
        state = state or self._state
        return bool(state is not None and state.game_over)

    def get_final_rewards(self, state: CoupState | None = None) -> list[float]:
        state = state or self._state
        if state is None:
            return [0.0] * self._num_players
        n = state.num_players
        return [1.0 if i == state.winner else 0.0 for i in range(n)]

    def get_winner(self, state: CoupState | None = None) -> int:
        state = state or self._state
        return state.winner if state is not None else -1

    # ===== 观察与动作编码 =====

    def state_to_observation(self, state: CoupState, player_id: int):
        return self._encoder.encode(state, player_id)

    def action_to_index(self, action: CoupAction, state: CoupState | None = None) -> int:
        n = self._num_players
        base = 3 * n + 1
        k = action.kind
        if k == "income":
            return 0
        if k == "foreign_aid":
            return 1
        if k == "tax":
            return 2
        if k == "exchange":
            return 3
        if k == "coup":
            return 4 + (action.target - 1)
        if k == "assassinate":
            return 4 + (n - 1) + (action.target - 1)
        if k == "steal":
            return 4 + 2 * (n - 1) + (action.target - 1)
        if k == "pass":
            return base
        if k == "challenge":
            return base + 1
        if k == "pass_block":
            return base + 2
        if k == "block":
            return base + 3 + BLOCK_ROLE_ORDER.index(action.role)
        if k == "reveal":
            return base + 7 + action.reveal_slot
        if k == "exchange_keep":
            return base + 9 + EXCHANGE_KEEP_COMBOS.index(action.keep)
        raise ValueError(f"未知动作 kind: {k}")

    def index_to_action(self, index: int, state: CoupState | None = None) -> CoupAction:
        n = self._num_players
        base = 3 * n + 1
        if index < 0 or index >= self.action_space_size:
            raise IndexError(f"动作索引 {index} 超出范围 [0, {self.action_space_size})")
        if index == 0:
            return CoupAction("income")
        if index == 1:
            return CoupAction("foreign_aid")
        if index == 2:
            return CoupAction("tax")
        if index == 3:
            return CoupAction("exchange")
        if index < 4 + (n - 1):
            return CoupAction("coup", target=index - 4 + 1)
        if index < 4 + 2 * (n - 1):
            return CoupAction("assassinate", target=index - (4 + (n - 1)) + 1)
        if index < 4 + 3 * (n - 1):
            return CoupAction("steal", target=index - (4 + 2 * (n - 1)) + 1)
        if index == base:
            return CoupAction("pass")
        if index == base + 1:
            return CoupAction("challenge")
        if index == base + 2:
            return CoupAction("pass_block")
        if index < base + 7:
            return CoupAction("block", role=BLOCK_ROLE_ORDER[index - (base + 3)])
        if index < base + 9:
            return CoupAction("reveal", reveal_slot=index - (base + 7))
        return CoupAction("exchange_keep", keep=EXCHANGE_KEEP_COMBOS[index - (base + 9)])

    # ===== 属性 =====

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (self._encoder.observation_dim,)

    @property
    def action_space_size(self) -> int:
        return 3 * self._num_players + 16

    @property
    def auxiliary_shape(self) -> tuple[int, ...]:
        """辅助任务：预测 (n-1) 个相对对手的第一张手牌角色（5 类）。"""
        return ((self._num_players - 1) * NUM_ROLES,)

    @property
    def encoder_params(self) -> dict:
        return self._encoder.model_encoder_params

    def get_auxiliary_labels(self, state: CoupState, player_id: int):
        import numpy as np
        n = state.num_players
        labels = np.full(n - 1, -1, dtype=np.int64)
        for r in range(1, n):
            other = (player_id + r) % n
            if state.alive[other] and state.hands[other]:
                labels[r - 1] = sorted(state.hands[other])[0]
        return labels

    def game_kwargs(self) -> dict:
        kwargs = super().game_kwargs()
        if self._seed is not None:
            kwargs["seed"] = self._seed
        kwargs["reward_config"] = dict(self._rewards)
        return kwargs

    # ===== 渲染 =====

    def render(self, state: CoupState | None = None, mode: str = "human") -> str | None:
        state = state or self._state
        if state is None:
            return None
        n = state.num_players
        lines = [f"=== 政变疑云 (金币: {state.coins} | 牌堆 {len(state.deck)} | 阶段 {state.phase}) ==="]
        for i in range(n):
            mark = "*" if i == state.current_player else " "
            status = [] if state.alive[i] else ["出局"]
            hand = [ROLE_NAMES.get(c, c) for c in state.hands[i]]
            rev = [ROLE_NAMES.get(c, c) for c in state.revealed[i]]
            lines.append(f"{mark} P{i}: 手牌={hand} 弃牌={rev} 金币={state.coins[i]} {'/'.join(status)}")
        text = "\n".join(lines)
        if mode == "human":
            print(text)
            return None
        return text


__all__ = ["CoupGame"]
