"""
情书（Love Letter）游戏引擎。

规则要点（经典 Tempest 版，2-4 人）：
- 16 张牌（8 种：卫兵×5、神父×2、男爵×2、侍女×2、王子×2、国王×1、女伯爵×1、公主×1）。
- 每轮开始时移除 1 张牌（面朝下），给每位玩家发 1 张手牌。
- 每回合：抽 1 张牌（手牌 2 张）→ 打出 1 张并执行效果。
- 轮结束条件：只剩 1 名玩家存活，或牌堆在回合开始时为空（比手牌大小）。
- 赢得一轮得 1 个爱心标记，先攒够目标标记数者赢得整局（3 人 = 5 个）。

动作空间（固定槽位，共 11n - 3 个；target 为「相对玩家编号」，0=自己）：
    0-2   无目标牌：侍女、女伯爵、公主
    3-5   神父（1 空打 + n-1 个相对目标）
    6-8   男爵（1 空打 + n-1 个相对目标）
    9-11  国王（1 空打 + n-1 个相对目标）
    12..  王子（n 个相对目标，0=自己）
    ..    卫兵（1 空打 + (n-1)×7 相对目标×猜测）
"""

from __future__ import annotations

import random
from typing import Any

from core.exceptions import IllegalActionError
from core.game_interface import GameInterface
from games.registry import register_game

from games.love_letter.actions import PlayCardAction
from games.love_letter.constants import (
    BARON,
    COUNTESS,
    GUARD,
    HANDMAID,
    KING,
    PRIEST,
    PRINCE,
    PRINCESS,
    TARGET_TOKENS,
    CARD_NAMES,
    make_deck,
)
from games.love_letter.encoder import LoveLetterEncoder
from games.love_letter.state import LoveLetterState

# 目标类卡（需要指定另一个玩家）：神父/男爵/国王
_TARGET_OTHER_CARDS = (PRIEST, BARON, KING)


@register_game("love_letter")
class LoveLetterGame(GameInterface):
    """情书游戏（整局到目标标记数，支持 2-4 人）。"""

    def __init__(
        self,
        num_players: int = 3,
        seed: int | None = None,
        reward_config: dict | None = None,
    ):
        if not 2 <= num_players <= 4:
            raise ValueError(f"玩家数量必须在 2-4 之间，当前: {num_players}")

        self._num_players = num_players
        self._seed = seed
        self._rng = random.Random(seed)
        self._target_tokens = TARGET_TOKENS[num_players]

        # 默认奖励（PPO 稠密奖励，可被配置覆盖）
        self._rewards = {
            "round_win": 0.5,          # 打牌直接赢下本轮
            "eliminate_opponent": 0.3, # 消灭一个对手（卫兵猜中/男爵/王子弃公主）
            "step_penalty": -0.01,     # 每步小惩罚，鼓励尽快结束
        }
        if reward_config:
            self._rewards.update(reward_config)

        # 固定动作空间（槽位 → (card, target, guess)）
        self._action_slots = self._build_action_slots(num_players)
        self._action_index = {slot: i for i, slot in enumerate(self._action_slots)}

        # 回合上限，防止异常死循环
        self._max_total_turns = 200

        self._encoder = LoveLetterEncoder(num_players)
        self._state: LoveLetterState | None = None

    # ===== 动作空间 =====

    @staticmethod
    def _build_action_slots(num_players: int) -> list[tuple[int, int, int]]:
        """构建固定动作槽位，返回 [(card, target, guess), ...]。

        布局（target 为「相对玩家编号」：0=自己，r=顺时针第 r 个玩家）：
            - 无目标牌（侍女/女伯爵/公主）：各 1 个槽位
            - 神父/男爵/国王：1 个「空打」槽位 + (n-1) 个相对目标槽位
            - 王子：n 个相对目标槽位（0=自己，永不空打）
            - 卫兵：1 个「空打」槽位 + (n-1)×7 相对目标×猜测槽位
        总计 = 11n - 3。
        """
        slots: list[tuple[int, int, int]] = []
        for c in (HANDMAID, COUNTESS, PRINCESS):
            slots.append((c, -1, -1))
        for c in _TARGET_OTHER_CARDS:
            slots.append((c, -1, -1))  # 空打（无有效目标时）
            for r in range(1, num_players):  # 相对对手
                slots.append((c, r, -1))
        for r in range(num_players):  # 王子：0=自己
            slots.append((PRINCE, r, -1))
        slots.append((GUARD, -1, -1))  # 空打
        for r in range(1, num_players):
            for g in range(2, 9):
                slots.append((GUARD, r, g))
        return slots

    @property
    def action_space_size(self) -> int:
        return len(self._action_slots)

    # ===== 核心接口 =====

    def reset(self) -> LoveLetterState:
        n = self._num_players
        state = LoveLetterState(
            num_players=n,
            tokens=[0] * n,
            target_tokens=self._target_tokens,
            deck=[],
            hands=[[] for _ in range(n)],
            discards=[[] for _ in range(n)],
            eliminated=[False] * n,
            protected=[False] * n,
            current_player=0,
            round_number=1,
            turn_number=0,
            round_over=False,
            game_over=False,
            winner=-1,
        )
        self._start_round(state, 0)
        self._state = state
        return state

    def _start_round(self, state: LoveLetterState, start_player: int) -> None:
        """开始新一轮：洗牌、移除 1 张、发牌、start_player 抽第二张。"""
        deck = make_deck()
        self._rng.shuffle(deck)
        deck.pop()  # 移除 1 张（面朝下，无人可见）
        state.deck = deck
        state.hands = [[deck.pop()] for _ in range(state.num_players)]
        state.discards = [[] for _ in range(state.num_players)]
        state.eliminated = [False] * state.num_players
        state.protected = [False] * state.num_players
        state.known = [[None] * state.num_players for _ in range(state.num_players)]
        state.events = []
        state.current_player = start_player
        state.round_over = False
        # 起始玩家抽第二张牌（手牌 2 张）
        state.hands[start_player].append(deck.pop())

    def step(self, action: PlayCardAction) -> tuple[LoveLetterState, list[float], bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("游戏未初始化，请先调用 reset()")
        if self._state.game_over:
            raise RuntimeError("游戏已结束")

        legal_actions = self.get_legal_actions(self._state)
        if action not in legal_actions:
            raise IllegalActionError(action)

        state = self._state.clone()
        n = state.num_players
        p = state.current_player

        dense_reward = self._rewards.get("step_penalty", 0.0)
        info = {
            "player_id": p,
            "card": action.card,
            "target": action.target,
            "guess": action.guess,
        }

        # 1. 打牌并执行效果
        before_elim = list(state.eliminated)
        self._play_card(state, p, action)

        # 消灭对手奖励（卫兵猜中 / 男爵 / 王子弃公主）
        eliminated_opponents = [
            i for i in range(n)
            if i != p and not before_elim[i] and state.eliminated[i]
        ]
        if eliminated_opponents:
            dense_reward += self._rewards.get("eliminate_opponent", 0.0) * len(eliminated_opponents)

        # 2. 判定本轮结果
        alive = state.alive_players()
        round_winner = -1
        if len(alive) <= 1:
            # 只剩 1 人（或异常地 0 人）→ 本轮结束
            state.round_over = True
            round_winner = alive[0] if alive else -1
        else:
            # 推进到下一玩家（可能因牌堆空而结束本轮）
            round_winner = self._advance_player(state)

        # 3. 本轮结束 → 结算
        if state.round_over:
            if round_winner < 0:
                round_winner = self._round_winner_by_compare(state)
            self._resolve_round_end(state, round_winner)
            if round_winner == p:
                dense_reward += self._rewards.get("round_win", 0.0)

        # 4. 回合上限保护
        state.turn_number += 1
        if not state.game_over and state.turn_number >= self._max_total_turns:
            state.game_over = True
            state.round_over = True
            state.winner = self._winner_by_tokens(state)

        rewards = [0.0] * n
        rewards[p] = dense_reward
        done = state.game_over
        info["dense_reward"] = dense_reward
        info["round_over"] = state.round_over
        info["winner"] = state.winner if done else -1

        self._state = state
        return state, rewards, done, info

    def _discard(self, state: LoveLetterState, p: int, card: int) -> None:
        """把一张牌放入玩家 p 的弃牌堆，并记录进全局事件时间线。"""
        state.discards[p].append(card)
        state.events.append((p, card))

    def _play_card(self, state: LoveLetterState, p: int, action: PlayCardAction) -> None:
        """执行打牌效果（action 已通过合法性校验）。"""
        c = action.card
        hand = state.hands[p]
        if c not in hand:
            raise IllegalActionError(action, f"手牌中没有 {c}")
        hand.remove(c)
        self._discard(state, p, c)

        # 相对目标 → 绝对玩家；target < 0 表示「空打」，无效果
        n = state.num_players
        t = (p + action.target) % n if action.target >= 0 else -1

        # 玩家 p 打出了牌 c：若别人之前知道 p 手牌是 c，则 p 留的是新抽的牌（未知）
        for obs in range(n):
            if state.known[obs][p] == c:
                state.known[obs][p] = None

        if c == HANDMAID:
            state.protected[p] = True
        elif c == PRIEST:
            # p 看到 t 的手牌
            if t >= 0 and state.hands[t]:
                state.known[p][t] = state.hands[t][0]
        elif c == BARON:
            if t >= 0:
                # 双方互相看到对方手牌
                if state.hands[t]:
                    state.known[p][t] = state.hands[t][0]
                if state.hands[p]:
                    state.known[t][p] = state.hands[p][0]
                my_val = state.hands[p][0] if state.hands[p] else 0
                their_val = state.hands[t][0] if state.hands[t] else 0
                if my_val < their_val:
                    self._eliminate(state, p)
                elif their_val < my_val:
                    self._eliminate(state, t)
        elif c == PRINCE:
            # 王子必有有效目标（t >= 0）
            hand_t = state.hands[t]
            while hand_t:
                card = hand_t.pop()
                self._discard(state, t, card)
                if card == PRINCESS:
                    state.eliminated[t] = True
                    state.protected[t] = False
            # t 手牌公开弃掉并重抽 → 所有人对 t 的知识失效
            for obs in range(n):
                state.known[obs][t] = None
            if not state.eliminated[t] and state.deck:
                hand_t.append(state.deck.pop())
        elif c == KING:
            if t >= 0:
                # 双方看到交换前的手牌，然后交换
                p_card = state.hands[p][0] if state.hands[p] else None
                t_card = state.hands[t][0] if state.hands[t] else None
                state.hands[p], state.hands[t] = state.hands[t], state.hands[p]
                # 交换后：t 拿的是 p 的旧牌，p 拿的是 t 的旧牌
                state.known[p][t] = p_card
                state.known[t][p] = t_card
        elif c == GUARD:
            g = action.guess
            if t >= 0 and state.hands[t] and state.hands[t][0] == g:
                self._eliminate(state, t)
        elif c == PRINCESS:
            self._eliminate(state, p)

    def _eliminate(self, state: LoveLetterState, p: int) -> None:
        """玩家出局：揭示并弃掉手牌。"""
        state.eliminated[p] = True
        state.protected[p] = False
        hand = state.hands[p]
        while hand:
            self._discard(state, p, hand.pop())
        # 出局 → 手牌公开揭示，清除所有人对该玩家的私密知识
        for obs in range(state.num_players):
            state.known[obs][p] = None

    def _advance_player(self, state: LoveLetterState) -> int:
        """推进到下一个存活玩家；若牌堆空则本轮结束（返回轮胜者）。"""
        n = state.num_players
        nxt = state.current_player
        for _ in range(n):
            nxt = (nxt + 1) % n
            if not state.eliminated[nxt]:
                break
        state.current_player = nxt
        # 侍女保护在该玩家自己的回合开始时解除
        state.protected[nxt] = False

        if len(state.deck) == 0:
            state.round_over = True
            return self._round_winner_by_compare(state)
        state.hands[nxt].append(state.deck.pop())
        return -1

    def _round_winner_by_compare(self, state: LoveLetterState) -> int:
        """牌堆空时比手牌：值大者胜；平局比弃牌堆值总和。"""
        alive = state.alive_players()
        if not alive:
            return -1
        if len(alive) == 1:
            return alive[0]

        best_val = -1
        best: list[int] = []
        for i in alive:
            v = state.hands[i][0] if state.hands[i] else 0
            if v > best_val:
                best_val = v
                best = [i]
            elif v == best_val:
                best.append(i)
        if len(best) == 1:
            return best[0]

        best_sum = -1
        best2: list[int] = []
        for i in best:
            s = sum(state.discards[i])
            if s > best_sum:
                best_sum = s
                best2 = [i]
            elif s == best_sum:
                best2.append(i)
        return best2[0] if best2 else -1

    def _resolve_round_end(self, state: LoveLetterState, round_winner: int) -> None:
        """轮结束：发标记；若达到目标则整局结束，否则开下一轮。"""
        if round_winner >= 0:
            state.tokens[round_winner] += 1
            if state.tokens[round_winner] >= state.target_tokens:
                state.game_over = True
                state.winner = round_winner
        if not state.game_over:
            start = round_winner if round_winner >= 0 else state.current_player
            state.round_number += 1
            self._start_round(state, start)

    @staticmethod
    def _winner_by_tokens(state: LoveLetterState) -> int:
        """回合上限触发时按标记数定胜者（平局取最小编号）。"""
        m = max(state.tokens)
        return next(i for i, t in enumerate(state.tokens) if t == m)

    # ===== 合法动作 =====

    def get_legal_actions(self, state: LoveLetterState | None = None) -> list[PlayCardAction]:
        state = state or self._state
        if state is None or state.game_over or state.round_over:
            return []
        p = state.current_player
        hand = state.hands[p]
        if not hand:
            return []

        # 手上有女伯爵 + 国王/王子时，必须打出女伯爵
        if COUNTESS in hand and (KING in hand or PRINCE in hand):
            playable = [COUNTESS]
        else:
            playable = list(hand)

        actions: list[PlayCardAction] = []
        for c in playable:
            actions.extend(self._card_actions(state, p, c))
        return actions

    def _card_actions(self, state: LoveLetterState, p: int, c: int) -> list[PlayCardAction]:
        """为当前玩家的一张手牌 c 生成所有合法动作（target 为相对编号）。"""
        n = state.num_players
        if c in (HANDMAID, COUNTESS, PRINCESS):
            return [PlayCardAction(c)]

        if c in _TARGET_OTHER_CARDS:
            # 相对目标：1..n-1（绝对 = (p+r)%n），须存活且未被保护
            targets = [
                r for r in range(1, n)
                if not state.eliminated[(p + r) % n] and not state.protected[(p + r) % n]
            ]
            if not targets:
                return [PlayCardAction(c)]  # 空打：无有效目标
            return [PlayCardAction(c, r) for r in targets]

        if c == PRINCE:
            # 相对目标：0=自己，1..n-1=对手；自己被保护不影响自己的牌
            targets = [
                r for r in range(n)
                if not state.eliminated[(p + r) % n]
                and (r == 0 or not state.protected[(p + r) % n])
            ]
            return [PlayCardAction(c, r) for r in targets]

        if c == GUARD:
            targets = [
                r for r in range(1, n)
                if not state.eliminated[(p + r) % n] and not state.protected[(p + r) % n]
            ]
            if not targets:
                return [PlayCardAction(c)]  # 空打
            return [PlayCardAction(c, r, g) for r in targets for g in range(2, 9)]

        return []

    # ===== 状态查询 =====

    def get_current_player(self, state: LoveLetterState | None = None) -> int:
        state = state or self._state
        if state is None:
            return 0
        return state.current_player

    def is_terminal(self, state: LoveLetterState | None = None) -> bool:
        state = state or self._state
        if state is None:
            return False
        return state.game_over

    def get_final_rewards(self, state: LoveLetterState | None = None) -> list[float]:
        state = state or self._state
        if state is None:
            return [0.0] * self._num_players
        n = state.num_players
        return [1.0 if i == state.winner else 0.0 for i in range(n)]

    def get_winner(self, state: LoveLetterState | None = None) -> int:
        state = state or self._state
        if state is None:
            return -1
        return state.winner

    # ===== 观察与动作编码 =====

    def state_to_observation(self, state: LoveLetterState, player_id: int):
        return self._encoder.encode(state, player_id)

    def action_to_index(self, action: PlayCardAction, state: LoveLetterState | None = None) -> int:
        key = (action.card, action.target, action.guess)
        if key not in self._action_index:
            raise IllegalActionError(action, "动作不在固定动作空间内")
        return self._action_index[key]

    def index_to_action(self, index: int, state: LoveLetterState | None = None) -> PlayCardAction:
        card, target, guess = self._action_slots[index]
        return PlayCardAction(card=card, target=target, guess=guess)

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
    def auxiliary_shape(self) -> tuple[int, ...]:
        """辅助任务输出维度：预测 (n-1) 个相对对手的手牌值（8 类）。"""
        return ((self._num_players - 1) * 8,)

    @property
    def encoder_params(self) -> dict:
        """模型编码器所需的游戏特定参数（GRU 序列编码用）。"""
        return self._encoder.model_encoder_params

    def get_auxiliary_labels(self, state: LoveLetterState, player_id: int):
        """上帝视角辅助标签：每个相对对手的手牌值（0..7）或 -1（出局/无牌）。"""
        import numpy as np
        n = state.num_players
        labels = np.full(n - 1, -1, dtype=np.int64)
        for r in range(1, n):
            other = (player_id + r) % n
            if not state.eliminated[other] and state.hands[other]:
                labels[r - 1] = state.hands[other][0] - 1  # 牌值 1-8 → 类 0-7
        return labels

    def game_kwargs(self) -> dict:
        kwargs = super().game_kwargs()
        if self._seed is not None:
            kwargs["seed"] = self._seed
        kwargs["reward_config"] = dict(self._rewards)
        return kwargs

    # ===== 渲染 =====

    def render(self, state: LoveLetterState | None = None, mode: str = "human") -> str | None:
        state = state or self._state
        if state is None:
            return None
        n = state.num_players
        lines = []
        lines.append(f"=== 情书 第 {state.round_number} 轮 (标记: {state.tokens}) ===")
        lines.append(f"牌堆剩余: {len(state.deck)} | 回合: {state.turn_number}")
        for i in range(n):
            mark = "*" if i == state.current_player else " "
            hand = state.hands[i]
            status = []
            if state.eliminated[i]:
                status.append("出局")
            if state.protected[i]:
                status.append("受保护")
            lines.append(
                f"{mark} P{i}: 手牌={[CARD_NAMES.get(v, v) for v in hand]} "
                f"弃牌={state.discards[i]} {'/'.join(status)}"
            )
        text = "\n".join(lines)
        if mode == "human":
            print(text)
            return None
        return text


__all__ = ["LoveLetterGame"]
