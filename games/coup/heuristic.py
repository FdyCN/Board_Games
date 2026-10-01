"""
政变疑云（Coup）手写规则 bot（只用公开信息 + 自己手牌）。

作为「会玩的对手」放进对手池，用于联赛训练和 head-to-head 评测。分两个档次：
- heuristic_choose：弱 bot（贪心攒钱 + 诚实反制，很少质疑）
- strong_choose：强 bot（贝叶斯算牌 + 概率质疑，见 strong_bot.py）

bot 接口：choose(game, state, player) -> CoupAction，按 state.phase 分派到各阶段决策。
"""

from __future__ import annotations

import random

from games.coup.actions import CoupAction
from games.coup.constants import (
    DUKE, ASSASSIN, CAPTAIN, AMBASSADOR, CONTESSA, NUM_ROLES, ROLE_COPIES,
    ASSASSINATE_COST, COUP_COST, FORCED_COUP_COINS,
    PHASE_ACTION, PHASE_CHALLENGE, PHASE_BLOCK, PHASE_BLOCK_CHALLENGE,
    PHASE_REVEAL, PHASE_EXCHANGE, BLOCKABLE, EXCHANGE_KEEP_COMBOS,
)

# 角色「价值」：换牌保留高分、扣血翻开低分（保住强角色）
ROLE_VALUE = {ASSASSIN: 5, CAPTAIN: 4, DUKE: 3, AMBASSADOR: 2, CONTESSA: 1}


def role_value(role: int) -> int:
    return ROLE_VALUE.get(role, 0)


def _abs(state, me, rel):
    return (me + rel) % state.num_players


def _seen_counts(state, me):
    """我看到每种角色几张（自己手牌 + 所有公开弃牌）。"""
    seen = [0] * NUM_ROLES
    for r in state.hands[me]:
        seen[r] += 1
    for p in range(state.num_players):
        for r in state.revealed[p]:
            seen[r] += 1
    return seen


def _opponent_cards_remaining(state, me):
    """每个存活对手还持有几张未知牌（用于威胁估计）。"""
    return {q: len(state.hands[q]) for q in range(state.num_players)
            if q != me and state.alive[q]}


def _threat(state, me, rel):
    """目标威胁度：金币为主、影响力为辅（谁快赢就打谁）。"""
    q = _abs(state, me, rel)
    return (state.coins[q] * 10 + len(state.hands[q]) * 2)


def _best_target(state, me, candidates, key):
    """从相对目标候选中选 key 最大者。"""
    return max(candidates, key=lambda a: key(state, me, a.target))


def _action_choose(game, state, me):
    legal = game.get_legal_actions(state)
    coins = state.coins[me]

    coup = [a for a in legal if a.kind == "coup"]
    if coup:
        return _best_target(state, me, coup, _threat)

    steal = [a for a in legal if a.kind == "steal"]
    assassinate = [a for a in legal if a.kind == "assassinate"]
    tax = [a for a in legal if a.kind == "tax"]
    faid = [a for a in legal if a.kind == "foreign_aid"]
    income = [a for a in legal if a.kind == "income"]
    exchange = [a for a in legal if a.kind == "exchange"]

    # 有队长且目标有钱 → 偷
    if steal and CAPTAIN in state.hands[me]:
        rich = [a for a in steal if state.coins[_abs(state, me, a.target)] >= 2]
        if rich:
            return _best_target(state, me, rich, _threat)
    # 有刺客且钱够 → 刺杀威胁最大者
    if assassinate and ASSASSIN in state.hands[me] and coins >= ASSASSINATE_COST:
        return _best_target(state, me, assassinate, _threat)
    # 攒钱：征税 > 外援 > 收入
    if tax:
        return tax[0]
    if faid:
        return faid[0]
    if income:
        return income[0]
    if exchange:
        return exchange[0]
    return legal[0]


def _challenge_or_pass(state, me, strong=False):
    """质疑阶段：是否质疑当前宣称（state.pending_role）。"""
    claimed = state.pending_role
    seen = _seen_counts(state, me)
    # 我看到该角色 3 张都被占 → 对方必说谎 → 质疑
    if seen[claimed] >= ROLE_COPIES:
        return CoupAction("challenge")
    # 我手里已有 2 张该角色 → 对方大概率说谎 → 质疑
    if state.hands[me].count(claimed) >= 2:
        return CoupAction("challenge")
    if strong:
        # 其余情况：低概率质疑（打乱对方，也避免被白嫖）
        if random.random() < 0.15:
            return CoupAction("challenge")
    return CoupAction("pass")


def _block_or_pass(state, me, strong=False):
    """反制阶段：是否反制（宣角色）。诚实反制（真有才反）。"""
    roles = BLOCKABLE.get(state.pending_kind, [])
    for role in roles:
        if role in state.hands[me]:
            return CoupAction("block", role=role)
    if strong and random.random() < 0.1 and roles:
        return CoupAction("block", role=roles[0])  # 偶尔 bluff 反制
    return CoupAction("pass_block")


def _reveal_choose(state, me):
    """扣血：翻开价值最低的那张（保住强角色）。"""
    hand = state.hands[me]
    worst = min(range(len(hand)), key=lambda i: role_value(hand[i]))
    return CoupAction("reveal", reveal_slot=worst)


def _exchange_choose(state, me):
    """换牌：从 4 张里保留价值最高的 2 张。"""
    hand = state.exchange_hand
    ranked = sorted([i for i in range(len(hand)) if hand[i] >= 0],
                    key=lambda i: role_value(hand[i]), reverse=True)
    keep = tuple(sorted(ranked[:2]))
    return CoupAction("exchange_keep", keep=keep)


def heuristic_choose(game, state, player):
    """弱 bot：贪心 + 诚实反制 + 极少质疑。"""
    phase = state.phase
    if phase == PHASE_ACTION:
        return _action_choose(game, state, player)
    if phase in (PHASE_CHALLENGE, PHASE_BLOCK_CHALLENGE):
        return _challenge_or_pass(state, player, strong=False)
    if phase == PHASE_BLOCK:
        return _block_or_pass(state, player, strong=False)
    if phase == PHASE_REVEAL:
        return _reveal_choose(state, player)
    if phase == PHASE_EXCHANGE:
        return _exchange_choose(state, player)
    return CoupAction("pass")


__all__ = [
    "heuristic_choose", "role_value", "ROLE_VALUE",
    "_seen_counts", "_abs", "_threat",
]
