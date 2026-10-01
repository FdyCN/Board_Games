"""
政变疑云（Coup）「强」规则 bot。

相比 heuristic.py 的弱 bot，这里做：
- 贝叶斯算牌：估算「宣称者/反制者是否说谎」的后验，决定是否质疑
- 更聪明的金币管理与目标选择
- 偶尔 bluff 反制/刺杀（制造信息不确定性）

只使用公开信息 + 自己的手牌，不含上帝视角。
"""

from __future__ import annotations

import random

from games.coup.actions import CoupAction
from games.coup.constants import (
    DUKE, ASSASSIN, CAPTAIN, AMBASSADOR, CONTESSA, NUM_ROLES, ROLE_COPIES,
    ASSASSINATE_COST, COUP_COST,
    PHASE_ACTION, PHASE_CHALLENGE, PHASE_BLOCK, PHASE_BLOCK_CHALLENGE,
    PHASE_REVEAL, PHASE_EXCHANGE, BLOCKABLE,
)
from games.coup.heuristic import (
    _seen_counts, _abs, _threat, _best_target, _reveal_choose, _exchange_choose,
    role_value,
)


def _remaining(state, me):
    """每种角色还有几张「我没看到」（= 牌堆 + 对手手牌）。"""
    seen = _seen_counts(state, me)
    return [ROLE_COPIES - seen[r] for r in range(NUM_ROLES)]


def _p_bluff(state, me, target, role):
    """估算 target 宣称持有 role 是在说谎的概率（0-1）。"""
    remaining = _remaining(state, me)
    if remaining[role] <= 0:
        return 1.0  # 该角色 3 张都被我看到 → 必说谎
    if state.hands[me].count(role) >= 2:
        return 0.85  # 我手里 2 张 → 只剩 1 张，大概率说谎
    # 粗略：剩余该角色张数 / 未知牌总数，越小越可疑
    total_unknown = sum(remaining) + 1e-9
    density = remaining[role] / total_unknown
    # 对方有 1-2 张手牌，持有概率 ≈ 1 - (1-density)^hand_size
    hand_size = max(1, len(state.hands[target]))
    p_hold = 1 - (1 - density) ** hand_size
    return 1 - p_hold


def _challenge_decision(state, me, target, role):
    """是否质疑 target 宣称 role。"""
    p = _p_bluff(state, me, target, role)
    if p >= 0.9:
        return CoupAction("challenge")
    if p >= 0.7 and random.random() < 0.6:
        return CoupAction("challenge")
    if random.random() < 0.10:
        return CoupAction("challenge")
    return CoupAction("pass")


def _action_choose(game, state, me):
    legal = game.get_legal_actions(state)
    coins = state.coins[me]
    hand = state.hands[me]

    coup = [a for a in legal if a.kind == "coup"]
    if coup:
        return _best_target(state, me, coup, _threat)

    steal = [a for a in legal if a.kind == "steal"]
    assassinate = [a for a in legal if a.kind == "assassinate"]
    tax = [a for a in legal if a.kind == "tax"]
    faid = [a for a in legal if a.kind == "foreign_aid"]
    income = [a for a in legal if a.kind == "income"]
    exchange = [a for a in legal if a.kind == "exchange"]

    # 有队长偷有钱的目标
    if steal and CAPTAIN in hand:
        rich = [a for a in steal if state.coins[_abs(state, me, a.target)] >= 2]
        if rich:
            return _best_target(state, me, rich, _threat)
    # 刺杀：有刺客且钱够；或偶尔 bluff 刺杀（威慑）
    if assassinate and coins >= ASSASSINATE_COST:
        if ASSASSIN in hand:
            return _best_target(state, me, assassinate, _threat)
        if coins >= 5 and random.random() < 0.2:
            return _best_target(state, me, assassinate, _threat)
    # 手牌都很弱 → 换牌
    if exchange and len(hand) == 2 and all(role_value(c) <= 2 for c in hand):
        return exchange[0]
    # 攒钱
    if tax:
        return tax[0]
    if faid:
        return faid[0]
    if income:
        return income[0]
    if exchange:
        return exchange[0]
    return legal[0]


def strong_choose(game, state, me):
    """强 bot：贝叶斯质疑 + 聪明攒钱。"""
    phase = state.phase
    if phase == PHASE_ACTION:
        return _action_choose(game, state, me)
    if phase == PHASE_CHALLENGE:
        return _challenge_decision(state, me, state.turn_player, state.pending_role)
    if phase == PHASE_BLOCK:
        return _block_or_pass(state, me)
    if phase == PHASE_BLOCK_CHALLENGE:
        return _challenge_decision(state, me, state.block_seat, state.block_role)
    if phase == PHASE_REVEAL:
        return _reveal_choose(state, me)
    if phase == PHASE_EXCHANGE:
        return _exchange_choose(state, me)
    return CoupAction("pass")


def _block_or_pass(state, me):
    roles = BLOCKABLE.get(state.pending_kind, [])
    for role in roles:
        if role in state.hands[me]:
            return CoupAction("block", role=role)
    # 偶尔 bluff 反制（尤其自己只剩 1 张、快输时）
    if roles and random.random() < 0.15:
        return CoupAction("block", role=roles[0])
    return CoupAction("pass_block")


__all__ = ["strong_choose"]
