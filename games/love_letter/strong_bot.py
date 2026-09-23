"""
情书（Love Letter）「认真版」强规则 bot。

相比 heuristic.py 的简单版，这个版本做：
- 精确贝叶斯算牌：对每个存活对手算「手里是每张牌」的后验概率（含私密知识 known）
- 卫兵：猜「使淘汰概率最大的那张牌」（而非简单数数）
- 男爵：只在 P(我的保留牌 > 对手牌) 明显占优时打
- 王子/国王：针对「推断手牌期望值最高」的对手
- 终局：保留高牌比大小

只使用公开信息 + 自己的私密知识（state.known[me][*]），不含上帝视角。
"""

from __future__ import annotations

import math
from collections import Counter

from games.love_letter.constants import (
    BARON, COUNTESS, GUARD, HANDMAID, KING, PRIEST, PRINCE, PRINCESS, CARD_COPIES,
)
from games.love_letter.heuristic import unseen_counts


def _abs_target(state, me, rel):
    return (me + rel) % state.num_players


def opponent_beliefs(state, me):
    """对每个存活对手 q，返回 {q: {牌值: 概率}}（贝叶斯后验，含私密知识）。"""
    n = state.num_players
    unseen = unseen_counts(state, me)  # {v: 还有几张没露面}

    pinned = Counter()
    known = {}
    for q in range(n):
        if q != me and not state.eliminated[q] and state.known[me][q] is not None:
            known[q] = state.known[me][q]
            pinned[state.known[me][q]] += 1

    available = {v: max(0, unseen[v] - pinned.get(v, 0)) for v in range(1, 9)}
    total_avail = sum(available.values())

    beliefs = {}
    for q in range(n):
        if q == me or state.eliminated[q]:
            continue
        if q in known:
            beliefs[q] = {known[q]: 1.0}
        elif total_avail > 0:
            beliefs[q] = {v: available[v] / total_avail for v in range(1, 9)}
        else:
            beliefs[q] = {}
    return beliefs


def expected_value(belief):
    """期望手牌值。"""
    return sum(v * p for v, p in belief.items())


def strong_choose(game, state, me):
    """返回一个合法的 PlayCardAction（贝叶斯强策略）。"""
    n = state.num_players
    hand = state.hands[me]
    legal = game.get_legal_actions(state)
    cards = sorted({a.card for a in legal})

    # 被迫打女伯爵
    if cards == [COUNTESS]:
        return legal[0]
    # 绝不主动打公主
    if PRINCESS in cards and len(cards) > 1:
        cards = [c for c in cards if c != PRINCESS]

    beliefs = opponent_beliefs(state, me)
    exp_val = {q: expected_value(b) for q, b in beliefs.items()}

    keep_for = {}
    for c in hand:
        others = [x for x in hand if x != c]
        keep_for[c] = others[0] if others else 0

    def abs_t(a):
        return _abs_target(state, me, a.target)

    def card_score(c):
        keep = keep_for[c]
        if c == GUARD:
            best = 0.0
            for a in legal:
                if a.card == GUARD:
                    best = max(best, beliefs.get(abs_t(a), {}).get(a.guess, 0.0))
            return best * 10 + keep
        if c == PRIEST:
            return 4 + keep
        if c == BARON:
            best = -100.0
            for a in legal:
                if a.card == BARON:
                    b = beliefs.get(abs_t(a), {})
                    p_win = sum(p for v, p in b.items() if v < keep)
                    p_lose = sum(p for v, p in b.items() if v > keep)
                    best = max(best, p_win * 8 - p_lose * 10)
            return best + keep * 0.5
        if c == HANDMAID:
            return keep + 3
        if c == PRINCE:
            # 保留牌差 → 优先丢自己重抽
            if keep <= 3:
                return 20
            best_exp = max([exp_val[q] for q in exp_val] + [0])
            return best_exp * 2 + keep
        if c == KING:
            best_exp = max([exp_val[q] for q in exp_val] + [0])
            return (best_exp - keep) * 3 + keep
        if c == COUNTESS:
            return 0
        if c == PRINCESS:
            return -999
        return 0

    card = max(cards, key=card_score)
    candidates = [a for a in legal if a.card == card]

    if card in (HANDMAID, COUNTESS, PRINCESS):
        return candidates[0]
    if card == GUARD:
        return max(candidates, key=lambda a: beliefs.get(abs_t(a), {}).get(a.guess, 0.0))
    if card == PRIEST:
        # 看「最不确定」的对手（熵最大），信息增益最大
        def ent(a):
            b = beliefs.get(abs_t(a), {})
            return -sum(p * math.log(p) for p in b.values() if p > 0)
        return max(candidates, key=ent)
    if card == BARON:
        # 打期望手牌最低的对手（最容易赢）
        return min(candidates, key=lambda a: exp_val.get(abs_t(a), 99))
    if card in (PRINCE, KING):
        # 打期望手牌最高的对手（王子废掉强牌 / 国王偷强牌）
        return max(candidates, key=lambda a: exp_val.get(abs_t(a), 0))
    return candidates[0]


__all__ = ["opponent_beliefs", "expected_value", "strong_choose"]
