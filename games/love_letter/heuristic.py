"""
情书（Love Letter）手写规则 bot（只使用公开信息 + 自己手牌）。

作为「会玩的对手」放进对手池，用于联赛训练和 head-to-head 评测。

启发式（都是明显正确的策略，不含上帝视角）：
- 绝不主动打公主；被迫打女伯爵时打女伯爵
- 优先打卫兵（猜「还没露面最多的非卫兵牌」），其次侍女保护、国王换牌
- 男爵只在手里另一张 ≥6 时打；王子优先丢「牌最少的对手」的手牌
- 高牌（6/7/8）留到终局比大小
"""

from __future__ import annotations

from games.love_letter.constants import (
    BARON, COUNTESS, GUARD, HANDMAID, KING, PRIEST, PRINCE, PRINCESS, CARD_COPIES,
)


def unseen_counts(state, player):
    """每种牌还有几张「我没看到」（= 牌堆 + 对手手牌 + 移除牌）。"""
    n = state.num_players
    seen = [0] * 9  # 索引 1..8
    for v in state.hands[player]:
        seen[v] += 1
    for p in range(n):
        for v in state.discards[p]:
            seen[v] += 1
    return {v: CARD_COPIES[v] - seen[v] for v in range(1, 9)}


def _abs_target(state, player, rel):
    return (player + rel) % state.num_players


def _opponent_discard_len(state, player, rel):
    t = _abs_target(state, player, rel)
    return len(state.discards[t])


def heuristic_choose(game, state, player):
    """返回一个合法的 PlayCardAction（只用公开信息 + 自己手牌）。"""
    n = state.num_players
    hand = state.hands[player]
    legal = game.get_legal_actions(state)
    cards = sorted({a.card for a in legal})

    # 被迫打女伯爵（get_legal_actions 已经只返回女伯爵）
    if cards == [COUNTESS]:
        return legal[0]

    # 绝不主动打公主
    if PRINCESS in cards and len(cards) > 1:
        cards = [c for c in cards if c != PRINCESS]

    unseen = unseen_counts(state, player)

    keep_for = {}
    for c in hand:
        others = [x for x in hand if x != c]
        keep_for[c] = others[0] if others else 0

    def score(c):
        keep = keep_for[c]
        if c == GUARD:
            return 90
        if c == PRIEST:
            return 50
        if c == BARON:
            return 70 if keep >= 6 else -100
        if c == HANDMAID:
            return 60 if keep >= 5 else 40
        if c == PRINCE:
            return 55
        if c == KING:
            return 75 if keep <= 3 else 15
        if c == COUNTESS:
            return 5
        if c == PRINCESS:
            return -1000
        return 0

    card = max(cards, key=score)
    candidates = [a for a in legal if a.card == card]

    if card in (HANDMAID, COUNTESS, PRINCESS):
        return candidates[0]

    if card == GUARD:
        def guard_key(a):
            guess_bonus = unseen.get(a.guess, 0)
            disc_len = _opponent_discard_len(state, player, a.target)
            return (guess_bonus, disc_len)
        return max(candidates, key=guard_key)

    if card == PRIEST:
        return max(candidates, key=lambda a: -_opponent_discard_len(state, player, a.target))

    if card == BARON:
        return max(candidates, key=lambda a: _opponent_discard_len(state, player, a.target))

    if card == KING:
        return max(candidates, key=lambda a: -_opponent_discard_len(state, player, a.target))

    if card == PRINCE:
        keep = keep_for[card]
        if keep <= 3:
            for a in candidates:
                if a.target == 0:
                    return a
        return max(candidates, key=lambda a: -_opponent_discard_len(state, player, a.target))

    return candidates[0]


__all__ = ["unseen_counts", "heuristic_choose"]
