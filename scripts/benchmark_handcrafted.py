#!/usr/bin/env python3
"""
手写规则 bot 基准（情书 3 人）。

目的：校准「77% vs 随机」到底是强还是弱——用一个只使用公开信息 +
自己手牌的规则 bot，测它对随机的胜率，作为「天花板」的参照。

规则 bot 使用的启发式（都是「明显正确」的策略，不含上帝视角）：
- 绝不主动打公主；被迫打女伯爵时打女伯爵
- 优先打卫兵（猜「还没露面最多的非卫兵牌」），其次侍女保护、国王换牌
- 男爵只在手里另一张 ≥6 时打；王子优先丢「牌最少的对手」的手牌
- 高牌（6/7/8）留到终局比大小

用法:
    python scripts/benchmark_handcrafted.py --games 200
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from games.love_letter.game import LoveLetterGame
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

    # 若打出 c，手里留下的是哪张
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
        # 猜「还没露面最多的非卫兵牌」；目标选弃牌最多的对手（信息最多）
        def guard_key(a):
            guess_bonus = unseen.get(a.guess, 0)
            disc_len = _opponent_discard_len(state, player, a.target)
            return (guess_bonus, disc_len)
        return max(candidates, key=guard_key)

    if card == PRIEST:
        # 看信息最少的对手（弃牌最少 = 最未知）
        return max(candidates, key=lambda a: -_opponent_discard_len(state, player, a.target))

    if card == BARON:
        # 打弃牌最多的对手（更可能已经把手里的强牌打光了）
        return max(candidates, key=lambda a: _opponent_discard_len(state, player, a.target))

    if card == KING:
        # 换「弃牌最少」的对手（更可能还攥着高牌）
        return max(candidates, key=lambda a: -_opponent_discard_len(state, player, a.target))

    if card == PRINCE:
        # 我的保留牌差 → 丢自己重抽；否则丢弃牌最少的对手（可能攥着高牌）
        keep = keep_for[card]
        if keep <= 3:
            for a in candidates:
                if a.target == 0:  # 相对目标 0 = 自己
                    return a
        return max(candidates, key=lambda a: -_opponent_discard_len(state, player, a.target))

    return candidates[0]


def benchmark(games, seed):
    game = LoveLetterGame(num_players=3, seed=seed)
    rng = random.Random(seed)
    n = game.num_players
    wins = 0
    total_steps = 0
    for _ in range(games):
        state = game.reset()
        bot_seat = rng.randrange(n)
        while True:
            p = game.get_current_player(state)
            legal = game.get_legal_actions(state)
            if not legal:
                break
            if p == bot_seat:
                action = heuristic_choose(game, state, p)
            else:
                action = rng.choice(legal)
            state, _, done, _ = game.step(action)
            total_steps += 1
            if done:
                break
        if game.get_winner(state) == bot_seat:
            wins += 1
    return wins / games, total_steps / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rate, avg_steps = benchmark(args.games, args.seed)
    print(f"手写规则 bot vs 2 随机（{args.games} 局，随机座位）:")
    print(f"  胜率: {rate:.1%}")
    print(f"  平均回合数: {avg_steps:.1f}")
    print(f"\n参照：PPO 系列 ≈ 75~77%，NFSP ≈ 59%，随机基线 ≈ 33%")


if __name__ == "__main__":
    main()
