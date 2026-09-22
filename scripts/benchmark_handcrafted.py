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
from games.love_letter.heuristic import heuristic_choose


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
