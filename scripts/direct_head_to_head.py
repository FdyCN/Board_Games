#!/usr/bin/env python3
"""
通用 head-to-head 对比（直接计数，2/3/4 人，随机座位）。

用于衡量模型的真实胜率。注意：不要用 Arena 的胜率（历史上在隐藏信息
游戏里被「价值函数 EV≈0」的自对弈模型虚高过），本脚本直接数每局谁赢。

玩家 spec 格式：
    random              随机
    heuristic           情书手写规则 bot（只适用于 love_letter）
    strong              情书强贝叶斯 bot（只适用于 love_letter）
    neural:<ckpt>:<enc> 神经网络（enc 为 mlp/attention/gru）

用法:
    # 情书 2 人：模型 vs 随机
    python scripts/direct_head_to_head.py --game love_letter \
        --players neural:data/love_letter/checkpoints/love_letter_2p_league_strong_v1/latest.pth:gru random \
        --games 300

    # 情书 4 人：模型 vs 3 个强 bot
    python scripts/direct_head_to_head.py --game love_letter \
        --players neural:...:gru strong strong strong --games 300

    # Splendor 2 人：模型 vs 随机
    python scripts/direct_head_to_head.py --game splendor \
        --players neural:data/splendor/checkpoints/mlp_medium_2p_v1/latest.pth:mlp random \
        --games 300
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from games.registry import create_game
from models.model_factory import create_model
from agents.neural_agent import NeuralAgent
from core.checkpoints import load_model_state_dict


def make_selector(spec, game, game_name, device, min_iteration=None):
    """根据 spec 返回一个 select(game, state, player, legal_idx) -> action_idx 函数。"""
    if spec == "random":
        def sel(game, state, player, legal_idx):
            return random.choice(legal_idx)
        return sel

    if spec in ("heuristic", "strong"):
        if game_name != "love_letter":
            raise ValueError(f"spec '{spec}' 只适用于 love_letter，当前游戏: {game_name}")
        from games.love_letter.heuristic import heuristic_choose
        from games.love_letter.strong_bot import strong_choose
        chooser = heuristic_choose if spec == "heuristic" else strong_choose

        def sel(game, state, player, legal_idx):
            action = chooser(game, state, player)
            return game.action_to_index(action, state)
        return sel

    if spec.startswith("neural:"):
        _, ckpt, enc = spec.split(":")
        model = create_model(
            obs_dim=game.observation_shape[0], action_size=game.action_space_size,
            encoder_type=enc, config="medium",
            aux_dim=game.auxiliary_shape[0] if game.auxiliary_shape else None,
            encoder_params=game.encoder_params,
        )
        model.load_state_dict(load_model_state_dict(ckpt, device=device, min_iteration=min_iteration))
        model.eval()
        agent = NeuralAgent(model, device=device, name=spec)

        def sel(game, state, player, legal_idx):
            obs = game.state_to_observation(state, player)
            action_idx, _ = agent.select_action(obs, legal_idx, deterministic=True)
            return action_idx
        return sel

    raise ValueError(f"未知玩家 spec: {spec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, choices=["love_letter", "splendor"])
    ap.add_argument("--players", nargs="+", required=True, help="玩家 spec 列表（数量 = 玩家人数）")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-iteration", type=int, default=None,
                    help="neural checkpoint 的 metadata.iteration 下限；低于它则报错，防止加载被短跑覆盖的 latest.pth")
    args = ap.parse_args()

    n = len(args.players)
    if not 2 <= n <= 4:
        print(f"玩家 spec 数量必须是 2-4，当前: {n}")
        sys.exit(1)

    game = create_game(args.game, num_players=n)
    device = torch.device("cpu")
    selectors = [make_selector(spec, game, args.game, device, min_iteration=args.min_iteration) for spec in args.players]
    rng = random.Random(args.seed)

    wins = defaultdict(int)
    total_steps = 0
    for _ in range(args.games):
        state = game.reset()
        # 随机座位：spec i 坐到 seat[i]
        seats = list(range(n))
        rng.shuffle(seats)
        seat_to_spec = {seat: i for i, seat in enumerate(seats)}
        while True:
            p = game.get_current_player(state)
            legal = game.get_legal_actions(state)
            if not legal:
                break
            legal_idx = [game.action_to_index(a, state) for a in legal]
            spec_i = seat_to_spec[p]
            action_idx = selectors[spec_i](game, state, p, legal_idx)
            action = game.index_to_action(action_idx, state)
            state, _, done, _ = game.step(action)
            total_steps += 1
            if done:
                break
        winner = game.get_winner(state)
        if winner >= 0 and winner in seat_to_spec:
            wins[seat_to_spec[winner]] += 1

    print(f"=== {args.game} head-to-head（{n} 人，{args.games} 局，随机座位）===")
    for i, spec in enumerate(args.players):
        rate = wins[i] / args.games
        print(f"  {spec:<55} 胜率 {rate:6.1%}  ({wins[i]} 胜)")
    print(f"平均回合数: {total_steps / args.games:.1f}")


if __name__ == "__main__":
    main()
