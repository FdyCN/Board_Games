#!/usr/bin/env python3
"""
情书 head-to-head 对比（3 人同场对打，随机座位）。

用于衡量「私密知识 + GRU」这些修复的真实收益——因为 vs 随机几乎测不出
隐藏信息推理的价值，必须让「会玩的对手」同场。

玩家 spec 格式：
    random              随机
    heuristic           手写规则 bot
    neural:<ckpt>:<enc> 神经网络（enc 为 mlp/attention/gru）

用法:
    python scripts/love_letter_head_to_head.py \
        --players random heuristic neural:data/love_letter/checkpoints/love_letter_3p_gru_v1/latest.pth:gru \
        --games 300
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from games.registry import create_game
from games.love_letter.heuristic import heuristic_choose
from games.love_letter.strong_bot import strong_choose
from models.model_factory import create_model
from agents.neural_agent import NeuralAgent


def make_selector(spec, game, device):
    """根据 spec 返回一个 select(game, state, player, legal_idx) -> action_idx 函数。"""
    if spec == "random":
        def sel(game, state, player, legal_idx):
            return random.choice(legal_idx)
        return sel

    if spec == "heuristic":
        def sel(game, state, player, legal_idx):
            action = heuristic_choose(game, state, player)
            return game.action_to_index(action, state)
        return sel

    if spec == "strong":
        def sel(game, state, player, legal_idx):
            action = strong_choose(game, state, player)
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
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
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
    ap.add_argument("--players", nargs="+", required=True, help="玩家 spec 列表（3 个）")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if len(args.players) != 3:
        print("需要恰好 3 个玩家 spec（3 人对局）")
        sys.exit(1)

    game = create_game("love_letter", num_players=3)
    device = torch.device("cpu")
    selectors = [make_selector(spec, game, device) for spec in args.players]
    rng = random.Random(args.seed)

    wins = defaultdict(int)
    total_steps = 0
    for _ in range(args.games):
        state = game.reset()
        # 随机座位：spec i 坐到 seat[i]
        seats = list(range(3))
        rng.shuffle(seats)
        seat_to_spec = {seat: i for i, seat in enumerate(seats)}
        while True:
            p = game.get_current_player(state)
            legal = game.get_legal_actions(state)
            legal_idx = [game.action_to_index(a, state) for a in legal]
            spec_i = seat_to_spec[p]
            action_idx = selectors[spec_i](game, state, p, legal_idx)
            action = game.index_to_action(action_idx, state)
            state, _, done, _ = game.step(action)
            total_steps += 1
            if done:
                break
        winner = game.get_winner(state)
        if winner >= 0:
            wins[seat_to_spec[winner]] += 1

    print(f"=== 情书 head-to-head（{args.games} 局，随机座位）===")
    for i, spec in enumerate(args.players):
        rate = wins[i] / args.games
        print(f"  {spec:<55} 胜率 {rate:6.1%}  ({wins[i]} 胜)")
    print(f"平均回合数: {total_steps / args.games:.1f}")


if __name__ == "__main__":
    main()
