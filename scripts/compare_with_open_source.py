#!/usr/bin/env python3
"""
对比开源 alpha-zero-general 的 Splendor 3 人预训练模型 vs 随机对手。

用法:
    python scripts/compare_with_open_source.py --num-games 50

说明:
    用开源模型的"贪婪策略"（argmax 策略，不用 MCTS）作为玩家 0，
    对战 2 个随机玩家，统计胜率。与本地 PPO 模型（确定性策略 vs 2 随机）
    的 78% 胜率对齐比较。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "alpha-zero-general"))

from splendor.SplendorLogicNumba import Board


def load_model():
    ckpt_path = project_root / "third_party/alpha-zero-general/splendor/pretrained_3players.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ckpt["full_model"]
    model.eval()
    return model


def model_action(board, model, player):
    """用开源模型的贪婪策略（argmax）选动作。"""
    valid = board.valid_moves(player)
    state = board.get_state()
    inp = torch.FloatTensor(state.astype(np.float32)).unsqueeze(0)
    val = torch.BoolTensor(valid.astype(bool)).unsqueeze(0)
    with torch.no_grad():
        log_pi, _ = model(inp, val)
    probs = torch.exp(log_pi).cpu().numpy()[0]
    probs = probs * valid.astype(np.float32)
    return int(np.argmax(probs))


def random_action(board, player):
    valid = board.valid_moves(player)
    idxs = np.where(valid)[0]
    return int(np.random.choice(idxs))


def play_one_game(model, model_seat=0):
    board = Board(3)
    current = 0
    while True:
        result = board.check_end_game()
        if np.any(result != 0.0):
            winner = int(np.argmax(result))
            return winner, board.get_round()
        if current == model_seat:
            a = model_action(board, model, current)
        else:
            a = random_action(board, current)
        current = board.make_move(a, current, 0)  # numba jitclass 不支持关键字参数


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-games", type=int, default=50)
    args = ap.parse_args()

    model = load_model()
    print(f"开源模型 nn_version={model.version}, 玩家数=3, 动作空间={model.action_size}")

    wins = 0
    lengths = []
    for g in range(args.num_games):
        winner, rounds = play_one_game(model, model_seat=0)
        lengths.append(rounds)
        if winner == 0:
            wins += 1
        if (g + 1) % 10 == 0:
            print(f"  {g+1}/{args.num_games} 局, 当前胜率 {wins/(g+1):.1%}")

    win_rate = wins / args.num_games
    print(f"\n开源模型(贪婪) vs 2 随机: 胜率 {win_rate:.1%} ({wins}/{args.num_games})")
    print(f"平均回合数: {np.mean(lengths):.1f}")
    print("(对比: 本地 PPO 模型确定性策略 vs 2 随机 = 98%)")


if __name__ == "__main__":
    main()
