#!/usr/bin/env python3
"""
AlphaZero 训练脚本（替换 PPO，游戏无关）。

参考 suragnair/alpha-zero-general 的训练循环：
    1. 自对弈：每步用 MCTS 搜索，把 (状态, MCTS 访问分布 π, 终局胜负 z) 存为训练样本；
    2. 训练：策略损失 = CE(policy_net, π)，价值损失 = MSE(value_net, z)；
    3. 不再使用 PPO 的 GAE / clipped surrogate。

用法:
    python scripts/train_alphazero.py \
        --config configs/splendor/splendor_ppo_mlp_medium_3p.yaml \
        --iterations 50 --episodes 32 --simulations 25 \
        --log-file data/splendor/logs/alphazero_3p.jsonl
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config_loader import Config, build_game_kwargs
from games.registry import create_game
from models.model_factory import create_model
from training.mcts import MCTS


def self_play(game, model, mcts, num_simulations, temp_threshold, device):
    """执行一局自对弈，返回 (obs, pi, z, legal_mask) 列表。"""
    examples = []
    state = game.reset()
    episode_step = 0

    while not game.is_terminal(state):
        current_player = game.get_current_player(state)
        legal_actions = game.get_legal_actions(state)
        if not legal_actions:
            break

        # 短实验阶段始终用温度 1（随机采样），避免弱模型被 temp=0 锁死到"不买卡"。
        # 参考实现是在训练充分后才用 temp=0 做确定性采样，这里先全程随机。
        temp = 1.0
        action_probs, legal_actions = mcts.search(
            state=state, model=model, num_simulations=num_simulations, temperature=temp,
        )

        # 构建完整 46 维策略（对齐规范动作空间）
        pi = np.zeros(game.action_space_size, dtype=np.float32)
        legal_mask = np.zeros(game.action_space_size, dtype=np.float32)
        for a, p in zip(legal_actions, action_probs):
            idx = game.action_to_index(a, state)
            pi[idx] = p
            legal_mask[idx] = 1.0

        obs = game.state_to_observation(state, current_player)
        examples.append((obs, pi, current_player, legal_mask))

        action_idx = np.random.choice(len(legal_actions), p=action_probs)
        action = legal_actions[action_idx]
        state, _, done, _ = game.step(action)
        episode_step += 1
        if done:
            break

    # 终局价值：胜者 +1，其余 -1（平局 0）
    winner = game.get_winner(state) if game.is_terminal(state) else -1
    z = np.zeros(game.num_players, dtype=np.float32)
    if winner >= 0:
        z[:] = -1.0
        z[winner] = 1.0

    return [(obs, pi, z[player], mask) for obs, pi, player, mask in examples]


def _worker_self_play(args):
    """多进程 worker：创建模型+游戏+MCTS，加载权重，跑 num_games 局自对弈。

    返回 (examples, ep_lens)，其中 examples 是训练样本列表。
    每个 worker 进程独立持有模型/游戏/MCTS，互不干扰。
    """
    (state_dict, num_games, num_simulations, mcts_batch_size,
     game_name, num_players, encoder_type, model_config, reward_config) = args

    game_kwargs = {"num_players": num_players}
    if reward_config:
        game_kwargs["reward_config"] = reward_config
    game = create_game(game_name, **game_kwargs)
    model = create_model(
        obs_dim=game.observation_shape[0], action_size=game.action_space_size,
        encoder_type=encoder_type, config=model_config,
    )
    model.load_state_dict(state_dict)
    model.eval()
    mcts = MCTS(game=game, c_puct=1.5, add_noise=True, device="cpu", batch_size=mcts_batch_size)

    examples = []
    ep_lens = []
    for _ in range(num_games):
        eps = self_play(game, model, mcts, num_simulations, temp_threshold=10**9, device=torch.device("cpu"))
        examples.extend(eps)
        ep_lens.append(len(eps))
    return examples, ep_lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--simulations", type=int, default=25)
    ap.add_argument("--log-file", required=True)
    ap.add_argument("--temp-threshold", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--mcts-batch-size", type=int, default=64, help="MCTS 叶节点批量评估大小（GPU/MPS 建议 64-256）")
    ap.add_argument("--num-workers", type=int, default=None, help="自对弈并行进程数（默认 CPU 核心数-1）")
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--checkpoint-interval", type=int, default=10)
    args = ap.parse_args()

    c = Config.from_yaml(args.config)
    reward_config = dict(c.algorithm.dense_rewards)

    num_workers = args.num_workers or max(1, (os.cpu_count() or 4) - 1)

    game = create_game(c.game.name, **build_game_kwargs(c))
    model = create_model(
        obs_dim=game.observation_shape[0], action_size=game.action_space_size,
        encoder_type=c.model.encoder_type, config=c.model.config,
    )
    # 强制 CPU：MPS 对小模型无益（见 docs/MPS_OPTIMIZATION.md）
    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    f.write(json.dumps({
        "type": "header", "config": args.config, "num_players": c.game.num_players,
        "iterations": args.iterations, "episodes": args.episodes,
        "simulations": args.simulations, "lr": args.lr, "num_workers": num_workers,
    }) + "\n")
    f.flush()

    def write(record):
        f.write(json.dumps(record) + "\n")
        f.flush()
        print(json.dumps(record), flush=True)

    # 持久进程池（spawn 一次，避免每迭代重复 spawn 开销）
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(num_workers)

    for it in range(1, args.iterations + 1):
        t0 = time.time()

        # 1. 并行自对弈收集（每个 worker 一份模型权重 + 独立 game/MCTS）
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        games_per_worker = args.episodes // num_workers
        remainder = args.episodes % num_workers
        tasks = []
        for w in range(num_workers):
            ng = games_per_worker + (1 if w < remainder else 0)
            if ng > 0:
                tasks.append((
                    state_dict, ng, args.simulations, args.mcts_batch_size,
                    c.game.name, c.game.num_players, c.model.encoder_type, c.model.config, reward_config,
                ))

        examples = []
        ep_lens = []
        for res_examples, res_lens in pool.map(_worker_self_play, tasks):
            examples.extend(res_examples)
            ep_lens.extend(res_lens)

        # 2. 训练
        obs = np.stack([e[0] for e in examples])
        pi = np.stack([e[1] for e in examples])
        z = np.array([e[2] for e in examples], dtype=np.float32)
        mask = np.stack([e[3] for e in examples])

        obs_t = torch.from_numpy(obs).float().to(device)
        pi_t = torch.from_numpy(pi).float().to(device)
        z_t = torch.from_numpy(z).float().to(device)
        mask_t = torch.from_numpy(mask).bool().to(device)

        model.train()
        policy_loss_total = 0.0
        value_loss_total = 0.0
        n_batches = 0

        idx = np.arange(len(examples))
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), args.batch_size):
                b = idx[s:s + args.batch_size]
                logits, values = model(obs_t[b], mask_t[b])
                # 策略 CE（MCTS 目标）：只对合法动作（pi>0）求和，避免 0 * -inf = NaN
                log_probs = torch.log_softmax(logits, dim=-1)
                legal = pi_t[b] > 0
                policy_loss = -(pi_t[b][legal] * log_probs[legal]).sum() / pi_t[b].shape[0]
                # 价值 MSE（终局胜负 +1/-1）
                value_loss = ((values.squeeze(-1) - z_t[b]) ** 2).mean()
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                n_batches += 1

        write({
            "type": "metric", "iteration": it,
            "mean_episode_length": round(float(np.mean(ep_lens)), 2),
            "policy_loss": round(policy_loss_total / n_batches, 6),
            "value_loss": round(value_loss_total / n_batches, 6),
            "seconds": round(time.time() - t0, 2),
        })

        if it % args.checkpoint_interval == 0:
            torch.save({"model_state_dict": model.state_dict(), "iteration": it},
                       Path(c.training.checkpoint_dir) / f"alphazero_iter_{it}.pth")

    pool.close()
    pool.join()
    f.close()
    print(json.dumps({"type": "done", "iterations": args.iterations}), flush=True)


if __name__ == "__main__":
    main()
