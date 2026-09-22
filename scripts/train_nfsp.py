#!/usr/bin/env python3
"""
NFSP（Neural Fictitious Self-Play，神经虚拟自对弈）训练脚本。

针对不完美信息博弈（情书）。参考 Heinrich & Silver '16 与 OpenSpiel 的
nfsp 实现（OpenSpiel 里 NFSP 是「充分测试」状态，PPO 只是「轻量测试」）。

每个（共享的）智能体维护两个网络：
- 最佳响应网络 Q（DQN 式训练）：近似「对平均策略的最佳响应」
- 平均策略网络 π（监督学习训练）：拟合历史最佳响应行为，收敛到纳什均衡

每一步：
- 以概率 η 用「最佳响应」动作（ε-greedy 于 Q），并把该 (s, a) 存入 SL 缓冲；
- 以概率 1-η 用「平均策略」动作（采样于 π）。

用法:
    python scripts/train_nfsp.py \
        --config configs/love_letter/love_letter_ppo.yaml \
        --iterations 1000 --episodes 128 \
        --log-file data/love_letter/logs/nfsp_3p.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config_loader import Config, build_game_kwargs
from games.registry import create_game
from models.model_factory import create_model


def _masked_logits(model, obs, mask, device):
    """返回 (B, action_size) 的已掩码 logits（非法动作 = -inf）。"""
    logits, _ = model(obs, mask)
    return logits


def _ap_probs(model, obs, mask, device):
    """平均策略的动作概率分布（softmax，非法动作概率为 0）。"""
    logits = _masked_logits(model, obs, mask, device)
    return F.softmax(logits, dim=-1)


def self_play_episode(game, q_net, pi_net, replay, sl, eta, epsilon, device):
    """打一局，按 NFSP 规则把转换写入 replay / SL 缓冲，返回本局步数。"""
    state = game.reset()
    n = game.num_players
    action_space = game.action_space_size
    steps = 0

    # 每个玩家「上一次」的 (obs, mask, action)，用于拼接该玩家自己的 next_obs
    pending: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

    while True:
        p = game.get_current_player(state)
        obs = game.state_to_observation(state, p)
        legal = game.get_legal_actions(state)
        legal_idx = [game.action_to_index(a, state) for a in legal]
        mask = np.zeros(action_space, dtype=np.float32)
        mask[legal_idx] = 1.0

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(device)

        if random.random() < eta:
            # 最佳响应：ε-greedy 于 Q
            with torch.no_grad():
                q = _masked_logits(q_net, obs_t, mask_t, device)[0]
            if random.random() < epsilon:
                a_idx = random.choice(legal_idx)
            else:
                a_idx = legal_idx[int(torch.argmax(q[legal_idx]).item())]
            # 最佳响应行为存入 SL 缓冲（用于训练平均策略）
            sl.append((obs, mask, a_idx))
        else:
            # 平均策略：采样于 π
            with torch.no_grad():
                probs = _ap_probs(pi_net, obs_t, mask_t, device)[0]
            p_legal = probs[legal_idx].detach().numpy()
            p_legal = p_legal / (p_legal.sum() + 1e-12)
            a_idx = int(np.random.choice(legal_idx, p=p_legal))

        action = game.index_to_action(a_idx, state)
        next_state, _, done, _ = game.step(action)

        # 关闭该玩家上一次的转换（next_obs = 本次 obs，即时奖励 0）
        if p in pending:
            prev_obs, prev_mask, prev_a = pending[p]
            replay.append((prev_obs, prev_mask, prev_a, 0.0, obs, mask, 0.0))
        pending[p] = (obs, mask, a_idx)

        state = next_state
        steps += 1
        if done:
            break

    # 终局：关闭所有未完成的转换，注入零和终局奖励
    winner = game.get_winner(state) if game.is_terminal(state) else -1
    zero_obs = np.zeros_like(obs)
    zero_mask = np.zeros(action_space, dtype=np.float32)
    for p, (obs_p, mask_p, a_idx) in pending.items():
        r = 0.0
        if winner >= 0:
            r = 1.0 if p == winner else -1.0 / (n - 1)
        replay.append((obs_p, mask_p, a_idx, r, zero_obs, zero_mask, 1.0))

    return steps


def train_q(q_net, q_target, opt, replay, gamma, batch_size, device):
    """DQN 更新最佳响应网络，返回损失。"""
    if len(replay) < batch_size:
        return 0.0
    batch = random.sample(replay, batch_size)
    obs = np.stack([b[0] for b in batch])
    mask = np.stack([b[1] for b in batch])
    acts = np.array([b[2] for b in batch], dtype=np.int64)
    rews = np.array([b[3] for b in batch], dtype=np.float32)
    next_obs = np.stack([b[4] for b in batch])
    next_mask = np.stack([b[5] for b in batch])
    dones = np.array([b[6] for b in batch], dtype=np.float32)

    obs_t = torch.FloatTensor(obs).to(device)
    mask_t = torch.BoolTensor(mask).to(device)
    acts_t = torch.LongTensor(acts).to(device)
    rews_t = torch.FloatTensor(rews).to(device)
    next_obs_t = torch.FloatTensor(next_obs).to(device)
    next_mask_t = torch.BoolTensor(next_mask).to(device)
    dones_t = torch.FloatTensor(dones).to(device)

    with torch.no_grad():
        next_q = _masked_logits(q_target, next_obs_t, next_mask_t, device)
        next_q_max = next_q.max(dim=-1).values
        # 用 torch.where 避免 done 样本出现 0 * (-inf) = NaN
        target = torch.where(dones_t.bool(), rews_t, rews_t + gamma * next_q_max)

    q_all = _masked_logits(q_net, obs_t, mask_t, device)
    q_pred = q_all.gather(1, acts_t.unsqueeze(-1)).squeeze(-1)
    loss = F.mse_loss(q_pred, target)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
    opt.step()
    return loss.item()


def train_pi(pi_net, opt, sl, batch_size, device):
    """监督学习更新平均策略网络（拟合最佳响应行为），返回损失。"""
    if len(sl) < batch_size:
        return 0.0
    batch = random.sample(sl, batch_size)
    obs = np.stack([b[0] for b in batch])
    mask = np.stack([b[1] for b in batch])
    acts = np.array([b[2] for b in batch], dtype=np.int64)

    obs_t = torch.FloatTensor(obs).to(device)
    mask_t = torch.BoolTensor(mask).to(device)
    acts_t = torch.LongTensor(acts).to(device)

    logits = _masked_logits(pi_net, obs_t, mask_t, device)
    loss = F.cross_entropy(logits, acts_t)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pi_net.parameters(), 1.0)
    opt.step()
    return loss.item()


def eval_ap_vs_random(game, pi_net, num_games, device):
    """评估平均策略 vs 随机对手（确定性，随机座位），返回胜率。"""
    n = game.num_players
    action_space = game.action_space_size
    wins = 0
    for _ in range(num_games):
        state = game.reset()
        ap_seat = random.randrange(n)
        while True:
            p = game.get_current_player(state)
            legal = game.get_legal_actions(state)
            legal_idx = [game.action_to_index(a, state) for a in legal]
            if p == ap_seat:
                obs = game.state_to_observation(state, p)
                mask = np.zeros(action_space, dtype=np.float32)
                mask[legal_idx] = 1.0
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                mask_t = torch.BoolTensor(mask).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = _ap_probs(pi_net, obs_t, mask_t, device)[0]
                a_idx = legal_idx[int(torch.argmax(probs[legal_idx]).item())]
            else:
                a_idx = random.choice(legal_idx)
            action = game.index_to_action(a_idx, state)
            state, _, done, _ = game.step(action)
            if done:
                break
        winner = game.get_winner(state)
        if winner == ap_seat:
            wins += 1
    return wins / num_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--log-file", required=True)
    ap.add_argument("--eta", type=float, default=0.1, help="最佳响应动作概率")
    ap.add_argument("--epsilon", type=float, default=0.06, help="BR 的 ε-greedy 探索")
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--replay-size", type=int, default=200000)
    ap.add_argument("--sl-size", type=int, default=2000000)
    ap.add_argument("--q-updates", type=int, default=256, help="每次迭代 Q 更新步数")
    ap.add_argument("--pi-updates", type=int, default=256, help="每次迭代 π 更新步数")
    ap.add_argument("--target-update", type=int, default=200, help="目标网络更新间隔（迭代）")
    ap.add_argument("--eval-games", type=int, default=100, help="周期性评估的局数")
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--checkpoint-interval", type=int, default=100)
    args = ap.parse_args()

    c = Config.from_yaml(args.config)
    episodes_per_iter = args.episodes or c.training.episodes_per_iteration
    game = create_game(c.game.name, **build_game_kwargs(c))
    n = game.num_players
    aux_dim = game.auxiliary_shape[0] if game.auxiliary_shape else None

    device = torch.device("cpu")

    def make_model():
        return create_model(
            obs_dim=game.observation_shape[0],
            action_size=game.action_space_size,
            encoder_type=c.model.encoder_type,
            config=c.model.config,
            aux_dim=aux_dim,  # 保持与 evaluate.py 的模型结构一致
            encoder_params=game.encoder_params,
        )

    q_net = make_model()
    q_target = make_model()
    q_target.load_state_dict(q_net.state_dict())
    pi_net = make_model()
    q_opt = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    pi_opt = torch.optim.Adam(pi_net.parameters(), lr=args.lr)

    replay: deque = deque(maxlen=args.replay_size)
    sl: deque = deque(maxlen=args.sl_size)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    f.write(json.dumps({
        "type": "header", "config": args.config, "game": c.game.name,
        "num_players": n, "iterations": args.iterations,
        "episodes_per_iteration": episodes_per_iter,
        "eta": args.eta, "epsilon": args.epsilon, "lr": args.lr, "gamma": args.gamma,
    }) + "\n")
    f.flush()

    def write(record):
        f.write(json.dumps(record) + "\n")
        f.flush()
        print(json.dumps(record), flush=True)

    q_net.train()
    pi_net.train()
    q_target.eval()

    for it in range(1, args.iterations + 1):
        t0 = time.time()

        # 1. 自对弈收集
        ep_lens = []
        for _ in range(episodes_per_iter):
            ep_lens.append(self_play_episode(
                game, q_net, pi_net, replay, sl, args.eta, args.epsilon, device))

        # 2. 训练 Q（最佳响应）与 π（平均策略）
        q_loss = float(np.mean([train_q(q_net, q_target, q_opt, replay,
                                        args.gamma, args.batch_size, device)
                                for _ in range(args.q_updates)]))
        pi_loss = float(np.mean([train_pi(pi_net, pi_opt, sl, args.batch_size, device)
                                 for _ in range(args.pi_updates)]))

        # 3. 目标网络更新
        if it % args.target_update == 0:
            q_target.load_state_dict(q_net.state_dict())

        # 4. 记录 + 周期性评估 AP vs 随机
        record = {
            "type": "metric", "iteration": it,
            "mean_episode_length": round(float(np.mean(ep_lens)), 2),
            "q_loss": round(q_loss, 6), "pi_loss": round(pi_loss, 6),
            "replay_size": len(replay), "sl_size": len(sl),
            "seconds": round(time.time() - t0, 2),
        }
        if it % args.eval_interval == 0 or it == 1:
            pi_net.eval()
            win_rate = eval_ap_vs_random(game, pi_net, args.eval_games, device)
            pi_net.train()
            record["ap_win_rate_vs_random"] = round(win_rate, 4)
        write(record)

        # 5. 保存 π（平均策略）为检查点
        if it % args.checkpoint_interval == 0:
            ckpt_dir = Path(c.training.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": pi_net.state_dict(), "iteration": it},
                       ckpt_dir / f"nfsp_iter_{it}.pth")
            torch.save({"model_state_dict": pi_net.state_dict(), "iteration": it},
                       ckpt_dir / "latest.pth")

    f.close()
    print(json.dumps({"type": "done", "iterations": args.iterations}), flush=True)


if __name__ == "__main__":
    main()
