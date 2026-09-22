#!/usr/bin/env python3
"""
对手池（League）训练脚本 —— 情书 3 人。

关键：不只 vs 随机，也不只 vs 自己，而是和一个「多样化对手池」对打：
    - 随机 agent（弱基线）
    - 手写规则 bot（会玩的基线，~71%）
    - 冻结的历史 checkpoint（本模型过去的快照）

当前模型每局随机占一个座位，其余座位从池里采样对手；只有当前模型的
经验用于 PPO 更新。定期把当前模型快照加入池子（AlphaStar 联赛思路）。

用法:
    python scripts/train_league.py \
        --config configs/love_letter/love_letter_ppo.yaml \
        --iterations 500 --episodes 128 \
        --log-file data/love_letter/logs/league_3p.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config_loader import Config, build_game_kwargs
from games.registry import create_game
from games.love_letter.heuristic import heuristic_choose
from models.model_factory import create_model
from agents.neural_agent import NeuralAgent
from training.algorithms.ppo import PPO
from training.experience import ExperienceBatch, compute_advantages_for_episode
from core.types import Experience, Episode


def league_episode(game, seats, current_agent, gamma, gae_lambda, rng, device):
    """打一局：seats[p] = (type, agent_or_None)；返回当前模型的 Experience 列表（已算优势）。"""
    state = game.reset()
    n = game.num_players
    action_space = game.action_space_size
    experiences: list[Experience] = []

    while True:
        p = game.get_current_player(state)
        obs = game.state_to_observation(state, p)
        legal = game.get_legal_actions(state)
        legal_idx = [game.action_to_index(a, state) for a in legal]
        mask = np.zeros(action_space, dtype=np.float32)
        mask[legal_idx] = 1.0

        spec_type, spec_agent = seats[p]
        pending = None

        if spec_type == "current":
            action_idx, info = current_agent.select_action(obs, legal_idx)
            aux_targets = game.get_auxiliary_labels(state, p)
            pending = (p, obs, action_idx, info["log_prob"], info["value"], mask, aux_targets)
        elif spec_type == "heuristic":
            action = heuristic_choose(game, state, p)
            action_idx = game.action_to_index(action, state)
        elif spec_type == "random":
            action_idx = rng.choice(legal_idx)
        elif spec_type == "neural":
            action_idx, _ = spec_agent.select_action(obs, legal_idx)
        else:
            raise ValueError(f"未知对手类型: {spec_type}")

        action = game.index_to_action(action_idx, state)
        next_state, _, done, step_info = game.step(action)

        if pending is not None:
            experiences.append(Experience(
                player_id=pending[0],
                observation=pending[1],
                action=pending[2],
                reward=step_info.get("dense_reward", 0.0),
                log_prob=pending[3],
                value=pending[4],
                legal_actions_mask=pending[5],
                aux_targets=pending[6],
            ))

        state = next_state
        if done:
            break

    final_rewards = game.get_final_rewards(state)
    winner = game.get_winner(state) if game.is_terminal(state) else -1
    for exp in experiences:
        exp.outcome = 1.0 if (winner >= 0 and exp.player_id == winner) else 0.0

    episode = Episode(
        experiences=experiences,
        total_rewards=final_rewards,
        winner=winner,
        num_steps=len(experiences),
    )
    compute_advantages_for_episode(episode, gamma=gamma, gae_lambda=gae_lambda)
    return episode.experiences


def sample_seats(game, current_agent, pool, rng):
    """随机指定当前模型座位，其余座位从池里采样对手。"""
    n = game.num_players
    current_seat = rng.randrange(n)
    seats = []
    for p in range(n):
        if p == current_seat:
            seats.append(("current", current_agent))
        else:
            seats.append(rng.choice(pool))
    return seats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--log-file", required=True)
    ap.add_argument("--snapshot-interval", type=int, default=20, help="每隔 N 迭代把当前模型加入池子")
    ap.add_argument("--pool-size", type=int, default=5, help="池子里最多保留多少个冻结快照")
    ap.add_argument("--checkpoint-interval", type=int, default=50)
    args = ap.parse_args()

    c = Config.from_yaml(args.config)
    episodes_per_iter = args.episodes or c.training.episodes_per_iteration
    game = create_game(c.game.name, **build_game_kwargs(c))
    n = game.num_players
    device = torch.device(c.training.device)

    model = create_model(
        obs_dim=game.observation_shape[0],
        action_size=game.action_space_size,
        encoder_type=c.model.encoder_type,
        config=c.model.config,
        aux_dim=game.auxiliary_shape[0] if game.auxiliary_shape else None,
        encoder_params=game.encoder_params,
    )
    current_agent = NeuralAgent(model, device=c.training.device, name="current")
    ppo = PPO(
        model=model,
        learning_rate=c.algorithm.learning_rate,
        clip_epsilon=c.algorithm.clip_epsilon,
        value_coef=c.algorithm.value_coef,
        entropy_coef=c.algorithm.entropy_coef,
        max_grad_norm=c.algorithm.max_grad_norm,
        device=c.training.device,
        use_value_clip=c.algorithm.use_value_clip,
        value_clip_epsilon=c.algorithm.value_clip_epsilon,
        outcome_coef=c.algorithm.outcome_coef,
    )

    # 对手池：随机 + 规则 bot + （后续加入）冻结快照
    pool: list[tuple[str, object | None]] = [("random", None), ("heuristic", None)]
    rng = random.Random(c.game.seed)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    f.write(json.dumps({
        "type": "header", "config": args.config, "game": c.game.name,
        "num_players": n, "iterations": args.iterations,
        "episodes_per_iteration": episodes_per_iter,
        "encoder": c.model.encoder_type,
    }) + "\n")
    f.flush()

    def write(record):
        f.write(json.dumps(record) + "\n")
        f.flush()
        print(json.dumps(record), flush=True)

    for it in range(1, args.iterations + 1):
        t0 = time.time()

        # 1. 收集（当前模型 vs 对手池）
        all_exps = []
        ep_lens = []
        for _ in range(episodes_per_iter):
            seats = sample_seats(game, current_agent, pool, rng)
            exps = league_episode(game, seats, current_agent,
                                  c.algorithm.gamma, c.algorithm.gae_lambda, rng, device)
            all_exps.extend(exps)
            ep_lens.append(len(exps))

        # 2. PPO 更新
        batch = ExperienceBatch.from_experiences(
            all_exps, device=c.training.device, normalize_advantages=True, num_players=n,
        )
        ppo_metrics = ppo.update(batch, epochs=c.training.update_epochs,
                                 minibatch_size=c.training.minibatch_size)

        # 3. 把当前模型快照加入池子
        if it % args.snapshot_interval == 0:
            snap_model = create_model(
                obs_dim=game.observation_shape[0], action_size=game.action_space_size,
                encoder_type=c.model.encoder_type, config=c.model.config,
                aux_dim=game.auxiliary_shape[0] if game.auxiliary_shape else None,
                encoder_params=game.encoder_params,
            )
            snap_model.load_state_dict(copy.deepcopy(model.state_dict()))
            snap_model.eval()
            snap_agent = NeuralAgent(snap_model, device=c.training.device, name=f"snap_{it}")
            pool.append(("neural", snap_agent))
            # 只保留最近 pool_size 个快照（前面始终保留 random + heuristic）
            neural_entries = [e for e in pool if e[0] == "neural"]
            if len(neural_entries) > args.pool_size:
                # 移除最旧的 neural 快照
                for e in neural_entries[:-args.pool_size]:
                    pool.remove(e)

        write({
            "type": "metric", "iteration": it,
            "mean_episode_length": round(float(np.mean(ep_lens)), 2),
            "policy_loss": round(ppo_metrics["policy_loss"], 6),
            "value_loss": round(ppo_metrics["value_loss"], 6),
            "outcome_loss": round(ppo_metrics.get("outcome_loss", 0.0), 6),
            "aux_loss": round(ppo_metrics.get("aux_loss", 0.0), 6),
            "explained_variance": round(ppo_metrics.get("explained_variance", 0.0), 6),
            "entropy": round(ppo_metrics["entropy"], 6),
            "pool_size": len(pool),
            "seconds": round(time.time() - t0, 2),
        })

        # 4. 保存检查点
        if it % args.checkpoint_interval == 0:
            ckpt_dir = Path(c.training.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ppo.save_checkpoint(ckpt_dir / f"checkpoint_iter_{it}.pth", metadata={"iteration": it})
            ppo.save_checkpoint(ckpt_dir / "latest.pth", metadata={"iteration": it})

    f.close()
    print(json.dumps({"type": "done", "iterations": args.iterations}), flush=True)


if __name__ == "__main__":
    main()
