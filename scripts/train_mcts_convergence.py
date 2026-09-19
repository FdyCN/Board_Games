#!/usr/bin/env python3
"""
MCTS 增强训练的 JSONL 收敛日志脚本（单进程）。

用法:
    python scripts/train_mcts_convergence.py \
        --config configs/splendor_mcts_ppo_3p.yaml \
        --iterations 100 --log-file data/logs/mcts_3p.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config_loader import Config
from games.registry import create_game
from models.model_factory import create_model
from training.trainer import Trainer
from training.experience import ExperienceBatch, split_episodes_by_player

REWARD_KEYS = [
    "take_gem", "discard_gem", "reserve_card", "get_gold",
    "buy_card_points", "buy_card_bonus", "noble_visit", "win", "step_penalty",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--log-file", required=True)
    args = ap.parse_args()

    c = Config.from_yaml(args.config)
    reward_config = {k: getattr(c.algorithm.dense_rewards, k) for k in REWARD_KEYS}

    game = create_game("splendor", num_players=c.game.num_players, reward_config=reward_config)
    model = create_model(
        obs_dim=game.observation_shape[0], action_size=game.action_space_size,
        encoder_type=c.model.encoder_type, config=c.model.config,
    )

    mcts_kwargs = {}
    if c.algorithm.mcts.enabled:
        mcts_kwargs = {
            "use_mcts": True,
            "mcts_simulations": c.algorithm.mcts.simulations,
            "mcts_c_puct": c.algorithm.mcts.c_puct,
            "mcts_add_noise": c.algorithm.mcts.add_noise,
            "mcts_temperature": c.algorithm.mcts.temperature,
        }
        if c.algorithm.mcts.scheduler.enabled:
            mcts_kwargs["mcts_scheduler"] = c.algorithm.mcts.scheduler.schedule

    trainer = Trainer(
        game=game, model=model,
        learning_rate=c.algorithm.learning_rate,
        gamma=c.algorithm.gamma, gae_lambda=c.algorithm.gae_lambda,
        clip_epsilon=c.algorithm.clip_epsilon,
        value_coef=c.algorithm.value_coef,
        entropy_coef=c.algorithm.entropy_coef,
        max_grad_norm=c.algorithm.max_grad_norm,
        device=c.training.device,
        checkpoint_dir=c.training.checkpoint_dir,
        verbose=False, num_workers=c.training.num_workers,
        use_tensorboard=False,
        use_value_clip=c.algorithm.use_value_clip,
        value_clip_epsilon=c.algorithm.value_clip_epsilon,
        outcome_coef=c.algorithm.outcome_coef,
        **mcts_kwargs,
    )

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    f.write(json.dumps({
        "type": "header", "config": args.config, "num_players": c.game.num_players,
        "iterations": args.iterations, "episodes_per_iteration": c.training.episodes_per_iteration,
        "reward_config": reward_config, "mcts_simulations": c.algorithm.mcts.simulations,
    }) + "\n")
    f.flush()

    n = game.num_players
    sims = c.algorithm.mcts.simulations

    for it in range(1, args.iterations + 1):
        trainer.iteration = it
        t0 = time.time()

        # MCTS 数据收集
        episodes = trainer._collect_with_mcts(
            num_episodes=c.training.episodes_per_iteration, mcts_simulations=sims,
        )
        trainer.total_episodes += len(episodes)
        trainer.total_steps += sum(ep.num_steps for ep in episodes)

        # 组装批次
        player_experiences = split_episodes_by_player(episodes)
        all_experiences = [e for pe in player_experiences for e in pe]
        batch = ExperienceBatch.from_experiences(
            all_experiences, device=c.training.device,
            normalize_advantages=True, num_players=n,
        )
        ppo_metrics = trainer.ppo.update(
            batch, epochs=c.training.update_epochs, minibatch_size=c.training.minibatch_size,
        )

        episode_lengths = [ep.num_steps for ep in episodes]
        win_counts = [sum(1 for ep in episodes if ep.winner == i) for i in range(n)]
        record = {
            "type": "metric", "iteration": it,
            "mean_episode_length": round(float(np.mean(episode_lengths)), 2),
            "win_rates": [round(x / len(episodes), 4) for x in win_counts],
            "policy_loss": round(ppo_metrics["policy_loss"], 6),
            "value_loss": round(ppo_metrics["value_loss"], 6),
            "outcome_loss": round(ppo_metrics.get("outcome_loss", 0.0), 6),
            "explained_variance": round(ppo_metrics.get("explained_variance", 0.0), 6),
            "entropy": round(ppo_metrics["entropy"], 6),
            "seconds": round(time.time() - t0, 2),
        }
        f.write(json.dumps(record) + "\n")
        f.flush()
        print(json.dumps(record), flush=True)

        if it % c.training.checkpoint_interval == 0:
            trainer._save_checkpoint()

    f.close()
    print(json.dumps({"type": "done", "iterations": args.iterations}), flush=True)


if __name__ == "__main__":
    main()
