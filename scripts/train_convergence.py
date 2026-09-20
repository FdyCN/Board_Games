#!/usr/bin/env python3
"""
训练并输出结构化 JSONL 指标日志，用于收敛分析（游戏无关）。

用法:
    python scripts/train_convergence.py \
        --config configs/splendor/splendor_ppo_mlp_medium_3p.yaml \
        --iterations 300 \
        --episodes 128 \
        --log-file data/splendor/logs/convergence_3p.jsonl \
        --checkpoint-interval 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config_loader import Config, build_game_kwargs
from games.registry import create_game
from models.model_factory import create_model
from training.trainer import Trainer
from training.experience import ExperienceBatch, split_episodes_by_player


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--log-file", required=True)
    ap.add_argument("--checkpoint-interval", type=int, default=50)
    args = ap.parse_args()

    c = Config.from_yaml(args.config)
    episodes_per_iter = args.episodes if args.episodes is not None else c.training.episodes_per_iteration

    reward_config = dict(c.algorithm.dense_rewards)

    game = create_game(c.game.name, **build_game_kwargs(c))
    model = create_model(
        obs_dim=game.observation_shape[0],
        action_size=game.action_space_size,
        encoder_type=c.model.encoder_type,
        config=c.model.config,
        aux_dim=game.auxiliary_shape[0] if game.auxiliary_shape else None,
    )
    trainer = Trainer(
        game=game,
        model=model,
        learning_rate=c.algorithm.learning_rate,
        gamma=c.algorithm.gamma,
        gae_lambda=c.algorithm.gae_lambda,
        clip_epsilon=c.algorithm.clip_epsilon,
        value_coef=c.algorithm.value_coef,
        entropy_coef=c.algorithm.entropy_coef,
        max_grad_norm=c.algorithm.max_grad_norm,
        device=c.training.device,
        checkpoint_dir=c.training.checkpoint_dir,
        verbose=False,
        num_workers=c.training.num_workers,
        use_tensorboard=False,
        use_value_clip=c.algorithm.use_value_clip,
        value_clip_epsilon=c.algorithm.value_clip_epsilon,
        outcome_coef=c.algorithm.outcome_coef,
    )

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")

    header = {
        "config": args.config,
        "num_players": c.game.num_players,
        "model": c.model.encoder_type,
        "iterations": args.iterations,
        "episodes_per_iteration": episodes_per_iter,
        "reward_config": reward_config,
        "learning_rate": c.algorithm.learning_rate,
        "gamma": c.algorithm.gamma,
        "gae_lambda": c.algorithm.gae_lambda,
        "clip_epsilon": c.algorithm.clip_epsilon,
        "entropy_coef": c.algorithm.entropy_coef,
        "value_coef": c.algorithm.value_coef,
    }
    f.write(json.dumps({"type": "header", **header}) + "\n")
    f.flush()

    n = game.num_players

    for it in range(1, args.iterations + 1):
        trainer.iteration = it
        t0 = time.time()

        # 1. 收集数据
        episodes = trainer.worker.collect(num_episodes=episodes_per_iter)
        trainer.episode_buffer.add_batch(episodes)
        trainer.total_episodes += len(episodes)
        trainer.total_steps += sum(ep.num_steps for ep in episodes)

        # 2. 组装批次
        player_experiences = split_episodes_by_player(episodes)
        all_experiences = []
        for player_exps in player_experiences:
            all_experiences.extend(player_exps)
        batch = ExperienceBatch.from_experiences(
            all_experiences,
            device=trainer.device,
            normalize_advantages=True,
            use_position_augmentation=False,
            num_players=n,
        )

        # 3. PPO 更新
        ppo_metrics = trainer.ppo.update(
            batch,
            epochs=c.training.update_epochs,
            minibatch_size=c.training.minibatch_size,
        )

        # 4. 统计指标
        episode_lengths = [ep.num_steps for ep in episodes]
        win_counts = [sum(1 for ep in episodes if ep.winner == i) for i in range(n)]

        record = {
            "type": "metric",
            "iteration": it,
            "mean_episode_length": round(float(np.mean(episode_lengths)), 2),
            "win_rates": [round(c / len(episodes), 4) for c in win_counts],
            "policy_loss": round(ppo_metrics["policy_loss"], 6),
            "value_loss": round(ppo_metrics["value_loss"], 6),
            "outcome_loss": round(ppo_metrics.get("outcome_loss", 0.0), 6),
            "aux_loss": round(ppo_metrics.get("aux_loss", 0.0), 6),
            "explained_variance": round(ppo_metrics.get("explained_variance", 0.0), 6),
            "entropy": round(ppo_metrics["entropy"], 6),
            "kl_div": round(ppo_metrics["kl_div"], 6),
            "clip_fraction": round(ppo_metrics["clip_fraction"], 6),
            "seconds": round(time.time() - t0, 2),
        }
        f.write(json.dumps(record) + "\n")
        f.flush()
        print(json.dumps(record), flush=True)

        # 5. 定期保存检查点
        if it % args.checkpoint_interval == 0:
            trainer._save_checkpoint()

    f.close()
    print(json.dumps({"type": "done", "iterations": args.iterations}), flush=True)


if __name__ == "__main__":
    main()
