#!/usr/bin/env python3
"""
对手池（Opponent Pool）训练脚本。

核心思想：自对弈时，除了用"当前正在训练的模型"，还随机混合一批
"历史检查点"作为对手。这样：
1. 引入实力不对称 → 价值函数（value）有东西可学（explained_variance 会转正）；
2. 对手固定（冻结） → 降低自对弈的非平稳性，训练更稳定；
3. 对手多样性 → 避免策略循环/崩塌。

只把"当前模型"产生的经验用于 PPO 更新，历史快照只当对手、不参与更新。

用法:
    python scripts/train_opponent_pool.py \
        --config configs/splendor_ppo_mlp_medium_3p.yaml \
        --iterations 500 --episodes 128 \
        --log-file data/logs/convergence_3p_pool.jsonl \
        --snapshot-interval 20 --pool-size 5 --opponent-prob 0.5 \
        --checkpoint-interval 50
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config_loader import Config
from games.registry import create_game
from models.model_factory import create_model
from agents.neural_agent import NeuralAgent
from training.trainer import Trainer
from training.experience import ExperienceBatch
from training.self_play_worker import collect_episode

REWARD_KEYS = [
    "take_gem", "discard_gem", "reserve_card", "get_gold",
    "buy_card_points", "buy_card_bonus", "noble_visit", "win", "step_penalty",
]


def _make_model(obs_dim, action_size, encoder_type, model_config):
    return create_model(
        obs_dim=obs_dim, action_size=action_size,
        encoder_type=encoder_type, config=model_config,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--log-file", required=True)
    ap.add_argument("--snapshot-interval", type=int, default=20)
    ap.add_argument("--pool-size", type=int, default=5)
    ap.add_argument("--opponent-prob", type=float, default=0.5)
    ap.add_argument("--checkpoint-interval", type=int, default=50)
    args = ap.parse_args()

    c = Config.from_yaml(args.config)
    episodes_per_iter = args.episodes or c.training.episodes_per_iteration
    reward_config = {k: getattr(c.algorithm.dense_rewards, k) for k in REWARD_KEYS}

    game = create_game("splendor", num_players=c.game.num_players, reward_config=reward_config)
    model = _make_model(game.observation_shape[0], game.action_space_size,
                        c.model.encoder_type, c.model.config)

    trainer = Trainer(
        game=game, model=model,
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
        num_workers=1,  # 对手池用单进程（每局玩家模型不同，不便于多进程序列化）
        use_tensorboard=False,
        use_value_clip=c.algorithm.use_value_clip,
        value_clip_epsilon=c.algorithm.value_clip_epsilon,
        outcome_coef=c.algorithm.outcome_coef,
    )

    # 当前模型 agent（与 trainer.ppo.model 共享同一个 model 对象）
    current_agent = NeuralAgent(model, device=c.training.device, name="current")
    pool_agents: list[NeuralAgent] = []  # 冻结的历史快照对手

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    header = {
        "type": "header", "config": args.config, "num_players": c.game.num_players,
        "iterations": args.iterations, "episodes_per_iteration": episodes_per_iter,
        "reward_config": reward_config, "learning_rate": c.algorithm.learning_rate,
        "gamma": c.algorithm.gamma, "gae_lambda": c.algorithm.gae_lambda,
        "value_coef": c.algorithm.value_coef, "entropy_coef": c.algorithm.entropy_coef,
        "snapshot_interval": args.snapshot_interval, "pool_size": args.pool_size,
        "opponent_prob": args.opponent_prob,
    }
    f.write(json.dumps(header) + "\n")
    f.flush()

    n = game.num_players

    for it in range(1, args.iterations + 1):
        trainer.iteration = it
        t0 = time.time()

        # 1. 收集数据：混合当前模型与对手池
        episodes = []
        current_experiences = []
        for _ in range(episodes_per_iter):
            use_current = [random.random() < args.opponent_prob for _ in range(n)]
            if not any(use_current):
                use_current[random.randrange(n)] = True

            agents = []
            for p in range(n):
                if use_current[p]:
                    agents.append(current_agent)
                elif pool_agents:
                    agents.append(random.choice(pool_agents))
                else:
                    agents.append(current_agent)  # 池为空时退化为普通自对弈

            ep = collect_episode(game, agents, gamma=c.algorithm.gamma,
                                 gae_lambda=c.algorithm.gae_lambda, deterministic=False)
            episodes.append(ep)
            for exp in ep.experiences:
                if use_current[exp.player_id]:
                    current_experiences.append(exp)

        trainer.total_episodes += len(episodes)
        trainer.total_steps += sum(ep.num_steps for ep in episodes)

        # 2. 只用当前模型的经验训练
        if not current_experiences:
            print(json.dumps({"type": "metric", "iteration": it, "warning": "no current experiences"}))
            continue
        batch = ExperienceBatch.from_experiences(
            current_experiences, device=c.training.device,
            normalize_advantages=True, num_players=n,
        )
        ppo_metrics = trainer.ppo.update(
            batch, epochs=c.training.update_epochs, minibatch_size=c.training.minibatch_size,
        )

        # 3. 统计
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
            "kl_div": round(ppo_metrics["kl_div"], 6),
            "clip_fraction": round(ppo_metrics["clip_fraction"], 6),
            "pool_size": len(pool_agents),
            "seconds": round(time.time() - t0, 2),
        }
        f.write(json.dumps(record) + "\n")
        f.flush()
        print(json.dumps(record), flush=True)

        # 4. 定期把当前模型快照加入对手池
        if it % args.snapshot_interval == 0:
            snap_model = _make_model(game.observation_shape[0], game.action_space_size,
                                     c.model.encoder_type, c.model.config)
            snap_model.load_state_dict(copy.deepcopy(model.state_dict()))
            snap_model.eval()
            pool_agents.append(NeuralAgent(snap_model, device=c.training.device, name=f"pool_{it}"))
            if len(pool_agents) > args.pool_size:
                pool_agents.pop(0)

        # 5. 保存检查点
        if it % args.checkpoint_interval == 0:
            trainer._save_checkpoint()

    f.close()
    print(json.dumps({"type": "done", "iterations": args.iterations}), flush=True)


if __name__ == "__main__":
    main()
