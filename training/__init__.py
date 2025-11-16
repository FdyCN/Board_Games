"""
Training Module

提供训练相关的功能，包括:
- 经验数据批处理
- 经验回放池
- 自对弈数据收集
- PPO 算法实现
- 训练循环
"""

from training.experience import (
    ExperienceBatch,
    compute_advantages_for_episode,
    split_episodes_by_player,
    merge_experience_batches,
)
from training.replay_buffer import ReplayBuffer, EpisodeBuffer
from training.self_play_worker import (
    collect_episode,
    collect_episodes,
    SelfPlayWorker,
)
from training.algorithms.ppo import PPO
from training.trainer import Trainer

__all__ = [
    # Experience
    "ExperienceBatch",
    "compute_advantages_for_episode",
    "split_episodes_by_player",
    "merge_experience_batches",
    # Replay Buffer
    "ReplayBuffer",
    "EpisodeBuffer",
    # Self-Play Worker
    "collect_episode",
    "collect_episodes",
    "SelfPlayWorker",
    # Algorithms
    "PPO",
    # Trainer
    "Trainer",
]
