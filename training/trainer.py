"""
训练循环 (Trainer)

整合自对弈数据收集、PPO 更新、日志记录等功能。
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np

from core.game_interface import GameInterface
from core.agent_interface import AgentInterface
from core.types import TrainingMetrics
from agents.neural_agent import NeuralAgent
from training.algorithms.ppo import PPO
from training.self_play_worker import SelfPlayWorker, collect_episodes
from training.replay_buffer import EpisodeBuffer
from training.experience import (
    ExperienceBatch,
    split_episodes_by_player,
)


class Trainer:
    """
    PPO 训练器

    整合数据收集和模型更新的完整训练循环。

    Attributes:
        game: 游戏实例
        model: Actor-Critic 模型
        ppo: PPO 训练器
        agents: Neural Agent 列表
        worker: 自对弈 Worker
        episode_buffer: Episode 缓冲区

    Examples:
        >>> from games.splendor import SplendorGame
        >>> from models.model_factory import create_splendor_model
        >>> game = SplendorGame(num_players=4)
        >>> model = create_splendor_model(encoder_type="mlp", config="medium")
        >>> trainer = Trainer(
        ...     game=game,
        ...     model=model,
        ...     learning_rate=3e-4,
        ...     gamma=0.99,
        ...     gae_lambda=0.95,
        ... )
        >>> trainer.train(num_iterations=100, episodes_per_iteration=50)
    """

    def __init__(
        self,
        game: GameInterface,
        model,  # ActorCritic model
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        初始化训练器

        Args:
            game: 游戏实例
            model: Actor-Critic 模型
            learning_rate: 学习率
            gamma: 折扣因子
            gae_lambda: GAE 参数
            clip_epsilon: PPO clip 参数
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪阈值
            device: 设备
            checkpoint_dir: 检查点保存目录
            verbose: 是否打印详细信息
        """
        self.game = game
        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose

        # 创建 PPO 训练器
        self.ppo = PPO(
            model=model,
            learning_rate=learning_rate,
            clip_epsilon=clip_epsilon,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            device=device,
        )

        # 创建 Neural Agents（所有玩家共享同一个模型）
        self.agents: List[AgentInterface] = [
            NeuralAgent(model, device=device, name=f"Agent_{i}")
            for i in range(game.num_players)
        ]

        # 创建自对弈 Worker
        self.worker = SelfPlayWorker(
            game=game,
            agents=self.agents,
            gamma=gamma,
            gae_lambda=gae_lambda,
            deterministic=False,
            verbose=False,
        )

        # Episode 缓冲区
        self.episode_buffer = EpisodeBuffer(capacity=1000)

        # 训练统计
        self.iteration = 0
        self.total_episodes = 0
        self.total_steps = 0
        self.start_time = None

        # 创建检查点目录
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train(
        self,
        num_iterations: int,
        episodes_per_iteration: int = 100,
        update_epochs: int = 4,
        minibatch_size: Optional[int] = 256,
        checkpoint_interval: int = 10,
        log_interval: int = 1,
    ) -> List[TrainingMetrics]:
        """
        执行训练循环

        Args:
            num_iterations: 训练迭代次数
            episodes_per_iteration: 每次迭代收集的 episode 数
            update_epochs: PPO 更新轮数
            minibatch_size: 小批次大小
            checkpoint_interval: 保存检查点的间隔（迭代数）
            log_interval: 打印日志的间隔（迭代数）

        Returns:
            训练指标列表
        """
        if self.start_time is None:
            self.start_time = time.time()

        all_metrics = []

        if self.verbose:
            print("=" * 60)
            print("开始训练")
            print(f"迭代次数: {num_iterations}")
            print(f"每次迭代 episodes: {episodes_per_iteration}")
            print(f"PPO 更新轮数: {update_epochs}")
            print(f"小批次大小: {minibatch_size}")
            print("=" * 60)

        for iteration in range(num_iterations):
            self.iteration = iteration + 1

            # === 1. 收集数据 ===
            if self.verbose and iteration % log_interval == 0:
                print(f"\n迭代 {self.iteration}/{num_iterations}: 收集数据...")

            collection_start = time.time()
            episodes = self.worker.collect(num_episodes=episodes_per_iteration)
            collection_time = time.time() - collection_start

            # 保存 episodes
            self.episode_buffer.add_batch(episodes)
            self.total_episodes += len(episodes)
            self.total_steps += sum(ep.num_steps for ep in episodes)

            # === 2. 处理经验 ===
            # 按玩家分组（在自对弈中，所有玩家使用相同模型）
            player_experiences = split_episodes_by_player(episodes)

            # 合并所有玩家的经验（因为共享模型）
            all_experiences = []
            for player_exps in player_experiences:
                all_experiences.extend(player_exps)

            # 创建批次
            batch = ExperienceBatch.from_experiences(
                all_experiences,
                device=self.device,
                normalize_advantages=True,
            )

            # === 3. 更新模型 ===
            if self.verbose and iteration % log_interval == 0:
                print(f"更新模型 (batch_size={batch.batch_size})...")

            update_start = time.time()
            ppo_metrics = self.ppo.update(
                batch, epochs=update_epochs, minibatch_size=minibatch_size
            )
            update_time = time.time() - update_start

            # === 4. 计算统计指标 ===
            episode_lengths = [ep.num_steps for ep in episodes]
            episode_rewards = [sum(ep.total_rewards) for ep in episodes]
            win_counts = [sum(1 for ep in episodes if ep.winner == i) for i in range(self.game.num_players)]

            metrics = TrainingMetrics(
                episode=self.iteration,
                policy_loss=ppo_metrics["policy_loss"],
                value_loss=ppo_metrics["value_loss"],
                entropy=ppo_metrics["entropy"],
                mean_reward=np.mean(episode_rewards),
                mean_episode_length=np.mean(episode_lengths),
                learning_rate=self.ppo.get_learning_rate(),
                kl_divergence=ppo_metrics["kl_div"],
                clip_fraction=ppo_metrics["clip_fraction"],
            )

            all_metrics.append(metrics)

            # === 5. 日志输出 ===
            if self.verbose and iteration % log_interval == 0:
                elapsed_time = time.time() - self.start_time
                print(f"\n迭代 {self.iteration} 统计:")
                print(f"  收集: {len(episodes)} episodes, {batch.batch_size} 经验")
                print(f"  收集时间: {collection_time:.2f}s, 更新时间: {update_time:.2f}s")
                print(f"  平均步数: {metrics.mean_episode_length:.1f}")
                print(f"  平均奖励: {metrics.mean_reward:.3f}")
                print(f"  胜率分布: {[f'{c/len(episodes):.2%}' for c in win_counts]}")
                print(f"  Policy Loss: {metrics.policy_loss:.4f}")
                print(f"  Value Loss: {metrics.value_loss:.4f}")
                print(f"  Entropy: {metrics.entropy:.4f}")
                print(f"  KL Div: {metrics.kl_divergence:.4f}")
                print(f"  Clip Fraction: {metrics.clip_fraction:.4f}")
                print(f"  总时间: {elapsed_time:.1f}s")

            # === 6. 保存检查点 ===
            if self.checkpoint_dir and (iteration + 1) % checkpoint_interval == 0:
                self._save_checkpoint()

        if self.verbose:
            total_time = time.time() - self.start_time
            print("\n" + "=" * 60)
            print("训练完成!")
            print(f"总迭代: {self.iteration}")
            print(f"总 episodes: {self.total_episodes}")
            print(f"总步数: {self.total_steps}")
            print(f"总时间: {total_time:.1f}s")
            print("=" * 60)

        return all_metrics

    def _save_checkpoint(self) -> None:
        """保存训练检查点"""
        if not self.checkpoint_dir:
            return

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_iter_{self.iteration}.pth"
        )

        metadata = {
            "iteration": self.iteration,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0,
        }

        self.ppo.save_checkpoint(checkpoint_path, metadata=metadata)

        if self.verbose:
            print(f"  保存检查点: {checkpoint_path}")

        # 同时保存最新模型
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        self.ppo.save_checkpoint(latest_path, metadata=metadata)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载训练检查点

        Args:
            checkpoint_path: 检查点路径
        """
        metadata = self.ppo.load_checkpoint(checkpoint_path)

        # 恢复训练状态
        self.iteration = metadata.get("iteration", 0)
        self.total_episodes = metadata.get("total_episodes", 0)
        self.total_steps = metadata.get("total_steps", 0)

        if self.verbose:
            print(f"加载检查点: {checkpoint_path}")
            print(f"  迭代: {self.iteration}")
            print(f"  总 episodes: {self.total_episodes}")
            print(f"  总步数: {self.total_steps}")

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        评估当前模型

        Args:
            num_episodes: 评估的 episode 数量

        Returns:
            评估指标字典
        """
        # 切换到评估模式
        self.model.eval()
        for agent in self.agents:
            agent.set_training_mode(False)

        # 收集评估数据（使用确定性策略）
        eval_worker = SelfPlayWorker(
            game=self.game,
            agents=self.agents,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            deterministic=True,  # 评估时使用确定性策略
            verbose=False,
        )

        episodes = eval_worker.collect(num_episodes=num_episodes)

        # 计算指标
        episode_rewards = [sum(ep.total_rewards) for ep in episodes]
        episode_lengths = [ep.num_steps for ep in episodes]
        win_counts = [sum(1 for ep in episodes if ep.winner == i) for i in range(self.game.num_players)]

        metrics = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
            "win_rates": [count / len(episodes) for count in win_counts],
        }

        # 恢复训练模式
        self.model.train()
        for agent in self.agents:
            agent.set_training_mode(True)

        return metrics


# ===== 导出 =====
__all__ = ["Trainer"]
