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

# TensorBoard 支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


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
        num_workers: int = 1,
        use_tensorboard: bool = True,
        tensorboard_dir: Optional[str] = None,
        use_value_clip: bool = True,
        value_clip_epsilon: float = 0.4,
        outcome_coef: float = 1.0,
        # MCTS 参数 (新增)
        use_mcts: bool = False,
        mcts_simulations: int = 100,
        mcts_c_puct: float = 1.5,
        mcts_add_noise: bool = True,
        mcts_temperature: float = 1.0,
        mcts_scheduler: Optional[List] = None,
    ):
        """
        初始化训练器

        Args:
            game: 游戏实例
            model: Actor-Critic 模型
            learning_rate: 学习率
            gamma: 折扣因子
            gae_lambda: GAE 参数
            clip_epsilon: PPO clip 参数（用于策略）
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪阈值
            device: 设备
            checkpoint_dir: 检查点保存目录
            verbose: 是否打印详细信息
            num_workers: 并行进程数（1=单进程，>1=多进程）
            use_tensorboard: 是否启用 TensorBoard
            tensorboard_dir: TensorBoard 日志目录（默认为 checkpoint_dir/tensorboard）
            use_value_clip: 是否使用价值损失裁剪（推荐启用以稳定训练）
            value_clip_epsilon: 价值损失裁剪参数（通常 0.3-0.5，比 clip_epsilon 更大）
            outcome_coef: 终局胜负辅助任务损失系数
            use_mcts: 是否启用 MCTS 增强训练
            mcts_simulations: MCTS 模拟次数 (0=禁用MCTS)
            mcts_c_puct: MCTS UCB 探索常数
            mcts_add_noise: 是否添加 Dirichlet 噪声
            mcts_temperature: 动作采样温度
            mcts_scheduler: MCTS 调度表 [(iteration, simulations), ...]
        """
        self.game = game
        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        self.num_workers = num_workers

        # 创建 PPO 训练器
        self.ppo = PPO(
            model=model,
            learning_rate=learning_rate,
            clip_epsilon=clip_epsilon,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            use_value_clip=use_value_clip,
            value_clip_epsilon=value_clip_epsilon,
            outcome_coef=outcome_coef,
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
            num_workers=num_workers,
        )

        # Episode 缓冲区
        self.episode_buffer = EpisodeBuffer(capacity=1000)

        # MCTS 配置 (新增)
        self.use_mcts = use_mcts
        self.mcts = None
        self.mcts_scheduler = None

        if use_mcts:
            from training.mcts import MCTS, MCTSScheduler

            self.mcts = MCTS(
                game=game,
                c_puct=mcts_c_puct,
                add_noise=mcts_add_noise,
                device=device,
            )
            self.mcts_simulations = mcts_simulations
            self.mcts_temperature = mcts_temperature

            # 创建调度器
            if mcts_scheduler:
                self.mcts_scheduler = MCTSScheduler(mcts_scheduler)
                if verbose:
                    print("MCTS 调度器:")
                    print(self.mcts_scheduler.get_schedule_info())

        # 训练统计
        self.iteration = 0
        self.total_episodes = 0
        self.total_steps = 0
        self.start_time = None

        # 创建检查点目录
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # TensorBoard 设置
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            if tensorboard_dir is None:
                tensorboard_dir = os.path.join(checkpoint_dir or "runs", "tensorboard")
            Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_dir)
            if self.verbose:
                print(f"TensorBoard 日志目录: {tensorboard_dir}")
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            if self.verbose:
                print("警告: TensorBoard 不可用，请安装 tensorboard: pip install tensorboard")

    def train(
        self,
        num_iterations: int,
        episodes_per_iteration: int = 100,
        update_epochs: int = 4,
        minibatch_size: Optional[int] = 256,
        checkpoint_interval: int = 10,
        log_interval: int = 1,
        use_position_augmentation: bool = False,
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
            use_position_augmentation: 是否使用位置旋转数据增强（减少位置偏差）

        Returns:
            训练指标列表
        """
        if self.start_time is None:
            self.start_time = time.time()

        all_metrics = []

        # 保存起始迭代次数（用于恢复训练）
        start_iteration = self.iteration

        if self.verbose:
            print("=" * 60)
            print("📊 训练配置")
            print("=" * 60)
            print(f"起始迭代: {start_iteration + 1}")
            print(f"目标迭代: {start_iteration + num_iterations}")
            print(f"每次迭代 episodes: {episodes_per_iteration}")
            print(f"PPO 更新轮数: {update_epochs}")
            print(f"小批次大小: {minibatch_size}")
            print(f"并行进程数: {self.worker.num_workers}")
            print(f"位置增强: {'启用' if use_position_augmentation else '禁用'}")
            print(f"MCTS 增强: {'启用' if self.use_mcts else '禁用'}")
            if self.use_mcts and not self.mcts_scheduler:
                print(f"MCTS 模拟次数: {self.mcts_simulations}")
            print(f"设备: {self.device}")
            print("=" * 60)

        for iteration in range(num_iterations):
            self.iteration = start_iteration + iteration + 1

            # 获取当前迭代的 MCTS 模拟次数 (如果使用调度器)
            current_mcts_sims = 0
            if self.use_mcts:
                if self.mcts_scheduler:
                    current_mcts_sims = self.mcts_scheduler.get_simulations(self.iteration)
                else:
                    current_mcts_sims = self.mcts_simulations

            # === 1. 收集数据 ===
            if self.verbose and iteration % log_interval == 0:
                mcts_info = f" (MCTS: {current_mcts_sims} sims)" if current_mcts_sims > 0 else ""
                print(f"\n迭代 {self.iteration}/{start_iteration + num_iterations}: 收集数据...{mcts_info}")

            collection_start = time.time()

            # 选择数据收集方式
            if current_mcts_sims > 0:
                # 使用 MCTS 增强的数据收集
                episodes = self._collect_with_mcts(
                    num_episodes=episodes_per_iteration,
                    mcts_simulations=current_mcts_sims
                )
            else:
                # 原始 PPO 数据收集
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
                use_position_augmentation=use_position_augmentation,
                num_players=self.game.num_players,
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

            # === 5.5 TensorBoard 记录 ===
            if self.writer is not None:
                # 损失指标
                self.writer.add_scalar("Loss/policy", metrics.policy_loss, self.iteration)
                self.writer.add_scalar("Loss/value", metrics.value_loss, self.iteration)
                self.writer.add_scalar("Loss/entropy", metrics.entropy, self.iteration)

                # 性能指标
                self.writer.add_scalar("Performance/mean_reward", metrics.mean_reward, self.iteration)
                self.writer.add_scalar("Performance/mean_episode_length", metrics.mean_episode_length, self.iteration)

                # PPO 指标
                self.writer.add_scalar("PPO/kl_divergence", metrics.kl_divergence, self.iteration)
                self.writer.add_scalar("PPO/clip_fraction", metrics.clip_fraction, self.iteration)
                self.writer.add_scalar("PPO/learning_rate", metrics.learning_rate, self.iteration)

                # 位置偏差指标
                win_rates = [c / len(episodes) for c in win_counts]
                for i, rate in enumerate(win_rates):
                    self.writer.add_scalar(f"WinRate/position_{i}", rate, self.iteration)

                # 位置偏差统计
                win_rate_std = np.std(win_rates)
                win_rate_range = max(win_rates) - min(win_rates)
                self.writer.add_scalar("PositionBias/win_rate_std", win_rate_std, self.iteration)
                self.writer.add_scalar("PositionBias/win_rate_range", win_rate_range, self.iteration)

                # 时间指标
                self.writer.add_scalar("Time/collection_time", collection_time, self.iteration)
                self.writer.add_scalar("Time/update_time", update_time, self.iteration)

                # MCTS 指标
                if self.use_mcts:
                    self.writer.add_scalar("MCTS/simulations", current_mcts_sims, self.iteration)

                # 定期刷新，确保训练中可以实时查看
                if self.iteration % 10 == 0:
                    self.writer.flush()

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

        # 刷新 TensorBoard 日志
        if self.writer is not None:
            self.writer.flush()

        return all_metrics

    def close(self) -> None:
        """关闭训练器资源"""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

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

    def _collect_with_mcts(
        self,
        num_episodes: int,
        mcts_simulations: int
    ):
        """
        使用 MCTS 增强的数据收集

        Args:
            num_episodes: 要收集的 episode 数量
            mcts_simulations: MCTS 模拟次数

        Returns:
            Episode 列表
        """
        import torch
        from training.experience import Episode, Experience

        episodes = []

        for ep_idx in range(num_episodes):
            state = self.game.reset()
            trajectory = []

            while not self.game.is_terminal(state):
                current_player = self.game.get_current_player(state)

                # 获取合法动作 (先检查是否有合法动作)
                legal_actions = self.game.get_legal_actions(state)
                if not legal_actions:
                    # 没有合法动作,游戏应该结束
                    break

                # MCTS 搜索获取改进的动作概率
                action_probs, legal_actions = self.mcts.search(
                    state=state,
                    model=self.model,
                    num_simulations=mcts_simulations,
                    temperature=self.mcts_temperature,
                    verbose=False,
                )

                # 采样动作（用 MCTS 改进后的访问分布做探索）
                action_idx = np.random.choice(len(legal_actions), p=action_probs)
                action = legal_actions[action_idx]
                canonical_idx = self.game.action_to_index(action, state)

                # 用策略网络（被训练的那个）计算 log_prob 和价值。
                # 注意：old_log_prob 必须来自策略网络，而不是 MCTS 访问概率，
                # 否则 PPO 的重要性采样比率 ratio 会算错（比较的是两种不同分布）。
                obs = self.game.state_to_observation(state, current_player)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                legal_mask = np.zeros(self.game.action_space_size, dtype=bool)
                for a in legal_actions:
                    legal_mask[self.game.action_to_index(a, state)] = True
                legal_mask_tensor = torch.BoolTensor(legal_mask).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits, value = self.model(obs_tensor, legal_mask_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    log_prob = torch.log(probs[0, canonical_idx] + 1e-8).item()
                    value = value.item()

                # 执行动作
                next_state, rewards, done, info = self.game.step(action)

                # 保存经验
                # 只保存稠密奖励；终局奖励在 _trajectory_to_episode 中按玩家注入
                trajectory.append({
                    'player_id': current_player,
                    'observation': obs,
                    'action': canonical_idx,
                    'log_prob': log_prob,
                    'value': value,
                    'reward': info.get("dense_reward", rewards[current_player]),
                    'legal_mask': legal_mask,
                })

                state = next_state

            # 从真实终局状态确定胜者和终局奖励
            winner = self.game.get_winner(state) if self.game.is_terminal(state) else -1
            total_rewards = self.game.get_final_rewards(state)

            # 转换为 Episode 对象
            episode = self._trajectory_to_episode(
                trajectory, winner=winner, total_rewards=total_rewards
            )
            episodes.append(episode)

        return episodes

    def _trajectory_to_episode(self, trajectory: list, winner: int, total_rewards: list):
        """
        将轨迹转换为 Episode 对象

        Args:
            trajectory: 轨迹数据
            winner: 真实终局胜者（来自游戏状态）
            total_rewards: 每个玩家的终局奖励（用于记录/统计）

        Returns:
            Episode 对象
        """
        from training.experience import Episode, Experience, compute_advantages_for_episode

        # 按时间顺序展开所有玩家的经验（保持交错顺序，与标准路径一致）
        experiences = []
        for step in trajectory:
            exp = Experience(
                player_id=step['player_id'],
                observation=step['observation'],
                action=step['action'],
                reward=step['reward'],
                log_prob=step['log_prob'],
                value=step['value'],
                legal_actions_mask=step['legal_mask'],
                outcome=1.0 if (winner >= 0 and step['player_id'] == winner) else 0.0,
            )
            experiences.append(exp)

        episode = Episode(
            experiences=experiences,
            total_rewards=total_rewards,
            winner=winner,
            num_steps=len(trajectory),
        )

        # 复用统一的按玩家 GAE 计算 + 终局零和奖励注入
        compute_advantages_for_episode(
            episode, gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        return episode

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
