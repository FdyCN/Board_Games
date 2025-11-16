"""
经验回放池 (Replay Buffer)

用于存储和采样经验数据，支持 PPO 的 on-policy 训练。
"""

from collections import deque
from typing import List, Optional
import numpy as np

from core.types import Experience, Episode


class ReplayBuffer:
    """
    经验回放池

    存储游戏经验，支持批量采样和清空。
    PPO 是 on-policy 算法，通常在每次更新后清空缓冲区。

    Attributes:
        capacity: 最大容量（经验数量）
        experiences: 经验列表
        episodes: Episode 列表（可选，用于统计）

    Examples:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.add_episode(episode)
        >>> if buffer.is_ready(min_size=1000):
        ...     experiences = buffer.sample_all()
        ...     # 训练模型
        ...     buffer.clear()
    """

    def __init__(self, capacity: int = 100000):
        """
        初始化回放池

        Args:
            capacity: 最大容量（经验数量）
        """
        self.capacity = capacity
        self.experiences: List[Experience] = []
        self.episodes: List[Episode] = []

    def add_experience(self, experience: Experience) -> None:
        """
        添加单个经验

        Args:
            experience: Experience 对象
        """
        self.experiences.append(experience)

        # 如果超过容量，移除最旧的经验
        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)

    def add_experiences(self, experiences: List[Experience]) -> None:
        """
        批量添加经验

        Args:
            experiences: 经验列表
        """
        for exp in experiences:
            self.add_experience(exp)

    def add_episode(self, episode: Episode) -> None:
        """
        添加完整的 Episode

        Args:
            episode: Episode 对象
        """
        self.episodes.append(episode)
        self.add_experiences(episode.experiences)

    def sample_all(self) -> List[Experience]:
        """
        采样所有经验

        Returns:
            所有经验的列表
        """
        return self.experiences.copy()

    def sample_random(self, batch_size: int) -> List[Experience]:
        """
        随机采样经验（用于 off-policy 算法）

        Args:
            batch_size: 批次大小

        Returns:
            随机采样的经验列表

        Raises:
            ValueError: 如果缓冲区经验不足
        """
        if len(self.experiences) < batch_size:
            raise ValueError(
                f"缓冲区经验不足: 需要 {batch_size}, 当前 {len(self.experiences)}"
            )

        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        return [self.experiences[i] for i in indices]

    def clear(self) -> None:
        """清空缓冲区"""
        self.experiences.clear()
        self.episodes.clear()

    def is_ready(self, min_size: int) -> bool:
        """
        检查是否有足够的经验

        Args:
            min_size: 最小经验数量

        Returns:
            是否准备好
        """
        return len(self.experiences) >= min_size

    def __len__(self) -> int:
        """返回当前经验数量"""
        return len(self.experiences)

    def get_statistics(self) -> dict:
        """
        获取缓冲区统计信息

        Returns:
            统计信息字典
        """
        if not self.episodes:
            return {
                "num_experiences": len(self.experiences),
                "num_episodes": 0,
                "mean_episode_length": 0.0,
                "mean_reward": 0.0,
            }

        episode_lengths = [ep.num_steps for ep in self.episodes]
        total_rewards = [sum(ep.total_rewards) for ep in self.episodes]

        return {
            "num_experiences": len(self.experiences),
            "num_episodes": len(self.episodes),
            "mean_episode_length": np.mean(episode_lengths),
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
        }


class EpisodeBuffer:
    """
    Episode 缓冲区

    专门用于存储完整的 Episode，支持按 Episode 采样。
    适用于需要完整轨迹的算法。

    Examples:
        >>> buffer = EpisodeBuffer(capacity=1000)
        >>> buffer.add(episode)
        >>> episodes = buffer.sample_recent(n=100)
    """

    def __init__(self, capacity: int = 1000):
        """
        初始化 Episode 缓冲区

        Args:
            capacity: 最大容量（Episode 数量）
        """
        self.capacity = capacity
        self.episodes: deque[Episode] = deque(maxlen=capacity)

    def add(self, episode: Episode) -> None:
        """
        添加 Episode

        Args:
            episode: Episode 对象
        """
        self.episodes.append(episode)

    def add_batch(self, episodes: List[Episode]) -> None:
        """
        批量添加 Episodes

        Args:
            episodes: Episode 列表
        """
        for episode in episodes:
            self.add(episode)

    def sample_recent(self, n: int) -> List[Episode]:
        """
        采样最近的 n 个 Episodes

        Args:
            n: 采样数量

        Returns:
            Episode 列表
        """
        n = min(n, len(self.episodes))
        return list(self.episodes)[-n:]

    def sample_all(self) -> List[Episode]:
        """
        获取所有 Episodes

        Returns:
            所有 Episode 的列表
        """
        return list(self.episodes)

    def clear(self) -> None:
        """清空缓冲区"""
        self.episodes.clear()

    def __len__(self) -> int:
        """返回当前 Episode 数量"""
        return len(self.episodes)

    def get_statistics(self) -> dict:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        if not self.episodes:
            return {
                "num_episodes": 0,
                "mean_episode_length": 0.0,
                "mean_reward": 0.0,
            }

        episode_lengths = [ep.num_steps for ep in self.episodes]
        total_rewards = [sum(ep.total_rewards) for ep in self.episodes]

        return {
            "num_episodes": len(self.episodes),
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
        }


# ===== 导出 =====
__all__ = ["ReplayBuffer", "EpisodeBuffer"]
