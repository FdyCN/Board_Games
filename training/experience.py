"""
经验数据批处理模块

提供经验数据的批处理、转换和预处理功能，用于 PPO 训练。
"""

import numpy as np
import torch
from typing import List, Generator

from core.types import Experience, Episode, compute_gae


class ExperienceBatch:
    """
    经验批次

    将多个经验样本组织成张量批次，用于高效训练。

    Attributes:
        observations: 观察张量 (batch_size, obs_dim)
        actions: 动作索引 (batch_size,)
        old_log_probs: 旧策略的对数概率 (batch_size,)
        advantages: 优势函数 (batch_size,)
        returns: 回报 (batch_size,)
        values: 状态价值 (batch_size,)

    Examples:
        >>> experiences = [exp1, exp2, exp3, ...]
        >>> batch = ExperienceBatch.from_experiences(experiences, device="cpu")
        >>> for mini_batch in batch.iterate_minibatches(batch_size=64):
        ...     # 训练模型
        ...     pass
    """

    def __init__(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
    ):
        self.observations = observations
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.advantages = advantages
        self.returns = returns
        self.values = values

        # 验证形状一致性
        batch_size = len(observations)
        assert len(actions) == batch_size
        assert len(old_log_probs) == batch_size
        assert len(advantages) == batch_size
        assert len(returns) == batch_size
        assert len(values) == batch_size

    @property
    def batch_size(self) -> int:
        """批次大小"""
        return len(self.observations)

    @classmethod
    def from_experiences(
        cls,
        experiences: List[Experience],
        device: str = "cpu",
        normalize_advantages: bool = True,
    ) -> "ExperienceBatch":
        """
        从经验列表创建批次

        Args:
            experiences: 经验列表
            device: 设备 ('cpu', 'cuda', 'mps')
            normalize_advantages: 是否归一化优势函数

        Returns:
            ExperienceBatch 实例

        Raises:
            ValueError: 如果经验列表为空或数据不完整
        """
        if not experiences:
            raise ValueError("经验列表不能为空")

        # 验证所有经验都有必要的字段
        for exp in experiences:
            if exp.log_prob is None or exp.value is None:
                raise ValueError("经验必须包含 log_prob 和 value")
            if exp.advantage is None or exp.returns is None:
                raise ValueError("经验必须先计算优势和回报（使用 compute_advantages）")

        # 提取数据
        observations = np.stack([exp.observation for exp in experiences])
        actions = np.array([exp.action for exp in experiences], dtype=np.int64)
        old_log_probs = np.array([exp.log_prob for exp in experiences], dtype=np.float32)
        advantages = np.array([exp.advantage for exp in experiences], dtype=np.float32)
        returns = np.array([exp.returns for exp in experiences], dtype=np.float32)
        values = np.array([exp.value for exp in experiences], dtype=np.float32)

        # 归一化优势函数
        if normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转换为张量
        device_obj = torch.device(device)
        return cls(
            observations=torch.from_numpy(observations).float().to(device_obj),
            actions=torch.from_numpy(actions).long().to(device_obj),
            old_log_probs=torch.from_numpy(old_log_probs).float().to(device_obj),
            advantages=torch.from_numpy(advantages).float().to(device_obj),
            returns=torch.from_numpy(returns).float().to(device_obj),
            values=torch.from_numpy(values).float().to(device_obj),
        )

    def iterate_minibatches(
        self, batch_size: int, shuffle: bool = True
    ) -> Generator["ExperienceBatch", None, None]:
        """
        迭代生成小批次

        Args:
            batch_size: 小批次大小
            shuffle: 是否打乱顺序

        Yields:
            ExperienceBatch: 小批次
        """
        indices = np.arange(self.batch_size)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.batch_size, batch_size):
            end_idx = min(start_idx + batch_size, self.batch_size)
            batch_indices = indices[start_idx:end_idx]

            yield ExperienceBatch(
                observations=self.observations[batch_indices],
                actions=self.actions[batch_indices],
                old_log_probs=self.old_log_probs[batch_indices],
                advantages=self.advantages[batch_indices],
                returns=self.returns[batch_indices],
                values=self.values[batch_indices],
            )

    def to(self, device: str) -> "ExperienceBatch":
        """
        移动到指定设备

        Args:
            device: 目标设备

        Returns:
            新的 ExperienceBatch 实例
        """
        device_obj = torch.device(device)
        return ExperienceBatch(
            observations=self.observations.to(device_obj),
            actions=self.actions.to(device_obj),
            old_log_probs=self.old_log_probs.to(device_obj),
            advantages=self.advantages.to(device_obj),
            returns=self.returns.to(device_obj),
            values=self.values.to(device_obj),
        )


def compute_advantages_for_episode(
    episode: Episode,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> None:
    """
    为 Episode 中的所有经验计算优势函数和回报

    使用 GAE (Generalized Advantage Estimation) 计算优势函数。
    直接修改 episode.experiences 中的 advantage 和 returns 字段。

    Args:
        episode: Episode 对象
        gamma: 折扣因子
        gae_lambda: GAE 参数

    Examples:
        >>> episode = collect_episode(game, agent)
        >>> compute_advantages_for_episode(episode, gamma=0.99, gae_lambda=0.95)
        >>> # 现在 episode.experiences 中的每个经验都有 advantage 和 returns
    """
    experiences = episode.experiences
    if not experiences:
        return

    # 提取数据
    rewards = [exp.reward for exp in experiences]
    values = [exp.value for exp in experiences]
    dones = [exp.done for exp in experiences]

    # 计算 next_values
    next_values = values[1:] + [0.0]  # 最后一个 next_value 为 0

    # 使用 GAE 计算优势和回报
    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        dones=dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    # 填充到经验中
    for exp, adv, ret in zip(experiences, advantages, returns):
        exp.advantage = adv
        exp.returns = ret


def split_episodes_by_player(
    episodes: List[Episode],
) -> List[List[Experience]]:
    """
    按玩家分割 episodes

    将多局游戏的经验按玩家 ID 分组，用于多智能体训练。

    Args:
        episodes: Episode 列表

    Returns:
        每个玩家的经验列表 [player0_exps, player1_exps, ...]

    Examples:
        >>> episodes = [episode1, episode2, episode3]
        >>> player_experiences = split_episodes_by_player(episodes)
        >>> # player_experiences[0] 包含所有玩家0的经验
    """
    if not episodes:
        return []

    # 确定玩家数量（从第一个 episode 推断）
    num_players = len(episodes[0].total_rewards)
    player_experiences = [[] for _ in range(num_players)]

    # 按玩家分组
    for episode in episodes:
        for player_id in range(num_players):
            player_exps = episode.get_player_experiences(player_id)
            player_experiences[player_id].extend(player_exps)

    return player_experiences


def merge_experience_batches(
    batches: List[ExperienceBatch],
) -> ExperienceBatch:
    """
    合并多个经验批次

    Args:
        batches: ExperienceBatch 列表

    Returns:
        合并后的 ExperienceBatch
    """
    if not batches:
        raise ValueError("批次列表不能为空")

    if len(batches) == 1:
        return batches[0]

    # 合并张量
    return ExperienceBatch(
        observations=torch.cat([b.observations for b in batches], dim=0),
        actions=torch.cat([b.actions for b in batches], dim=0),
        old_log_probs=torch.cat([b.old_log_probs for b in batches], dim=0),
        advantages=torch.cat([b.advantages for b in batches], dim=0),
        returns=torch.cat([b.returns for b in batches], dim=0),
        values=torch.cat([b.values for b in batches], dim=0),
    )


# ===== 导出 =====
__all__ = [
    "ExperienceBatch",
    "compute_advantages_for_episode",
    "split_episodes_by_player",
    "merge_experience_batches",
]
