"""
通用类型定义和数据结构

本模块定义了框架中使用的核心类型、数据类和类型别名。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import numpy as np

# ===== 类型变量 =====
StateType = TypeVar("StateType")  # 游戏状态类型
ActionType = TypeVar("ActionType")  # 动作类型


# ===== 基础类型别名 =====
ObservationType = np.ndarray  # 观察（神经网络输入）
ActionIndex = int  # 动作索引
Reward = float  # 奖励
PlayerID = int  # 玩家 ID


# ===== 枚举类型 =====
class GameResult(Enum):
    """游戏结果"""

    WIN = 1.0  # 获胜
    LOSS = 0.0  # 失败
    DRAW = 0.5  # 平局


class AgentType(Enum):
    """Agent 类型"""

    RANDOM = "random"  # 随机策略
    NEURAL = "neural"  # 神经网络
    HUMAN = "human"  # 人类玩家
    MCTS = "mcts"  # 蒙特卡洛树搜索


# ===== 数据类 =====
@dataclass
class Experience:
    """
    单个经验样本（用于训练）

    Attributes:
        player_id: 玩家 ID
        observation: 观察向量
        action: 动作索引
        reward: 即时奖励
        next_observation: 下一个观察（可选）
        done: 是否终止
        log_prob: 动作的对数概率（PPO 用）
        value: 状态价值估计（PPO 用）
        advantage: 优势函数（计算后填充）
    """

    player_id: PlayerID
    observation: ObservationType
    action: ActionIndex
    reward: Reward
    next_observation: ObservationType | None = None
    done: bool = False
    log_prob: float | None = None
    value: float | None = None
    advantage: float | None = None
    returns: float | None = None  # 回报（用于价值函数训练）

    def __post_init__(self):
        """验证数据有效性"""
        assert isinstance(self.observation, np.ndarray), "observation 必须是 numpy array"
        assert isinstance(self.action, int), "action 必须是整数索引"
        assert isinstance(self.reward, (int, float)), "reward 必须是数值"


@dataclass
class Episode:
    """
    完整的一局游戏记录

    Attributes:
        experiences: 经验列表
        total_rewards: 每个玩家的总奖励
        winner: 获胜者 ID（-1 表示平局）
        num_steps: 总步数
        metadata: 额外信息
    """

    experiences: list[Experience]
    total_rewards: list[Reward]
    winner: PlayerID
    num_steps: int
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get_player_experiences(self, player_id: PlayerID) -> list[Experience]:
        """获取指定玩家的所有经验"""
        return [exp for exp in self.experiences if exp.player_id == player_id]


@dataclass
class ActionInfo:
    """
    Agent 选择动作时返回的额外信息

    Attributes:
        action_index: 选择的动作索引
        log_prob: 动作的对数概率
        value: 状态价值估计
        policy: 完整的策略分布（可选）
        entropy: 策略熵（可选）
    """

    action_index: ActionIndex
    log_prob: float
    value: float
    policy: np.ndarray | None = None
    entropy: float | None = None


@dataclass
class TrainingMetrics:
    """
    训练过程中的指标

    Attributes:
        episode: 当前 episode 数
        policy_loss: 策略损失
        value_loss: 价值损失
        entropy: 平均熵
        mean_reward: 平均奖励
        mean_episode_length: 平均 episode 长度
        learning_rate: 当前学习率
        explained_variance: 价值函数的解释方差
    """

    episode: int
    policy_loss: float
    value_loss: float
    entropy: float
    mean_reward: float
    mean_episode_length: float
    learning_rate: float
    explained_variance: float | None = None
    kl_divergence: float | None = None  # KL 散度（PPO 用）
    clip_fraction: float | None = None  # Clip 比例（PPO 用）

    def to_dict(self) -> dict[str, float]:
        """转换为字典格式（用于日志）"""
        return {
            "episode": self.episode,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "mean_reward": self.mean_reward,
            "mean_episode_length": self.mean_episode_length,
            "learning_rate": self.learning_rate,
            "explained_variance": self.explained_variance or 0.0,
            "kl_divergence": self.kl_divergence or 0.0,
            "clip_fraction": self.clip_fraction or 0.0,
        }


@dataclass
class EvaluationResult:
    """
    评估结果

    Attributes:
        win_rate: 胜率（对于多人游戏，是第一名的比例）
        mean_reward: 平均奖励
        mean_rank: 平均排名（1.0 是最好）
        games_played: 对局数量
        elo_rating: ELO 分数（可选）
    """

    win_rate: float
    mean_reward: float
    mean_rank: float
    games_played: int
    elo_rating: float | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GameConfig:
    """
    游戏配置

    Attributes:
        name: 游戏名称
        num_players: 玩家数量
        params: 游戏特定参数
    """

    name: str
    num_players: int
    params: dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class ModelConfig:
    """
    模型配置

    Attributes:
        encoder_type: 编码器类型 ('mlp', 'attention', 'cnn')
        hidden_size: 隐藏层大小
        num_layers: 层数
        dropout: Dropout 比例
        activation: 激活函数
        params: 额外参数
    """

    encoder_type: str
    hidden_size: int
    num_layers: int = 2
    dropout: float = 0.0
    activation: str = "relu"
    params: dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class TrainingConfig:
    """
    训练配置

    Attributes:
        algorithm: 算法名称 ('ppo', 'a2c', 'dqn')
        learning_rate: 学习率
        gamma: 折扣因子
        num_workers: 并行 worker 数量
        episodes_per_update: 每次更新收集的 episode 数
        batch_size: 批大小
        device: 设备 ('cpu', 'cuda', 'mps')
        params: 算法特定参数
    """

    algorithm: str
    learning_rate: float
    gamma: float
    num_workers: int = 1
    episodes_per_update: int = 100
    batch_size: int = 256
    device: str = "cpu"
    params: dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


# ===== 辅助函数 =====
def normalize_rewards(rewards: list[Reward]) -> list[Reward]:
    """
    归一化奖励（零均值，单位方差）

    Args:
        rewards: 奖励列表

    Returns:
        归一化后的奖励列表
    """
    rewards_arr = np.array(rewards)
    mean = rewards_arr.mean()
    std = rewards_arr.std() + 1e-8  # 避免除零
    return ((rewards_arr - mean) / std).tolist()


def compute_gae(
    rewards: list[Reward],
    values: list[float],
    next_values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    """
    计算广义优势估计 (Generalized Advantage Estimation, GAE)

    Args:
        rewards: 奖励列表
        values: 状态价值列表
        next_values: 下一状态价值列表
        dones: 是否终止列表
        gamma: 折扣因子
        gae_lambda: GAE 参数

    Returns:
        advantages: 优势函数列表
        returns: 回报列表
    """
    advantages = []
    gae = 0.0

    # 从后向前计算
    for t in reversed(range(len(rewards))):
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * gae_lambda * gae

        advantages.insert(0, gae)

    # 计算回报
    returns = [adv + val for adv, val in zip(advantages, values)]

    return advantages, returns


def compute_returns(
    rewards: list[Reward], gamma: float = 0.99, normalize: bool = True
) -> list[float]:
    """
    计算折扣回报

    Args:
        rewards: 奖励列表
        gamma: 折扣因子
        normalize: 是否归一化

    Returns:
        回报列表
    """
    returns = []
    r = 0.0

    for reward in reversed(rewards):
        r = reward + gamma * r
        returns.insert(0, r)

    if normalize and len(returns) > 1:
        returns_arr = np.array(returns)
        mean = returns_arr.mean()
        std = returns_arr.std() + 1e-8
        returns = ((returns_arr - mean) / std).tolist()

    return returns
