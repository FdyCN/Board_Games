"""
Agent 抽象接口

本模块定义了所有 Agent 必须实现的标准接口 AgentInterface。
Agent 负责根据观察选择动作，可以是随机策略、神经网络、MCTS 或人类玩家。
"""

from abc import ABC, abstractmethod

import numpy as np

from core.types import ActionIndex, Experience, ObservationType


class AgentInterface(ABC):
    """
    Agent 抽象基类

    所有 Agent（随机、神经网络、人类等）都必须实现这个接口。

    设计原则：
    - 无状态或显式状态管理：Agent 不应该隐式依赖历史，如需历史需显式传入
    - 合法动作过滤：select_action 必须只从 legal_actions 中选择
    - 信息返回：返回动作及额外信息（概率、价值等）用于训练

    典型使用流程：
        >>> agent = NeuralAgent(model)
        >>> obs = game.state_to_observation(state, player_id=0)
        >>> legal_actions = [game.action_to_index(a) for a in game.get_legal_actions(state)]
        >>> action_idx, info = agent.select_action(obs, legal_actions)
        >>> action = game.index_to_action(action_idx)
        >>> state, rewards, done, _ = game.step(action)
    """

    # ===== 核心方法 =====

    @abstractmethod
    def select_action(
        self,
        observation: ObservationType,
        legal_actions: list[ActionIndex],
        deterministic: bool = False,
    ) -> tuple[ActionIndex, dict[str, any]]:
        """
        根据观察选择动作

        Args:
            observation: 游戏观察（由 game.state_to_observation 生成）
            legal_actions: 合法动作索引列表（必须从中选择）
            deterministic: 是否使用确定性策略
                - True: 选择概率最高/价值最大的动作（评估时用）
                - False: 按概率分布采样（训练时用）

        Returns:
            action_index: 选择的动作索引（必须在 legal_actions 中）
            info: 额外信息字典，建议包含：
                - 'log_prob' (float): 动作的对数概率（PPO 需要）
                - 'value' (float): 状态价值估计（Actor-Critic 需要）
                - 'policy' (np.ndarray): 完整策略分布（可选，用于分析）
                - 'entropy' (float): 策略熵（可选，用于监控探索）

        Raises:
            ValueError: 如果 legal_actions 为空

        Examples:
            >>> # 训练时：采样动作
            >>> action_idx, info = agent.select_action(obs, legal_actions, deterministic=False)
            >>> print(f"采样动作: {action_idx}, log_prob: {info['log_prob']:.3f}")

            >>> # 评估时：选择最优动作
            >>> action_idx, info = agent.select_action(obs, legal_actions, deterministic=True)
            >>> print(f"最优动作: {action_idx}, value: {info['value']:.3f}")
        """
        pass

    # ===== 可选方法（提供默认实现） =====

    def reset(self) -> None:
        """
        重置 Agent 内部状态（如 RNN 隐状态）

        在每局游戏开始前调用

        Examples:
            >>> agent.reset()  # 清空 RNN 隐状态
        """
        pass

    def learn(self, experiences: list[Experience]) -> dict[str, float]:
        """
        从经验中学习（可选，用于在线学习）

        大多数情况下训练是离线的（由 Trainer 负责），但某些 Agent
        （如 DQN）可能需要在线更新

        Args:
            experiences: 经验样本列表

        Returns:
            训练指标字典，例如：
                - 'loss': 总损失
                - 'policy_loss': 策略损失
                - 'value_loss': 价值损失

        Examples:
            >>> metrics = agent.learn(experiences)
            >>> print(f"训练损失: {metrics['loss']:.4f}")
        """
        return {}

    def save(self, path: str) -> None:
        """
        保存 Agent（模型权重、配置等）

        Args:
            path: 保存路径（文件或目录）

        Examples:
            >>> agent.save("data/checkpoints/agent_v1.pth")
        """
        raise NotImplementedError(f"{self.__class__.__name__} 未实现 save 方法")

    def load(self, path: str) -> None:
        """
        加载 Agent

        Args:
            path: 加载路径

        Examples:
            >>> agent.load("data/checkpoints/agent_v1.pth")
        """
        raise NotImplementedError(f"{self.__class__.__name__} 未实现 load 方法")

    def get_name(self) -> str:
        """
        获取 Agent 名称（用于日志和评估）

        Returns:
            Agent 名称

        Examples:
            >>> print(agent.get_name())  # "NeuralAgent_v1"
        """
        return self.__class__.__name__

    def set_training_mode(self, training: bool) -> None:
        """
        设置训练/评估模式

        Args:
            training: True 为训练模式，False 为评估模式

        Examples:
            >>> agent.set_training_mode(True)   # 启用 dropout、batch norm 等
            >>> agent.set_training_mode(False)  # 禁用 dropout
        """
        pass

    def __str__(self) -> str:
        """返回 Agent 的字符串表示"""
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """返回 Agent 的详细表示"""
        return f"{self.__class__.__name__}()"


# ===== 辅助函数 =====


def sample_action_from_policy(
    policy: np.ndarray, legal_actions: list[ActionIndex], deterministic: bool = False
) -> tuple[ActionIndex, float]:
    """
    从策略分布中采样动作

    Args:
        policy: 策略分布（未归一化的 logits 或已归一化的概率）
        legal_actions: 合法动作索引列表
        deterministic: 是否选择最优动作

    Returns:
        action_index: 选择的动作索引
        log_prob: 动作的对数概率

    Examples:
        >>> policy = np.array([0.1, 0.3, 0.4, 0.2])
        >>> legal_actions = [0, 2, 3]  # 动作 1 非法
        >>> action, log_prob = sample_action_from_policy(policy, legal_actions)
    """
    if len(legal_actions) == 0:
        raise ValueError("legal_actions 不能为空")

    # 创建掩码
    mask = np.ones_like(policy) * -np.inf
    mask[legal_actions] = 0.0

    # 应用掩码
    masked_policy = policy + mask

    # Softmax 归一化
    exp_policy = np.exp(masked_policy - np.max(masked_policy))
    probs = exp_policy / exp_policy.sum()

    # 选择动作
    if deterministic:
        action_index = legal_actions[np.argmax(probs[legal_actions])]
    else:
        action_index = np.random.choice(len(probs), p=probs)

    # 计算对数概率
    log_prob = np.log(probs[action_index] + 1e-8)

    return action_index, log_prob


def compute_entropy(policy: np.ndarray) -> float:
    """
    计算策略分布的熵（衡量探索程度）

    Args:
        policy: 策略概率分布（已归一化）

    Returns:
        熵值（越大表示越随机，越小表示越确定）

    Examples:
        >>> policy = np.array([0.25, 0.25, 0.25, 0.25])
        >>> entropy = compute_entropy(policy)
        >>> print(f"熵: {entropy:.3f}")  # 均匀分布熵最大
    """
    # 避免 log(0)
    policy = np.clip(policy, 1e-8, 1.0)
    entropy = -np.sum(policy * np.log(policy))
    return entropy
