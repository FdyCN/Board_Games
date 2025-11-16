"""
Neural Agent

基于神经网络的智能体，封装 Actor-Critic 模型实现 AgentInterface。
"""

import numpy as np
import torch

from core.agent_interface import AgentInterface, compute_entropy
from core.types import ActionIndex, ObservationType
from models.actor_critic import ActorCritic


class NeuralAgent(AgentInterface):
    """
    基于神经网络的智能体

    封装 Actor-Critic 模型，提供标准的 Agent 接口。
    支持训练模式和评估模式，可以进行确定性或随机动作选择。

    Args:
        model: Actor-Critic 模型
        device: 运行设备 ("cpu" 或 "cuda")
        name: Agent 名称（可选，用于日志）

    Examples:
        >>> from models import create_splendor_model
        >>> model = create_splendor_model(encoder_type="mlp", config="medium")
        >>> agent = NeuralAgent(model, device="cpu", name="NeuralAgent_v1")
        >>>
        >>> # 选择动作
        >>> obs = np.random.randn(384)
        >>> legal_actions = [0, 1, 2, 5, 10]
        >>> action_idx, info = agent.select_action(obs, legal_actions)
        >>> print(f"选择动作: {action_idx}, 价值: {info['value']:.3f}")
    """

    def __init__(
        self,
        model: ActorCritic,
        device: str = "cpu",
        name: str | None = None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self._name = name or "NeuralAgent"
        self._training = False

        # 默认设置为评估模式
        self.model.eval()

    def select_action(
        self,
        observation: ObservationType,
        legal_actions: list[ActionIndex],
        deterministic: bool = False,
    ) -> tuple[ActionIndex, dict[str, any]]:
        """
        根据观察选择动作

        Args:
            observation: 游戏观察向量 (numpy array)
            legal_actions: 合法动作索引列表
            deterministic: 是否使用确定性策略

        Returns:
            action_index: 选择的动作索引
            info: 包含 log_prob, value, policy, entropy 的字典
        """
        if len(legal_actions) == 0:
            raise ValueError("legal_actions 不能为空")

        # 转换观察为 tensor
        obs_tensor = self._obs_to_tensor(observation)

        # 创建合法动作掩码
        legal_mask = self._create_legal_mask(legal_actions, self.model.action_size)

        # 使用模型选择动作
        with torch.no_grad():
            action_tensor, log_prob_tensor, value_tensor = self.model.get_action_and_value(
                obs_tensor, legal_mask, deterministic=deterministic
            )

        # 转换为 numpy
        action_idx = int(action_tensor.item())
        log_prob = float(log_prob_tensor.item())
        value = float(value_tensor.item())

        # 获取完整策略分布（用于分析和熵计算）
        with torch.no_grad():
            logits, _ = self.model(obs_tensor, legal_mask)
            probs = torch.softmax(logits, dim=-1)
            policy = probs.cpu().numpy().flatten()

        # 计算熵
        entropy = compute_entropy(policy)

        # 验证动作合法性
        if action_idx not in legal_actions:
            raise RuntimeError(
                f"模型输出非法动作 {action_idx}，合法动作: {legal_actions}"
            )

        info = {
            "log_prob": log_prob,
            "value": value,
            "policy": policy,
            "entropy": entropy,
        }

        return action_idx, info

    def set_training_mode(self, training: bool) -> None:
        """
        设置训练/评估模式

        Args:
            training: True 为训练模式，False 为评估模式
        """
        self._training = training
        if training:
            self.model.train()
        else:
            self.model.eval()

    def save(self, path: str) -> None:
        """
        保存模型权重

        Args:
            path: 保存路径（.pth 文件）

        Examples:
            >>> agent.save("data/checkpoints/agent_v1.pth")
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": {
                    "obs_dim": self.model.obs_dim,
                    "action_size": self.model.action_size,
                    "encoder_type": self.model.encoder_type,
                    "hidden_dim": self.model.hidden_dim,
                },
                "agent_name": self._name,
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        加载模型权重

        Args:
            path: 加载路径（.pth 文件）

        Examples:
            >>> agent.load("data/checkpoints/agent_v1.pth")
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # 如果保存了名称，则恢复
        if "agent_name" in checkpoint:
            self._name = checkpoint["agent_name"]

    def get_name(self) -> str:
        """
        获取 Agent 名称

        Returns:
            Agent 名称
        """
        return self._name

    def _obs_to_tensor(self, observation: ObservationType) -> torch.Tensor:
        """
        将观察转换为 tensor

        Args:
            observation: numpy array 或 tensor

        Returns:
            torch.Tensor (1, obs_dim)
        """
        if isinstance(observation, np.ndarray):
            obs = torch.from_numpy(observation).float()
        else:
            obs = observation.float()

        # 确保是 (1, obs_dim) 形状
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        return obs.to(self.device)

    def _create_legal_mask(
        self, legal_actions: list[ActionIndex], action_size: int
    ) -> torch.Tensor:
        """
        创建合法动作掩码

        Args:
            legal_actions: 合法动作索引列表
            action_size: 总动作空间大小

        Returns:
            torch.Tensor (1, action_size), True 表示合法，False 表示非法
        """
        mask = torch.zeros(1, action_size, dtype=torch.bool, device=self.device)
        mask[0, legal_actions] = True
        return mask

    def __str__(self) -> str:
        """返回 Agent 的字符串表示"""
        return f"{self._name}(encoder={self.model.encoder_type}, device={self.device})"

    def __repr__(self) -> str:
        """返回 Agent 的详细表示"""
        param_count = self.model.count_parameters()["total"]
        return (
            f"{self.__class__.__name__}("
            f"name={self._name}, "
            f"encoder={self.model.encoder_type}, "
            f"params={param_count:,}, "
            f"device={self.device})"
        )


# ===== 导出 =====
__all__ = ["NeuralAgent"]
