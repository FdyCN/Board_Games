"""
Actor-Critic Model

结合编码器、策略头和价值头的完整 Actor-Critic 模型。
支持多种编码器类型（MLP、Attention）。
"""

import torch
import torch.nn as nn
from typing import Literal

from models.encoders.mlp_encoder import MLPEncoder
from models.encoders.attention_encoder import AttentionEncoder
from models.heads.policy_head import PolicyHead
from models.heads.value_head import ValueHead


class ActorCritic(nn.Module):
    """
    Actor-Critic 模型

    Architecture:
        observation → Encoder → features
                                ├→ PolicyHead → action_logits
                                └→ ValueHead → state_value

    Args:
        obs_dim: 观察空间维度
        action_size: 动作空间大小
        encoder_type: 编码器类型 ("mlp" 或 "attention")
        hidden_dim: 编码器输出维度（默认 256）
        encoder_intermediate_dim: 编码器中间层维度（默认 320）
        head_intermediate_dim: 头部中间层维度（默认 128）
        num_attention_heads: 注意力头数（仅用于 attention 编码器，默认 4）
        dropout: Dropout 概率（默认 0.0）
    """

    def __init__(
        self,
        obs_dim: int,
        action_size: int,
        encoder_type: Literal["mlp", "attention"] = "mlp",
        hidden_dim: int = 256,
        encoder_intermediate_dim: int = 320,
        head_intermediate_dim: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_size = action_size
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim

        # 创建编码器
        if encoder_type == "mlp":
            self.encoder = MLPEncoder(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                intermediate_dim=encoder_intermediate_dim,
                dropout=dropout,
            )
        elif encoder_type == "attention":
            self.encoder = AttentionEncoder(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                intermediate_dim=encoder_intermediate_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"未知的编码器类型: {encoder_type}")

        # 创建策略头（Actor）
        self.policy_head = PolicyHead(
            hidden_dim=hidden_dim,
            action_size=action_size,
            intermediate_dim=head_intermediate_dim,
            dropout=dropout,
        )

        # 创建价值头（Critic）
        self.value_head = ValueHead(
            hidden_dim=hidden_dim,
            intermediate_dim=head_intermediate_dim,
            dropout=dropout,
        )

    def forward(
        self,
        obs: torch.Tensor,
        legal_actions_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            obs: 观察向量 (batch_size, obs_dim)
            legal_actions_mask: 合法动作掩码 (batch_size, action_size)
                                True 表示合法，False 表示非法

        Returns:
            logits: 动作 logits (batch_size, action_size)
            values: 状态价值 (batch_size, 1)
        """
        # 编码观察
        features = self.encoder(obs)

        # 获取策略和价值
        logits = self.policy_head(features, legal_actions_mask)
        values = self.value_head(features)

        return logits, values

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        legal_actions_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作、对数概率和状态价值（用于训练）

        Args:
            obs: 观察向量 (batch_size, obs_dim)
            legal_actions_mask: 合法动作掩码 (batch_size, action_size)
            deterministic: 是否确定性选择动作

        Returns:
            actions: 采样的动作 (batch_size,)
            log_probs: 动作的对数概率 (batch_size,)
            values: 状态价值 (batch_size, 1)
        """
        # 编码观察
        features = self.encoder(obs)

        # 采样动作
        actions, log_probs = self.policy_head.sample_action(
            features, legal_actions_mask, deterministic
        )

        # 获取价值
        values = self.value_head(features)

        return actions, log_probs, values

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        仅获取状态价值（用于优势估计）

        Args:
            obs: 观察向量 (batch_size, obs_dim)

        Returns:
            values: 状态价值 (batch_size, 1)
        """
        features = self.encoder(obs)
        values = self.value_head(features)
        return values

    def get_action_probs(
        self,
        obs: torch.Tensor,
        legal_actions_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        获取动作概率分布（用于评估）

        Args:
            obs: 观察向量 (batch_size, obs_dim)
            legal_actions_mask: 合法动作掩码 (batch_size, action_size)

        Returns:
            probs: 动作概率 (batch_size, action_size)
        """
        features = self.encoder(obs)
        probs = self.policy_head.get_action_probs(features, legal_actions_mask)
        return probs

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        legal_actions_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作（用于 PPO 更新）

        Args:
            obs: 观察向量 (batch_size, obs_dim)
            actions: 动作索引 (batch_size,)
            legal_actions_mask: 合法动作掩码 (batch_size, action_size)

        Returns:
            values: 状态价值 (batch_size, 1)
            log_probs: 动作的对数概率 (batch_size,)
            entropy: 策略熵 (batch_size,)
        """
        # 编码观察
        features = self.encoder(obs)

        # 获取 logits 和价值
        logits = self.policy_head(features, legal_actions_mask)
        values = self.value_head(features)

        # 计算概率分布
        probs = torch.softmax(logits, dim=-1)

        # 计算对数概率
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)

        # 计算熵（用于鼓励探索）
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        return values, log_probs, entropy

    def count_parameters(self) -> dict[str, int]:
        """
        统计模型参数量

        Returns:
            参数统计字典
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        value_params = sum(p.numel() for p in self.value_head.parameters())
        total_params = encoder_params + policy_params + value_params

        return {
            "encoder": encoder_params,
            "policy_head": policy_params,
            "value_head": value_params,
            "total": total_params,
        }


# ===== 导出 =====
__all__ = ["ActorCritic"]
