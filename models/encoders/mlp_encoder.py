"""
MLP Encoder

简单的多层感知机编码器，用于将观察向量编码为特征向量。
"""

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    多层感知机编码器

    Architecture:
        input (obs_dim) → Linear → ReLU → Dropout →
        Linear → ReLU → Dropout → output (hidden_dim)

    Args:
        obs_dim: 观察空间维度
        hidden_dim: 输出特征维度
        intermediate_dim: 中间层维度（默认 512）
        dropout: Dropout 概率（默认 0.0）
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        intermediate_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        # 网络层
        self.fc1 = nn.Linear(obs_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 层归一化（可选，提升训练稳定性）
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            obs: 观察向量 (batch_size, obs_dim)

        Returns:
            features: 特征向量 (batch_size, hidden_dim)
        """
        x = obs

        # 第一层
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 第二层
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 层归一化
        x = self.layer_norm(x)

        return x

    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.hidden_dim


# ===== 导出 =====
__all__ = ["MLPEncoder"]
