"""
Value Head

价值头，将编码后的特征向量映射到状态价值估计。
"""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    价值头

    Architecture:
        input (hidden_dim) → Linear → ReLU → Dropout →
        Linear → output (1)

    输出单个标量值，表示当前状态的价值估计。

    Args:
        hidden_dim: 输入特征维度
        intermediate_dim: 中间层维度（默认 128）
        dropout: Dropout 概率（默认 0.0）
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        # 网络层
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, 1)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: 特征向量 (batch_size, hidden_dim)

        Returns:
            values: 状态价值 (batch_size, 1)
        """
        x = features

        # 第一层
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 第二层（输出价值）
        values = self.fc2(x)

        return values


# ===== 导出 =====
__all__ = ["ValueHead"]
