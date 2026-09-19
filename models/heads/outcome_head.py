"""
Outcome Head

预测"当前玩家最终是否获胜"的辅助头。

为什么需要它：
    在对称自对弈中，所有玩家共享同一个模型、实力相等，PPO 的 value 网络
    几乎学不到"谁领先"（explained_variance ≈ 0）。这个头用**终局胜负**做
    强监督（二分类），给共享编码器一个清晰可学的信号，从而帮助 value 和
    policy 一起学到"谁占优"的特征。

输出：单个 logit（配合 BCEWithLogitsLoss），表示 P(当前玩家获胜)。
"""

import torch
import torch.nn as nn


class OutcomeHead(nn.Module):
    """
    终局胜负预测头

    Architecture:
        input (hidden_dim) → Linear → ReLU → Dropout → Linear → logit (1)

    Args:
        hidden_dim: 输入特征维度
        intermediate_dim: 中间层维度（默认 128）
        dropout: Dropout 概率（默认 0.0）
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int = 128, dropout: float = 0.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, 1)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: 编码器输出特征 (batch_size, hidden_dim)

        Returns:
            logits: (batch_size, 1)，未经 sigmoid 的胜率 logit
        """
        x = self.fc1(features)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


# ===== 导出 =====
__all__ = ["OutcomeHead"]
