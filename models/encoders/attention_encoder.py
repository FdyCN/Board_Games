"""
Attention Encoder

使用自注意力机制的编码器，能够捕捉观察向量中不同部分之间的关系。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionEncoder(nn.Module):
    """
    注意力机制编码器

    Architecture:
        input (obs_dim) → Linear → Self-Attention →
        Residual → LayerNorm → Linear → output (hidden_dim)

    Args:
        obs_dim: 观察空间维度
        hidden_dim: 输出特征维度
        intermediate_dim: 中间层维度（默认 512）
        num_heads: 注意力头数（默认 4）
        dropout: Dropout 概率（默认 0.0）
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        intermediate_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads

        # 输入投影
        self.input_proj = nn.Linear(obs_dim, intermediate_dim)

        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=intermediate_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # 输出投影
        self.output_proj = nn.Linear(intermediate_dim, hidden_dim)

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(intermediate_dim)
        self.layer_norm2 = nn.LayerNorm(intermediate_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            obs: 观察向量 (batch_size, obs_dim)

        Returns:
            features: 特征向量 (batch_size, hidden_dim)
        """
        batch_size = obs.size(0)

        # 输入投影 (batch_size, obs_dim) → (batch_size, intermediate_dim)
        x = self.input_proj(obs)

        # 为注意力机制添加序列维度 (batch_size, 1, intermediate_dim)
        x = x.unsqueeze(1)

        # 自注意力 + 残差连接
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))

        # 移除序列维度 (batch_size, intermediate_dim)
        x = x.squeeze(1)

        # 输出投影
        x = self.output_proj(x)
        x = self.layer_norm3(x)

        return x

    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.hidden_dim


# ===== 导出 =====
__all__ = ["AttentionEncoder"]
