"""
GRU 序列编码器（用于不完美信息博弈的「有序信息状态」）。

把观察向量拆成两段：
- 静态段（当前状态快照）→ 小 MLP
- 序列段（按时间顺序的事件时间线）→ GRU（带因果/时序归纳偏置）

两者拼接后输出 hidden_dim 特征。参考 OpenSpiel「信息状态 + 序列模型」的思路。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """GRU 序列编码器：静态特征（MLP）+ 有序事件（GRU）。"""

    def __init__(
        self,
        static_dim: int,
        seq_len: int,
        evt_dim: int,
        hidden_dim: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.seq_len = seq_len
        self.evt_dim = evt_dim
        self.hidden_dim = hidden_dim

        # 事件序列：GRU（batch_first）
        self.gru = nn.GRU(
            input_size=evt_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # 静态特征：小 MLP
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 拼接后投影
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, static_dim + seq_len * evt_dim)

        Returns:
            features: (batch, hidden_dim)
        """
        static = obs[:, : self.static_dim]
        seq = obs[:, self.static_dim :].reshape(-1, self.seq_len, self.evt_dim)

        # GRU 处理事件序列，取最后一层隐藏状态
        _, h = self.gru(seq)          # h: (num_layers, batch, hidden)
        h = h[-1]                      # 最后一层： (batch, hidden)

        # MLP 处理静态特征
        s = self.static_mlp(static)    # (batch, hidden)

        # 拼接
        out = self.combine(torch.cat([h, s], dim=-1))
        return self.activation(out)


__all__ = ["GRUEncoder"]
