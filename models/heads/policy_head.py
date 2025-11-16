"""
Policy Head

策略头，将编码后的特征向量映射到动作概率分布。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """
    策略头

    Architecture:
        input (hidden_dim) → Linear → ReLU → Dropout →
        Linear → output (action_size)

    输出经过 softmax 后得到动作概率分布。
    支持合法动作掩码（将非法动作概率设为 0）。

    Args:
        hidden_dim: 输入特征维度
        action_size: 动作空间大小
        intermediate_dim: 中间层维度（默认 128）
        dropout: Dropout 概率（默认 0.0）
    """

    def __init__(
        self,
        hidden_dim: int,
        action_size: int,
        intermediate_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_size = action_size
        self.intermediate_dim = intermediate_dim

        # 网络层
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, action_size)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self, features: torch.Tensor, legal_actions_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            features: 特征向量 (batch_size, hidden_dim)
            legal_actions_mask: 合法动作掩码 (batch_size, action_size)
                                True 表示合法，False 表示非法
                                如果为 None，则所有动作都合法

        Returns:
            logits: 动作 logits (batch_size, action_size)
        """
        x = features

        # 第一层
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 第二层（输出 logits）
        logits = self.fc2(x)

        # 应用合法动作掩码
        if legal_actions_mask is not None:
            # 将非法动作的 logits 设为 -inf
            logits = torch.where(legal_actions_mask, logits, torch.tensor(float("-inf")))

        return logits

    def get_action_probs(
        self, features: torch.Tensor, legal_actions_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        获取动作概率分布

        Args:
            features: 特征向量 (batch_size, hidden_dim)
            legal_actions_mask: 合法动作掩码 (batch_size, action_size)

        Returns:
            probs: 动作概率 (batch_size, action_size)
        """
        logits = self.forward(features, legal_actions_mask)
        probs = F.softmax(logits, dim=-1)
        return probs

    def sample_action(
        self,
        features: torch.Tensor,
        legal_actions_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作

        Args:
            features: 特征向量 (batch_size, hidden_dim)
            legal_actions_mask: 合法动作掩码 (batch_size, action_size)
            deterministic: 是否确定性选择（选择概率最大的动作）

        Returns:
            actions: 采样的动作索引 (batch_size,)
            log_probs: 动作的对数概率 (batch_size,)
        """
        logits = self.forward(features, legal_actions_mask)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            # 确定性：选择概率最大的动作
            actions = torch.argmax(probs, dim=-1)
        else:
            # 随机采样
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # 计算对数概率
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)

        return actions, log_probs


# ===== 导出 =====
__all__ = ["PolicyHead"]
