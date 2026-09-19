"""
Policy Head（结构化 + 逐卡共享评估器）

关键改进（参考 alpha-zero-general Splendor 的 StructuredPolicyHead）：
    "购买明牌"、"保留明牌"、"购买保留牌"这三个动作头，**共享同一个卡价值评估器**
    （同一个 Linear(card_dim→1) 施加到每一张卡的 embedding 上）。

    这样模型学到的"这张卡值不值得买/保留"是一个**与卡槽位置无关**的概念，
    从而能泛化到"高价值卡值得攒宝石去买"，而不是记住某个绝对槽位。

动作槽位布局（games/splendor/game.py ACTION_SPACE_SIZE=46）：
    0-4   拿2同色 | 5-14 拿3不同色 | 15-26 保留明牌(12) | 27-29 保留牌堆(3)
    30-41 买明牌(12) | 42-44 买保留(3) | 45 pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GlobalHead(nn.Module):
    """基于全局特征的输出头：Linear(hidden) -> ReLU -> Linear(out)"""

    def __init__(self, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


class PolicyHead(nn.Module):
    """结构化策略头：全局头 + 逐卡共享评估器"""

    def __init__(
        self,
        hidden_dim: int,
        action_size: int,
        card_dim: int = 32,
        intermediate_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_size = action_size
        self.card_dim = card_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 是否走结构化（action_size == 46）
        self._structured = action_size == 46

        if self._structured:
            # 共享中间层（全局特征）
            self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
            self.activation = nn.ReLU()

            # 全局动作头
            self.take2_head = _GlobalHead(intermediate_dim, 5, dropout)
            self.take3_head = _GlobalHead(intermediate_dim, 10, dropout)
            self.rsv_deck_head = _GlobalHead(intermediate_dim, 3, dropout)
            self.pass_head = _GlobalHead(intermediate_dim, 1, dropout)

            # 逐卡共享评估器（作用于每张卡的 embedding）
            self.rsv_visible_eval = nn.Linear(card_dim, 1)   # 保留明牌
            self.buy_visible_eval = nn.Linear(card_dim, 1)   # 买明牌
            self.buy_reserved_eval = nn.Linear(card_dim, 1)  # 买保留牌
        else:
            # 退回普通 flat 头（通用模型/测试兼容）
            self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
            self.activation = nn.ReLU()
            self.fc2 = nn.Linear(intermediate_dim, action_size)

    def forward(
        self,
        global_features: torch.Tensor,
        open_card_embeds: torch.Tensor | None = None,
        reserved_card_embeds: torch.Tensor | None = None,
        legal_actions_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            global_features: (batch, hidden_dim)
            open_card_embeds: (batch, 12, card_dim)
            reserved_card_embeds: (batch, 3, card_dim)
            legal_actions_mask: (batch, action_size)

        Returns:
            logits: (batch, action_size)
        """
        if not self._structured:
            x = self.fc1(global_features)
            x = self.activation(x)
            x = self.dropout(x)
            logits = self.fc2(x)
        else:
            x = self.fc1(global_features)
            x = self.activation(x)
            x = self.dropout(x)

            # 全局动作 logits
            take2 = self.take2_head(x)          # (B, 5)
            take3 = self.take3_head(x)          # (B, 10)
            rsv_deck = self.rsv_deck_head(x)    # (B, 3)
            pass_logit = self.pass_head(x)      # (B, 1)

            # 逐卡 logits（共享评估器，广播到所有卡）
            rsv_visible = self.rsv_visible_eval(open_card_embeds).squeeze(-1)   # (B, 12)
            buy_visible = self.buy_visible_eval(open_card_embeds).squeeze(-1)   # (B, 12)
            buy_reserved = self.buy_reserved_eval(reserved_card_embeds).squeeze(-1)  # (B, 3)

            logits = torch.cat(
                [take2, take3, rsv_visible, rsv_deck, buy_visible, buy_reserved, pass_logit],
                dim=-1,
            )

        if legal_actions_mask is not None:
            logits = torch.where(
                legal_actions_mask.bool(),
                logits,
                torch.tensor(float("-inf"), device=logits.device),
            )

        return logits

    def get_action_probs(
        self,
        global_features: torch.Tensor,
        open_card_embeds: torch.Tensor | None = None,
        reserved_card_embeds: torch.Tensor | None = None,
        legal_actions_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.forward(global_features, open_card_embeds, reserved_card_embeds, legal_actions_mask)
        return F.softmax(logits, dim=-1)

    def sample_action(
        self,
        global_features: torch.Tensor,
        open_card_embeds: torch.Tensor | None = None,
        reserved_card_embeds: torch.Tensor | None = None,
        legal_actions_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(global_features, open_card_embeds, reserved_card_embeds, legal_actions_mask)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            actions = torch.argmax(probs, dim=-1)
        else:
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

        log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)

        return actions, log_probs


# ===== 导出 =====
__all__ = ["PolicyHead"]
