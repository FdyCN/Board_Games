"""
PPO (Proximal Policy Optimization) 算法实现

实现 PPO 的核心训练逻辑，包括 clipped surrogate objective、
价值函数损失和熵正则化。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import numpy as np

from models.actor_critic import ActorCritic
from training.experience import ExperienceBatch


class PPO:
    """
    PPO 训练器

    实现 PPO 算法的核心逻辑，用于更新 Actor-Critic 模型。

    Attributes:
        model: Actor-Critic 模型
        optimizer: 优化器
        clip_epsilon: PPO clip 参数
        value_coef: 价值损失系数
        entropy_coef: 熵正则化系数
        max_grad_norm: 梯度裁剪阈值

    Examples:
        >>> model = create_splendor_model(encoder_type="mlp", config="medium")
        >>> ppo = PPO(model, learning_rate=3e-4, clip_epsilon=0.2)
        >>> # 收集经验
        >>> batch = ExperienceBatch.from_experiences(experiences)
        >>> # 更新模型
        >>> metrics = ppo.update(batch, epochs=4)
        >>> print(f"Policy Loss: {metrics['policy_loss']:.4f}")
    """

    def __init__(
        self,
        model: ActorCritic,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        """
        初始化 PPO 训练器

        Args:
            model: Actor-Critic 模型
            learning_rate: 学习率
            clip_epsilon: PPO clip 参数（通常 0.1-0.3）
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪阈值
            device: 设备
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练统计
        self.update_count = 0

    def update(
        self,
        batch: ExperienceBatch,
        epochs: int = 4,
        minibatch_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        使用 PPO 更新模型

        Args:
            batch: 经验批次
            epochs: 更新轮数
            minibatch_size: 小批次大小（None 表示使用整个批次）

        Returns:
            训练指标字典
        """
        # 切换到训练模式
        self.model.train()

        # 移动批次到正确的设备
        batch = batch.to(self.device)

        # 累积指标
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl_div = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        # 多轮更新
        for epoch in range(epochs):
            # 如果指定了 minibatch_size，则分批更新
            if minibatch_size is not None and minibatch_size < batch.batch_size:
                minibatches = batch.iterate_minibatches(
                    batch_size=minibatch_size, shuffle=True
                )
            else:
                # 否则使用整个批次
                minibatches = [batch]

            for mb in minibatches:
                metrics = self._update_minibatch(mb)

                total_policy_loss += metrics["policy_loss"]
                total_value_loss += metrics["value_loss"]
                total_entropy += metrics["entropy"]
                total_kl_div += metrics["kl_div"]
                total_clip_fraction += metrics["clip_fraction"]
                num_updates += 1

        # 平均指标
        avg_metrics = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "kl_div": total_kl_div / num_updates,
            "clip_fraction": total_clip_fraction / num_updates,
            "total_loss": (total_policy_loss + total_value_loss) / num_updates,
        }

        self.update_count += 1

        return avg_metrics

    def _update_minibatch(self, batch: ExperienceBatch) -> Dict[str, float]:
        """
        更新一个小批次

        Args:
            batch: 小批次

        Returns:
            训练指标
        """
        observations = batch.observations
        actions = batch.actions
        old_log_probs = batch.old_log_probs
        advantages = batch.advantages
        returns = batch.returns

        # 使用保存的合法动作掩码（用于正确计算熵）
        legal_mask = batch.legal_actions_masks

        new_values, new_log_probs, entropy = self.model.evaluate_actions(
            observations, actions, legal_mask
        )

        # === 1. 策略损失 (Clipped Surrogate Objective) ===
        # 计算重要性采样比率
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # === 2. 价值损失 ===
        # 使用 MSE 损失
        value_loss = 0.5 * ((new_values - returns) ** 2).mean()

        # === 3. 熵正则化 ===
        # 鼓励探索
        entropy_loss = -entropy.mean()

        # === 总损失 ===
        total_loss = (
            policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        )

        # 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # === 计算统计指标 ===
        with torch.no_grad():
            # KL 散度（近似）
            kl_div = (old_log_probs - new_log_probs).mean()

            # Clip 比例（衡量有多少比例的样本被 clip）
            clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "kl_div": kl_div.item(),
            "clip_fraction": clip_fraction.item(),
            "grad_norm": grad_norm.item(),
        }

    def save_checkpoint(self, path: str, metadata: Optional[dict] = None) -> None:
        """
        保存训练检查点

        Args:
            path: 保存路径
            metadata: 额外的元数据（如训练步数、epoch 等）
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "hyperparameters": {
                "clip_epsilon": self.clip_epsilon,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "max_grad_norm": self.max_grad_norm,
            },
        }

        if metadata:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> dict:
        """
        加载训练检查点

        Args:
            path: 检查点路径

        Returns:
            元数据字典
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint.get("update_count", 0)

        # 恢复超参数
        hyperparams = checkpoint.get("hyperparameters", {})
        self.clip_epsilon = hyperparams.get("clip_epsilon", self.clip_epsilon)
        self.value_coef = hyperparams.get("value_coef", self.value_coef)
        self.entropy_coef = hyperparams.get("entropy_coef", self.entropy_coef)
        self.max_grad_norm = hyperparams.get("max_grad_norm", self.max_grad_norm)

        return checkpoint.get("metadata", {})

    def get_learning_rate(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]["lr"]

    def set_learning_rate(self, lr: float) -> None:
        """设置学习率"""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


# ===== 导出 =====
__all__ = ["PPO"]
