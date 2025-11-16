"""
随机 Agent 实现

用于测试和基准比较的随机策略 Agent
"""

import random
from typing import Any

import numpy as np

from core.agent_interface import AgentInterface
from core.types import ActionIndex, ObservationType


class RandomAgent(AgentInterface):
    """
    随机策略 Agent

    从合法动作中随机选择一个动作执行
    """

    def __init__(self, player_id: int = 0, seed: int | None = None):
        """
        初始化随机 Agent

        Args:
            player_id: 玩家 ID
            seed: 随机种子
        """
        self.player_id = player_id
        self.seed = seed
        self.rng = random.Random(seed)

    def select_action(
        self,
        observation: ObservationType,
        legal_actions: list[ActionIndex],
        deterministic: bool = False,
    ) -> tuple[ActionIndex, dict[str, Any]]:
        """
        从合法动作中随机选择

        Args:
            observation: 观察向量（未使用，随机策略不需要）
            legal_actions: 合法动作索引列表
            deterministic: 是否确定性（随机策略忽略此参数）

        Returns:
            action_index: 选择的动作索引
            info: 额外信息字典
        """
        if not legal_actions:
            raise ValueError("合法动作列表为空")

        action_index = self.rng.choice(legal_actions)

        info = {
            "agent_type": "random",
            "num_legal_actions": len(legal_actions),
            "log_prob": 0.0,  # 随机策略没有 log_prob，设为 0
            "value": 0.0,  # 随机策略没有价值估计，设为 0
        }

        return action_index, info

    def reset(self):
        """重置 Agent 状态"""
        pass

    def __str__(self) -> str:
        return f"RandomAgent(player_id={self.player_id})"


# ===== 导出 =====
__all__ = ["RandomAgent"]
