"""
MCTS 节点实现

每个节点代表游戏树中的一个状态，存储:
- 访问统计 (visit_count, total_value)
- 先验概率 (prior, 来自策略网络)
- 子节点 (children)
"""

import numpy as np
from typing import Optional, Dict, Any


class MCTSNode:
    """
    MCTS 搜索树节点

    Attributes:
        state: 游戏状态 (延迟加载以节省内存)
        parent: 父节点
        action: 从父节点到此节点的动作
        prior: 先验概率 (来自策略网络)
        visit_count: 访问次数 N(s,a)
        total_value: 累计价值 W(s,a)
        children: 子节点字典 {action_index: MCTSNode}
    """

    def __init__(
        self,
        state: Any = None,
        parent: Optional["MCTSNode"] = None,
        action: Optional[Any] = None,
        action_index: Optional[int] = None,
        prior: float = 0.0,
    ):
        """
        初始化 MCTS 节点

        Args:
            state: 游戏状态 (可选,支持延迟加载)
            parent: 父节点
            action: 从父节点到此节点的动作对象
            action_index: 动作索引 (在合法动作列表中的位置)
            prior: 先验概率 P(s,a)
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.action_index = action_index

        # MCTS 统计量
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0

        # 子节点: {action_index: MCTSNode}
        self.children: Dict[int, "MCTSNode"] = {}

    def q_value(self) -> float:
        """
        平均动作价值 Q(s,a)

        Returns:
            平均价值 (如果未访问则返回 0)
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float = 1.5) -> float:
        """
        UCB (Upper Confidence Bound) 分数

        用于平衡探索和利用:
        UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            c_puct: 探索常数 (越大越倾向探索未访问节点)

        Returns:
            UCB 分数
        """
        if self.parent is None:
            return 0.0

        # Q 值 (利用)
        q = self.q_value()

        # U 值 (探索)
        u = c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)

        return q + u

    def select_child(self, c_puct: float = 1.5) -> "MCTSNode":
        """
        选择 UCB 分数最高的子节点

        Args:
            c_puct: 探索常数

        Returns:
            UCB 最高的子节点

        Raises:
            ValueError: 如果没有子节点
        """
        if not self.children:
            raise ValueError("Node has no children to select from")

        return max(self.children.values(), key=lambda node: node.ucb_score(c_puct))

    def expand(self, action_probs: np.ndarray, legal_actions: list) -> None:
        """
        扩展节点,为所有合法动作创建子节点

        Args:
            action_probs: 动作概率分布 (长度 = len(legal_actions))
            legal_actions: 合法动作列表
        """
        for action_index, (action, prob) in enumerate(zip(legal_actions, action_probs)):
            if action_index not in self.children:
                self.children[action_index] = MCTSNode(
                    state=None,  # 延迟加载
                    parent=self,
                    action=action,
                    action_index=action_index,
                    prior=prob,
                )

    def update(self, value: float) -> None:
        """
        更新节点统计量并向上传播

        在多人游戏中,价值从不同玩家视角会取反。

        Args:
            value: 当前玩家视角下的价值 (1 = 胜利, -1 = 失败, 0 = 平局)
        """
        self.visit_count += 1
        self.total_value += value

    def backpropagate(self, value: float) -> None:
        """
        从当前节点向上回溯更新所有祖先节点

        Args:
            value: 叶节点的价值 (当前玩家视角)
        """
        node = self
        while node is not None:
            node.update(value)
            # 多人游戏: 切换到对手视角 (价值取反)
            value = -value
            node = node.parent

    def is_leaf(self) -> bool:
        """检查是否为叶节点 (未扩展)"""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """检查是否为根节点"""
        return self.parent is None

    def get_visit_counts(self) -> Dict[int, int]:
        """
        获取所有子节点的访问次数

        Returns:
            {action_index: visit_count}
        """
        return {action_idx: child.visit_count for action_idx, child in self.children.items()}

    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """
        根据访问次数计算动作概率分布

        Args:
            temperature: 温度参数
                - temperature = 0: 选择访问次数最多的动作 (确定性)
                - temperature = 1: 概率正比于访问次数 (随机)
                - temperature > 1: 更均匀的分布 (更多探索)

        Returns:
            动作概率分布 (长度 = len(children))
        """
        if not self.children:
            return np.array([])

        # 获取访问次数
        action_indices = sorted(self.children.keys())
        visit_counts = np.array([self.children[idx].visit_count for idx in action_indices])

        if temperature == 0:
            # 确定性: 选择访问最多的动作
            probs = np.zeros_like(visit_counts, dtype=np.float32)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            # 随机: 根据访问次数加温度调整
            visit_counts = visit_counts ** (1.0 / temperature)
            probs = visit_counts / np.sum(visit_counts)

        return probs

    def __repr__(self) -> str:
        """节点的字符串表示"""
        return (
            f"MCTSNode(visit_count={self.visit_count}, "
            f"q_value={self.q_value():.3f}, "
            f"prior={self.prior:.3f}, "
            f"children={len(self.children)})"
        )


# ===== 导出 =====
__all__ = ["MCTSNode"]
