"""
MCTS 调度器

用于在训练过程中渐进式增加 MCTS 模拟次数。

示例:
    >>> scheduler = MCTSScheduler([(0, 0), (200, 50), (500, 200)])
    >>> scheduler.get_simulations(150)  # 返回 0
    >>> scheduler.get_simulations(300)  # 返回 50
    >>> scheduler.get_simulations(600)  # 返回 200
"""

from typing import List, Tuple


class MCTSScheduler:
    """
    MCTS 模拟次数调度器

    根据训练迭代次数动态调整 MCTS 搜索深度。

    Attributes:
        schedule: 调度表 [(iteration, simulations), ...]
    """

    def __init__(self, schedule: List[Tuple[int, int]]):
        """
        初始化调度器

        Args:
            schedule: 调度表,格式为 [(迭代次数, 模拟次数), ...]
                     例如: [(0, 0), (200, 50), (500, 200)]
                     表示:
                     - 0-199 次迭代: 0 次模拟 (纯 PPO)
                     - 200-499 次迭代: 50 次模拟
                     - 500+ 次迭代: 200 次模拟

        Raises:
            ValueError: 如果调度表为空或格式错误
        """
        if not schedule:
            raise ValueError("Schedule cannot be empty")

        # 按迭代次数排序
        self.schedule = sorted(schedule, key=lambda x: x[0])

        # 验证格式
        for iteration, simulations in self.schedule:
            if not isinstance(iteration, int) or not isinstance(simulations, int):
                raise ValueError(f"Invalid schedule entry: ({iteration}, {simulations})")
            if iteration < 0 or simulations < 0:
                raise ValueError(f"Negative values not allowed: ({iteration}, {simulations})")

    def get_simulations(self, iteration: int) -> int:
        """
        获取指定迭代次数对应的 MCTS 模拟次数

        Args:
            iteration: 当前迭代次数

        Returns:
            MCTS 模拟次数
        """
        # 找到最后一个不大于 iteration 的阈值
        simulations = 0
        for iter_threshold, sim_count in self.schedule:
            if iteration >= iter_threshold:
                simulations = sim_count
            else:
                break

        return simulations

    def get_schedule_info(self) -> str:
        """
        获取调度表的可读描述

        Returns:
            调度表描述字符串
        """
        lines = ["MCTS Schedule:"]
        for i, (iteration, simulations) in enumerate(self.schedule):
            if i == len(self.schedule) - 1:
                lines.append(f"  Iteration {iteration}+: {simulations} simulations")
            else:
                next_iteration = self.schedule[i + 1][0]
                lines.append(
                    f"  Iteration {iteration}-{next_iteration-1}: {simulations} simulations"
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MCTSScheduler(schedule={self.schedule})"


# ===== 导出 =====
__all__ = ["MCTSScheduler"]
