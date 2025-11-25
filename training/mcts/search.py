"""
MCTS 搜索引擎实现

核心搜索流程:
1. Selection: 根据 UCB 选择子节点
2. Expansion: 扩展叶节点
3. Simulation: 使用神经网络评估价值
4. Backpropagation: 反向传播价值
"""

import numpy as np
import torch
from typing import Optional, Tuple, List
from copy import deepcopy

from core.game_interface import GameInterface
from training.mcts.node import MCTSNode


class MCTS:
    """
    MCTS 搜索引擎

    用于增强策略训练的蒙特卡洛树搜索。

    Attributes:
        game: 游戏实例
        c_puct: UCB 探索常数
        add_noise: 是否在根节点添加 Dirichlet 噪声
        noise_epsilon: 噪声混合系数
        noise_alpha: Dirichlet 分布参数
    """

    def __init__(
        self,
        game: GameInterface,
        c_puct: float = 1.5,
        add_noise: bool = True,
        noise_epsilon: float = 0.25,
        noise_alpha: float = 0.3,
        device: str = "cpu",
    ):
        """
        初始化 MCTS 搜索引擎

        Args:
            game: 游戏实例
            c_puct: UCB 探索常数 (推荐范围: 1.0-2.5)
            add_noise: 是否添加 Dirichlet 噪声增加探索
            noise_epsilon: 噪声混合系数 (推荐: 0.25)
            noise_alpha: Dirichlet 参数 (推荐: 10/avg_branching_factor)
            device: 设备 ("cpu", "cuda", "mps")
        """
        self.game = game
        self.c_puct = c_puct
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.noise_alpha = noise_alpha
        self.device = torch.device(device)

    def search(
        self,
        state,
        model,
        num_simulations: int = 100,
        temperature: float = 1.0,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, List]:
        """
        执行 MCTS 搜索

        Args:
            state: 当前游戏状态
            model: Actor-Critic 模型
            num_simulations: MCTS 模拟次数 (越多越准确但越慢)
            temperature: 采样温度 (0=确定性, 1=随机, >1=更随机)
            verbose: 是否打印调试信息

        Returns:
            action_probs: 动作概率分布 (numpy array)
            legal_actions: 对应的合法动作列表
        """
        # 创建根节点
        root = MCTSNode(state=state)

        # 获取合法动作
        legal_actions = self.game.get_legal_actions(state)
        if not legal_actions:
            # 检查是否为终局
            if self.game.is_terminal(state):
                raise ValueError("Cannot search on terminal state (game is over)")
            else:
                raise ValueError("No legal actions available (possible game state error)")

        # 获取策略网络的先验概率
        action_probs = self._get_policy_probs(state, legal_actions, model)

        # 在根节点添加 Dirichlet 噪声 (增加探索)
        if self.add_noise:
            action_probs = self._add_dirichlet_noise(action_probs)

        # 扩展根节点
        root.expand(action_probs, legal_actions)

        # 执行 MCTS 模拟
        for sim in range(num_simulations):
            self._simulate(root, model)

            if verbose and (sim + 1) % 20 == 0:
                print(f"  MCTS Simulation {sim + 1}/{num_simulations}")

        # 根据访问次数计算动作概率
        action_probs = root.get_action_probs(temperature=temperature)

        if verbose:
            visit_counts = root.get_visit_counts()
            print(f"\n  MCTS Search Results:")
            print(f"    Total visits: {root.visit_count}")
            print(f"    Action probs: {action_probs}")
            print(f"    Visit counts: {list(visit_counts.values())}")

        return action_probs, legal_actions

    def _simulate(self, root: MCTSNode, model) -> None:
        """
        单次 MCTS 模拟

        执行流程:
        1. Selection: 从根节点开始,选择 UCB 最高的子节点
        2. Expansion: 到达叶节点后扩展
        3. Evaluation: 使用神经网络评估价值
        4. Backpropagation: 向上传播价值

        Args:
            root: 根节点
            model: Actor-Critic 模型
        """
        node = root
        search_path = [node]
        current_state = root.state

        # === 1. Selection: 选择叶节点 ===
        while not node.is_leaf():
            node = node.select_child(self.c_puct)
            search_path.append(node)

            # 延迟状态计算 (节省内存)
            if node.state is None:
                current_state = self._apply_action(current_state, node.action)
                node.state = current_state
            else:
                current_state = node.state

        # === 2. Expansion & Evaluation ===
        # 检查是否为终局
        if self.game.is_terminal(current_state):
            # 终局: 使用真实游戏结果
            value = self._get_terminal_value(current_state, root.state)
        else:
            # 非终局: 使用神经网络评估并扩展
            legal_actions = self.game.get_legal_actions(current_state)

            if legal_actions:
                # 获取策略和价值
                action_probs = self._get_policy_probs(current_state, legal_actions, model)
                value = self._get_value(current_state, model)

                # 扩展节点
                node.expand(action_probs, legal_actions)
            else:
                # 没有合法动作 (异常情况)
                value = 0.0

        # === 3. Backpropagation: 回溯更新 ===
        # 从叶节点开始回溯
        for path_node in reversed(search_path):
            path_node.update(value)
            value = -value  # 多人游戏: 切换到对手视角

    def _apply_action(self, state, action):
        """
        在状态副本上执行动作

        Args:
            state: 当前状态
            action: 动作对象

        Returns:
            新状态
        """
        # 克隆游戏和状态
        game_copy = self.game.clone()
        game_copy._state = deepcopy(state)

        # 执行动作
        new_state, _, _, _ = game_copy.step(action)

        return new_state

    def _get_policy_probs(self, state, legal_actions: list, model) -> np.ndarray:
        """
        使用策略网络获取动作概率分布

        Args:
            state: 游戏状态
            legal_actions: 合法动作列表
            model: Actor-Critic 模型

        Returns:
            动作概率 (numpy array, 长度 = len(legal_actions))
        """
        # 获取当前玩家
        current_player = self.game.get_current_player(state)

        # 编码状态
        obs = self.game.state_to_observation(state, current_player)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # 创建合法动作掩码
        legal_mask = self._create_legal_mask(state, legal_actions)
        legal_mask_tensor = torch.BoolTensor(legal_mask).unsqueeze(0).to(self.device)

        # 获取策略网络输出
        model.eval()
        with torch.no_grad():
            action_probs_full = model.get_action_probs(obs_tensor, legal_mask_tensor)

        # action_probs_full 的形状是 (1, action_space_size)
        # 其中非法动作的概率为 0, 合法动作的概率已归一化
        # 我们需要提取出对应于 legal_actions 的概率

        # 方式: 遍历 legal_actions, 找到每个动作在 action_space 中的位置
        legal_probs = []
        for i, action in enumerate(legal_actions):
            # 找到这个动作在整个 legal_actions 列表中的索引 (就是 i)
            # 然后找到它在 action_space 中对应的全局索引
            # 这里有个问题: action_to_index 返回的是在 legal_actions 中的索引
            # 我们需要一个能返回全局索引的方法...

            # 临时方案: 利用 legal_mask 来提取
            # legal_mask 中 True 的位置对应全局 action_space 中合法动作的索引
            pass

        # 更简单的方法: 直接从 action_probs_full 中提取非零概率
        probs_np = action_probs_full[0].cpu().numpy()
        legal_indices = np.where(legal_mask)[0]  # 找到所有 True 的位置
        legal_probs = probs_np[legal_indices]

        # 归一化 (确保和为 1, 虽然理论上已经归一化了)
        legal_probs = legal_probs / (np.sum(legal_probs) + 1e-8)

        return legal_probs

    def _get_value(self, state, model) -> float:
        """
        使用价值网络评估状态价值

        Args:
            state: 游戏状态
            model: Actor-Critic 模型

        Returns:
            状态价值 (当前玩家视角)
        """
        current_player = self.game.get_current_player(state)
        obs = self.game.state_to_observation(state, current_player)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        model.eval()
        with torch.no_grad():
            value = model.get_value(obs_tensor)

        return value.item()

    def _get_terminal_value(self, terminal_state, root_state) -> float:
        """
        获取终局状态的真实价值

        Args:
            terminal_state: 终局状态
            root_state: 根节点状态 (用于确定评估视角)

        Returns:
            价值 (根节点玩家视角: 1=胜利, -1=失败, 0=平局)
        """
        # 获取游戏结果
        results = self.game.get_result(terminal_state)

        # 确定根节点的玩家
        root_player = self.game.get_current_player(root_state)

        # 返回该玩家的结果 (1=赢, 0=输)
        # 转换为 MCTS 价值 (1=赢, -1=输)
        if results[root_player] > 0.5:
            return 1.0  # 胜利
        elif results[root_player] < 0.5:
            return -1.0  # 失败
        else:
            return 0.0  # 平局

    def _create_legal_mask(self, state, legal_actions: list) -> np.ndarray:
        """
        创建合法动作掩码

        Args:
            state: 游戏状态
            legal_actions: 合法动作列表

        Returns:
            掩码数组 (长度 = action_space_size)
        """
        mask = np.zeros(self.game.action_space_size, dtype=bool)

        for action in legal_actions:
            action_idx = self.game.action_to_index(action, legal_actions)
            mask[action_idx] = True

        return mask

    def _add_dirichlet_noise(self, action_probs: np.ndarray) -> np.ndarray:
        """
        添加 Dirichlet 噪声增加探索

        噪声公式:
        P'(a) = (1 - ε) * P(a) + ε * η(a)
        其中 η ~ Dir(α)

        Args:
            action_probs: 原始动作概率

        Returns:
            添加噪声后的概率
        """
        noise = np.random.dirichlet([self.noise_alpha] * len(action_probs))
        noisy_probs = (1 - self.noise_epsilon) * action_probs + self.noise_epsilon * noise

        # 归一化
        noisy_probs = noisy_probs / (np.sum(noisy_probs) + 1e-8)

        return noisy_probs


# ===== 导出 =====
__all__ = ["MCTS"]
