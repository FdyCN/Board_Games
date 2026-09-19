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
        batch_size: int = 64,
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
            batch_size: 批量叶节点评估的大小（越大越能利用 GPU/MPS 并行）
        """
        self.game = game
        self.c_puct = c_puct
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.noise_alpha = noise_alpha
        self.device = torch.device(device)
        self.batch_size = batch_size

        # 搜索专用游戏实例（复用，避免每次动作都 clone/deepcopy）
        self._search_game = game.clone()

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

        # 批量执行 MCTS 模拟（虚拟损失 + 批量叶节点评估）
        pending_leaves = []  # (leaf_node, leaf_state, leaf_legal_actions, search_path)
        for sim in range(num_simulations):
            # 选择叶节点（带虚拟损失）
            leaf_node, leaf_state, search_path = self._select_leaf(root)
            leaf_legal_actions = self.game.get_legal_actions(leaf_state)

            if self.game.is_terminal(leaf_state):
                value = self._get_terminal_value(leaf_state)
                for p in reversed(search_path):
                    p.revert_virtual_loss(value)
            elif not leaf_legal_actions:
                # 无合法动作（异常/死锁），价值按 0 处理
                for p in reversed(search_path):
                    p.revert_virtual_loss(0.0)
            else:
                pending_leaves.append((leaf_node, leaf_state, leaf_legal_actions, search_path))

            # 攒够一批就批量评估
            if len(pending_leaves) >= self.batch_size:
                self._evaluate_batch(pending_leaves, model)
                pending_leaves = []

        # 处理剩余未评估的叶节点
        if pending_leaves:
            self._evaluate_batch(pending_leaves, model)

        # 根据访问次数计算动作概率
        action_probs = root.get_action_probs(temperature=temperature)

        if verbose:
            visit_counts = root.get_visit_counts()
            print(f"\n  MCTS Search Results:")
            print(f"    Total visits: {root.visit_count}")
            print(f"    Action probs: {action_probs}")
            print(f"    Visit counts: {list(visit_counts.values())}")

        return action_probs, legal_actions

    def _select_leaf(self, root: MCTSNode):
        """
        Selection：从根节点沿 UCB 选择到叶节点，沿途添加虚拟损失。

        Returns:
            (leaf_node, leaf_state, search_path)
        """
        node = root
        search_path = [node]
        current_state = root.state

        node.add_virtual_loss()

        while not node.is_leaf():
            node = node.select_child(self.c_puct)
            search_path.append(node)
            node.add_virtual_loss()

            # 延迟状态计算
            if node.state is None:
                current_state = self._apply_action(current_state, node.action)
                node.state = current_state
            else:
                current_state = node.state

        return node, current_state, search_path

    def _evaluate_batch(self, leaves: list, model) -> None:
        """
        批量评估一组叶节点：一次前向得到所有叶子的策略先验和价值，
        然后逐个扩展并回溯。

        Args:
            leaves: [(leaf_node, leaf_state, leaf_legal_actions, search_path), ...]
            model: Actor-Critic 模型
        """
        if not leaves:
            return

        # 构建批量输入
        obs_list = []
        mask_list = []
        for _, leaf_state, leaf_legal_actions, _ in leaves:
            cp = self.game.get_current_player(leaf_state)
            obs_list.append(self.game.state_to_observation(leaf_state, cp))
            mask_list.append(self._create_legal_mask(leaf_state, leaf_legal_actions))

        obs_batch = torch.FloatTensor(np.stack(obs_list)).to(self.device)
        mask_batch = torch.BoolTensor(np.stack(mask_list)).to(self.device)

        model.eval()
        with torch.no_grad():
            logits, values = model(obs_batch, mask_batch)  # (B, action_size), (B, 1)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        vals = values.squeeze(-1).cpu().numpy()

        for i, (leaf_node, leaf_state, leaf_legal_actions, search_path) in enumerate(leaves):
            # 提取该叶子各合法动作的先验概率
            leaf_probs = np.array(
                [probs[i][self.game.action_to_index(a, leaf_state)] for a in leaf_legal_actions],
                dtype=np.float32,
            )
            s = leaf_probs.sum()
            if s > 1e-8:
                leaf_probs = leaf_probs / s
            else:
                leaf_probs = np.full(len(leaf_legal_actions), 1.0 / len(leaf_legal_actions), dtype=np.float32)

            leaf_node.expand(leaf_probs, leaf_legal_actions)

            value = float(vals[i])
            # 回溯：抵消虚拟损失 + 加入真实价值
            for p in reversed(search_path):
                p.revert_virtual_loss(value)

    def _apply_action(self, state, action):
        """
        在状态副本上执行动作

        Args:
            state: 当前状态
            action: 动作对象

        Returns:
            新状态
        """
        # 复用搜索专用游戏实例：直接设置状态，step 内部做快速浅拷贝
        self._search_game._state = state
        new_state, _, _, _ = self._search_game.step(action)

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

        # action_probs_full 的形状是 (1, action_space_size)，其中非法动作概率为 0。
        # 需要按 legal_actions 的**顺序**提取每个动作在规范槽位上的概率，
        # 保证 legal_probs[i] 与 legal_actions[i] 一一对应。
        probs_np = action_probs_full[0].cpu().numpy()
        legal_probs = np.array(
            [probs_np[self.game.action_to_index(a, state)] for a in legal_actions],
            dtype=np.float32,
        )

        # 归一化 (确保和为 1)
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

    def _get_terminal_value(self, terminal_state) -> float:
        """
        获取终局状态的真实价值（当前玩家视角）

        Args:
            terminal_state: 终局状态

        Returns:
            价值 (终局当前玩家视角: 胜利=+1, 失败=-1/(n-1)，与 PPO 零和终局奖励一致)
        """
        results = self.game.get_result(terminal_state)  # winner=1.0, others=0.0
        leaf_player = terminal_state.current_player
        n = self.game.num_players

        if results[leaf_player] > 0.5:
            return 1.0  # 胜利
        return -1.0 / (n - 1)  # 失败（零和：失败者平分负奖励）

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
            action_idx = self.game.action_to_index(action, state)
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
