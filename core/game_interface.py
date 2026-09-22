"""
游戏抽象接口

本模块定义了所有桌游必须实现的标准接口 GameInterface。
所有游戏实现都应该继承这个抽象基类并实现所有抽象方法。
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core.types import ActionIndex, ActionType, ObservationType, PlayerID, Reward, StateType


class GameInterface(ABC):
    """
    桌游抽象基类

    所有游戏必须实现这个接口，以便与训练框架、评估系统等其他组件集成。

    设计原则：
    - 游戏状态不可变：每次 step 返回新状态，不修改原状态
    - 观察者模式：state_to_observation 根据玩家视角返回观察
    - 离散动作空间：所有动作映射到 [0, action_space_size) 的整数索引

    典型使用流程：
        >>> game = SplendorGame(num_players=4)
        >>> state = game.reset()
        >>> while not game.is_terminal(state):
        ...     player_id = game.get_current_player(state)
        ...     legal_actions = game.get_legal_actions(state)
        ...     action = choose_action(legal_actions)  # 由 Agent 选择
        ...     state, rewards, done, info = game.step(action)
        >>> final_rewards = game.get_final_rewards(state)
    """

    # ===== 核心游戏循环方法 =====

    @abstractmethod
    def reset(self) -> StateType:
        """
        重置游戏到初始状态

        Returns:
            StateType: 游戏初始状态对象

        Examples:
            >>> game = SplendorGame(num_players=4)
            >>> state = game.reset()
            >>> print(f"游戏开始，当前玩家: {game.get_current_player(state)}")
        """
        pass

    @abstractmethod
    def step(self, action: ActionType) -> tuple[StateType, list[Reward], bool, dict[str, Any]]:
        """
        执行一个动作，推进游戏状态

        Args:
            action: 动作对象（类型由具体游戏定义）

        Returns:
            new_state: 执行动作后的新状态（原状态不变）
            rewards: 所有玩家的即时奖励列表，长度为 num_players
            done: 游戏是否结束
            info: 额外信息字典，可包含：
                - 'legal_actions': 新状态下的合法动作
                - 'current_player': 当前玩家 ID
                - 'game_event': 游戏事件描述
                - 其他游戏特定信息

        Raises:
            ValueError: 如果动作非法

        Examples:
            >>> action = TakeGemsAction(gems=[2, 2, 0, 0, 0])
            >>> new_state, rewards, done, info = game.step(action)
            >>> if done:
            ...     print(f"游戏结束！最终奖励: {rewards}")
        """
        pass

    # ===== 状态查询方法 =====

    @abstractmethod
    def get_legal_actions(self, state: StateType) -> list[ActionType]:
        """
        获取当前状态下的所有合法动作

        Args:
            state: 当前游戏状态

        Returns:
            合法动作对象列表

        Examples:
            >>> legal_actions = game.get_legal_actions(state)
            >>> print(f"当前有 {len(legal_actions)} 个合法动作")
        """
        pass

    @abstractmethod
    def get_current_player(self, state: StateType) -> PlayerID:
        """
        获取当前应该行动的玩家 ID

        Args:
            state: 当前游戏状态

        Returns:
            玩家 ID (0, 1, 2, ...)

        Examples:
            >>> player_id = game.get_current_player(state)
            >>> print(f"轮到玩家 {player_id} 行动")
        """
        pass

    @abstractmethod
    def is_terminal(self, state: StateType) -> bool:
        """
        判断游戏是否结束

        Args:
            state: 当前游戏状态

        Returns:
            True 如果游戏结束，False 否则

        Examples:
            >>> if game.is_terminal(state):
            ...     winner = game.get_winner(state)
            ...     print(f"游戏结束，获胜者: Player {winner}")
        """
        pass

    @abstractmethod
    def get_final_rewards(self, state: StateType) -> list[Reward]:
        """
        获取游戏结束时的最终奖励

        注意：只在终局状态调用此方法

        Args:
            state: 终局状态

        Returns:
            所有玩家的最终奖励列表
            - 通常获胜者为 1.0，其他为 0.0
            - 平局时所有玩家可能都是 0.5
            - 也可以根据排名分配不同奖励

        Examples:
            >>> if game.is_terminal(state):
            ...     final_rewards = game.get_final_rewards(state)
            ...     # [1.0, 0.0, 0.0, 0.0] 表示玩家 0 获胜
        """
        pass

    # ===== 观察和动作编码方法 =====

    @abstractmethod
    def state_to_observation(self, state: StateType, player_id: PlayerID) -> ObservationType:
        """
        将游戏状态转换为指定玩家的观察（神经网络输入）

        重要设计要点：
        1. 玩家视角：观察应该只包含该玩家可见的信息
        2. 归一化：数值应归一化到合理范围（通常 [0, 1] 或 [-1, 1]）
        3. 固定维度：无论游戏进行到哪一步，观察维度必须固定
        4. 位置编码：可以包含当前玩家的相对位置信息

        Args:
            state: 游戏状态
            player_id: 观察者玩家 ID

        Returns:
            观察向量/矩阵（numpy array），形状为 observation_shape

        Examples:
            >>> obs = game.state_to_observation(state, player_id=0)
            >>> print(f"观察形状: {obs.shape}")  # (256,)
            >>> assert obs.shape == game.observation_shape
        """
        pass

    @abstractmethod
    def action_to_index(self, action: ActionType) -> ActionIndex:
        """
        将动作对象转换为动作索引（用于神经网络输出层）

        Args:
            action: 动作对象

        Returns:
            动作索引 (0 到 action_space_size-1)

        Examples:
            >>> action = BuyCardAction(card_id=5, tier=2)
            >>> action_idx = game.action_to_index(action)
            >>> print(f"动作索引: {action_idx}")  # 例如 42
        """
        pass

    @abstractmethod
    def index_to_action(self, index: ActionIndex) -> ActionType:
        """
        将动作索引转换为动作对象

        Args:
            index: 动作索引

        Returns:
            动作对象

        Examples:
            >>> action_idx = 42
            >>> action = game.index_to_action(action_idx)
            >>> print(f"动作: {action}")  # BuyCardAction(card_id=5, tier=2)
        """
        pass

    # ===== 属性（必须实现） =====

    @property
    @abstractmethod
    def num_players(self) -> int:
        """
        游戏玩家数量

        Returns:
            玩家数量

        Examples:
            >>> print(f"这是一个 {game.num_players} 人游戏")
        """
        pass

    @property
    @abstractmethod
    def observation_shape(self) -> tuple[int, ...]:
        """
        观察空间的形状

        Returns:
            观察形状元组
            - 1D: (feature_dim,) 例如 (256,)
            - 2D: (height, width) 例如 (8, 8)
            - 3D: (height, width, channels) 例如 (8, 8, 12)

        Examples:
            >>> print(f"观察形状: {game.observation_shape}")
            >>> # 输出: (256,) 或 (8, 8, 12)
        """
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """
        离散动作空间大小（所有可能动作的数量）

        注意：不是所有状态下所有动作都合法，需要用 get_legal_actions 查询

        Returns:
            动作空间大小

        Examples:
            >>> print(f"动作空间大小: {game.action_space_size}")
            >>> # 输出: 82（Splendor 约 80+ 个可能动作）
        """
        pass

    # ===== 可选方法（提供默认实现） =====

    def clone_state(self, state: StateType) -> StateType:
        """
        深拷贝游戏状态（用于 MCTS 等搜索算法）

        默认实现使用 copy.deepcopy，子类可以提供更高效的实现

        Args:
            state: 原始状态

        Returns:
            状态的深拷贝

        Examples:
            >>> state_copy = game.clone_state(state)
            >>> # 修改 state_copy 不会影响 state
        """
        import copy

        return copy.deepcopy(state)

    def render(self, state: StateType, mode: str = "human") -> Any | None:
        """
        可视化游戏状态（可选实现）

        Args:
            state: 当前状态
            mode: 渲染模式
                - 'human': 终端输出（ASCII 艺术）
                - 'rgb_array': 返回图像数组 (H, W, 3)
                - 'ansi': 返回 ANSI 字符串

        Returns:
            根据 mode 返回不同类型：
            - 'human': None（直接打印）
            - 'rgb_array': np.ndarray
            - 'ansi': str

        Examples:
            >>> game.render(state, mode='human')
            >>> # 终端显示游戏状态
        """
        print(f"游戏状态渲染未实现 (mode={mode})")
        return None

    def game_kwargs(self) -> dict[str, Any]:
        """
        返回用于重建一个等价游戏实例的构造参数（用于多进程数据收集）。

        多进程自对弈时，子进程需要用相同的参数重建游戏实例。默认只返回
        num_players；带有种子或稠密奖励的游戏应覆盖本方法，把 seed /
        reward_config 等一并返回，以保证子进程行为与主进程一致。

        Returns:
            可直接解包传给 create_game(game_name, **kwargs) 的参数字典。

        Examples:
            >>> game = SplendorGame(num_players=3, seed=42, reward_config={...})
            >>> game.game_kwargs()
            {'num_players': 3, 'seed': 42, 'reward_config': {...}}
        """
        return {"num_players": self.num_players}

    def get_winner(self, state: StateType) -> PlayerID:
        """
        获取获胜者 ID

        默认实现：返回最终奖励最高的玩家

        Args:
            state: 终局状态

        Returns:
            获胜者玩家 ID，如果平局返回 -1

        Examples:
            >>> if game.is_terminal(state):
            ...     winner = game.get_winner(state)
            ...     if winner >= 0:
            ...         print(f"获胜者: Player {winner}")
            ...     else:
            ...         print("平局")
        """
        if not self.is_terminal(state):
            raise ValueError("只能在终局状态调用 get_winner")

        final_rewards = self.get_final_rewards(state)
        max_reward = max(final_rewards)

        # 检查是否平局
        winners = [i for i, r in enumerate(final_rewards) if r == max_reward]
        if len(winners) > 1:
            return -1  # 平局

        return winners[0]

    def get_legal_action_mask(self, state: StateType) -> np.ndarray:
        """
        获取合法动作掩码（用于神经网络）

        Returns:
            布尔数组，形状为 (action_space_size,)
            True 表示合法动作，False 表示非法动作

        Examples:
            >>> mask = game.get_legal_action_mask(state)
            >>> # 在神经网络中使用掩码
            >>> policy[~mask] = -np.inf  # 非法动作概率设为 0
            >>> policy = torch.softmax(policy, dim=-1)
        """
        mask = np.zeros(self.action_space_size, dtype=bool)
        legal_actions = self.get_legal_actions(state)
        legal_indices = [self.action_to_index(action) for action in legal_actions]
        mask[legal_indices] = True
        return mask

    def get_symmetries(
        self, state: StateType, policy: np.ndarray
    ) -> list[tuple[StateType, np.ndarray]]:
        """
        获取状态的对称变换（用于数据增强）

        对于具有对称性的游戏（如围棋、象棋），可以通过旋转、翻转等操作增强训练数据

        Args:
            state: 游戏状态
            policy: 策略分布

        Returns:
            [(state, policy)] 对称状态和对应策略的列表

        Examples:
            >>> # 对于围棋，可以返回 8 种对称（4 个旋转 + 4 个翻转）
            >>> symmetries = game.get_symmetries(state, policy)
            >>> # 默认实现：只返回原始状态
        """
        return [(state, policy)]

    @property
    def auxiliary_shape(self) -> tuple[int, ...] | None:
        """
        辅助监督标签的观察侧形状（可选）。
        """
        return None

    @property
    def encoder_params(self) -> dict | None:
        """
        模型编码器所需的游戏特定参数（可选）。

        例如序列编码器需要知道「静态段维度 / 序列长度 / 事件维度」才能把
        观察向量拆分成（静态特征 + 有序事件时间线）。默认 None 表示无需
        额外参数（普通 MLP/Attention 编码器）。
        """
        return None

    def get_auxiliary_labels(self, state: StateType, player_id: PlayerID) -> np.ndarray | None:
        """
        返回该玩家视角的辅助监督标签（可选，用于「上帝视角」辅助任务）。

        训练数据收集时，主进程能看到完整状态（包括其他玩家的隐藏信息），
        因此可以生成「预测对手隐藏状态」等辅助标签，帮助模型学习从公开
        信息推断隐藏信息（信念状态）。

        Args:
            state: 当前游戏状态（可能是完整状态）。
            player_id: 观察者玩家 ID。

        Returns:
            辅助标签数组（形状见 auxiliary_shape），或 None 表示该状态无标签。

        Examples:
            >>> # 情书：预测每个相对对手的手牌值
            >>> labels = game.get_auxiliary_labels(state, player_id)
            >>> # labels 形状 ((num_players-1),)，值为 0..7 或 -1
        """
        return None

    def __str__(self) -> str:
        """返回游戏的字符串表示"""
        return f"{self.__class__.__name__}(num_players={self.num_players})"

    def __repr__(self) -> str:
        """返回游戏的详细表示"""
        return (
            f"{self.__class__.__name__}("
            f"num_players={self.num_players}, "
            f"obs_shape={self.observation_shape}, "
            f"action_size={self.action_space_size})"
        )
