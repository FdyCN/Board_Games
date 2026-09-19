"""
自对弈 Worker

负责收集自对弈数据，支持单进程和多进程模式。
"""

import time
import multiprocessing as mp
from typing import List, Optional, Callable, Tuple, Any
import numpy as np
import torch

from core.game_interface import GameInterface
from core.agent_interface import AgentInterface
from core.types import Experience, Episode
from training.experience import compute_advantages_for_episode


def collect_episode(
    game: GameInterface,
    agents: List[AgentInterface],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    deterministic: bool = False,
    verbose: bool = False,
) -> Episode:
    """
    收集一局完整的游戏经验

    Args:
        game: 游戏实例
        agents: Agent 列表（每个玩家一个）
        gamma: 折扣因子
        gae_lambda: GAE 参数
        deterministic: 是否使用确定性策略
        verbose: 是否打印详细信息

    Returns:
        Episode 对象

    Examples:
        >>> game = SplendorGame(num_players=4)
        >>> agents = [NeuralAgent(model) for _ in range(4)]
        >>> episode = collect_episode(game, agents, gamma=0.99)
        >>> print(f"收集了 {len(episode.experiences)} 个经验")
    """
    if len(agents) != game.num_players:
        raise ValueError(
            f"Agent 数量 ({len(agents)}) 与游戏玩家数 ({game.num_players}) 不匹配"
        )

    # 重置游戏
    state = game.reset()
    done = False
    experiences = []
    step_count = 0
    start_time = time.time()

    if verbose:
        print(f"开始自对弈游戏 (玩家数: {game.num_players})...")

    # 游戏循环
    while not done:
        # 获取当前玩家
        current_player = game.get_current_player(state)
        agent = agents[current_player]

        # 获取观察和合法动作
        observation = game.state_to_observation(state, current_player)
        legal_actions = game.get_legal_actions(state)

        if not legal_actions:
            if verbose:
                print(f"步数 {step_count}: 玩家 {current_player} 没有合法动作，游戏结束")
            done = True
            break

        # 转换为规范动作索引（固定槽位）
        legal_action_indices = [
            game.action_to_index(action, state) for action in legal_actions
        ]

        # 创建合法动作掩码 (用于保存到经验中)
        legal_actions_mask = np.zeros(game.action_space_size, dtype=np.float32)
        legal_actions_mask[legal_action_indices] = 1.0

        # Agent 选择动作
        action_idx, info = agent.select_action(
            observation, legal_action_indices, deterministic=deterministic
        )

        # 转换回动作对象
        action = game.index_to_action(action_idx, state)

        # 执行动作
        next_state, rewards, done, game_info = game.step(action)

        # 创建经验
        # 注意：这里只保存"稠密奖励"（不含终局胜利奖励）。
        # 终局胜利/失败奖励在 compute_advantages_for_episode 中按玩家注入，
        # 避免"胜利奖励只落在最后一位行动玩家身上"而丢失。
        experience = Experience(
            player_id=current_player,
            observation=observation,
            action=action_idx,
            reward=game_info.get("dense_reward", rewards[current_player]),
            next_observation=game.state_to_observation(next_state, current_player)
            if not done
            else None,
            done=done,
            log_prob=info["log_prob"],
            value=info["value"],
            legal_actions_mask=legal_actions_mask,  # 保存合法动作掩码
        )

        experiences.append(experience)

        # 更新状态
        state = next_state
        step_count += 1

        if verbose and step_count % 10 == 0:
            print(f"步数 {step_count}, 当前玩家: {current_player}")

    # 获取最终奖励
    final_rewards = game.get_final_rewards(state)

    # 获取获胜者（仅在终局状态）
    if game.is_terminal(state):
        winner = game.get_winner(state)
    else:
        # 游戏未正常结束，使用 final_rewards 确定获胜者
        if verbose:
            print("警告: 游戏未达到终局状态，使用当前分数确定获胜者")
        max_reward = max(final_rewards)
        winners = [i for i, r in enumerate(final_rewards) if r == max_reward]
        winner = winners[0] if len(winners) == 1 else -1  # 平局返回 -1

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"游戏结束! 步数: {step_count}, 获胜者: {winner}, 耗时: {elapsed_time:.2f}s")
        print(f"最终奖励: {final_rewards}")

    # 填充终局胜负标签（辅助任务用）：本玩家获胜=1，否则=0
    # winner=-1（平局/死锁）时，所有玩家都视为"未获胜"（0）。
    for exp in experiences:
        exp.outcome = 1.0 if (winner >= 0 and exp.player_id == winner) else 0.0

    # 创建 Episode
    episode = Episode(
        experiences=experiences,
        total_rewards=final_rewards,
        winner=winner,
        num_steps=step_count,
        metadata={"elapsed_time": elapsed_time},
    )

    # 计算优势函数
    compute_advantages_for_episode(episode, gamma=gamma, gae_lambda=gae_lambda)

    return episode


def collect_episodes(
    game: GameInterface,
    agents: List[AgentInterface],
    num_episodes: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    deterministic: bool = False,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Episode]:
    """
    收集多局游戏经验

    Args:
        game: 游戏实例
        agents: Agent 列表
        num_episodes: 收集的 episode 数量
        gamma: 折扣因子
        gae_lambda: GAE 参数
        deterministic: 是否使用确定性策略
        verbose: 是否打印详细信息
        progress_callback: 进度回调函数 callback(current, total)

    Returns:
        Episode 列表

    Examples:
        >>> game = SplendorGame(num_players=4)
        >>> agents = [NeuralAgent(model) for _ in range(4)]
        >>> episodes = collect_episodes(game, agents, num_episodes=100)
        >>> print(f"收集了 {len(episodes)} 个 episodes")
    """
    episodes = []
    start_time = time.time()

    for i in range(num_episodes):
        if verbose:
            print(f"\n=== Episode {i + 1}/{num_episodes} ===")

        episode = collect_episode(
            game=game,
            agents=agents,
            gamma=gamma,
            gae_lambda=gae_lambda,
            deterministic=deterministic,
            verbose=verbose,
        )

        episodes.append(episode)

        # 进度回调
        if progress_callback:
            progress_callback(i + 1, num_episodes)

    elapsed_time = time.time() - start_time

    if verbose:
        avg_steps = np.mean([ep.num_steps for ep in episodes])
        avg_reward = np.mean([sum(ep.total_rewards) for ep in episodes])
        print(f"\n收集完成!")
        print(f"总耗时: {elapsed_time:.2f}s")
        print(f"平均步数: {avg_steps:.1f}")
        print(f"平均奖励: {avg_reward:.3f}")
        print(f"吞吐: {num_episodes / elapsed_time:.2f} episodes/s")

    return episodes


def _worker_init(model_state_dict: Optional[dict], seed: int) -> None:
    """
    多进程 Worker 初始化函数

    在每个子进程启动时设置随机种子，确保每个进程的随机性独立。

    Args:
        model_state_dict: 模型状态字典（暂未使用，预留给未来）
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_collect_episode(args: Tuple) -> Episode:
    """
    Worker 函数：在子进程中收集单个 episode

    Args:
        args: 包含以下元素的元组：
            - game_class: 游戏类
            - game_kwargs: 游戏初始化参数
            - agent_data_list: 序列化的 agent 数据列表（pickle格式）
            - gamma: 折扣因子
            - gae_lambda: GAE 参数
            - deterministic: 是否确定性策略
            - seed: 随机种子

    Returns:
        Episode 对象
    """
    (
        game_class,
        game_kwargs,
        agent_data_list,
        gamma,
        gae_lambda,
        deterministic,
        seed,
    ) = args

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 创建游戏实例
    game = game_class(**game_kwargs)

    # 反序列化 agents
    import pickle

    agents = []
    for agent_data in agent_data_list:
        agent = pickle.loads(agent_data)
        # 确保模型处于评估模式
        if hasattr(agent, "model"):
            agent.model.eval()
        agents.append(agent)

    # 收集一个 episode
    episode = collect_episode(
        game=game,
        agents=agents,
        gamma=gamma,
        gae_lambda=gae_lambda,
        deterministic=deterministic,
        verbose=False,  # 多进程模式下不打印详细信息
    )

    return episode


def collect_episodes_multiprocess(
    game_class: type,
    game_kwargs: dict,
    agents: List[AgentInterface],
    num_episodes: int,
    num_workers: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    deterministic: bool = False,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Episode]:
    """
    使用多进程收集多个 episodes

    Args:
        game_class: 游戏类
        game_kwargs: 游戏初始化参数
        agents: Agent 列表（将被序列化传递给子进程）
        num_episodes: Episode 数量
        num_workers: 并行进程数
        gamma: 折扣因子
        gae_lambda: GAE 参数
        deterministic: 是否确定性策略
        seed: 随机种子（如果为 None，使用随机种子）
        progress_callback: 进度回调

    Returns:
        Episode 列表

    Examples:
        >>> from games.splendor import SplendorGame
        >>> from agents.neural_agent import NeuralAgent
        >>> model = create_model(...)
        >>> agents = [NeuralAgent(model) for _ in range(4)]
        >>> episodes = collect_episodes_multiprocess(
        ...     game_class=SplendorGame,
        ...     game_kwargs={"num_players": 4},
        ...     agents=agents,
        ...     num_episodes=100,
        ...     num_workers=4,
        ... )
    """
    import pickle

    start_time = time.time()

    # 生成每个 episode 的随机种子
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)

    rng = np.random.RandomState(seed)
    episode_seeds = [rng.randint(0, 2**31 - 1) for _ in range(num_episodes)]

    # 序列化 agents（用于传递给子进程）
    agent_data_list = []
    for agent in agents:
        # 确保模型处于评估模式
        if hasattr(agent, "model"):
            agent.model.eval()
        agent_data = pickle.dumps(agent)
        agent_data_list.append(agent_data)

    # 准备参数
    args_list = [
        (
            game_class,
            game_kwargs,
            agent_data_list,
            gamma,
            gae_lambda,
            deterministic,
            episode_seeds[i],
        )
        for i in range(num_episodes)
    ]

    # 使用进程池收集 episodes
    with mp.Pool(processes=num_workers) as pool:
        episodes = []
        for i, episode in enumerate(pool.imap(_worker_collect_episode, args_list)):
            episodes.append(episode)

            # 进度回调
            if progress_callback:
                progress_callback(i + 1, num_episodes)

    elapsed_time = time.time() - start_time

    return episodes


class SelfPlayWorker:
    """
    自对弈 Worker

    封装自对弈逻辑，支持配置和状态管理。

    Attributes:
        game: 游戏实例
        agents: Agent 列表
        gamma: 折扣因子
        gae_lambda: GAE 参数
        deterministic: 是否使用确定性策略
        verbose: 是否打印详细信息

    Examples:
        >>> game = SplendorGame(num_players=4)
        >>> agents = [NeuralAgent(model) for _ in range(4)]
        >>> worker = SelfPlayWorker(game, agents, gamma=0.99, gae_lambda=0.95)
        >>> episodes = worker.collect(num_episodes=100)
    """

    def __init__(
        self,
        game: GameInterface,
        agents: List[AgentInterface],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        deterministic: bool = False,
        verbose: bool = False,
        num_workers: int = 1,
    ):
        """
        初始化 Worker

        Args:
            game: 游戏实例
            agents: Agent 列表
            gamma: 折扣因子
            gae_lambda: GAE 参数
            deterministic: 是否使用确定性策略
            verbose: 是否打印详细信息
            num_workers: 并行进程数（1 表示单进程，>1 表示多进程）
        """
        if len(agents) != game.num_players:
            raise ValueError(
                f"Agent 数量 ({len(agents)}) 与游戏玩家数 ({game.num_players}) 不匹配"
            )

        self.game = game
        self.agents = agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.deterministic = deterministic
        self.verbose = verbose
        self.num_workers = num_workers

        self.episodes_collected = 0
        self.total_steps = 0

        # 保存游戏类和参数（用于多进程）
        self._game_class = type(game)
        self._game_kwargs = {"num_players": game.num_players}
        if hasattr(game, "seed"):
            self._game_kwargs["seed"] = game.seed
        # 关键：多进程子进程会重新创建游戏实例，必须把 reward_config 一并传下去，
        # 否则子进程会退回 DEFAULT_REWARDS，导致自定义奖励/稠密奖励系数失效。
        if hasattr(game, "_rewards"):
            self._game_kwargs["reward_config"] = dict(game._rewards)

    def collect_one(self) -> Episode:
        """
        收集一个 Episode

        Returns:
            Episode 对象
        """
        episode = collect_episode(
            game=self.game,
            agents=self.agents,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            deterministic=self.deterministic,
            verbose=self.verbose,
        )

        self.episodes_collected += 1
        self.total_steps += episode.num_steps

        return episode

    def collect(
        self,
        num_episodes: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Episode]:
        """
        收集多个 Episodes

        根据 num_workers 设置，自动选择单进程或多进程模式。

        Args:
            num_episodes: Episode 数量
            progress_callback: 进度回调函数

        Returns:
            Episode 列表
        """
        # 根据 num_workers 选择收集模式
        if self.num_workers > 1:
            # 多进程模式
            if self.verbose:
                print(f"使用多进程模式收集数据 (workers={self.num_workers})...")

            episodes = collect_episodes_multiprocess(
                game_class=self._game_class,
                game_kwargs=self._game_kwargs,
                agents=self.agents,
                num_episodes=num_episodes,
                num_workers=self.num_workers,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                deterministic=self.deterministic,
                seed=None,
                progress_callback=progress_callback,
            )
        else:
            # 单进程模式
            episodes = collect_episodes(
                game=self.game,
                agents=self.agents,
                num_episodes=num_episodes,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                deterministic=self.deterministic,
                verbose=self.verbose,
                progress_callback=progress_callback,
            )

        # 更新统计信息
        self.episodes_collected += len(episodes)
        self.total_steps += sum(ep.num_steps for ep in episodes)

        return episodes

    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.episodes_collected = 0
        self.total_steps = 0

    def get_statistics(self) -> dict:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "episodes_collected": self.episodes_collected,
            "total_steps": self.total_steps,
            "avg_steps_per_episode": (
                self.total_steps / self.episodes_collected
                if self.episodes_collected > 0
                else 0.0
            ),
        }


# ===== 导出 =====
__all__ = [
    "collect_episode",
    "collect_episodes",
    "collect_episodes_multiprocess",
    "SelfPlayWorker",
]
