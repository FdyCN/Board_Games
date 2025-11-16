"""
自对弈 Worker

负责收集自对弈数据，支持单进程和多进程模式。
"""

import time
from typing import List, Optional, Callable
import numpy as np

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

        # 转换为索引
        legal_action_indices = [
            game.action_to_index(action, legal_actions) for action in legal_actions
        ]

        # Agent 选择动作
        action_idx, info = agent.select_action(
            observation, legal_action_indices, deterministic=deterministic
        )

        # 转换回动作对象
        action = game.index_to_action(action_idx, legal_actions)

        # 执行动作
        next_state, rewards, done, game_info = game.step(action)

        # 创建经验
        experience = Experience(
            player_id=current_player,
            observation=observation,
            action=action_idx,
            reward=rewards[current_player],
            next_observation=game.state_to_observation(next_state, current_player)
            if not done
            else None,
            done=done,
            log_prob=info["log_prob"],
            value=info["value"],
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

        self.episodes_collected = 0
        self.total_steps = 0

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

        Args:
            num_episodes: Episode 数量
            progress_callback: 进度回调函数

        Returns:
            Episode 列表
        """
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
    "SelfPlayWorker",
]
