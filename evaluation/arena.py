"""
Arena 对战系统

提供 Agent 之间的竞技场对战，支持多种比赛模式。
"""

import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np

from core.game_interface import GameInterface
from core.agent_interface import AgentInterface
from evaluation.metrics import GameResult, MetricsCollector
from evaluation.elo_system import EloSystem


@dataclass
class MatchConfig:
    """
    比赛配置

    Attributes:
        num_games: 每对选手对战的局数
        shuffle_positions: 是否随机打乱玩家位置
        verbose: 是否显示详细信息
        progress_callback: 进度回调函数
    """

    num_games: int = 100
    shuffle_positions: bool = True
    verbose: bool = True
    progress_callback: Optional[Callable[[int, int], None]] = None


class Arena:
    """
    竞技场

    管理多个 Agent 之间的对战，支持循环赛等模式。

    Attributes:
        game: 游戏实例
        agents: 参赛 Agent 列表
        agent_names: Agent 名称列表
        metrics_collector: 指标收集器
        elo_system: ELO 评分系统

    Examples:
        >>> from games.registry import create_game
        >>> from agents.random_agent import RandomAgent
        >>> game = create_game("splendor", num_players=4)
        >>> agents = [RandomAgent(player_id=i) for i in range(4)]
        >>> agent_names = [f"Agent{i}" for i in range(4)]
        >>> arena = Arena(game, agents, agent_names)
        >>> results = arena.run_tournament(num_games=100)
        >>> print(f"Winner: {results['winner']}")
    """

    def __init__(
        self,
        game: GameInterface,
        agents: List[AgentInterface],
        agent_names: Optional[List[str]] = None,
        use_elo: bool = True,
        elo_k_factor: float = 32.0,
    ):
        """
        初始化 Arena

        Args:
            game: 游戏实例
            agents: Agent 列表
            agent_names: Agent 名称列表（如果为 None，则自动生成）
            use_elo: 是否使用 ELO 评分系统
            elo_k_factor: ELO K 因子
        """
        if len(agents) != game.num_players:
            raise ValueError(
                f"Agent 数量 ({len(agents)}) 必须等于游戏玩家数 ({game.num_players})"
            )

        self.game = game
        self.agents = agents
        self.agent_names = (
            agent_names
            if agent_names is not None
            else [f"Agent{i}" for i in range(len(agents))]
        )

        if len(self.agent_names) != len(agents):
            raise ValueError("agent_names 长度必须与 agents 长度一致")

        self.metrics_collector = MetricsCollector()
        self.elo_system = EloSystem(k_factor=elo_k_factor) if use_elo else None

        # 统计信息
        self.total_games = 0
        self.total_steps = 0

    def play_game(
        self,
        game_id: int,
        agent_order: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> GameResult:
        """
        进行一局游戏

        Args:
            game_id: 游戏 ID
            agent_order: Agent 顺序（索引列表）。如果为 None，则按默认顺序
            verbose: 是否显示详细信息

        Returns:
            GameResult 对象
        """
        # 确定 Agent 顺序
        if agent_order is None:
            agent_order = list(range(len(self.agents)))

        # 重新排列 Agent 和名称
        ordered_agents = [self.agents[i] for i in agent_order]
        ordered_names = [self.agent_names[i] for i in agent_order]

        # 重置游戏
        state = self.game.reset()
        done = False
        num_steps = 0
        start_time = time.time()

        if verbose:
            print(f"\n游戏 {game_id} 开始")
            print(f"玩家顺序: {ordered_names}")

        # 游戏循环
        while not done:
            current_player = self.game.get_current_player(state)
            agent = ordered_agents[current_player]

            # 获取合法动作
            legal_actions = self.game.get_legal_actions(state)
            if not legal_actions:
                if verbose:
                    print(f"  步骤 {num_steps}: 玩家 {current_player} 无合法动作，游戏结束")
                break

            # 编码状态
            observation = self.game.state_to_observation(state, current_player)
            legal_action_indices = [
                self.game.action_to_index(action, legal_actions) for action in legal_actions
            ]

            # Agent 选择动作
            action_idx, info = agent.select_action(observation, legal_action_indices)
            action = legal_actions[legal_action_indices.index(action_idx)]

            if verbose:
                print(
                    f"  步骤 {num_steps}: 玩家 {current_player} 执行动作 {action}"
                )

            # 执行动作
            state, rewards, done, step_info = self.game.step(action)
            num_steps += 1

            # 防止无限循环
            if num_steps > 10000:
                if verbose:
                    print(f"  游戏超过 10000 步，强制结束")
                break

        # 游戏结束
        duration = time.time() - start_time
        scores = [state.players[i].get_score() for i in range(len(state.players))]
        winner_idx = scores.index(max(scores))

        if verbose:
            print(f"\n游戏结束！")
            print(f"  总步数: {num_steps}")
            print(f"  耗时: {duration:.2f} 秒")
            print(f"  分数: {scores}")
            print(f"  获胜者: {ordered_names[winner_idx]}")

        # 创建 GameResult（使用原始名称）
        result = GameResult(
            game_id=game_id,
            players=ordered_names,
            scores=scores,
            winner=winner_idx,
            num_steps=num_steps,
            duration=duration,
        )

        # 更新统计
        self.total_games += 1
        self.total_steps += num_steps

        return result

    def run_games(
        self,
        num_games: int,
        shuffle_positions: bool = True,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[GameResult]:
        """
        运行多局游戏

        Args:
            num_games: 游戏局数
            shuffle_positions: 是否随机打乱玩家位置
            verbose: 是否显示详细信息
            progress_callback: 进度回调函数 (current, total)

        Returns:
            GameResult 列表
        """
        results = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"开始运行 {num_games} 局游戏")
            print(f"参赛 Agent: {self.agent_names}")
            print(f"{'='*60}")

        for i in range(num_games):
            # 确定玩家顺序
            if shuffle_positions:
                agent_order = np.random.permutation(len(self.agents)).tolist()
            else:
                agent_order = None

            # 进行游戏
            result = self.play_game(
                game_id=i,
                agent_order=agent_order,
                verbose=False,  # 单局不显示详细信息
            )

            results.append(result)

            # 添加到指标收集器
            self.metrics_collector.add_result(result)

            # 更新 ELO 分数
            if self.elo_system is not None:
                self.elo_system.update_ratings(
                    players=result.players,
                    scores=result.scores,
                    winner_idx=result.winner,
                )

            # 进度回调
            if progress_callback is not None:
                progress_callback(i + 1, num_games)

            # 显示进度
            if verbose and (i + 1) % max(1, num_games // 10) == 0:
                print(f"  进度: {i + 1}/{num_games} ({(i + 1) / num_games * 100:.1f}%)")

        if verbose:
            print(f"\n所有游戏完成！")
            self._print_summary()

        return results

    def run_tournament(
        self,
        num_games: int = 100,
        shuffle_positions: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        运行锦标赛（循环赛）

        Args:
            num_games: 总游戏局数
            shuffle_positions: 是否随机打乱玩家位置
            verbose: 是否显示详细信息

        Returns:
            锦标赛结果字典
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"锦标赛开始")
            print(f"{'='*60}")

        # 运行游戏
        results = self.run_games(
            num_games=num_games,
            shuffle_positions=shuffle_positions,
            verbose=verbose,
        )

        # 获取排行榜
        leaderboard = self.metrics_collector.get_leaderboard(sort_by="win_rate")

        # 构建结果字典
        tournament_results = {
            "num_games": num_games,
            "results": results,
            "leaderboard": leaderboard,
            "winner": leaderboard[0].name if leaderboard else None,
            "metrics": self.metrics_collector.get_summary(),
        }

        # 添加 ELO 排行榜
        if self.elo_system is not None:
            tournament_results["elo_leaderboard"] = self.elo_system.get_leaderboard()

        return tournament_results

    def get_head_to_head(self, agent1_name: str, agent2_name: str) -> Dict[str, int]:
        """
        获取两个 Agent 的对战统计

        Args:
            agent1_name: Agent1 名称
            agent2_name: Agent2 名称

        Returns:
            对战统计字典
        """
        return self.metrics_collector.get_head_to_head(agent1_name, agent2_name)

    def get_statistics(self) -> Dict:
        """
        获取整体统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "total_games": self.total_games,
            "total_steps": self.total_steps,
            "avg_steps_per_game": (
                self.total_steps / self.total_games if self.total_games > 0 else 0
            ),
            "metrics": self.metrics_collector.get_summary(),
        }

        if self.elo_system is not None:
            stats["elo_ratings"] = {
                name: self.elo_system.get_rating(name) for name in self.agent_names
            }

        return stats

    def reset_statistics(self) -> None:
        """重置所有统计信息"""
        self.metrics_collector.clear()
        if self.elo_system is not None:
            self.elo_system.reset()
        self.total_games = 0
        self.total_steps = 0

    def _print_summary(self) -> None:
        """打印统计摘要"""
        print(f"\n{'='*60}")
        print(f"统计摘要")
        print(f"{'='*60}")

        # 基本统计
        print(f"\n总游戏数: {self.total_games}")
        print(f"总步数: {self.total_steps}")
        print(
            f"平均步数: {self.total_steps / self.total_games if self.total_games > 0 else 0:.1f}"
        )

        # 玩家统计
        print(f"\n玩家统计:")
        leaderboard = self.metrics_collector.get_leaderboard(sort_by="win_rate")
        for i, stats in enumerate(leaderboard, start=1):
            print(
                f"  {i}. {stats.name}: "
                f"{stats.wins} 胜 / {stats.games_played} 局 "
                f"({stats.win_rate:.2%}), "
                f"平均分数: {stats.avg_score:.2f}, "
                f"平均排名: {stats.avg_rank:.2f}"
            )

        # ELO 分数
        if self.elo_system is not None:
            print(f"\nELO 评分:")
            elo_leaderboard = self.elo_system.get_leaderboard()
            for i, rating in enumerate(elo_leaderboard, start=1):
                print(
                    f"  {i}. {rating.player_name}: "
                    f"ELO {rating.rating:.0f}, "
                    f"{rating.wins} 胜 / {rating.games_played} 局"
                )

        print(f"{'='*60}\n")


# ===== 导出 =====
__all__ = ["Arena", "MatchConfig"]
