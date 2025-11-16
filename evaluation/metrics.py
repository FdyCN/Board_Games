"""
评估指标模块

提供各种评估指标的计算，包括胜率、平均奖励、排名等。
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class GameResult:
    """
    单局游戏结果

    Attributes:
        game_id: 游戏 ID
        players: 玩家名称列表
        scores: 玩家分数列表
        winner: 获胜者索引
        num_steps: 游戏步数
        duration: 游戏时长（秒）
        metadata: 额外信息
    """

    game_id: int
    players: List[str]
    scores: List[float]
    winner: int
    num_steps: int
    duration: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def get_ranks(self) -> List[int]:
        """
        获取玩家排名（1 为最好）

        Returns:
            排名列表
        """
        # 分数越高排名越高
        sorted_indices = np.argsort(self.scores)[::-1]
        ranks = [0] * len(self.scores)
        for rank, idx in enumerate(sorted_indices, start=1):
            ranks[idx] = rank
        return ranks


@dataclass
class PlayerStats:
    """
    玩家统计信息

    Attributes:
        name: 玩家名称
        games_played: 参与游戏数
        wins: 获胜次数
        total_score: 总分数
        total_steps: 总步数
        ranks: 排名列表
    """

    name: str
    games_played: int = 0
    wins: int = 0
    total_score: float = 0.0
    total_steps: int = 0
    ranks: List[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """胜率"""
        return self.wins / self.games_played if self.games_played > 0 else 0.0

    @property
    def avg_score(self) -> float:
        """平均分数"""
        return self.total_score / self.games_played if self.games_played > 0 else 0.0

    @property
    def avg_rank(self) -> float:
        """平均排名（1.0 是最好）"""
        return np.mean(self.ranks) if self.ranks else 0.0

    @property
    def avg_steps(self) -> float:
        """平均步数"""
        return self.total_steps / self.games_played if self.games_played > 0 else 0.0

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "name": self.name,
            "games_played": self.games_played,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "total_score": self.total_score,
            "avg_score": self.avg_score,
            "avg_rank": self.avg_rank,
            "avg_steps": self.avg_steps,
        }


class MetricsCollector:
    """
    指标收集器

    收集和计算多局游戏的统计指标。

    Examples:
        >>> collector = MetricsCollector()
        >>> result = GameResult(
        ...     game_id=1,
        ...     players=["Agent1", "Agent2", "Agent3", "Agent4"],
        ...     scores=[15, 12, 10, 8],
        ...     winner=0,
        ...     num_steps=100,
        ... )
        >>> collector.add_result(result)
        >>> stats = collector.get_player_stats("Agent1")
        >>> print(f"Win rate: {stats.win_rate:.2%}")
    """

    def __init__(self):
        self.results: List[GameResult] = []
        self.player_stats: Dict[str, PlayerStats] = {}

    def add_result(self, result: GameResult) -> None:
        """
        添加游戏结果

        Args:
            result: GameResult 对象
        """
        self.results.append(result)

        # 更新玩家统计
        ranks = result.get_ranks()
        for i, player_name in enumerate(result.players):
            if player_name not in self.player_stats:
                self.player_stats[player_name] = PlayerStats(name=player_name)

            stats = self.player_stats[player_name]
            stats.games_played += 1
            stats.total_score += result.scores[i]
            stats.total_steps += result.num_steps
            stats.ranks.append(ranks[i])

            if i == result.winner:
                stats.wins += 1

    def add_results(self, results: List[GameResult]) -> None:
        """批量添加游戏结果"""
        for result in results:
            self.add_result(result)

    def get_player_stats(self, player_name: str) -> Optional[PlayerStats]:
        """获取玩家统计"""
        return self.player_stats.get(player_name)

    def get_all_stats(self) -> Dict[str, PlayerStats]:
        """获取所有玩家统计"""
        return self.player_stats

    def get_leaderboard(self, sort_by: str = "win_rate") -> List[PlayerStats]:
        """
        获取排行榜

        Args:
            sort_by: 排序依据 ("win_rate", "avg_score", "avg_rank", "wins")

        Returns:
            排序后的玩家统计列表
        """
        stats_list = list(self.player_stats.values())

        if sort_by == "win_rate":
            stats_list.sort(key=lambda s: s.win_rate, reverse=True)
        elif sort_by == "avg_score":
            stats_list.sort(key=lambda s: s.avg_score, reverse=True)
        elif sort_by == "avg_rank":
            stats_list.sort(key=lambda s: s.avg_rank)  # 排名越小越好
        elif sort_by == "wins":
            stats_list.sort(key=lambda s: s.wins, reverse=True)
        else:
            raise ValueError(f"未知的排序依据: {sort_by}")

        return stats_list

    def get_head_to_head(
        self, player1: str, player2: str
    ) -> Dict[str, int]:
        """
        获取两个玩家的对战统计

        Args:
            player1: 玩家1名称
            player2: 玩家2名称

        Returns:
            对战统计字典
        """
        stats = {
            "player1_wins": 0,
            "player2_wins": 0,
            "draws": 0,
            "total_games": 0,
        }

        for result in self.results:
            if player1 in result.players and player2 in result.players:
                idx1 = result.players.index(player1)
                idx2 = result.players.index(player2)

                stats["total_games"] += 1

                if result.winner == idx1:
                    stats["player1_wins"] += 1
                elif result.winner == idx2:
                    stats["player2_wins"] += 1
                else:
                    # 平局（如果分数相同）
                    if result.scores[idx1] == result.scores[idx2]:
                        stats["draws"] += 1

        return stats

    def get_summary(self) -> Dict:
        """
        获取整体统计摘要

        Returns:
            统计摘要字典
        """
        if not self.results:
            return {
                "total_games": 0,
                "total_players": 0,
                "avg_game_length": 0.0,
                "avg_game_duration": 0.0,
            }

        total_steps = sum(r.num_steps for r in self.results)
        total_duration = sum(r.duration for r in self.results)

        return {
            "total_games": len(self.results),
            "total_players": len(self.player_stats),
            "avg_game_length": total_steps / len(self.results),
            "avg_game_duration": total_duration / len(self.results),
            "player_stats": {
                name: stats.to_dict() for name, stats in self.player_stats.items()
            },
        }

    def clear(self) -> None:
        """清空所有统计"""
        self.results.clear()
        self.player_stats.clear()


def compute_win_matrix(results: List[GameResult]) -> np.ndarray:
    """
    计算胜率矩阵

    Args:
        results: 游戏结果列表

    Returns:
        胜率矩阵 (n_players, n_players)
        matrix[i][j] 表示玩家 i 对玩家 j 的胜率
    """
    # 收集所有玩家
    all_players = set()
    for result in results:
        all_players.update(result.players)

    players = sorted(list(all_players))
    n = len(players)
    player_to_idx = {name: i for i, name in enumerate(players)}

    # 初始化计数矩阵
    win_counts = np.zeros((n, n))
    total_games = np.zeros((n, n))

    # 统计对战结果
    for result in results:
        winner_name = result.players[result.winner]
        winner_idx = player_to_idx[winner_name]

        for loser_name in result.players:
            if loser_name != winner_name:
                loser_idx = player_to_idx[loser_name]
                win_counts[winner_idx][loser_idx] += 1
                total_games[winner_idx][loser_idx] += 1
                total_games[loser_idx][winner_idx] += 1

    # 计算胜率
    win_matrix = np.divide(
        win_counts,
        total_games,
        out=np.zeros_like(win_counts),
        where=total_games != 0,
    )

    return win_matrix


# ===== 导出 =====
__all__ = [
    "GameResult",
    "PlayerStats",
    "MetricsCollector",
    "compute_win_matrix",
]
