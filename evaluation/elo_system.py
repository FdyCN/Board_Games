"""
ELO 评分系统

实现标准的 ELO 评分算法，用于评估 Agent 的相对实力。
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class EloRating:
    """
    ELO 评分

    Attributes:
        player_name: 玩家名称
        rating: ELO 分数
        games_played: 参与游戏数
        wins: 获胜次数
        losses: 失败次数
        draws: 平局次数
    """

    player_name: str
    rating: float = 1500.0  # 初始 ELO 分数
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        """胜率"""
        return self.wins / self.games_played if self.games_played > 0 else 0.0

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "player_name": self.player_name,
            "rating": self.rating,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
        }


class EloSystem:
    """
    ELO 评分系统

    支持多人游戏的 ELO 评分计算和持久化。

    Attributes:
        k_factor: K 因子（控制分数变化幅度）
        initial_rating: 初始 ELO 分数
        ratings: 玩家评分字典

    Examples:
        >>> elo = EloSystem(k_factor=32)
        >>> elo.update_ratings(
        ...     players=["Agent1", "Agent2", "Agent3", "Agent4"],
        ...     scores=[15, 12, 10, 8],
        ...     winner_idx=0,
        ... )
        >>> print(f"Agent1 rating: {elo.get_rating('Agent1'):.0f}")
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_rating: float = 1500.0,
    ):
        """
        初始化 ELO 系统

        Args:
            k_factor: K 因子（通常 10-40）
            initial_rating: 初始 ELO 分数
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, EloRating] = {}

    def get_rating(self, player_name: str) -> float:
        """获取玩家 ELO 分数"""
        if player_name not in self.ratings:
            self.ratings[player_name] = EloRating(
                player_name=player_name, rating=self.initial_rating
            )
        return self.ratings[player_name].rating

    def get_elo_rating(self, player_name: str) -> EloRating:
        """获取玩家 ELO 评分对象"""
        if player_name not in self.ratings:
            self.ratings[player_name] = EloRating(
                player_name=player_name, rating=self.initial_rating
            )
        return self.ratings[player_name]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        计算期望得分

        Args:
            rating_a: 玩家 A 的 ELO 分数
            rating_b: 玩家 B 的 ELO 分数

        Returns:
            玩家 A 的期望得分（0-1 之间）
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def update_two_players(
        self,
        player1: str,
        player2: str,
        result: float,
    ) -> tuple[float, float]:
        """
        更新两个玩家的 ELO 分数

        Args:
            player1: 玩家1名称
            player2: 玩家2名称
            result: 玩家1的实际得分（1.0=赢, 0.5=平, 0.0=输）

        Returns:
            (玩家1新分数, 玩家2新分数)
        """
        # 获取当前分数
        rating1 = self.get_rating(player1)
        rating2 = self.get_rating(player2)

        # 计算期望得分
        expected1 = self.expected_score(rating1, rating2)
        expected2 = 1.0 - expected1

        # 计算新分数
        new_rating1 = rating1 + self.k_factor * (result - expected1)
        new_rating2 = rating2 + self.k_factor * (1.0 - result - expected2)

        # 更新分数
        elo1 = self.get_elo_rating(player1)
        elo2 = self.get_elo_rating(player2)

        elo1.rating = new_rating1
        elo2.rating = new_rating2

        elo1.games_played += 1
        elo2.games_played += 1

        if result == 1.0:
            elo1.wins += 1
            elo2.losses += 1
        elif result == 0.0:
            elo1.losses += 1
            elo2.wins += 1
        else:
            elo1.draws += 1
            elo2.draws += 1

        return new_rating1, new_rating2

    def update_ratings(
        self,
        players: List[str],
        scores: List[float],
        winner_idx: int,
    ) -> Dict[str, float]:
        """
        更新多人游戏的 ELO 分数

        使用所有玩家两两对战的方式计算 ELO 变化。

        Args:
            players: 玩家名称列表
            scores: 玩家分数列表
            winner_idx: 获胜者索引

        Returns:
            更新后的 ELO 分数字典
        """
        n = len(players)
        if n < 2:
            raise ValueError("至少需要 2 个玩家")

        if len(scores) != n:
            raise ValueError("分数列表长度必须与玩家数量一致")

        # 计算每个玩家的排名（分数相同则排名相同）
        sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
        ranks = [0] * n
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank

        # 两两更新 ELO 分数
        delta_ratings = {player: 0.0 for player in players}

        for i in range(n):
            for j in range(i + 1, n):
                player_i = players[i]
                player_j = players[j]

                # 确定对战结果
                if ranks[i] < ranks[j]:  # i 排名更高（分数更大）
                    result_i = 1.0
                elif ranks[i] > ranks[j]:
                    result_i = 0.0
                else:  # 平局
                    result_i = 0.5

                # 获取当前分数
                rating_i = self.get_rating(player_i)
                rating_j = self.get_rating(player_j)

                # 计算期望得分
                expected_i = self.expected_score(rating_i, rating_j)

                # 计算分数变化（使用较小的 K 因子避免变化过大）
                k = self.k_factor / (n - 1)  # 分散到多次对战
                delta_i = k * (result_i - expected_i)
                delta_j = -delta_i

                delta_ratings[player_i] += delta_i
                delta_ratings[player_j] += delta_j

        # 应用分数变化
        new_ratings = {}
        for player in players:
            elo = self.get_elo_rating(player)
            elo.rating += delta_ratings[player]
            elo.games_played += 1

            idx = players.index(player)
            if idx == winner_idx:
                elo.wins += 1
            else:
                elo.losses += 1

            new_ratings[player] = elo.rating

        return new_ratings

    def get_leaderboard(self) -> List[EloRating]:
        """
        获取排行榜

        Returns:
            按 ELO 分数排序的玩家列表
        """
        return sorted(
            self.ratings.values(),
            key=lambda r: r.rating,
            reverse=True,
        )

    def save(self, file_path: str) -> None:
        """
        保存 ELO 分数到文件

        Args:
            file_path: 保存路径（JSON 文件）
        """
        data = {
            "k_factor": self.k_factor,
            "initial_rating": self.initial_rating,
            "ratings": {
                name: rating.to_dict() for name, rating in self.ratings.items()
            },
        }

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, file_path: str) -> None:
        """
        从文件加载 ELO 分数

        Args:
            file_path: 加载路径（JSON 文件）
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.k_factor = data["k_factor"]
        self.initial_rating = data["initial_rating"]

        self.ratings = {}
        for name, rating_dict in data["ratings"].items():
            self.ratings[name] = EloRating(
                player_name=rating_dict["player_name"],
                rating=rating_dict["rating"],
                games_played=rating_dict["games_played"],
                wins=rating_dict["wins"],
                losses=rating_dict["losses"],
                draws=rating_dict["draws"],
            )

    def reset(self) -> None:
        """重置所有 ELO 分数"""
        self.ratings.clear()


# ===== 导出 =====
__all__ = ["EloRating", "EloSystem"]
