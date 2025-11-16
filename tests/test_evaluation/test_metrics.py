"""
测试统计指标模块
"""

import pytest
import numpy as np

from evaluation.metrics import (
    GameResult,
    PlayerStats,
    MetricsCollector,
    compute_win_matrix,
)


class TestGameResult:
    """测试 GameResult"""

    def test_create_game_result(self):
        """测试创建 GameResult"""
        result = GameResult(
            game_id=1,
            players=["A", "B", "C", "D"],
            scores=[15, 12, 10, 8],
            winner=0,
            num_steps=100,
            duration=60.5,
        )

        assert result.game_id == 1
        assert result.players == ["A", "B", "C", "D"]
        assert result.scores == [15, 12, 10, 8]
        assert result.winner == 0
        assert result.num_steps == 100
        assert result.duration == 60.5

    def test_get_ranks(self):
        """测试获取排名"""
        result = GameResult(
            game_id=1,
            players=["A", "B", "C", "D"],
            scores=[15, 12, 10, 8],
            winner=0,
            num_steps=100,
        )

        ranks = result.get_ranks()

        assert ranks == [1, 2, 3, 4]  # A=1st, B=2nd, C=3rd, D=4th

    def test_get_ranks_with_ties(self):
        """测试平局排名"""
        result = GameResult(
            game_id=1,
            players=["A", "B", "C", "D"],
            scores=[15, 12, 12, 8],
            winner=0,
            num_steps=100,
        )

        ranks = result.get_ranks()

        # A=1st, B and C tied for 2nd (one gets 2, one gets 3), D=4th
        assert ranks[0] == 1  # A is first
        assert ranks[1] in [2, 3]  # B is 2nd or 3rd
        assert ranks[2] in [2, 3]  # C is 2nd or 3rd
        assert ranks[3] == 4  # D is fourth


class TestPlayerStats:
    """测试 PlayerStats"""

    def test_create_player_stats(self):
        """测试创建 PlayerStats"""
        stats = PlayerStats(
            name="Player1",
            games_played=10,
            wins=5,
            total_score=150.0,
            total_steps=1000,
            ranks=[1, 2, 1, 3, 2, 1, 4, 2, 1, 3],
        )

        assert stats.name == "Player1"
        assert stats.games_played == 10
        assert stats.wins == 5

    def test_win_rate(self):
        """测试胜率计算"""
        stats = PlayerStats(name="Player1", games_played=10, wins=5)
        assert stats.win_rate == 0.5

        stats = PlayerStats(name="Player2", games_played=0, wins=0)
        assert stats.win_rate == 0.0

    def test_avg_score(self):
        """测试平均分数"""
        stats = PlayerStats(name="Player1", games_played=10, total_score=150.0)
        assert stats.avg_score == 15.0

        stats = PlayerStats(name="Player2", games_played=0, total_score=0.0)
        assert stats.avg_score == 0.0

    def test_avg_rank(self):
        """测试平均排名"""
        stats = PlayerStats(name="Player1", ranks=[1, 2, 1, 3, 2])
        assert stats.avg_rank == pytest.approx(1.8)

        stats = PlayerStats(name="Player2", ranks=[])
        assert stats.avg_rank == 0.0

    def test_to_dict(self):
        """测试转换为字典"""
        stats = PlayerStats(
            name="Player1",
            games_played=10,
            wins=5,
            total_score=150.0,
            total_steps=1000,
            ranks=[1, 2, 3],
        )

        result = stats.to_dict()

        assert result["name"] == "Player1"
        assert result["games_played"] == 10
        assert result["wins"] == 5
        assert result["win_rate"] == 0.5
        assert result["avg_score"] == 15.0


class TestMetricsCollector:
    """测试 MetricsCollector"""

    def test_create_collector(self):
        """测试创建收集器"""
        collector = MetricsCollector()

        assert len(collector.results) == 0
        assert len(collector.player_stats) == 0

    def test_add_result(self):
        """测试添加结果"""
        collector = MetricsCollector()

        result = GameResult(
            game_id=1,
            players=["A", "B", "C", "D"],
            scores=[15, 12, 10, 8],
            winner=0,
            num_steps=100,
        )

        collector.add_result(result)

        assert len(collector.results) == 1
        assert len(collector.player_stats) == 4

        # 检查玩家 A 的统计
        stats_a = collector.get_player_stats("A")
        assert stats_a.games_played == 1
        assert stats_a.wins == 1
        assert stats_a.total_score == 15

    def test_add_multiple_results(self):
        """测试添加多个结果"""
        collector = MetricsCollector()

        # 第一局
        result1 = GameResult(
            game_id=1,
            players=["A", "B", "C", "D"],
            scores=[15, 12, 10, 8],
            winner=0,
            num_steps=100,
        )
        collector.add_result(result1)

        # 第二局
        result2 = GameResult(
            game_id=2,
            players=["A", "B", "C", "D"],
            scores=[10, 15, 12, 8],
            winner=1,
            num_steps=120,
        )
        collector.add_result(result2)

        # 检查统计
        stats_a = collector.get_player_stats("A")
        assert stats_a.games_played == 2
        assert stats_a.wins == 1
        assert stats_a.total_score == 25

        stats_b = collector.get_player_stats("B")
        assert stats_b.games_played == 2
        assert stats_b.wins == 1
        assert stats_b.total_score == 27

    def test_get_leaderboard(self):
        """测试获取排行榜"""
        collector = MetricsCollector()

        # 添加多局结果
        for i in range(10):
            result = GameResult(
                game_id=i,
                players=["A", "B", "C", "D"],
                scores=[15, 12, 10, 8],
                winner=0,  # A always wins
                num_steps=100,
            )
            collector.add_result(result)

        # 按胜率排序
        leaderboard = collector.get_leaderboard(sort_by="win_rate")

        assert leaderboard[0].name == "A"
        assert leaderboard[0].win_rate == 1.0
        assert leaderboard[1].win_rate == 0.0

    def test_get_leaderboard_by_avg_score(self):
        """测试按平均分数排序"""
        collector = MetricsCollector()

        result = GameResult(
            game_id=1,
            players=["A", "B", "C", "D"],
            scores=[15, 12, 10, 8],
            winner=0,
            num_steps=100,
        )
        collector.add_result(result)

        leaderboard = collector.get_leaderboard(sort_by="avg_score")

        assert leaderboard[0].name == "A"
        assert leaderboard[0].avg_score == 15
        assert leaderboard[3].name == "D"
        assert leaderboard[3].avg_score == 8

    def test_get_head_to_head(self):
        """测试对战统计"""
        collector = MetricsCollector()

        # 添加一些结果
        for i in range(10):
            result = GameResult(
                game_id=i,
                players=["A", "B", "C", "D"],
                scores=[15, 12, 10, 8] if i < 7 else [12, 15, 10, 8],
                winner=0 if i < 7 else 1,
                num_steps=100,
            )
            collector.add_result(result)

        # A vs B
        h2h = collector.get_head_to_head("A", "B")

        assert h2h["total_games"] == 10
        assert h2h["player1_wins"] == 7  # A wins 7
        assert h2h["player2_wins"] == 3  # B wins 3

    def test_get_summary(self):
        """测试获取摘要"""
        collector = MetricsCollector()

        result = GameResult(
            game_id=1,
            players=["A", "B", "C", "D"],
            scores=[15, 12, 10, 8],
            winner=0,
            num_steps=100,
            duration=60.0,
        )
        collector.add_result(result)

        summary = collector.get_summary()

        assert summary["total_games"] == 1
        assert summary["total_players"] == 4
        assert summary["avg_game_length"] == 100
        assert summary["avg_game_duration"] == 60.0

    def test_clear(self):
        """测试清空统计"""
        collector = MetricsCollector()

        result = GameResult(
            game_id=1,
            players=["A", "B"],
            scores=[15, 12],
            winner=0,
            num_steps=100,
        )
        collector.add_result(result)

        assert len(collector.results) == 1

        collector.clear()

        assert len(collector.results) == 0
        assert len(collector.player_stats) == 0


class TestComputeWinMatrix:
    """测试 compute_win_matrix"""

    def test_compute_win_matrix_simple(self):
        """测试简单的胜率矩阵"""
        results = [
            GameResult(1, ["A", "B"], [15, 12], 0, 100),
            GameResult(2, ["A", "B"], [15, 12], 0, 100),
            GameResult(3, ["A", "B"], [10, 15], 1, 100),
        ]

        matrix = compute_win_matrix(results)

        # A vs B: A wins 2 out of 3
        # matrix[A_idx][B_idx] = 2/3 (A beats B in 2 out of 3 games)
        # matrix[B_idx][A_idx] = 1/3 (B beats A in 1 out of 3 games)

        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 0.0  # A vs A
        assert matrix[0, 1] == pytest.approx(2.0 / 3.0)  # A vs B
        assert matrix[1, 0] == pytest.approx(1.0 / 3.0)  # B vs A (B wins 1 out of 3)
        assert matrix[1, 1] == 0.0  # B vs B

    def test_compute_win_matrix_multiplayer(self):
        """测试多人游戏胜率矩阵"""
        results = [
            GameResult(1, ["A", "B", "C"], [15, 12, 10], 0, 100),
            GameResult(2, ["A", "B", "C"], [12, 15, 10], 1, 100),
            GameResult(3, ["A", "B", "C"], [10, 12, 15], 2, 100),
        ]

        matrix = compute_win_matrix(results)

        assert matrix.shape == (3, 3)

        # A wins game 1, so A beats B and C
        # B wins game 2, so B beats A and C
        # C wins game 3, so C beats A and B

        # Each pair plays 2 games (as participants), each wins 1
        # So the win rate should be 0.5 for all pairs

        assert matrix[0, 1] == pytest.approx(0.5)  # A vs B (A wins 1 out of 2)
        assert matrix[1, 0] == pytest.approx(0.5)  # B vs A (B wins 1 out of 2)
        assert matrix[0, 2] == pytest.approx(0.5)  # A vs C (A wins 1 out of 2)
        assert matrix[2, 0] == pytest.approx(0.5)  # C vs A (C wins 1 out of 2)
        assert matrix[1, 2] == pytest.approx(0.5)  # B vs C (B wins 1 out of 2)
        assert matrix[2, 1] == pytest.approx(0.5)  # C vs B (C wins 1 out of 2)
