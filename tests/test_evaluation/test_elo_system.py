"""
测试 ELO 评分系统
"""

import pytest
import tempfile
import json
from pathlib import Path

from evaluation.elo_system import EloRating, EloSystem


class TestEloRating:
    """测试 EloRating"""

    def test_create_elo_rating(self):
        """测试创建 EloRating"""
        rating = EloRating(player_name="Player1", rating=1600.0, games_played=10, wins=5)

        assert rating.player_name == "Player1"
        assert rating.rating == 1600.0
        assert rating.games_played == 10
        assert rating.wins == 5

    def test_default_values(self):
        """测试默认值"""
        rating = EloRating(player_name="Player1")

        assert rating.rating == 1500.0
        assert rating.games_played == 0
        assert rating.wins == 0
        assert rating.losses == 0
        assert rating.draws == 0

    def test_win_rate(self):
        """测试胜率"""
        rating = EloRating(player_name="Player1", games_played=10, wins=6)
        assert rating.win_rate == 0.6

        rating = EloRating(player_name="Player2", games_played=0, wins=0)
        assert rating.win_rate == 0.0

    def test_to_dict(self):
        """测试转换为字典"""
        rating = EloRating(
            player_name="Player1",
            rating=1600.0,
            games_played=10,
            wins=6,
            losses=3,
            draws=1,
        )

        result = rating.to_dict()

        assert result["player_name"] == "Player1"
        assert result["rating"] == 1600.0
        assert result["games_played"] == 10
        assert result["wins"] == 6
        assert result["losses"] == 3
        assert result["draws"] == 1
        assert result["win_rate"] == 0.6


class TestEloSystem:
    """测试 EloSystem"""

    def test_create_elo_system(self):
        """测试创建 EloSystem"""
        elo = EloSystem(k_factor=32.0, initial_rating=1500.0)

        assert elo.k_factor == 32.0
        assert elo.initial_rating == 1500.0
        assert len(elo.ratings) == 0

    def test_get_rating_new_player(self):
        """测试获取新玩家的分数"""
        elo = EloSystem(initial_rating=1500.0)

        rating = elo.get_rating("Player1")

        assert rating == 1500.0
        assert "Player1" in elo.ratings

    def test_expected_score(self):
        """测试期望得分计算"""
        elo = EloSystem()

        # 同等分数，期望得分应该是 0.5
        expected = elo.expected_score(1500.0, 1500.0)
        assert expected == pytest.approx(0.5)

        # 高分玩家期望得分应该更高
        expected = elo.expected_score(1600.0, 1400.0)
        assert expected > 0.5

        # 低分玩家期望得分应该更低
        expected = elo.expected_score(1400.0, 1600.0)
        assert expected < 0.5

    def test_update_two_players_win(self):
        """测试两人对战更新（获胜）"""
        elo = EloSystem(k_factor=32.0)

        # Player1 beats Player2
        new_rating1, new_rating2 = elo.update_two_players("Player1", "Player2", 1.0)

        # Player1 应该增加分数
        assert new_rating1 > 1500.0
        # Player2 应该减少分数
        assert new_rating2 < 1500.0

        # 检查统计
        rating1 = elo.get_elo_rating("Player1")
        assert rating1.wins == 1
        assert rating1.losses == 0

        rating2 = elo.get_elo_rating("Player2")
        assert rating2.wins == 0
        assert rating2.losses == 1

    def test_update_two_players_draw(self):
        """测试两人对战更新（平局）"""
        elo = EloSystem(k_factor=32.0)

        # Draw between equal players
        new_rating1, new_rating2 = elo.update_two_players("Player1", "Player2", 0.5)

        # 同等分数平局，分数不应改变
        assert new_rating1 == pytest.approx(1500.0)
        assert new_rating2 == pytest.approx(1500.0)

        # 检查统计
        rating1 = elo.get_elo_rating("Player1")
        assert rating1.draws == 1

        rating2 = elo.get_elo_rating("Player2")
        assert rating2.draws == 1

    def test_update_ratings_multiplayer(self):
        """测试多人游戏 ELO 更新"""
        elo = EloSystem(k_factor=32.0)

        players = ["Player1", "Player2", "Player3", "Player4"]
        scores = [15, 12, 10, 8]
        winner_idx = 0

        new_ratings = elo.update_ratings(players, scores, winner_idx)

        # Player1 应该增加分数（获胜）
        assert new_ratings["Player1"] > 1500.0

        # Player4 应该减少分数（最后一名）
        assert new_ratings["Player4"] < 1500.0

        # 检查所有玩家的游戏数
        for player in players:
            rating = elo.get_elo_rating(player)
            assert rating.games_played == 1

        # 检查获胜者统计
        rating1 = elo.get_elo_rating("Player1")
        assert rating1.wins == 1

    def test_update_ratings_invalid_input(self):
        """测试无效输入"""
        elo = EloSystem()

        # 玩家数量不足
        with pytest.raises(ValueError):
            elo.update_ratings(["Player1"], [15], 0)

        # 分数列表长度不匹配
        with pytest.raises(ValueError):
            elo.update_ratings(["Player1", "Player2"], [15], 0)

    def test_get_leaderboard(self):
        """测试获取排行榜"""
        elo = EloSystem(k_factor=32.0)

        # Player1 获胜多次
        for _ in range(10):
            elo.update_two_players("Player1", "Player2", 1.0)

        leaderboard = elo.get_leaderboard()

        assert len(leaderboard) == 2
        assert leaderboard[0].player_name == "Player1"
        assert leaderboard[0].rating > leaderboard[1].rating

    def test_save_and_load(self):
        """测试保存和加载"""
        elo = EloSystem(k_factor=32.0, initial_rating=1500.0)

        # 更新一些分数
        elo.update_two_players("Player1", "Player2", 1.0)
        elo.update_two_players("Player1", "Player3", 1.0)

        # 保存到临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "elo_ratings.json"

            # 保存
            elo.save(str(save_path))

            # 创建新的 EloSystem 并加载
            elo2 = EloSystem()
            elo2.load(str(save_path))

            # 验证数据一致
            assert elo2.k_factor == elo.k_factor
            assert elo2.initial_rating == elo.initial_rating
            assert len(elo2.ratings) == len(elo.ratings)

            # 验证玩家数据
            for player_name in ["Player1", "Player2", "Player3"]:
                rating1 = elo.get_elo_rating(player_name)
                rating2 = elo2.get_elo_rating(player_name)

                assert rating2.rating == rating1.rating
                assert rating2.games_played == rating1.games_played
                assert rating2.wins == rating1.wins

    def test_reset(self):
        """测试重置"""
        elo = EloSystem()

        # 添加一些数据
        elo.update_two_players("Player1", "Player2", 1.0)

        assert len(elo.ratings) == 2

        # 重置
        elo.reset()

        assert len(elo.ratings) == 0

    def test_elo_rating_convergence(self):
        """测试 ELO 分数收敛"""
        elo = EloSystem(k_factor=32.0)

        # 强玩家一直击败弱玩家
        for _ in range(100):
            elo.update_two_players("Strong", "Weak", 1.0)

        strong_rating = elo.get_rating("Strong")
        weak_rating = elo.get_rating("Weak")

        # 强玩家分数应该远高于弱玩家
        assert strong_rating > 1700
        assert weak_rating < 1300

    def test_multiplayer_elo_consistency(self):
        """测试多人 ELO 更新的一致性"""
        elo = EloSystem(k_factor=32.0)

        # 多次相同结果应该产生稳定的分数
        players = ["A", "B", "C", "D"]

        for _ in range(50):
            elo.update_ratings(players, [15, 12, 10, 8], winner_idx=0)

        # A 应该有最高分数
        ratings = {player: elo.get_rating(player) for player in players}

        assert ratings["A"] > ratings["B"]
        assert ratings["B"] > ratings["C"]
        assert ratings["C"] > ratings["D"]

    def test_elo_symmetry(self):
        """测试 ELO 对称性"""
        elo1 = EloSystem(k_factor=32.0)
        elo2 = EloSystem(k_factor=32.0)

        # Player1 beats Player2
        new_rating1_a, new_rating2_a = elo1.update_two_players("Player1", "Player2", 1.0)

        # Player2 loses to Player1
        new_rating2_b, new_rating1_b = elo2.update_two_players("Player2", "Player1", 0.0)

        # 分数应该相同
        assert new_rating1_a == pytest.approx(new_rating1_b)
        assert new_rating2_a == pytest.approx(new_rating2_b)
