"""
测试 Arena 对战系统
"""

import pytest

# 导入 Splendor 以注册游戏
import games.splendor  # noqa: F401

from games.registry import create_game
from agents.random_agent import RandomAgent
from evaluation.arena import Arena, MatchConfig
from evaluation.metrics import GameResult


class TestMatchConfig:
    """测试 MatchConfig"""

    def test_create_match_config(self):
        """测试创建配置"""
        config = MatchConfig(
            num_games=100,
            shuffle_positions=True,
            verbose=False,
        )

        assert config.num_games == 100
        assert config.shuffle_positions is True
        assert config.verbose is False


class TestArena:
    """测试 Arena"""

    @pytest.fixture
    def game(self):
        """创建测试游戏"""
        return create_game("splendor", num_players=4)

    @pytest.fixture
    def agents(self):
        """创建测试 Agent"""
        return [RandomAgent(player_id=i) for i in range(4)]

    @pytest.fixture
    def agent_names(self):
        """Agent 名称"""
        return ["Agent1", "Agent2", "Agent3", "Agent4"]

    @pytest.fixture
    def arena(self, game, agents, agent_names):
        """创建 Arena"""
        return Arena(game, agents, agent_names, use_elo=True)

    def test_create_arena(self, game, agents, agent_names):
        """测试创建 Arena"""
        arena = Arena(game, agents, agent_names, use_elo=True)

        assert arena.game == game
        assert arena.agents == agents
        assert arena.agent_names == agent_names
        assert arena.elo_system is not None
        assert arena.total_games == 0

    def test_create_arena_without_elo(self, game, agents, agent_names):
        """测试创建不使用 ELO 的 Arena"""
        arena = Arena(game, agents, agent_names, use_elo=False)

        assert arena.elo_system is None

    def test_create_arena_auto_names(self, game, agents):
        """测试自动生成 Agent 名称"""
        arena = Arena(game, agents)

        assert arena.agent_names == ["Agent0", "Agent1", "Agent2", "Agent3"]

    def test_create_arena_invalid_agent_count(self, game):
        """测试 Agent 数量不匹配"""
        agents = [RandomAgent(player_id=i) for i in range(2)]

        with pytest.raises(ValueError, match="Agent 数量.*必须等于游戏玩家数"):
            Arena(game, agents)

    def test_create_arena_invalid_names_count(self, game, agents):
        """测试名称数量不匹配"""
        agent_names = ["Agent1", "Agent2"]  # 只有 2 个名称，但有 4 个 Agent

        with pytest.raises(ValueError, match="agent_names 长度必须与 agents 长度一致"):
            Arena(game, agents, agent_names)

    def test_play_game(self, arena):
        """测试进行一局游戏"""
        result = arena.play_game(game_id=1, verbose=False)

        assert isinstance(result, GameResult)
        assert result.game_id == 1
        assert len(result.players) == 4
        assert len(result.scores) == 4
        assert 0 <= result.winner < 4
        assert result.num_steps > 0

        # 检查统计
        assert arena.total_games == 1
        assert arena.total_steps > 0

    def test_play_game_with_agent_order(self, arena):
        """测试指定 Agent 顺序"""
        agent_order = [3, 2, 1, 0]  # 反向顺序
        result = arena.play_game(game_id=1, agent_order=agent_order, verbose=False)

        # 玩家名称应该按照指定顺序
        assert result.players == ["Agent4", "Agent3", "Agent2", "Agent1"]

    def test_run_games(self, arena):
        """测试运行多局游戏"""
        num_games = 10
        results = arena.run_games(num_games=num_games, verbose=False)

        assert len(results) == num_games
        assert arena.total_games == num_games

        # 检查所有结果都被添加到 MetricsCollector
        assert len(arena.metrics_collector.results) == num_games

        # 检查 ELO 系统有数据
        if arena.elo_system:
            for name in arena.agent_names:
                rating = arena.elo_system.get_rating(name)
                # 初始分数是 1500，经过对战应该有变化（除非所有游戏都是平局）
                # 但我们只检查玩家已注册
                assert rating >= 0

    def test_run_games_no_shuffle(self, arena):
        """测试不打乱位置"""
        results = arena.run_games(num_games=5, shuffle_positions=False, verbose=False)

        # 所有游戏应该使用相同的玩家顺序
        first_players = results[0].players
        for result in results:
            assert result.players == first_players

    def test_run_tournament(self, arena):
        """测试运行锦标赛"""
        tournament_results = arena.run_tournament(num_games=20, verbose=False)

        assert tournament_results["num_games"] == 20
        assert len(tournament_results["results"]) == 20
        assert "leaderboard" in tournament_results
        assert "winner" in tournament_results
        assert "metrics" in tournament_results

        # 检查获胜者
        assert tournament_results["winner"] in arena.agent_names

        # 检查 ELO 排行榜
        if arena.elo_system:
            assert "elo_leaderboard" in tournament_results
            assert len(tournament_results["elo_leaderboard"]) == len(arena.agent_names)

    def test_get_head_to_head(self, arena):
        """测试对战统计"""
        # 运行一些游戏
        arena.run_games(num_games=10, verbose=False)

        # 获取对战统计
        h2h = arena.get_head_to_head("Agent1", "Agent2")

        assert "player1_wins" in h2h
        assert "player2_wins" in h2h
        assert "draws" in h2h
        assert "total_games" in h2h
        assert h2h["total_games"] == 10

    def test_get_statistics(self, arena):
        """测试获取统计信息"""
        # 运行一些游戏
        arena.run_games(num_games=10, verbose=False)

        stats = arena.get_statistics()

        assert stats["total_games"] == 10
        assert stats["total_steps"] > 0
        assert stats["avg_steps_per_game"] > 0
        assert "metrics" in stats

        if arena.elo_system:
            assert "elo_ratings" in stats
            assert len(stats["elo_ratings"]) == len(arena.agent_names)

    def test_reset_statistics(self, arena):
        """测试重置统计"""
        # 运行一些游戏
        arena.run_games(num_games=5, verbose=False)

        assert arena.total_games == 5

        # 重置
        arena.reset_statistics()

        assert arena.total_games == 0
        assert len(arena.metrics_collector.results) == 0
        if arena.elo_system:
            assert len(arena.elo_system.ratings) == 0

    def test_progress_callback(self, arena):
        """测试进度回调"""
        callback_calls = []

        def progress_callback(current, total):
            callback_calls.append((current, total))

        arena.run_games(
            num_games=5,
            verbose=False,
            progress_callback=progress_callback,
        )

        # 应该被调用 5 次
        assert len(callback_calls) == 5

        # 检查调用顺序
        for i, (current, total) in enumerate(callback_calls):
            assert current == i + 1
            assert total == 5

    def test_metrics_collector_integration(self, arena):
        """测试 MetricsCollector 集成"""
        arena.run_games(num_games=10, verbose=False)

        # 检查每个玩家的统计
        for name in arena.agent_names:
            stats = arena.metrics_collector.get_player_stats(name)
            assert stats is not None
            assert stats.games_played == 10

        # 检查排行榜
        leaderboard = arena.metrics_collector.get_leaderboard(sort_by="win_rate")
        assert len(leaderboard) == len(arena.agent_names)

    def test_elo_system_integration(self, arena):
        """测试 ELO 系统集成"""
        if not arena.elo_system:
            pytest.skip("Arena created without ELO system")

        arena.run_games(num_games=20, verbose=False)

        # 检查每个玩家的 ELO 分数
        for name in arena.agent_names:
            rating = arena.elo_system.get_rating(name)
            # 分数应该在合理范围内
            assert 1000 <= rating <= 2000

        # 检查 ELO 排行榜
        elo_leaderboard = arena.elo_system.get_leaderboard()
        assert len(elo_leaderboard) == len(arena.agent_names)

    def test_game_loop_termination(self, arena):
        """测试游戏循环终止"""
        # 游戏应该正常结束，不会无限循环
        result = arena.play_game(game_id=1, verbose=False)

        # 检查步数在合理范围内（Splendor 通常 < 1000 步）
        assert result.num_steps < 10000

    def test_arena_consistency(self, game, agents, agent_names):
        """测试 Arena 一致性"""
        # 创建两个 Arena 实例
        arena1 = Arena(game, agents, agent_names, use_elo=True)
        arena2 = Arena(game, agents, agent_names, use_elo=True)

        # 运行相同数量的游戏
        arena1.run_games(num_games=10, verbose=False)
        arena2.run_games(num_games=10, verbose=False)

        # 统计应该一致（游戏数量）
        assert arena1.total_games == arena2.total_games

        # 但具体结果可能不同（因为随机性）
        # 这里只检查统计合理性
        stats1 = arena1.get_statistics()
        stats2 = arena2.get_statistics()

        assert stats1["total_games"] == stats2["total_games"]
