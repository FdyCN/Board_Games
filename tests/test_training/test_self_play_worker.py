"""
测试自对弈 Worker
"""

import pytest
import numpy as np

from games.splendor import SplendorGame
from agents.random_agent import RandomAgent
from training.self_play_worker import (
    collect_episode,
    collect_episodes,
    collect_episodes_multiprocess,
    SelfPlayWorker,
)


class TestCollectEpisode:
    """测试 collect_episode"""

    @pytest.fixture
    def game(self):
        """创建测试游戏"""
        return SplendorGame(num_players=2, seed=42)

    @pytest.fixture
    def agents(self):
        """创建测试 agents"""
        return [RandomAgent(player_id=i) for i in range(2)]

    def test_collect_episode_basic(self, game, agents):
        """测试基本的 episode 收集"""
        episode = collect_episode(
            game=game,
            agents=agents,
            gamma=0.99,
            gae_lambda=0.95,
            deterministic=False,
            verbose=False,
        )

        # 检查 episode 基本属性
        assert episode is not None
        assert len(episode.experiences) > 0
        assert episode.num_steps > 0
        assert 0 <= episode.winner < 2

    def test_collect_episode_advantages_computed(self, game, agents):
        """测试优势函数已计算"""
        episode = collect_episode(
            game=game,
            agents=agents,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # 所有经验应该有 advantage 和 returns
        for exp in episode.experiences:
            assert exp.advantage is not None
            assert exp.returns is not None

    def test_collect_episode_agent_mismatch(self, game):
        """测试 agent 数量不匹配"""
        agents = [RandomAgent()]  # 只有 1 个 agent，游戏需要 2 个

        with pytest.raises(ValueError, match="Agent 数量.*与游戏玩家数.*不匹配"):
            collect_episode(game=game, agents=agents)


class TestCollectEpisodes:
    """测试 collect_episodes"""

    @pytest.fixture
    def game(self):
        """创建测试游戏"""
        return SplendorGame(num_players=2, seed=42)

    @pytest.fixture
    def agents(self):
        """创建测试 agents"""
        return [RandomAgent(player_id=i) for i in range(2)]

    def test_collect_multiple_episodes(self, game, agents):
        """测试收集多个 episodes"""
        episodes = collect_episodes(
            game=game,
            agents=agents,
            num_episodes=5,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=False,
        )

        assert len(episodes) == 5

        for episode in episodes:
            assert len(episode.experiences) > 0
            assert episode.num_steps > 0

    def test_collect_episodes_progress_callback(self, game, agents):
        """测试进度回调"""
        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        collect_episodes(
            game=game,
            agents=agents,
            num_episodes=3,
            verbose=False,
            progress_callback=callback,
        )

        # 应该调用 3 次
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)


class TestSelfPlayWorker:
    """测试 SelfPlayWorker"""

    @pytest.fixture
    def game(self):
        """创建测试游戏"""
        return SplendorGame(num_players=2, seed=42)

    @pytest.fixture
    def agents(self):
        """创建测试 agents"""
        return [RandomAgent(player_id=i) for i in range(2)]

    def test_initialization(self, game, agents):
        """测试初始化"""
        worker = SelfPlayWorker(
            game=game,
            agents=agents,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert worker.game == game
        assert worker.agents == agents
        assert worker.gamma == 0.99
        assert worker.gae_lambda == 0.95

    def test_initialization_agent_mismatch(self, game):
        """测试 agent 数量不匹配"""
        agents = [RandomAgent()]

        with pytest.raises(ValueError, match="Agent 数量.*与游戏玩家数.*不匹配"):
            SelfPlayWorker(game=game, agents=agents)

    def test_collect_one(self, game, agents):
        """测试收集一个 episode"""
        worker = SelfPlayWorker(game=game, agents=agents)

        episode = worker.collect_one()

        assert episode is not None
        assert len(episode.experiences) > 0
        assert worker.episodes_collected == 1

    def test_collect_multiple(self, game, agents):
        """测试收集多个 episodes"""
        worker = SelfPlayWorker(game=game, agents=agents)

        episodes = worker.collect(num_episodes=5)

        assert len(episodes) == 5
        assert worker.episodes_collected == 5

    def test_statistics(self, game, agents):
        """测试统计信息"""
        worker = SelfPlayWorker(game=game, agents=agents)

        # 初始统计
        stats = worker.get_statistics()
        assert stats["episodes_collected"] == 0
        assert stats["total_steps"] == 0

        # 收集一些 episodes
        worker.collect(num_episodes=3)

        stats = worker.get_statistics()
        assert stats["episodes_collected"] == 3
        assert stats["total_steps"] > 0
        assert stats["avg_steps_per_episode"] > 0

    def test_reset_statistics(self, game, agents):
        """测试重置统计"""
        worker = SelfPlayWorker(game=game, agents=agents)

        worker.collect(num_episodes=3)
        assert worker.episodes_collected == 3

        worker.reset_statistics()
        assert worker.episodes_collected == 0
        assert worker.total_steps == 0


class TestMultiprocessing:
    """测试多进程功能"""

    @pytest.fixture
    def game(self):
        """创建测试游戏"""
        return SplendorGame(num_players=2, seed=42)

    @pytest.fixture
    def agents(self):
        """创建测试 agents"""
        return [RandomAgent(player_id=i) for i in range(2)]

    def test_collect_episodes_multiprocess_basic(self, game, agents):
        """测试基本的多进程收集"""
        episodes = collect_episodes_multiprocess(
            game_class=SplendorGame,
            game_kwargs={"num_players": 2, "seed": 42},
            agents=agents,
            num_episodes=4,
            num_workers=2,
            gamma=0.99,
            gae_lambda=0.95,
            seed=42,
        )

        # 验证收集到正确数量的 episodes
        assert len(episodes) == 4

        # 验证每个 episode 都有有效数据
        for episode in episodes:
            assert len(episode.experiences) > 0
            assert episode.num_steps > 0
            assert 0 <= episode.winner < 2

    def test_multiprocess_worker_initialization(self, game, agents):
        """测试多进程 Worker 初始化"""
        worker = SelfPlayWorker(
            game=game,
            agents=agents,
            gamma=0.99,
            gae_lambda=0.95,
            num_workers=2,
        )

        assert worker.num_workers == 2
        assert worker.game == game
        assert worker.agents == agents

    def test_worker_multiprocess_collect(self, game, agents):
        """测试 Worker 使用多进程收集"""
        worker = SelfPlayWorker(
            game=game,
            agents=agents,
            num_workers=2,
            verbose=False,
        )

        episodes = worker.collect(num_episodes=4)

        assert len(episodes) == 4
        assert worker.episodes_collected == 4

        # 验证统计信息
        stats = worker.get_statistics()
        assert stats["episodes_collected"] == 4
        assert stats["total_steps"] > 0

    def test_worker_single_vs_multiprocess(self, game, agents):
        """测试单进程和多进程结果一致性"""
        # 单进程
        worker_single = SelfPlayWorker(
            game=game,
            agents=agents,
            num_workers=1,
            verbose=False,
        )
        episodes_single = worker_single.collect(num_episodes=3)

        # 多进程
        worker_multi = SelfPlayWorker(
            game=game,
            agents=agents,
            num_workers=2,
            verbose=False,
        )
        episodes_multi = worker_multi.collect(num_episodes=3)

        # 验证都收集到了正确数量的 episodes
        assert len(episodes_single) == 3
        assert len(episodes_multi) == 3

        # 验证每个 episode 都有有效数据
        for eps in [episodes_single, episodes_multi]:
            for episode in eps:
                assert len(episode.experiences) > 0
                assert episode.num_steps > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
