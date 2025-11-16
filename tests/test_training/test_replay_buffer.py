"""
测试经验回放池
"""

import pytest
import numpy as np

from core.types import Experience, Episode
from training.replay_buffer import ReplayBuffer, EpisodeBuffer


class TestReplayBuffer:
    """测试 ReplayBuffer"""

    @pytest.fixture
    def sample_experience(self):
        """创建测试用经验"""
        return Experience(
            player_id=0,
            observation=np.random.randn(384),
            action=0,
            reward=1.0,
            done=False,
            log_prob=0.0,
            value=0.5,
        )

    @pytest.fixture
    def sample_episode(self):
        """创建测试用 episode"""
        experiences = []
        for i in range(10):
            exp = Experience(
                player_id=i % 2,
                observation=np.random.randn(384),
                action=i,
                reward=1.0,
                done=(i == 9),
                log_prob=0.0,
                value=0.5,
            )
            experiences.append(exp)

        return Episode(
            experiences=experiences,
            total_rewards=[5.0, 5.0],
            winner=0,
            num_steps=10,
        )

    def test_initialization(self):
        """测试初始化"""
        buffer = ReplayBuffer(capacity=1000)
        assert buffer.capacity == 1000
        assert len(buffer) == 0

    def test_add_experience(self, sample_experience):
        """测试添加单个经验"""
        buffer = ReplayBuffer(capacity=100)
        buffer.add_experience(sample_experience)

        assert len(buffer) == 1

    def test_add_experiences(self, sample_experience):
        """测试批量添加经验"""
        buffer = ReplayBuffer(capacity=100)
        experiences = [sample_experience for _ in range(10)]
        buffer.add_experiences(experiences)

        assert len(buffer) == 10

    def test_add_episode(self, sample_episode):
        """测试添加 episode"""
        buffer = ReplayBuffer(capacity=100)
        buffer.add_episode(sample_episode)

        assert len(buffer) == 10  # episode 有 10 个经验
        assert len(buffer.episodes) == 1

    def test_capacity_limit(self, sample_experience):
        """测试容量限制"""
        buffer = ReplayBuffer(capacity=5)

        # 添加 10 个经验
        for _ in range(10):
            buffer.add_experience(sample_experience)

        # 应该只保留最后 5 个
        assert len(buffer) == 5

    def test_sample_all(self, sample_episode):
        """测试采样所有经验"""
        buffer = ReplayBuffer(capacity=100)
        buffer.add_episode(sample_episode)

        sampled = buffer.sample_all()
        assert len(sampled) == 10

    def test_sample_random(self, sample_episode):
        """测试随机采样"""
        buffer = ReplayBuffer(capacity=100)
        buffer.add_episode(sample_episode)

        sampled = buffer.sample_random(batch_size=5)
        assert len(sampled) == 5

    def test_sample_random_insufficient(self, sample_experience):
        """测试采样数量不足抛出异常"""
        buffer = ReplayBuffer(capacity=100)
        buffer.add_experience(sample_experience)

        with pytest.raises(ValueError, match="缓冲区经验不足"):
            buffer.sample_random(batch_size=10)

    def test_clear(self, sample_episode):
        """测试清空缓冲区"""
        buffer = ReplayBuffer(capacity=100)
        buffer.add_episode(sample_episode)

        assert len(buffer) == 10

        buffer.clear()
        assert len(buffer) == 0
        assert len(buffer.episodes) == 0

    def test_is_ready(self, sample_experience):
        """测试检查是否准备好"""
        buffer = ReplayBuffer(capacity=100)

        assert not buffer.is_ready(min_size=10)

        for _ in range(10):
            buffer.add_experience(sample_experience)

        assert buffer.is_ready(min_size=10)
        assert not buffer.is_ready(min_size=20)

    def test_get_statistics_empty(self):
        """测试空缓冲区统计"""
        buffer = ReplayBuffer(capacity=100)
        stats = buffer.get_statistics()

        assert stats["num_experiences"] == 0
        assert stats["num_episodes"] == 0

    def test_get_statistics(self, sample_episode):
        """测试获取统计信息"""
        buffer = ReplayBuffer(capacity=100)
        buffer.add_episode(sample_episode)
        buffer.add_episode(sample_episode)

        stats = buffer.get_statistics()

        assert stats["num_experiences"] == 20
        assert stats["num_episodes"] == 2
        assert stats["mean_episode_length"] == 10.0
        assert "mean_reward" in stats


class TestEpisodeBuffer:
    """测试 EpisodeBuffer"""

    @pytest.fixture
    def sample_episode(self):
        """创建测试用 episode"""
        experiences = []
        for i in range(10):
            exp = Experience(
                player_id=0,
                observation=np.random.randn(384),
                action=i,
                reward=1.0,
                done=(i == 9),
            )
            experiences.append(exp)

        return Episode(
            experiences=experiences,
            total_rewards=[10.0],
            winner=0,
            num_steps=10,
        )

    def test_initialization(self):
        """测试初始化"""
        buffer = EpisodeBuffer(capacity=100)
        assert len(buffer) == 0

    def test_add(self, sample_episode):
        """测试添加 episode"""
        buffer = EpisodeBuffer(capacity=100)
        buffer.add(sample_episode)

        assert len(buffer) == 1

    def test_add_batch(self, sample_episode):
        """测试批量添加 episodes"""
        buffer = EpisodeBuffer(capacity=100)
        episodes = [sample_episode for _ in range(10)]
        buffer.add_batch(episodes)

        assert len(buffer) == 10

    def test_capacity_limit(self, sample_episode):
        """测试容量限制"""
        buffer = EpisodeBuffer(capacity=5)

        # 添加 10 个 episodes
        for _ in range(10):
            buffer.add(sample_episode)

        # 应该只保留最后 5 个
        assert len(buffer) == 5

    def test_sample_recent(self, sample_episode):
        """测试采样最近的 episodes"""
        buffer = EpisodeBuffer(capacity=100)

        for _ in range(10):
            buffer.add(sample_episode)

        recent = buffer.sample_recent(n=5)
        assert len(recent) == 5

    def test_sample_recent_more_than_available(self, sample_episode):
        """测试采样数量超过可用数量"""
        buffer = EpisodeBuffer(capacity=100)
        buffer.add(sample_episode)

        recent = buffer.sample_recent(n=10)
        assert len(recent) == 1  # 只有 1 个可用

    def test_sample_all(self, sample_episode):
        """测试采样所有 episodes"""
        buffer = EpisodeBuffer(capacity=100)

        for _ in range(10):
            buffer.add(sample_episode)

        all_episodes = buffer.sample_all()
        assert len(all_episodes) == 10

    def test_clear(self, sample_episode):
        """测试清空缓冲区"""
        buffer = EpisodeBuffer(capacity=100)
        buffer.add(sample_episode)

        assert len(buffer) == 1

        buffer.clear()
        assert len(buffer) == 0

    def test_get_statistics_empty(self):
        """测试空缓冲区统计"""
        buffer = EpisodeBuffer(capacity=100)
        stats = buffer.get_statistics()

        assert stats["num_episodes"] == 0
        assert stats["mean_episode_length"] == 0.0

    def test_get_statistics(self, sample_episode):
        """测试获取统计信息"""
        buffer = EpisodeBuffer(capacity=100)
        buffer.add(sample_episode)
        buffer.add(sample_episode)

        stats = buffer.get_statistics()

        assert stats["num_episodes"] == 2
        assert stats["mean_episode_length"] == 10.0
        assert "std_episode_length" in stats
        assert "mean_reward" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
