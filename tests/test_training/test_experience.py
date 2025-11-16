"""
测试经验批处理模块
"""

import pytest
import numpy as np
import torch

from core.types import Experience, Episode
from training.experience import (
    ExperienceBatch,
    compute_advantages_for_episode,
    split_episodes_by_player,
    merge_experience_batches,
)


class TestExperienceBatch:
    """测试 ExperienceBatch"""

    @pytest.fixture
    def sample_experiences(self):
        """创建测试用经验"""
        experiences = []
        for i in range(10):
            exp = Experience(
                player_id=i % 2,  # 2 个玩家
                observation=np.random.randn(384).astype(np.float32),
                action=i,
                reward=np.random.rand(),
                next_observation=np.random.randn(384).astype(np.float32),
                done=(i == 9),
                log_prob=np.random.rand(),
                value=np.random.rand(),
                advantage=np.random.rand(),
                returns=np.random.rand(),
            )
            experiences.append(exp)
        return experiences

    def test_from_experiences(self, sample_experiences):
        """测试从经验列表创建批次"""
        batch = ExperienceBatch.from_experiences(sample_experiences, device="cpu")

        assert batch.batch_size == 10
        assert batch.observations.shape == (10, 384)
        assert batch.actions.shape == (10,)
        assert batch.old_log_probs.shape == (10,)
        assert batch.advantages.shape == (10,)
        assert batch.returns.shape == (10,)
        assert batch.values.shape == (10,)

    def test_from_experiences_empty(self):
        """测试空经验列表"""
        with pytest.raises(ValueError, match="经验列表不能为空"):
            ExperienceBatch.from_experiences([], device="cpu")

    def test_from_experiences_missing_data(self):
        """测试缺失必要数据"""
        exp = Experience(
            player_id=0,
            observation=np.random.randn(384),
            action=0,
            reward=1.0,
            log_prob=None,  # 缺失
            value=None,  # 缺失
        )

        with pytest.raises(ValueError, match="经验必须包含 log_prob 和 value"):
            ExperienceBatch.from_experiences([exp], device="cpu")

    def test_iterate_minibatches(self, sample_experiences):
        """测试迭代小批次"""
        batch = ExperienceBatch.from_experiences(sample_experiences, device="cpu")

        minibatches = list(batch.iterate_minibatches(batch_size=3, shuffle=False))

        # 应该有 4 个小批次 (10 / 3 = 3 余 1)
        assert len(minibatches) == 4
        assert minibatches[0].batch_size == 3
        assert minibatches[1].batch_size == 3
        assert minibatches[2].batch_size == 3
        assert minibatches[3].batch_size == 1

    def test_to_device(self, sample_experiences):
        """测试移动到不同设备"""
        batch = ExperienceBatch.from_experiences(sample_experiences, device="cpu")

        # 测试移动到 CPU（应该不变）
        batch_cpu = batch.to("cpu")
        assert batch_cpu.observations.device == torch.device("cpu")


class TestComputeAdvantages:
    """测试优势计算"""

    def test_compute_advantages_for_episode(self):
        """测试为 episode 计算优势"""
        # 创建简单的 episode
        experiences = []
        for i in range(5):
            exp = Experience(
                player_id=0,
                observation=np.random.randn(384),
                action=i,
                reward=1.0,
                done=(i == 4),
                log_prob=0.0,
                value=0.5,
            )
            experiences.append(exp)

        episode = Episode(
            experiences=experiences,
            total_rewards=[5.0],
            winner=0,
            num_steps=5,
        )

        # 计算优势
        compute_advantages_for_episode(episode, gamma=0.99, gae_lambda=0.95)

        # 检查所有经验都有 advantage 和 returns
        for exp in episode.experiences:
            assert exp.advantage is not None
            assert exp.returns is not None

    def test_compute_advantages_empty_episode(self):
        """测试空 episode"""
        episode = Episode(
            experiences=[],
            total_rewards=[],
            winner=-1,
            num_steps=0,
        )

        # 应该不报错
        compute_advantages_for_episode(episode)


class TestSplitEpisodes:
    """测试 episode 分割"""

    def test_split_episodes_by_player(self):
        """测试按玩家分割 episodes"""
        # 创建 2 个 episodes，每个有 2 个玩家
        episodes = []
        for ep_idx in range(2):
            experiences = []
            for i in range(4):
                exp = Experience(
                    player_id=i % 2,
                    observation=np.random.randn(384),
                    action=i,
                    reward=1.0,
                    done=(i == 3),
                    log_prob=0.0,
                    value=0.5,
                    advantage=0.1,
                    returns=1.0,
                )
                experiences.append(exp)

            episode = Episode(
                experiences=experiences,
                total_rewards=[2.0, 2.0],
                winner=0,
                num_steps=4,
            )
            episodes.append(episode)

        # 分割
        player_exps = split_episodes_by_player(episodes)

        # 应该有 2 个玩家
        assert len(player_exps) == 2

        # 每个玩家应该有 4 个经验 (2 episodes * 2 experiences per player)
        assert len(player_exps[0]) == 4
        assert len(player_exps[1]) == 4

        # 检查玩家 ID
        for exp in player_exps[0]:
            assert exp.player_id == 0
        for exp in player_exps[1]:
            assert exp.player_id == 1

    def test_split_episodes_empty(self):
        """测试空 episodes"""
        player_exps = split_episodes_by_player([])
        assert player_exps == []


class TestMergeBatches:
    """测试批次合并"""

    def test_merge_experience_batches(self):
        """测试合并多个批次"""
        # 创建 3 个批次
        batches = []
        for _ in range(3):
            experiences = []
            for i in range(5):
                exp = Experience(
                    player_id=0,
                    observation=np.random.randn(384),
                    action=i,
                    reward=1.0,
                    log_prob=0.0,
                    value=0.5,
                    advantage=0.1,
                    returns=1.0,
                )
                experiences.append(exp)

            batch = ExperienceBatch.from_experiences(experiences, device="cpu")
            batches.append(batch)

        # 合并
        merged = merge_experience_batches(batches)

        # 合并后应该有 15 个经验 (3 * 5)
        assert merged.batch_size == 15

    def test_merge_single_batch(self):
        """测试合并单个批次"""
        experiences = []
        for i in range(5):
            exp = Experience(
                player_id=0,
                observation=np.random.randn(384),
                action=i,
                reward=1.0,
                log_prob=0.0,
                value=0.5,
                advantage=0.1,
                returns=1.0,
            )
            experiences.append(exp)

        batch = ExperienceBatch.from_experiences(experiences, device="cpu")
        merged = merge_experience_batches([batch])

        assert merged.batch_size == 5

    def test_merge_empty_raises_error(self):
        """测试合并空列表抛出异常"""
        with pytest.raises(ValueError, match="批次列表不能为空"):
            merge_experience_batches([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
