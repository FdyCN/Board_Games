"""
测试 PPO 算法
"""

import pytest
import tempfile
import os
import numpy as np

from models.model_factory import create_splendor_model
from training.algorithms.ppo import PPO
from training.experience import ExperienceBatch
from core.types import Experience


class TestPPO:
    """测试 PPO 训练器"""

    @pytest.fixture
    def model(self):
        """创建测试模型"""
        return create_splendor_model(encoder_type="mlp", config="small")

    @pytest.fixture
    def ppo(self, model):
        """创建 PPO 训练器"""
        return PPO(
            model=model,
            learning_rate=1e-3,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            device="cpu",
        )

    @pytest.fixture
    def sample_batch(self):
        """创建测试用经验批次"""
        experiences = []
        for i in range(32):
            exp = Experience(
                player_id=0,
                observation=np.random.randn(384).astype(np.float32),
                action=i % 50,
                reward=np.random.rand(),
                done=(i == 31),
                log_prob=np.random.randn(),
                value=np.random.rand(),
                advantage=np.random.randn(),
                returns=np.random.rand(),
            )
            experiences.append(exp)

        return ExperienceBatch.from_experiences(
            experiences, device="cpu", normalize_advantages=True
        )

    def test_initialization(self, model):
        """测试初始化"""
        ppo = PPO(
            model=model,
            learning_rate=3e-4,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
        )

        assert ppo.model == model
        assert ppo.clip_epsilon == 0.2
        assert ppo.value_coef == 0.5
        assert ppo.entropy_coef == 0.01
        assert ppo.update_count == 0

    def test_update(self, ppo, sample_batch):
        """测试更新模型"""
        metrics = ppo.update(sample_batch, epochs=2, minibatch_size=None)

        # 检查返回的指标
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "kl_div" in metrics
        assert "clip_fraction" in metrics
        assert "total_loss" in metrics

        # 检查更新计数
        assert ppo.update_count == 1

    def test_update_with_minibatches(self, ppo, sample_batch):
        """测试使用小批次更新"""
        metrics = ppo.update(sample_batch, epochs=2, minibatch_size=16)

        # 应该正常返回指标
        assert "policy_loss" in metrics
        assert "value_loss" in metrics

    def test_save_and_load_checkpoint(self, ppo, sample_batch):
        """测试保存和加载检查点"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pth")

            # 更新一次以改变模型状态
            ppo.update(sample_batch, epochs=1)
            original_update_count = ppo.update_count

            # 保存检查点
            metadata = {"iteration": 10, "total_steps": 1000}
            ppo.save_checkpoint(checkpoint_path, metadata=metadata)

            assert os.path.exists(checkpoint_path)

            # 重置更新计数
            ppo.update_count = 0

            # 加载检查点
            loaded_metadata = ppo.load_checkpoint(checkpoint_path)

            # 验证元数据
            assert loaded_metadata["iteration"] == 10
            assert loaded_metadata["total_steps"] == 1000

            # 验证更新计数已恢复
            assert ppo.update_count == original_update_count

    def test_get_learning_rate(self, ppo):
        """测试获取学习率"""
        lr = ppo.get_learning_rate()
        assert lr == 1e-3  # 初始化时设置的学习率

    def test_set_learning_rate(self, ppo):
        """测试设置学习率"""
        ppo.set_learning_rate(1e-4)
        assert ppo.get_learning_rate() == 1e-4

    def test_multiple_updates(self, ppo, sample_batch):
        """测试多次更新"""
        # 执行多次更新
        for _ in range(3):
            metrics = ppo.update(sample_batch, epochs=1)
            assert "policy_loss" in metrics

        # 更新计数应该累加
        assert ppo.update_count == 3

    def test_loss_values_reasonable(self, ppo, sample_batch):
        """测试损失值在合理范围内"""
        metrics = ppo.update(sample_batch, epochs=1)

        # 损失值应该是有限的数值
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
        assert np.isfinite(metrics["entropy"])
        assert np.isfinite(metrics["kl_div"])

        # Clip fraction 应该在 [0, 1] 范围内
        assert 0.0 <= metrics["clip_fraction"] <= 1.0


class TestPPOIntegration:
    """PPO 集成测试"""

    def test_full_training_loop(self):
        """测试完整的训练循环"""
        # 创建模型和 PPO
        model = create_splendor_model(encoder_type="mlp", config="small")
        ppo = PPO(model=model, learning_rate=1e-3, device="cpu")

        # 创建批次
        experiences = []
        for i in range(64):
            exp = Experience(
                player_id=0,
                observation=np.random.randn(384).astype(np.float32),
                action=i % 50,
                reward=1.0 if i % 10 == 0 else 0.0,
                done=(i == 63),
                log_prob=np.random.randn(),
                value=np.random.rand(),
                advantage=np.random.randn(),
                returns=np.random.rand(),
            )
            experiences.append(exp)

        batch = ExperienceBatch.from_experiences(experiences, device="cpu")

        # 执行多次更新
        for _ in range(5):
            metrics = ppo.update(batch, epochs=2, minibatch_size=32)

            # 验证所有指标都是有效的
            for key, value in metrics.items():
                assert np.isfinite(value), f"{key} 不是有限值: {value}"

        # 验证更新计数
        assert ppo.update_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
