"""
神经网络模型单元测试

测试编码器、头部、Actor-Critic 模型和模型工厂的功能。
"""

import pytest
import torch

from models.encoders.mlp_encoder import MLPEncoder
from models.encoders.attention_encoder import AttentionEncoder
from models.heads.policy_head import PolicyHead
from models.heads.value_head import ValueHead
from models.actor_critic import ActorCritic
from models.model_factory import (
    create_model,
    create_splendor_model,
    get_model_info,
    MODEL_CONFIGS,
)


class TestMLPEncoder:
    """测试 MLP 编码器"""

    def test_initialization(self):
        """测试编码器初始化"""
        encoder = MLPEncoder(obs_dim=384, hidden_dim=256)
        assert encoder.obs_dim == 384
        assert encoder.hidden_dim == 256
        assert encoder.get_output_dim() == 256

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        encoder = MLPEncoder(obs_dim=384, hidden_dim=256)
        obs = torch.randn(32, 384)
        features = encoder(obs)
        assert features.shape == (32, 256)

    def test_forward_single_batch(self):
        """测试单样本前向传播"""
        encoder = MLPEncoder(obs_dim=384, hidden_dim=256)
        obs = torch.randn(1, 384)
        features = encoder(obs)
        assert features.shape == (1, 256)

    def test_dropout(self):
        """测试 dropout 功能"""
        encoder = MLPEncoder(obs_dim=384, hidden_dim=256, dropout=0.5)
        encoder.eval()  # 评估模式下 dropout 应被禁用
        obs = torch.randn(2, 384)
        features1 = encoder(obs)
        features2 = encoder(obs)
        assert torch.allclose(features1, features2)


class TestAttentionEncoder:
    """测试注意力编码器"""

    def test_initialization(self):
        """测试编码器初始化"""
        encoder = AttentionEncoder(obs_dim=384, hidden_dim=256, num_heads=4)
        assert encoder.obs_dim == 384
        assert encoder.hidden_dim == 256
        assert encoder.num_heads == 4
        assert encoder.get_output_dim() == 256

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        encoder = AttentionEncoder(obs_dim=384, hidden_dim=256)
        obs = torch.randn(32, 384)
        features = encoder(obs)
        assert features.shape == (32, 256)

    def test_forward_single_batch(self):
        """测试单样本前向传播"""
        encoder = AttentionEncoder(obs_dim=384, hidden_dim=256)
        obs = torch.randn(1, 384)
        features = encoder(obs)
        assert features.shape == (1, 256)

    def test_different_num_heads(self):
        """测试不同的注意力头数"""
        for num_heads in [2, 4, 8]:
            encoder = AttentionEncoder(
                obs_dim=384, hidden_dim=256, intermediate_dim=512, num_heads=num_heads
            )
            obs = torch.randn(4, 384)
            features = encoder(obs)
            assert features.shape == (4, 256)


class TestPolicyHead:
    """测试策略头"""

    def test_initialization(self):
        """测试策略头初始化"""
        policy = PolicyHead(hidden_dim=256, action_size=50)
        assert policy.hidden_dim == 256
        assert policy.action_size == 50

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        policy = PolicyHead(hidden_dim=256, action_size=50)
        features = torch.randn(32, 256)
        logits = policy(features)
        assert logits.shape == (32, 50)

    def test_legal_actions_mask(self):
        """测试合法动作掩码"""
        policy = PolicyHead(hidden_dim=256, action_size=50)
        features = torch.randn(4, 256)

        # 创建掩码：只有前 10 个动作合法
        mask = torch.zeros(4, 50, dtype=torch.bool)
        mask[:, :10] = True

        logits = policy(features, mask)
        assert logits.shape == (4, 50)

        # 检查非法动作的 logits 为 -inf
        assert torch.all(torch.isinf(logits[:, 10:]))
        assert torch.all(logits[:, 10:] < 0)

    def test_get_action_probs(self):
        """测试获取动作概率"""
        policy = PolicyHead(hidden_dim=256, action_size=50)
        features = torch.randn(4, 256)
        probs = policy.get_action_probs(features)

        assert probs.shape == (4, 50)
        # 概率和应为 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-6)
        # 所有概率应为非负
        assert torch.all(probs >= 0)

    def test_get_action_probs_with_mask(self):
        """测试带掩码的动作概率"""
        policy = PolicyHead(hidden_dim=256, action_size=50)
        features = torch.randn(4, 256)

        # 只有前 10 个动作合法
        mask = torch.zeros(4, 50, dtype=torch.bool)
        mask[:, :10] = True

        probs = policy.get_action_probs(features, mask)

        # 非法动作的概率应为 0
        assert torch.allclose(probs[:, 10:], torch.zeros(4, 40))
        # 合法动作的概率和应为 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-6)

    def test_sample_action_deterministic(self):
        """测试确定性动作采样"""
        policy = PolicyHead(hidden_dim=256, action_size=50)
        features = torch.randn(4, 256)

        actions, log_probs = policy.sample_action(features, deterministic=True)

        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert torch.all(actions >= 0) and torch.all(actions < 50)

    def test_sample_action_stochastic(self):
        """测试随机动作采样"""
        policy = PolicyHead(hidden_dim=256, action_size=50)
        features = torch.randn(4, 256)

        actions, log_probs = policy.sample_action(features, deterministic=False)

        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert torch.all(actions >= 0) and torch.all(actions < 50)


class TestValueHead:
    """测试价值头"""

    def test_initialization(self):
        """测试价值头初始化"""
        value_head = ValueHead(hidden_dim=256)
        assert value_head.hidden_dim == 256

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        value_head = ValueHead(hidden_dim=256)
        features = torch.randn(32, 256)
        values = value_head(features)
        assert values.shape == (32, 1)

    def test_forward_single_batch(self):
        """测试单样本前向传播"""
        value_head = ValueHead(hidden_dim=256)
        features = torch.randn(1, 256)
        values = value_head(features)
        assert values.shape == (1, 1)


class TestActorCritic:
    """测试 Actor-Critic 模型"""

    @pytest.mark.parametrize("encoder_type", ["mlp", "attention"])
    def test_initialization(self, encoder_type):
        """测试模型初始化"""
        model = ActorCritic(
            obs_dim=384, action_size=50, encoder_type=encoder_type, hidden_dim=256
        )
        assert model.obs_dim == 384
        assert model.action_size == 50
        assert model.encoder_type == encoder_type
        assert model.hidden_dim == 256

    def test_invalid_encoder_type(self):
        """测试无效的编码器类型"""
        with pytest.raises(ValueError):
            ActorCritic(obs_dim=384, action_size=50, encoder_type="invalid")

    @pytest.mark.parametrize("encoder_type", ["mlp", "attention"])
    def test_forward_shape(self, encoder_type):
        """测试前向传播输出形状"""
        model = ActorCritic(obs_dim=384, action_size=50, encoder_type=encoder_type)
        obs = torch.randn(32, 384)
        logits, values = model(obs)

        assert logits.shape == (32, 50)
        assert values.shape == (32, 1)

    @pytest.mark.parametrize("encoder_type", ["mlp", "attention"])
    def test_get_action_and_value(self, encoder_type):
        """测试获取动作和价值"""
        model = ActorCritic(obs_dim=384, action_size=50, encoder_type=encoder_type)
        obs = torch.randn(4, 384)

        actions, log_probs, values = model.get_action_and_value(obs)

        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4, 1)
        assert torch.all(actions >= 0) and torch.all(actions < 50)

    @pytest.mark.parametrize("encoder_type", ["mlp", "attention"])
    def test_get_action_and_value_with_mask(self, encoder_type):
        """测试带掩码的动作和价值"""
        model = ActorCritic(obs_dim=384, action_size=50, encoder_type=encoder_type)
        obs = torch.randn(4, 384)

        # 只有前 10 个动作合法
        mask = torch.zeros(4, 50, dtype=torch.bool)
        mask[:, :10] = True

        actions, log_probs, values = model.get_action_and_value(obs, mask)

        assert actions.shape == (4,)
        assert values.shape == (4, 1)
        # 所有采样的动作应该在合法范围内
        assert torch.all(actions < 10)

    @pytest.mark.parametrize("encoder_type", ["mlp", "attention"])
    def test_get_value(self, encoder_type):
        """测试仅获取价值"""
        model = ActorCritic(obs_dim=384, action_size=50, encoder_type=encoder_type)
        obs = torch.randn(4, 384)
        values = model.get_value(obs)

        assert values.shape == (4, 1)

    @pytest.mark.parametrize("encoder_type", ["mlp", "attention"])
    def test_get_action_probs(self, encoder_type):
        """测试获取动作概率"""
        model = ActorCritic(obs_dim=384, action_size=50, encoder_type=encoder_type)
        obs = torch.randn(4, 384)
        probs = model.get_action_probs(obs)

        assert probs.shape == (4, 50)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-6)

    @pytest.mark.parametrize("encoder_type", ["mlp", "attention"])
    def test_evaluate_actions(self, encoder_type):
        """测试评估动作"""
        model = ActorCritic(obs_dim=384, action_size=50, encoder_type=encoder_type)
        obs = torch.randn(4, 384)
        actions = torch.randint(0, 50, (4,))

        values, log_probs, entropy = model.evaluate_actions(obs, actions)

        assert values.shape == (4, 1)
        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)

    @pytest.mark.parametrize("encoder_type", ["mlp", "attention"])
    def test_count_parameters(self, encoder_type):
        """测试参数统计"""
        model = ActorCritic(obs_dim=384, action_size=50, encoder_type=encoder_type)
        param_counts = model.count_parameters()

        assert "encoder" in param_counts
        assert "policy_head" in param_counts
        assert "value_head" in param_counts
        assert "total" in param_counts

        # 总参数量应该等于各部分之和
        assert param_counts["total"] == (
            param_counts["encoder"]
            + param_counts["policy_head"]
            + param_counts["value_head"]
        )

        # 参数量应该在合理范围内（100K - 1M）
        assert 100_000 <= param_counts["total"] <= 1_000_000


class TestModelFactory:
    """测试模型工厂"""

    def test_create_model_with_preset_config(self):
        """测试使用预设配置创建模型"""
        for config_name in ["small", "medium", "large"]:
            model = create_model(
                obs_dim=384, action_size=50, encoder_type="mlp", config=config_name
            )
            assert isinstance(model, ActorCritic)

    def test_create_model_with_custom_config(self):
        """测试使用自定义配置创建模型"""
        custom_config = {
            "hidden_dim": 256,
            "encoder_intermediate_dim": 512,
            "head_intermediate_dim": 128,
            "num_attention_heads": 4,
            "dropout": 0.05,
        }
        model = create_model(
            obs_dim=384, action_size=50, encoder_type="mlp", config=custom_config
        )
        assert isinstance(model, ActorCritic)
        assert model.hidden_dim == 256

    def test_create_model_default_config(self):
        """测试默认配置（应为 medium）"""
        model = create_model(obs_dim=384, action_size=50, encoder_type="mlp")
        assert isinstance(model, ActorCritic)
        assert model.hidden_dim == MODEL_CONFIGS["medium"]["hidden_dim"]

    def test_create_model_invalid_config(self):
        """测试无效的配置名称"""
        with pytest.raises(ValueError):
            create_model(
                obs_dim=384, action_size=50, encoder_type="mlp", config="invalid"
            )

    def test_create_splendor_model(self):
        """测试创建 Splendor 专用模型"""
        model = create_splendor_model(encoder_type="mlp", config="medium")
        assert isinstance(model, ActorCritic)
        assert model.obs_dim == 384
        assert model.action_size == 50

    @pytest.mark.parametrize("encoder_type", ["mlp", "attention"])
    @pytest.mark.parametrize("config", ["small", "medium", "large"])
    def test_create_splendor_model_combinations(self, encoder_type, config):
        """测试不同编码器和配置组合"""
        model = create_splendor_model(encoder_type=encoder_type, config=config)
        assert isinstance(model, ActorCritic)

    def test_get_model_info(self):
        """测试获取模型信息"""
        model = create_splendor_model(encoder_type="mlp", config="medium")
        info = get_model_info(model)

        assert "encoder_type" in info
        assert "obs_dim" in info
        assert "action_size" in info
        assert "hidden_dim" in info
        assert "parameters" in info

        assert info["encoder_type"] == "mlp"
        assert info["obs_dim"] == 384
        assert info["action_size"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
