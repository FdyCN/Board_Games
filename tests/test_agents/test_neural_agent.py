
"""
Neural Agent 单元测试

测试基于神经网络的智能体功能。
"""

import tempfile
import os
import pytest
import numpy as np
import torch

from agents.neural_agent import NeuralAgent
from models.model_factory import create_model


class TestNeuralAgent:
    """测试 NeuralAgent"""

    @pytest.fixture
    def model(self):
        """创建测试用模型"""
        return create_model(
            obs_dim=384,
            action_size=50,
            encoder_type="mlp",
            config="small",  # 使用小模型加快测试速度
        )

    @pytest.fixture
    def agent(self, model):
        """创建测试用 Agent"""
        return NeuralAgent(model, device="cpu", name="TestAgent")

    def test_initialization(self, model):
        """测试 Agent 初始化"""
        agent = NeuralAgent(model, device="cpu", name="TestAgent")
        assert agent.get_name() == "TestAgent"
        assert agent.model == model
        assert agent.device == torch.device("cpu")

    def test_initialization_default_name(self, model):
        """测试默认名称"""
        agent = NeuralAgent(model)
        assert agent.get_name() == "NeuralAgent"

    def test_select_action_shape(self, agent):
        """测试 select_action 返回值形状"""
        obs = np.random.randn(384)
        legal_actions = [0, 1, 2, 5, 10, 20, 30, 40]

        action_idx, info = agent.select_action(obs, legal_actions)

        # 检查动作索引
        assert isinstance(action_idx, int)
        assert action_idx in legal_actions

        # 检查 info 字典
        assert "log_prob" in info
        assert "value" in info
        assert "policy" in info
        assert "entropy" in info

        # 检查数值类型
        assert isinstance(info["log_prob"], (float, np.floating))
        assert isinstance(info["value"], (float, np.floating))
        assert isinstance(info["policy"], np.ndarray)
        assert isinstance(info["entropy"], (float, np.floating))

        # 检查策略分布形状
        assert info["policy"].shape == (50,)

    def test_select_action_legal_mask(self, agent):
        """测试合法动作掩码"""
        obs = np.random.randn(384)
        legal_actions = [5, 10, 15]

        # 多次采样，确保只选择合法动作
        for _ in range(20):
            action_idx, info = agent.select_action(obs, legal_actions, deterministic=False)
            assert action_idx in legal_actions

            # 检查非法动作的概率接近 0
            policy = info["policy"]
            illegal_actions = [i for i in range(50) if i not in legal_actions]
            assert np.allclose(policy[illegal_actions], 0.0, atol=1e-6)

    def test_select_action_deterministic(self, agent):
        """测试确定性动作选择"""
        obs = np.random.randn(384)
        legal_actions = [0, 1, 2, 5, 10]

        # 确定性模式下，相同输入应产生相同输出
        action1, info1 = agent.select_action(obs, legal_actions, deterministic=True)
        action2, info2 = agent.select_action(obs, legal_actions, deterministic=True)

        assert action1 == action2
        assert np.isclose(info1["value"], info2["value"])

    def test_select_action_stochastic(self, agent):
        """测试随机动作选择"""
        obs = np.random.randn(384)
        legal_actions = list(range(20))  # 更多合法动作以便观察随机性

        # 随机模式下，多次采样应产生不同结果（大概率）
        actions = []
        for _ in range(30):
            action_idx, _ = agent.select_action(obs, legal_actions, deterministic=False)
            actions.append(action_idx)

        # 至少应该有 2 种不同的动作（概率极高）
        unique_actions = len(set(actions))
        assert unique_actions >= 2, f"只采样到 {unique_actions} 种不同动作"

    def test_select_action_single_legal_action(self, agent):
        """测试只有一个合法动作的情况"""
        obs = np.random.randn(384)
        legal_actions = [7]

        action_idx, info = agent.select_action(obs, legal_actions)

        assert action_idx == 7
        # 唯一合法动作的概率应该接近 1
        assert info["policy"][7] > 0.99

    def test_select_action_empty_legal_actions(self, agent):
        """测试空合法动作列表"""
        obs = np.random.randn(384)
        legal_actions = []

        with pytest.raises(ValueError, match="legal_actions 不能为空"):
            agent.select_action(obs, legal_actions)

    def test_select_action_batch_observation(self, agent):
        """测试批量观察输入"""
        # 即使输入是批量形状，也应该正常工作
        obs = np.random.randn(1, 384)  # (1, 384)
        legal_actions = [0, 1, 2]

        action_idx, info = agent.select_action(obs, legal_actions)
        assert action_idx in legal_actions

    def test_select_action_tensor_observation(self, agent):
        """测试 tensor 观察输入"""
        obs = torch.randn(384)
        legal_actions = [0, 1, 2]

        action_idx, info = agent.select_action(obs, legal_actions)
        assert action_idx in legal_actions

    def test_set_training_mode(self, agent):
        """测试训练/评估模式切换"""
        # 默认应该是评估模式
        assert not agent.model.training

        # 设置为训练模式
        agent.set_training_mode(True)
        assert agent.model.training
        assert agent._training is True

        # 设置为评估模式
        agent.set_training_mode(False)
        assert not agent.model.training
        assert agent._training is False

    def test_save_and_load(self, agent):
        """测试保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "agent.pth")

            # 保存 Agent
            agent.save(save_path)
            assert os.path.exists(save_path)

            # 记录原始权重
            original_weights = agent.model.encoder.fc1.weight.data.clone()

            # 修改权重
            agent.model.encoder.fc1.weight.data.fill_(0.5)
            modified_weights = agent.model.encoder.fc1.weight.data.clone()
            assert not torch.allclose(original_weights, modified_weights)

            # 加载 Agent
            agent.load(save_path)
            loaded_weights = agent.model.encoder.fc1.weight.data

            # 验证权重已恢复
            assert torch.allclose(original_weights, loaded_weights)

    def test_save_and_load_different_agent(self, model):
        """测试在不同 Agent 实例间保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "agent.pth")

            # 创建第一个 Agent 并保存
            agent1 = NeuralAgent(model, name="Agent1")
            agent1.save(save_path)

            # 创建第二个 Agent（新模型）
            model2 = create_model(
                obs_dim=384, action_size=50, encoder_type="mlp", config="small"
            )
            agent2 = NeuralAgent(model2, name="Agent2")

            # 加载第一个 Agent 的权重
            agent2.load(save_path)

            # 验证权重一致
            assert torch.allclose(
                agent1.model.encoder.fc1.weight, agent2.model.encoder.fc1.weight
            )

            # 验证名称已恢复
            assert agent2.get_name() == "Agent1"

    def test_get_name(self, agent):
        """测试获取名称"""
        assert agent.get_name() == "TestAgent"

    def test_str_and_repr(self, agent):
        """测试字符串表示"""
        str_repr = str(agent)
        assert "TestAgent" in str_repr
        assert "mlp" in str_repr

        repr_str = repr(agent)
        assert "NeuralAgent" in repr_str
        assert "TestAgent" in repr_str
        assert "mlp" in repr_str
        assert "params=" in repr_str


class TestNeuralAgentIntegration:
    """测试 NeuralAgent 与游戏集成"""

    def test_with_splendor_game(self):
        """测试 NeuralAgent 与 Splendor 游戏集成"""
        from games.splendor import SplendorGame
        from models.model_factory import create_splendor_model

        # 创建游戏和 Agent
        game = SplendorGame(num_players=2)
        model = create_splendor_model(encoder_type="mlp", config="small")
        agent = NeuralAgent(model, name="SplendorAgent")

        # 重置游戏
        state = game.reset()

        # 玩 10 步
        for step in range(10):
            # 获取当前玩家
            current_player = state.current_player

            # 获取观察和合法动作
            obs = game.state_to_observation(state, current_player)
            legal_actions_objects = game.get_legal_actions(state)
            legal_actions = [
                game.action_to_index(a, state) for a in legal_actions_objects
            ]

            # Agent 选择动作
            action_idx, info = agent.select_action(obs, legal_actions)

            # 验证观察维度
            assert obs.shape == (384,)

            # 验证动作合法
            assert action_idx in legal_actions

            # 验证 info 包含必要信息
            assert "log_prob" in info
            assert "value" in info

            # 执行动作
            action = game.index_to_action(action_idx, state)
            state, rewards, done, game_info = game.step(action)

            if done:
                break

    def test_multiple_agents_different_encoders(self):
        """测试多个不同编码器的 Agent"""
        from models.model_factory import create_model

        # 创建 MLP Agent
        mlp_model = create_model(obs_dim=384, action_size=50, encoder_type="mlp")
        mlp_agent = NeuralAgent(mlp_model, name="MLP_Agent")

        # 创建 Attention Agent
        attn_model = create_model(obs_dim=384, action_size=50, encoder_type="attention")
        attn_agent = NeuralAgent(attn_model, name="Attention_Agent")

        # 相同输入
        obs = np.random.randn(384)
        legal_actions = [0, 1, 2, 5, 10]

        # 两个 Agent 都能正常选择动作
        action1, info1 = mlp_agent.select_action(obs, legal_actions)
        action2, info2 = attn_agent.select_action(obs, legal_actions)

        assert action1 in legal_actions
        assert action2 in legal_actions

        # 由于模型不同，输出可能不同
        # 但都应该是合法的


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
