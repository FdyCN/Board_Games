"""
测试核心类型和辅助函数
"""

import numpy as np
import pytest

from core.types import (
    ActionInfo,
    AgentType,
    Episode,
    EvaluationResult,
    Experience,
    GameResult,
    TrainingMetrics,
    compute_gae,
    compute_returns,
    normalize_rewards,
)


def test_experience_creation():
    """测试 Experience 创建"""
    obs = np.array([1.0, 2.0, 3.0])
    exp = Experience(
        player_id=0,
        observation=obs,
        action=5,
        reward=1.0,
        done=False,
        log_prob=-0.5,
        value=0.8,
    )

    assert exp.player_id == 0
    assert np.array_equal(exp.observation, obs)
    assert exp.action == 5
    assert exp.reward == 1.0
    assert exp.done is False
    assert exp.log_prob == -0.5
    assert exp.value == 0.8


def test_experience_validation():
    """测试 Experience 数据验证"""
    # 错误的 observation 类型
    with pytest.raises(AssertionError):
        Experience(
            player_id=0,
            observation=[1, 2, 3],  # 应该是 np.ndarray
            action=0,
            reward=0.0,
        )


def test_episode_creation():
    """测试 Episode 创建"""
    experiences = [
        Experience(player_id=0, observation=np.array([1.0]), action=0, reward=0.5, done=False),
        Experience(player_id=1, observation=np.array([2.0]), action=1, reward=0.3, done=True),
    ]

    episode = Episode(
        experiences=experiences,
        total_rewards=[1.0, 0.0],
        winner=0,
        num_steps=2,
    )

    assert len(episode.experiences) == 2
    assert episode.winner == 0
    assert episode.num_steps == 2


def test_episode_get_player_experiences():
    """测试获取玩家经验"""
    experiences = [
        Experience(player_id=0, observation=np.array([1.0]), action=0, reward=0.5),
        Experience(player_id=1, observation=np.array([2.0]), action=1, reward=0.3),
        Experience(player_id=0, observation=np.array([3.0]), action=2, reward=0.2),
    ]

    episode = Episode(experiences=experiences, total_rewards=[0.7, 0.3], winner=0, num_steps=3)

    player0_exp = episode.get_player_experiences(0)
    assert len(player0_exp) == 2
    assert player0_exp[0].reward == 0.5
    assert player0_exp[1].reward == 0.2


def test_action_info():
    """测试 ActionInfo"""
    policy = np.array([0.2, 0.5, 0.3])
    info = ActionInfo(action_index=1, log_prob=-0.693, value=0.75, policy=policy, entropy=1.03)

    assert info.action_index == 1
    assert info.value == 0.75
    assert np.array_equal(info.policy, policy)


def test_training_metrics():
    """测试 TrainingMetrics"""
    metrics = TrainingMetrics(
        episode=100,
        policy_loss=0.5,
        value_loss=0.3,
        entropy=1.5,
        mean_reward=10.0,
        mean_episode_length=50.0,
        learning_rate=3e-4,
        explained_variance=0.8,
    )

    # 测试转换为字典
    metrics_dict = metrics.to_dict()
    assert metrics_dict["episode"] == 100
    assert metrics_dict["policy_loss"] == 0.5
    assert metrics_dict["mean_reward"] == 10.0


def test_evaluation_result():
    """测试 EvaluationResult"""
    result = EvaluationResult(
        win_rate=0.75, mean_reward=12.5, mean_rank=1.2, games_played=100, elo_rating=1600
    )

    assert result.win_rate == 0.75
    assert result.games_played == 100
    assert result.elo_rating == 1600


def test_normalize_rewards():
    """测试奖励归一化"""
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    normalized = normalize_rewards(rewards)

    # 检查均值约为 0，标准差约为 1
    assert abs(np.mean(normalized)) < 1e-6
    assert abs(np.std(normalized) - 1.0) < 1e-6


def test_compute_returns():
    """测试计算折扣回报"""
    rewards = [1.0, 1.0, 1.0]
    gamma = 0.9

    returns = compute_returns(rewards, gamma=gamma, normalize=False)

    # R[0] = 1 + 0.9 * 1 + 0.9^2 * 1 = 2.71
    # R[1] = 1 + 0.9 * 1 = 1.9
    # R[2] = 1
    expected = [2.71, 1.9, 1.0]

    assert len(returns) == 3
    for r, exp in zip(returns, expected):
        assert abs(r - exp) < 1e-6


def test_compute_gae():
    """测试 GAE 计算"""
    rewards = [1.0, 1.0, 0.0]
    values = [0.5, 0.8, 0.3]
    next_values = [0.8, 0.3, 0.0]
    dones = [False, False, True]

    advantages, returns = compute_gae(
        rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95
    )

    assert len(advantages) == 3
    assert len(returns) == 3

    # 检查回报 = 优势 + 价值
    for i in range(3):
        assert abs(returns[i] - (advantages[i] + values[i])) < 1e-6


def test_game_result_enum():
    """测试 GameResult 枚举"""
    assert GameResult.WIN.value == 1.0
    assert GameResult.LOSS.value == 0.0
    assert GameResult.DRAW.value == 0.5


def test_agent_type_enum():
    """测试 AgentType 枚举"""
    assert AgentType.RANDOM.value == "random"
    assert AgentType.NEURAL.value == "neural"
    assert AgentType.HUMAN.value == "human"
    assert AgentType.MCTS.value == "mcts"
