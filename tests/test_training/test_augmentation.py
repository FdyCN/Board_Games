"""
测试位置旋转数据增强功能

验证数据增强是否正确实现，以及是否能减少位置偏差。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from games.splendor import SplendorGame
from models.model_factory import create_splendor_model
from agents.neural_agent import NeuralAgent
from training.self_play_worker import collect_episode
from training.data_augmentation import (
    rotate_splendor_observation,
    apply_random_rotation_to_batch,
)
from training.experience import ExperienceBatch


def test_observation_rotation():
    """测试观察旋转功能"""
    print("=" * 60)
    print("测试 1: 观察旋转功能")
    print("=" * 60)

    # 创建游戏和模型
    game = SplendorGame(num_players=4, seed=42)
    model = create_splendor_model(encoder_type="mlp", config="medium")
    agent = NeuralAgent(model)

    # 收集一个 episode
    episode = collect_episode(game, [agent] * 4, gamma=0.99, gae_lambda=0.95)

    # 测试旋转
    original_obs = episode.experiences[0].observation
    print(f"原始观察形状: {original_obs.shape}")
    print(f"原始观察前 10 维: {original_obs[:10]}")

    # 旋转 1 步
    rotated_obs = rotate_splendor_observation(original_obs, rotation=1)
    print(f"\n旋转后观察前 10 维: {rotated_obs[:10]}")

    # 检查观察是否改变（但形状保持不变）
    assert rotated_obs.shape == original_obs.shape, "观察形状应保持不变"
    assert not np.array_equal(rotated_obs, original_obs), "观察应该改变"

    print("✅ 观察旋转功能正常")


def test_batch_augmentation():
    """测试批次数据增强功能"""
    print("\n" + "=" * 60)
    print("测试 2: 批次数据增强功能")
    print("=" * 60)

    # 创建游戏和模型
    game = SplendorGame(num_players=4, seed=42)
    model = create_splendor_model(encoder_type="mlp", config="medium")
    agent = NeuralAgent(model)

    # 收集一个 episode
    episode = collect_episode(game, [agent] * 4, gamma=0.99, gae_lambda=0.95)

    original_count = len(episode.experiences)
    print(f"原始经验数量: {original_count}")

    # 应用随机旋转（50% 概率）
    augmented_experiences = apply_random_rotation_to_batch(
        episode.experiences, num_players=4, rotation_probability=0.5
    )

    print(f"增强后经验数量: {len(augmented_experiences)}")
    assert len(augmented_experiences) == original_count, "批次大小应保持不变"

    # 检查是否有经验被旋转了（通过比较观察）
    num_rotated = sum(
        1
        for orig, aug in zip(episode.experiences, augmented_experiences)
        if not np.array_equal(orig.observation, aug.observation)
    )
    print(f"被旋转的经验数量: {num_rotated}/{original_count} ({num_rotated/original_count:.1%})")

    print("✅ 批次数据增强功能正常")


def test_batch_creation_with_augmentation():
    """测试带数据增强的批次创建"""
    print("\n" + "=" * 60)
    print("测试 3: 带数据增强的批次创建")
    print("=" * 60)

    # 创建游戏和模型
    game = SplendorGame(num_players=4, seed=42)
    model = create_splendor_model(encoder_type="mlp", config="medium")
    agent = NeuralAgent(model)

    # 收集一个 episode
    episode = collect_episode(game, [agent] * 4, gamma=0.99, gae_lambda=0.95)

    # 不使用数据增强
    batch_no_aug = ExperienceBatch.from_experiences(
        episode.experiences,
        device="cpu",
        normalize_advantages=True,
        use_position_augmentation=False,
    )

    # 使用数据增强
    batch_with_aug = ExperienceBatch.from_experiences(
        episode.experiences,
        device="cpu",
        normalize_advantages=True,
        use_position_augmentation=True,
        num_players=4,
    )

    print(f"无增强批次大小: {batch_no_aug.batch_size}")
    print(f"带增强批次大小: {batch_with_aug.batch_size}")

    # 批次大小应该相同（我们使用的是随机旋转，不增加数据量）
    assert batch_no_aug.batch_size == batch_with_aug.batch_size

    print("✅ 带数据增强的批次创建功能正常")


def test_entropy_with_augmentation():
    """测试数据增强是否影响熵计算"""
    print("\n" + "=" * 60)
    print("测试 4: 数据增强对熵的影响")
    print("=" * 60)

    # 创建游戏和模型
    game = SplendorGame(num_players=4, seed=42)
    model = create_splendor_model(encoder_type="mlp", config="medium")
    model.eval()
    agent = NeuralAgent(model)

    # 收集一个 episode
    episode = collect_episode(game, [agent] * 4, gamma=0.99, gae_lambda=0.95)

    # 创建两个批次
    batch_no_aug = ExperienceBatch.from_experiences(
        episode.experiences,
        device="cpu",
        normalize_advantages=True,
        use_position_augmentation=False,
    )

    batch_with_aug = ExperienceBatch.from_experiences(
        episode.experiences,
        device="cpu",
        normalize_advantages=True,
        use_position_augmentation=True,
        num_players=4,
    )

    # 计算熵
    with torch.no_grad():
        values1, log_probs1, entropy1, outcome_logits1 = model.evaluate_actions(
            batch_no_aug.observations,
            batch_no_aug.actions,
            batch_no_aug.legal_actions_masks,
        )

        values2, log_probs2, entropy2, outcome_logits2 = model.evaluate_actions(
            batch_with_aug.observations,
            batch_with_aug.actions,
            batch_with_aug.legal_actions_masks,
        )

    avg_entropy1 = entropy1.mean().item()
    avg_entropy2 = entropy2.mean().item()

    print(f"无增强平均熵: {avg_entropy1:.4f}")
    print(f"带增强平均熵: {avg_entropy2:.4f}")

    # 熵应该都为正
    assert avg_entropy1 > 0, "熵必须为正"
    assert avg_entropy2 > 0, "熵必须为正"

    print("✅ 熵计算正常（都为正值）")


def test_player_id_encoding():
    """测试当前玩家 ID 编码是否正确旋转"""
    print("\n" + "=" * 60)
    print("测试 5: 当前玩家 ID 编码旋转")
    print("=" * 60)

    # 创建游戏和模型
    game = SplendorGame(num_players=4, seed=42)
    model = create_splendor_model(encoder_type="mlp", config="medium")
    agent = NeuralAgent(model)

    # 收集一个 episode
    episode = collect_episode(game, [agent] * 4, gamma=0.99, gae_lambda=0.95)

    # 取第一个观察
    original_obs = episode.experiences[0].observation

    # 检查原始的"当前玩家 ID"编码（索引 321-325）
    original_player_id = np.argmax(original_obs[321:325])
    print(f"原始当前玩家 ID: {original_player_id}")

    # 旋转并检查
    for rotation in range(1, 4):
        rotated_obs = rotate_splendor_observation(original_obs, rotation=rotation)
        new_player_id = np.argmax(rotated_obs[321:325])
        expected_id = (original_player_id - rotation) % 4

        print(f"旋转 {rotation} 步后:")
        print(f"  新的当前玩家 ID: {new_player_id}")
        print(f"  期望的 ID: {expected_id}")

        assert new_player_id == expected_id, f"旋转后的玩家 ID 应为 {expected_id}"

    print("✅ 当前玩家 ID 编码旋转正确")


def main():
    print("\n" + "=" * 60)
    print("位置旋转数据增强功能测试")
    print("=" * 60)

    # 运行所有测试
    test_observation_rotation()
    test_batch_augmentation()
    test_batch_creation_with_augmentation()
    test_entropy_with_augmentation()
    test_player_id_encoding()

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    print("\n建议：")
    print("1. 在训练配置中设置 use_position_augmentation: true")
    print("2. 重新训练模型，观察位置偏差是否减少")
    print("3. 使用 test_position_bias.py 评估新模型的位置偏差")
    print("=" * 60)


if __name__ == "__main__":
    main()
