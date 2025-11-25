#!/usr/bin/env python3
"""
快速测试 MCTS 训练流程
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from games.registry import create_game
from models.model_factory import create_model
from training.trainer import Trainer

print("=" * 60)
print("MCTS 训练流程集成测试")
print("=" * 60)

# 创建游戏
print("\n1. 创建游戏...")
game = create_game("splendor", num_players=4, reward_config={
    "take_gem": 0.0,
    "buy_card_points": 0.0,
    "win": 1.0,
})
print("   ✅ 游戏创建成功")

# 创建模型
print("\n2. 创建模型...")
model = create_model(
    obs_dim=game.observation_shape[0],
    action_size=game.action_space_size,
    encoder_type="mlp",
    config="small",
)
print("   ✅ 模型创建成功")

# 创建 Trainer (启用 MCTS)
print("\n3. 创建 Trainer (MCTS 模式)...")
trainer = Trainer(
    game=game,
    model=model,
    learning_rate=0.0001,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.1,
    value_coef=0.5,
    entropy_coef=0.01,
    device="cpu",
    verbose=False,
    use_mcts=True,
    mcts_simulations=10,  # 少量模拟以加快测试
    mcts_c_puct=1.5,
    mcts_add_noise=True,
    mcts_temperature=1.0,
)
print("   ✅ Trainer 创建成功")

# 训练 1 次迭代
print("\n4. 测试训练 1 次迭代...")
try:
    metrics = trainer.train(
        num_iterations=1,
        episodes_per_iteration=2,  # 只收集 2 个 episode
        update_epochs=1,
        minibatch_size=64,
        checkpoint_interval=100,
        log_interval=1,
        use_position_augmentation=False,
    )
    print("   ✅ 训练成功完成")
    print(f"   - Policy Loss: {metrics[0].policy_loss:.4f}")
    print(f"   - Value Loss: {metrics[0].value_loss:.4f}")
    print(f"   - Mean Reward: {metrics[0].mean_reward:.4f}")
except Exception as e:
    print(f"   ❌ 训练失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有集成测试通过!")
print("=" * 60)
print("\n你现在可以开始完整训练:")
print("  ./scripts/train_mcts.sh")
