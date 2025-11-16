"""
可视化 Batch Size 差异

演示自对弈阶段（batch_size=1）和训练阶段（batch_size=256）的区别
"""

import time
import torch
import numpy as np
from games.registry import create_game
from models.model_factory import create_splendor_model
from agents.neural_agent import NeuralAgent


def demo_self_play_phase(device: str):
    """演示自对弈阶段（batch_size=1）"""
    print(f"\n{'='*60}")
    print(f"阶段 1: 自对弈（数据收集）- Batch Size = 1")
    print(f"设备: {device.upper()}")
    print(f"{'='*60}")

    # 创建模型和 Agent
    model = create_splendor_model(encoder_type='mlp', config='small')
    model.to(device)
    agent = NeuralAgent(model=model, device=device)

    # 创建游戏
    game = create_game('splendor', num_players=4)
    state = game.reset()

    # 模拟 10 步游戏
    num_steps = 10
    print(f"\n模拟 {num_steps} 步游戏...")

    start_time = time.time()

    for step in range(num_steps):
        current_player = game.get_current_player(state)

        # 获取观察和合法动作
        observation = game.state_to_observation(state, current_player)
        legal_actions = game.get_legal_actions(state)
        legal_action_indices = [
            game.action_to_index(a, legal_actions) for a in legal_actions
        ]

        # 🔴 关键：每次只处理 1 个观察
        print(f"  步骤 {step + 1}:")
        print(f"    观察 shape: {observation.shape}")  # (384,)

        # Agent 选择动作
        action_idx, info = agent.select_action(observation, legal_action_indices)

        # 执行动作
        action = game.index_to_action(action_idx, legal_actions)
        state, rewards, done, _ = game.step(action)

        if done:
            break

    elapsed_time = time.time() - start_time

    print(f"\n总结:")
    print(f"  总步数: {num_steps}")
    print(f"  总耗时: {elapsed_time * 1000:.2f}ms")
    print(f"  平均每步: {elapsed_time / num_steps * 1000:.3f}ms")
    print(f"  🔴 Batch Size: 1（每次只处理 1 个样本）")
    print(f"  ⏱️  占总训练时间: ~85%")


def demo_training_phase(device: str):
    """演示训练阶段（batch_size=256）"""
    print(f"\n{'='*60}")
    print(f"阶段 2: PPO 训练（梯度更新）- Batch Size = 256")
    print(f"设备: {device.upper()}")
    print(f"{'='*60}")

    # 创建模型
    model = create_splendor_model(encoder_type='mlp', config='small')
    model.to(device)
    model.train()

    # 模拟收集的经验数据
    batch_size = 256
    obs_dim = 384
    action_size = 50

    print(f"\n模拟 PPO 更新（4 个 epochs）...")

    # 生成模拟数据
    observations = torch.randn(batch_size, obs_dim, device=device)
    actions = torch.randint(0, action_size, (batch_size,), device=device)
    old_log_probs = torch.randn(batch_size, device=device)
    advantages = torch.randn(batch_size, device=device)
    returns = torch.randn(batch_size, device=device)

    print(f"\n数据 shapes:")
    print(f"  observations: {observations.shape}")  # (256, 384)
    print(f"  actions: {actions.shape}")            # (256,)
    print(f"  🟢 Batch Size: 256（批量处理 256 个样本）")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    start_time = time.time()

    num_epochs = 4
    for epoch in range(num_epochs):
        # 前向传播
        legal_mask = torch.ones(batch_size, action_size, dtype=torch.bool, device=device)
        new_log_probs, entropy, new_values = model.evaluate_actions(
            observations, actions, legal_mask
        )

        # 计算损失（简化版）
        ratio = torch.exp(new_log_probs - old_log_probs)
        policy_loss = -torch.min(ratio * advantages, ratio * advantages).mean()
        value_loss = 0.5 * ((new_values.squeeze() - returns) ** 2).mean()
        entropy_loss = -entropy.mean()
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"  Epoch {epoch + 1}/{num_epochs}: loss = {total_loss.item():.4f}")

    elapsed_time = time.time() - start_time

    print(f"\n总结:")
    print(f"  Epochs: {num_epochs}")
    print(f"  总耗时: {elapsed_time * 1000:.2f}ms")
    print(f"  平均每 epoch: {elapsed_time / num_epochs * 1000:.3f}ms")
    print(f"  🟢 Batch Size: 256（GPU 可以发挥并行优势）")
    print(f"  ⏱️  占总训练时间: ~15%")


def compare_phases(device: str):
    """对比两个阶段"""
    print(f"\n{'='*60}")
    print(f"性能对比总结")
    print(f"{'='*60}")

    # 阶段 1: 自对弈
    model = create_splendor_model(encoder_type='mlp', config='small')
    model.to(device)
    agent = NeuralAgent(model=model, device=device)

    game = create_game('splendor', num_players=4)
    state = game.reset()

    # 测试 100 步
    num_steps = 100
    start_time = time.time()

    for _ in range(num_steps):
        current_player = game.get_current_player(state)
        observation = game.state_to_observation(state, current_player)
        legal_actions = game.get_legal_actions(state)
        legal_action_indices = [
            game.action_to_index(a, legal_actions) for a in legal_actions
        ]
        action_idx, _ = agent.select_action(observation, legal_action_indices)
        action = game.index_to_action(action_idx, legal_actions)
        state, _, done, _ = game.step(action)
        if done:
            state = game.reset()

    selfplay_time = time.time() - start_time

    # 阶段 2: 训练
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    batch_size = 256
    obs_dim = 384
    action_size = 50

    observations = torch.randn(batch_size, obs_dim, device=device)
    actions = torch.randint(0, action_size, (batch_size,), device=device)
    legal_mask = torch.ones(batch_size, action_size, dtype=torch.bool, device=device)
    returns = torch.randn(batch_size, device=device)
    advantages = torch.randn(batch_size, device=device)
    old_log_probs = torch.randn(batch_size, device=device)

    # 测试 10 次更新
    num_updates = 10
    start_time = time.time()

    for _ in range(num_updates):
        new_log_probs, entropy, new_values = model.evaluate_actions(
            observations, actions, legal_mask
        )
        ratio = torch.exp(new_log_probs - old_log_probs)
        policy_loss = -(ratio * advantages).mean()
        value_loss = 0.5 * ((new_values.squeeze() - returns) ** 2).mean()
        total_loss = policy_loss + value_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    training_time = time.time() - start_time

    # 计算实际训练时间分配
    # 假设收集 50 episodes，每个 150 steps = 7500 steps
    # PPO 更新: 7500 / 256 * 4 epochs ≈ 117 次
    estimated_selfplay = selfplay_time / num_steps * 7500
    estimated_training = training_time / num_updates * 117

    total_time = estimated_selfplay + estimated_training
    selfplay_pct = estimated_selfplay / total_time * 100
    training_pct = estimated_training / total_time * 100

    print(f"\n实际测量:")
    print(f"  自对弈 ({num_steps} 步): {selfplay_time * 1000:.2f}ms")
    print(f"  训练 ({num_updates} 次更新): {training_time * 1000:.2f}ms")

    print(f"\n预估完整训练时间分配:")
    print(f"  自对弈阶段 (7500 steps, batch_size=1):")
    print(f"    耗时: {estimated_selfplay:.2f}s ({selfplay_pct:.1f}%)")
    print(f"    🔴 Batch Size: 1")

    print(f"\n  PPO 训练阶段 (117 次更新, batch_size=256):")
    print(f"    耗时: {estimated_training:.2f}s ({training_pct:.1f}%)")
    print(f"    🟢 Batch Size: 256")

    print(f"\n  总时间: {total_time:.2f}s")

    print(f"\n结论:")
    if selfplay_pct > 70:
        print(f"  ⚠️  自对弈占 {selfplay_pct:.1f}%，是性能瓶颈！")
        print(f"  ⚠️  由于自对弈 batch_size=1，GPU 无法发挥优势")
    else:
        print(f"  ✅ 训练阶段占比较高，GPU 可能有优势")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Batch Size 差异演示")
    print("="*60)

    # 检查设备
    has_mps = torch.backends.mps.is_available()
    device = "mps" if has_mps else "cpu"

    print(f"\n使用设备: {device.upper()}")

    # 演示阶段 1: 自对弈
    demo_self_play_phase(device)

    # 演示阶段 2: 训练
    demo_training_phase(device)

    # 对比
    compare_phases(device)

    # 总结
    print(f"\n{'='*60}")
    print(f"关键理解")
    print(f"{'='*60}")

    print(f"""
1. 配置文件中的 minibatch_size: 256
   ✅ 用于 PPO 训练阶段
   ✅ 占总时间 ~15%
   ✅ GPU 在这个阶段有优势

2. 自对弈阶段的 batch_size: 1
   ⚠️  用于收集训练数据
   ⚠️  占总时间 ~85%（性能瓶颈）
   ⚠️  GPU 在这个阶段没优势（batch 太小）

3. 为什么 MPS 比 CPU 慢？
   ⚠️  因为 85% 的时间在自对弈（batch_size=1）
   ⚠️  MPS 的数据传输开销 > 计算收益
   ⚠️  训练阶段虽然 GPU 快，但占比太小

4. 解决方案：
   ✅ 使用 CPU 训练（当前最佳选择）
   或
   ⚡ 实现批量自对弈（让自对弈也用 batch_size=64+）
   或
   📈 使用更大的模型（增加计算量使 GPU 值得）
    """)

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
