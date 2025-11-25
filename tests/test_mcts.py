#!/usr/bin/env python3
"""
MCTS 单元测试

测试 MCTS 模块的基本功能。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from games.registry import create_game
from models.model_factory import create_model
from training.mcts import MCTS, MCTSNode, MCTSScheduler


def test_mcts_node():
    """测试 MCTS 节点"""
    print("=" * 60)
    print("测试 MCTS 节点")
    print("=" * 60)

    # 创建根节点
    root = MCTSNode(state="root_state")
    print(f"根节点: {root}")

    # 测试扩展
    action_probs = np.array([0.3, 0.5, 0.2])
    legal_actions = ["action_0", "action_1", "action_2"]
    root.expand(action_probs, legal_actions)

    print(f"扩展后子节点数: {len(root.children)}")
    for idx, child in root.children.items():
        print(f"  子节点 {idx}: prior={child.prior:.3f}")

    # 测试 UCB 选择
    selected = root.select_child(c_puct=1.5)
    print(f"选择的子节点: {selected.action_index}, prior={selected.prior:.3f}")

    # 测试更新
    root.children[0].update(1.0)
    root.children[0].update(0.5)
    root.children[1].update(-1.0)

    print(f"\n更新后的统计:")
    for idx, child in root.children.items():
        print(f"  子节点 {idx}: visits={child.visit_count}, q={child.q_value():.3f}")

    # 测试回溯
    root.children[0].backpropagate(1.0)
    print(f"\n回溯后根节点访问次数: {root.visit_count}")

    # 测试动作概率
    action_probs = root.get_action_probs(temperature=1.0)
    print(f"动作概率分布: {action_probs}")

    print("✅ MCTS 节点测试通过\n")


def test_mcts_scheduler():
    """测试 MCTS 调度器"""
    print("=" * 60)
    print("测试 MCTS 调度器")
    print("=" * 60)

    # 创建调度器
    schedule = [(0, 0), (200, 50), (500, 100), (800, 200)]
    scheduler = MCTSScheduler(schedule)

    print(scheduler.get_schedule_info())

    # 测试不同迭代次数
    test_iterations = [0, 100, 250, 600, 1000]
    for iteration in test_iterations:
        sims = scheduler.get_simulations(iteration)
        print(f"迭代 {iteration}: {sims} 次模拟")

    print("✅ MCTS 调度器测试通过\n")


def test_mcts_search():
    """测试 MCTS 搜索"""
    print("=" * 60)
    print("测试 MCTS 搜索")
    print("=" * 60)

    # 创建游戏
    print("创建 Splendor 游戏...")
    game = create_game("splendor", num_players=4)
    state = game.reset()

    # 创建模型
    print("创建神经网络模型...")
    model = create_model(
        obs_dim=game.observation_shape[0],
        action_size=game.action_space_size,
        encoder_type="mlp",
        config="small",  # 使用小模型加快测试
    )

    # 创建 MCTS
    print("创建 MCTS 搜索引擎...")
    mcts = MCTS(
        game=game,
        c_puct=1.5,
        add_noise=True,
        device="cpu",
    )

    # 执行搜索
    print("\n执行 MCTS 搜索 (20 次模拟)...")
    action_probs, legal_actions = mcts.search(
        state=state,
        model=model,
        num_simulations=20,  # 少量模拟以加快测试
        temperature=1.0,
        verbose=True,
    )

    print(f"\n合法动作数: {len(legal_actions)}")
    print(f"动作概率和: {np.sum(action_probs):.6f}")
    print(f"最高概率: {np.max(action_probs):.4f}")
    print(f"最低概率: {np.min(action_probs):.4f}")

    # 测试确定性采样
    print("\n测试确定性采样 (temperature=0)...")
    action_probs_det, _ = mcts.search(
        state=state,
        model=model,
        num_simulations=20,
        temperature=0.0,  # 确定性
        verbose=False,
    )

    selected_action_count = np.sum(action_probs_det > 0.5)
    print(f"确定性选择的动作数: {selected_action_count} (应该是 1)")

    print("✅ MCTS 搜索测试通过\n")


def test_mcts_game_simulation():
    """测试 MCTS 完整游戏模拟"""
    print("=" * 60)
    print("测试 MCTS 完整游戏模拟")
    print("=" * 60)

    # 创建游戏和模型
    game = create_game("splendor", num_players=4)
    model = create_model(
        obs_dim=game.observation_shape[0],
        action_size=game.action_space_size,
        encoder_type="mlp",
        config="small",
    )

    # 创建 MCTS
    mcts = MCTS(game=game, c_puct=1.5, add_noise=True, device="cpu")

    # 模拟一局游戏
    state = game.reset()
    step_count = 0
    max_steps = 50  # 限制最大步数以加快测试

    print("开始游戏模拟...")
    while not game.is_terminal(state) and step_count < max_steps:
        # MCTS 搜索
        action_probs, legal_actions = mcts.search(
            state=state,
            model=model,
            num_simulations=10,  # 少量模拟
            temperature=1.0,
            verbose=False,
        )

        # 采样动作
        action_idx = np.random.choice(len(legal_actions), p=action_probs)
        action = legal_actions[action_idx]

        # 执行动作
        state, rewards, done, info = game.step(action)
        step_count += 1

        if step_count % 10 == 0:
            print(f"  步数: {step_count}")

    print(f"\n游戏结束:")
    print(f"  总步数: {step_count}")
    print(f"  是否终局: {game.is_terminal(state)}")
    if game.is_terminal(state):
        results = game.get_result(state)
        winner = np.argmax(results)
        print(f"  胜者: 玩家 {winner}")

    print("✅ MCTS 游戏模拟测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("MCTS 单元测试")
    print("=" * 60 + "\n")

    try:
        # 1. 测试 MCTS 节点
        test_mcts_node()

        # 2. 测试调度器
        test_mcts_scheduler()

        # 3. 测试 MCTS 搜索
        test_mcts_search()

        # 4. 测试完整游戏模拟
        test_mcts_game_simulation()

        print("=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
