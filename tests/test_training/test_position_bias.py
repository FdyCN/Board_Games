"""
测试位置偏差 (Position Bias)

评估模型在不同玩家位置上的表现差异。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from typing import List, Dict
from collections import defaultdict

from games.splendor import SplendorGame
from models.model_factory import create_splendor_model
from agents.neural_agent import NeuralAgent
from agents.random_agent import RandomAgent
from training.self_play_worker import collect_episode


def test_position_bias(
    checkpoint_path: str,
    num_games: int = 100,
    encoder_type: str = "mlp",
    config: str = "medium",
    device: str = "cpu",
) -> Dict[int, Dict[str, float]]:
    """
    测试模型在不同位置的表现

    策略：
    - 将训练好的模型放在每个位置（0, 1, 2, 3）
    - 其他位置使用随机 agent
    - 测量每个位置的胜率

    Args:
        checkpoint_path: 检查点路径
        num_games: 每个位置测试的游戏数量
        encoder_type: 编码器类型
        config: 模型配置
        device: 设备

    Returns:
        每个位置的统计信息
    """
    print("=" * 60)
    print("位置偏差测试")
    print("=" * 60)
    print(f"检查点: {checkpoint_path}")
    print(f"每个位置测试: {num_games} 局")
    print("=" * 60)

    # 创建游戏
    game = SplendorGame(num_players=4)

    # 加载模型
    print("\n加载模型...")
    model = create_splendor_model(encoder_type=encoder_type, config=config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ 模型加载成功 (迭代: {checkpoint.get('iteration', 'unknown')})")

    # 统计结果
    position_stats = defaultdict(lambda: {"wins": 0, "rewards": [], "ranks": []})

    # 测试每个位置
    for test_position in range(4):
        print(f"\n{'='*60}")
        print(f"测试位置 {test_position}...")
        print(f"{'='*60}")

        # 创建 agents（测试位置使用神经网络，其他使用随机）
        agents = []
        for pos in range(4):
            if pos == test_position:
                agents.append(NeuralAgent(model, device=device))
            else:
                agents.append(RandomAgent())

        # 收集游戏数据
        wins = 0
        total_reward = 0.0
        ranks = []

        for game_idx in range(num_games):
            # 收集一局游戏
            episode = collect_episode(
                game=game,
                agents=agents,
                gamma=0.99,
                gae_lambda=0.95,
                deterministic=False,  # 使用随机策略以获得更稳定的评估
                verbose=False,
            )

            # 记录结果
            reward = episode.total_rewards[test_position]
            total_reward += reward

            # 计算排名（分数越高排名越好）
            scores = episode.total_rewards
            rank = sum(1 for s in scores if s > reward) + 1
            ranks.append(rank)

            # 记录胜利
            if episode.winner == test_position:
                wins += 1

            # 进度显示
            if (game_idx + 1) % 20 == 0:
                current_win_rate = wins / (game_idx + 1)
                print(
                    f"  进度: {game_idx + 1}/{num_games}, "
                    f"当前胜率: {current_win_rate:.2%}"
                )

        # 计算统计数据
        win_rate = wins / num_games
        avg_reward = total_reward / num_games
        avg_rank = np.mean(ranks)

        position_stats[test_position] = {
            "wins": wins,
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_rank": avg_rank,
            "games": num_games,
        }

        print(f"\n位置 {test_position} 结果:")
        print(f"  胜率: {win_rate:.2%} ({wins}/{num_games})")
        print(f"  平均奖励: {avg_reward:.3f}")
        print(f"  平均排名: {avg_rank:.2f} (1 = 最好)")

    return position_stats


def analyze_position_bias(stats: Dict[int, Dict[str, float]]) -> None:
    """
    分析位置偏差

    Args:
        stats: 位置统计信息
    """
    print("\n" + "=" * 60)
    print("位置偏差分析")
    print("=" * 60)

    # 提取胜率和排名
    positions = sorted(stats.keys())
    win_rates = [stats[pos]["win_rate"] for pos in positions]
    avg_ranks = [stats[pos]["avg_rank"] for pos in positions]
    avg_rewards = [stats[pos]["avg_reward"] for pos in positions]

    # 打印汇总表格
    print("\n位置性能汇总:")
    print("-" * 60)
    print(f"{'位置':<8} {'胜率':<12} {'平均奖励':<12} {'平均排名':<12}")
    print("-" * 60)
    for pos in positions:
        print(
            f"{pos:<8} "
            f"{stats[pos]['win_rate']:<12.2%} "
            f"{stats[pos]['avg_reward']:<12.3f} "
            f"{stats[pos]['avg_rank']:<12.2f}"
        )
    print("-" * 60)

    # 计算偏差指标
    win_rate_std = np.std(win_rates)
    win_rate_range = max(win_rates) - min(win_rates)
    rank_std = np.std(avg_ranks)

    print("\n偏差指标:")
    print(f"  胜率标准差: {win_rate_std:.4f}")
    print(f"  胜率范围: {win_rate_range:.4f} ({min(win_rates):.2%} - {max(win_rates):.2%})")
    print(f"  排名标准差: {rank_std:.4f}")

    # 判断偏差程度
    print("\n偏差评估:")
    if win_rate_range < 0.10:
        print("  ✅ 低偏差 (<10% 差异) - 模型在各位置表现均衡")
    elif win_rate_range < 0.20:
        print("  ⚠️  中等偏差 (10-20% 差异) - 可能存在轻微位置过拟合")
    else:
        print("  ❌ 高偏差 (>20% 差异) - 模型明显过拟合到特定位置")

    # 识别最强和最弱位置
    best_pos = positions[np.argmax(win_rates)]
    worst_pos = positions[np.argmin(win_rates)]

    print(f"\n  最强位置: {best_pos} (胜率: {stats[best_pos]['win_rate']:.2%})")
    print(f"  最弱位置: {worst_pos} (胜率: {stats[worst_pos]['win_rate']:.2%})")

    # 与基线比较
    baseline_win_rate = 0.25  # 随机情况下期望胜率（4人游戏）
    avg_win_rate = np.mean(win_rates)

    print(f"\n  平均胜率: {avg_win_rate:.2%}")
    print(f"  基线胜率: {baseline_win_rate:.2%} (随机)")
    print(f"  相对提升: {(avg_win_rate / baseline_win_rate - 1):.1%}")

    # 给出建议
    print("\n建议:")
    if win_rate_range > 0.15:
        print("  1. 考虑从观察中移除或混淆位置信息")
        print("  2. 增加训练时的位置轮换")
        print("  3. 使用数据增强（随机交换玩家顺序）")
    elif avg_win_rate < 0.30:
        print("  1. 模型整体表现较弱，建议继续训练")
        print("  2. 可能需要增加模型容量或调整超参数")
    else:
        print("  1. 模型表现良好且位置偏差可接受")
        print("  2. 可以继续训练以进一步提升性能")

    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="测试位置偏差")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型检查点路径",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="每个位置测试的游戏数量 (默认: 100)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="mlp",
        choices=["mlp", "attention"],
        help="编码器类型 (默认: mlp)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="模型配置 (默认: medium)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="设备 (默认: cpu)",
    )

    args = parser.parse_args()

    # 运行测试
    stats = test_position_bias(
        checkpoint_path=args.checkpoint,
        num_games=args.num_games,
        encoder_type=args.encoder,
        config=args.config,
        device=args.device,
    )

    # 分析结果
    analyze_position_bias(stats)


if __name__ == "__main__":
    main()
