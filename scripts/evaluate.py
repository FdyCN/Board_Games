#!/usr/bin/env python3
"""
模型评估脚本

用于评估训练好的模型，支持多种评估模式：
- 单模型 vs 随机 Agent
- 多模型锦标赛
- 模型进化曲线评估

用法:
    # 评估单个模型 vs 随机 Agent
    python scripts/evaluate.py --model data/splendor/checkpoints/mlp_medium_3p_v1/latest.pth --mode vs_random --games 100

    # 多个模型锦标赛
    python scripts/evaluate.py --models model1.pth model2.pth model3.pth --mode tournament --games 200

    # 评估模型进化
    python scripts/evaluate.py --checkpoints data/splendor/checkpoints/mlp_medium_3p_v1/ --mode evolution --games 50
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from games.registry import create_game
from models.model_factory import create_model
from agents.neural_agent import NeuralAgent
from agents.random_agent import RandomAgent
from evaluation.arena import Arena
from evaluation.elo_system import EloSystem


def load_model_agent(
    checkpoint_path: str,
    game,
    agent_name: str,
    device: str = "cpu",
) -> NeuralAgent:
    """
    从检查点加载模型 Agent

    Args:
        checkpoint_path: 检查点路径
        game: 游戏实例
        agent_name: Agent 名称
        device: 设备

    Returns:
        NeuralAgent 实例
    """
    print(f"加载模型: {checkpoint_path}")

    # 创建模型
    model = create_model(
        obs_dim=game.observation_shape[0],
        action_size=game.action_space_size,
        encoder_type="attention",  # 默认使用 attention
        config="medium",
    )

    # 创建 Agent
    agent = NeuralAgent(
        model=model,
        player_id=0,  # 临时 ID，稍后会被 Arena 覆盖
        device=device,
    )

    # 加载检查点
    agent.load(checkpoint_path)
    agent.set_training_mode(False)  # 评估模式

    print(f"  模型已加载: {agent_name}")

    return agent


def evaluate_vs_random(
    model_path: str,
    game_name: str = "splendor",
    num_players: int = 4,
    num_games: int = 100,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """
    评估模型 vs 随机 Agent

    Args:
        model_path: 模型检查点路径
        game_name: 游戏名称
        num_players: 玩家数
        num_games: 游戏局数
        device: 设备
        verbose: 是否显示详细信息

    Returns:
        评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"评估模式: 模型 vs 随机 Agent")
    print(f"{'='*60}")

    # 创建游戏
    game = create_game(game_name, num_players=num_players)

    # 创建 Agent
    agents = []
    agent_names = []

    # 加载训练好的模型
    model_agent = load_model_agent(model_path, game, "Trained Model", device)
    agents.append(model_agent)
    agent_names.append("Trained Model")

    # 其他位置使用随机 Agent
    for i in range(1, num_players):
        agents.append(RandomAgent(player_id=i))
        agent_names.append(f"Random {i}")

    # 创建 Arena
    arena = Arena(game, agents, agent_names, use_elo=True)

    # 运行游戏
    results = arena.run_games(
        num_games=num_games,
        shuffle_positions=True,  # 随机打乱位置，确保公平
        verbose=verbose,
    )

    # 获取统计
    stats = arena.get_statistics()

    return {
        "mode": "vs_random",
        "model_path": model_path,
        "num_games": num_games,
        "results": results,
        "statistics": stats,
    }


def evaluate_tournament(
    model_paths: List[str],
    game_name: str = "splendor",
    num_players: int = 4,
    num_games: int = 200,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """
    评估多个模型的锦标赛

    Args:
        model_paths: 模型检查点路径列表
        game_name: 游戏名称
        num_players: 玩家数
        num_games: 游戏局数
        device: 设备
        verbose: 是否显示详细信息

    Returns:
        评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"评估模式: 模型锦标赛")
    print(f"{'='*60}")

    if len(model_paths) != num_players:
        raise ValueError(f"模型数量 ({len(model_paths)}) 必须等于玩家数 ({num_players})")

    # 创建游戏
    game = create_game(game_name, num_players=num_players)

    # 加载所有模型
    agents = []
    agent_names = []

    for i, model_path in enumerate(model_paths):
        agent_name = f"Model {i+1}"
        agent = load_model_agent(model_path, game, agent_name, device)
        agents.append(agent)
        agent_names.append(agent_name)

    # 创建 Arena
    arena = Arena(game, agents, agent_names, use_elo=True)

    # 运行锦标赛
    tournament_results = arena.run_tournament(
        num_games=num_games,
        shuffle_positions=True,
        verbose=verbose,
    )

    return {
        "mode": "tournament",
        "model_paths": model_paths,
        "num_games": num_games,
        "tournament_results": tournament_results,
    }


def evaluate_evolution(
    checkpoint_dir: str,
    game_name: str = "splendor",
    num_players: int = 4,
    num_games: int = 50,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """
    评估模型进化曲线

    加载检查点目录中的所有检查点，评估每个版本 vs 随机 Agent。

    Args:
        checkpoint_dir: 检查点目录
        game_name: 游戏名称
        num_players: 玩家数
        num_games: 每个检查点的游戏局数
        device: 设备
        verbose: 是否显示详细信息

    Returns:
        评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"评估模式: 模型进化曲线")
    print(f"{'='*60}")

    # 查找所有检查点
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_path.glob("checkpoint_*.pth"))

    if not checkpoints:
        raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到检查点文件")

    print(f"找到 {len(checkpoints)} 个检查点")

    # 创建游戏
    game = create_game(game_name, num_players=num_players)

    # 评估每个检查点
    evolution_results = []

    for i, checkpoint in enumerate(checkpoints):
        print(f"\n评估检查点 {i+1}/{len(checkpoints)}: {checkpoint.name}")

        # 创建 Agent
        agents = []
        agent_names = []

        # 加载检查点模型
        model_agent = load_model_agent(str(checkpoint), game, checkpoint.name, device)
        agents.append(model_agent)
        agent_names.append(checkpoint.name)

        # 其他位置使用随机 Agent
        for j in range(1, num_players):
            agents.append(RandomAgent(player_id=j))
            agent_names.append(f"Random {j}")

        # 创建 Arena
        arena = Arena(game, agents, agent_names, use_elo=False)

        # 运行游戏
        results = arena.run_games(
            num_games=num_games,
            shuffle_positions=True,
            verbose=False,  # 不显示每个检查点的详细信息
        )

        # 获取统计
        stats = arena.get_statistics()
        model_stats = arena.metrics_collector.get_player_stats(checkpoint.name)

        evolution_results.append(
            {
                "checkpoint": checkpoint.name,
                "win_rate": model_stats.win_rate if model_stats else 0.0,
                "avg_score": model_stats.avg_score if model_stats else 0.0,
                "avg_rank": model_stats.avg_rank if model_stats else 0.0,
                "num_games": num_games,
            }
        )

        if verbose:
            print(
                f"  胜率: {model_stats.win_rate:.2%}, "
                f"平均分数: {model_stats.avg_score:.2f}, "
                f"平均排名: {model_stats.avg_rank:.2f}"
            )

    return {
        "mode": "evolution",
        "checkpoint_dir": checkpoint_dir,
        "num_checkpoints": len(checkpoints),
        "num_games_per_checkpoint": num_games,
        "evolution_results": evolution_results,
    }


def save_results(results: Dict, output_path: str) -> None:
    """
    保存评估结果到 JSON 文件

    Args:
        results: 评估结果字典
        output_path: 输出路径
    """
    # 移除不可序列化的对象
    serializable_results = results.copy()
    if "results" in serializable_results:
        del serializable_results["results"]
    if "tournament_results" in serializable_results:
        serializable_results["tournament_results"] = {
            "num_games": serializable_results["tournament_results"]["num_games"],
            "winner": serializable_results["tournament_results"]["winner"],
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n评估结果已保存到: {output_path}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="模型评估脚本")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["vs_random", "tournament", "evolution"],
        required=True,
        help="评估模式",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="单个模型检查点路径（用于 vs_random 模式）",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="多个模型检查点路径（用于 tournament 模式）",
    )

    parser.add_argument(
        "--checkpoints",
        type=str,
        help="检查点目录（用于 evolution 模式）",
    )

    parser.add_argument(
        "--game",
        type=str,
        default="splendor",
        help="游戏名称（默认: splendor）",
    )

    parser.add_argument(
        "--players",
        type=int,
        default=4,
        help="玩家数（默认: 4）",
    )

    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="游戏局数（默认: 100）",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备（默认: cpu）",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="输出结果的 JSON 文件路径",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息",
    )

    args = parser.parse_args()

    # 根据模式进行评估
    try:
        if args.mode == "vs_random":
            if not args.model:
                parser.error("vs_random 模式需要 --model 参数")

            results = evaluate_vs_random(
                model_path=args.model,
                game_name=args.game,
                num_players=args.players,
                num_games=args.games,
                device=args.device,
                verbose=args.verbose,
            )

        elif args.mode == "tournament":
            if not args.models:
                parser.error("tournament 模式需要 --models 参数")

            results = evaluate_tournament(
                model_paths=args.models,
                game_name=args.game,
                num_players=args.players,
                num_games=args.games,
                device=args.device,
                verbose=args.verbose,
            )

        elif args.mode == "evolution":
            if not args.checkpoints:
                parser.error("evolution 模式需要 --checkpoints 参数")

            results = evaluate_evolution(
                checkpoint_dir=args.checkpoints,
                game_name=args.game,
                num_players=args.players,
                num_games=args.games,
                device=args.device,
                verbose=args.verbose,
            )

        # 保存结果
        if args.output:
            save_results(results, args.output)

        print("\n评估完成!")

    except Exception as e:
        print(f"\n评估过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
