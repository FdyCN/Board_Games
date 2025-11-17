#!/usr/bin/env python3
"""
PPO 训练脚本

用于训练 Splendor 游戏的 PPO 智能体。

用法:
    python scripts/train.py --config configs/splendor_ppo.yaml
    python scripts/train.py --config configs/splendor_ppo.yaml --resume data/checkpoints/splendor_ppo/latest.pth
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config_loader import Config
from games.registry import create_game
from models.model_factory import create_model
from training.trainer import Trainer


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="PPO 训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径 (YAML)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的检查点路径",
    )
    args = parser.parse_args()

    # 加载配置
    print(f"加载配置: {args.config}")
    config = Config.from_yaml(args.config)

    print(f"\n{'='*60}")
    print(f"实验名称: {config.experiment.name}")
    print(f"游戏: {config.game.name}, 玩家数: {config.game.num_players}")
    print(f"模型: {config.model.encoder_type} ({config.model.config})")
    print(f"算法: {config.algorithm.name}")
    print(f"设备: {config.training.device}")
    print(f"{'='*60}\n")

    # 创建游戏
    print("创建游戏...")
    game = create_game(config.game.name, num_players=config.game.num_players)
    print(f"  观察维度: {game.observation_shape}")
    print(f"  动作空间: {game.action_space_size}")

    # 创建模型
    print("\n创建模型...")
    model = create_model(
        obs_dim=game.observation_shape[0],
        action_size=game.action_space_size,
        encoder_type=config.model.encoder_type,
        config=config.model.config,
    )

    param_count = model.count_parameters()
    print(f"  总参数量: {param_count['total']:,}")
    print(f"  编码器: {param_count['encoder']:,}")
    print(f"  策略头: {param_count['policy_head']:,}")
    print(f"  价值头: {param_count['value_head']:,}")

    # 创建训练器
    print("\n创建训练器...")
    trainer = Trainer(
        game=game,
        model=model,
        learning_rate=config.algorithm.learning_rate,
        gamma=config.algorithm.gamma,
        gae_lambda=config.algorithm.gae_lambda,
        clip_epsilon=config.algorithm.clip_epsilon,
        value_coef=config.algorithm.value_coef,
        entropy_coef=config.algorithm.entropy_coef,
        max_grad_norm=config.algorithm.max_grad_norm,
        device=config.training.device,
        checkpoint_dir=config.training.checkpoint_dir,
        verbose=config.training.verbose,
        num_workers=config.training.num_workers,
    )

    # 恢复训练（如果指定）
    if args.resume:
        print(f"\n{'='*60}")
        print(f"📥 恢复训练模式")
        print(f"{'='*60}")
        print(f"加载检查点: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"将从迭代 {trainer.iteration} 继续训练")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"🆕 全新训练模式 (从头开始)")
        print(f"{'='*60}")
        print(f"模型: {config.model.encoder_type} ({config.model.config})")
        print(f"参数量: {trainer.model.count_parameters()['total']:,}")
        print(f"检查点目录: {config.training.checkpoint_dir}")
        print(f"{'='*60}")

    # 开始训练
    print("\n🚀 开始训练...")
    try:
        metrics = trainer.train(
            num_iterations=config.training.num_iterations,
            episodes_per_iteration=config.training.episodes_per_iteration,
            update_epochs=config.training.update_epochs,
            minibatch_size=config.training.minibatch_size,
            checkpoint_interval=config.training.checkpoint_interval,
            log_interval=config.training.log_interval,
        )

        print("\n训练完成!")
        print(f"最终策略损失: {metrics[-1].policy_loss:.4f}")
        print(f"最终价值损失: {metrics[-1].value_loss:.4f}")
        print(f"最终平均奖励: {metrics[-1].mean_reward:.4f}")

    except KeyboardInterrupt:
        print("\n\n训练被用户中断!")
        print("保存最终检查点...")
        trainer._save_checkpoint()
        print("检查点已保存。")

    except Exception as e:
        print(f"\n\n训练过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
        print("\n保存紧急检查点...")
        trainer._save_checkpoint()
        sys.exit(1)


if __name__ == "__main__":
    main()
