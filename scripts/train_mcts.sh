#!/bin/bash
# MCTS-Enhanced PPO 训练快速启动脚本

echo "=========================================="
echo "MCTS-Enhanced PPO Training"
echo "=========================================="
echo ""

# 检查配置文件
CONFIG_FILE="configs/splendor_mcts_ppo.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "配置文件: $CONFIG_FILE"
echo ""

# 询问是否从检查点恢复
echo "是否从检查点恢复训练? (y/N)"
read -r RESUME_CHOICE

if [ "$RESUME_CHOICE" = "y" ] || [ "$RESUME_CHOICE" = "Y" ]; then
    # 查找最新的检查点
    CHECKPOINT_DIR=$(grep "checkpoint_dir:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
    LATEST_CHECKPOINT="$CHECKPOINT_DIR/latest.pth"

    if [ -f "$LATEST_CHECKPOINT" ]; then
        echo "找到检查点: $LATEST_CHECKPOINT"
        echo ""
        echo "开始恢复训练..."
        python scripts/train.py --config "$CONFIG_FILE" --resume "$LATEST_CHECKPOINT"
    else
        echo "警告: 未找到检查点 $LATEST_CHECKPOINT"
        echo "将开始全新训练..."
        python scripts/train.py --config "$CONFIG_FILE"
    fi
else
    echo "开始全新训练..."
    python scripts/train.py --config "$CONFIG_FILE"
fi
