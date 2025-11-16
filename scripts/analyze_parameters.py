"""
参数量分析脚本

分析不同配置下的模型参数量
"""

import torch
from models.model_factory import create_splendor_model, MODEL_CONFIGS


def analyze_parameters():
    """分析各种配置的参数量"""
    print("=" * 80)
    print("模型参数量分析".center(80))
    print("=" * 80)
    print()

    # 分析所有编码器类型和配置组合
    for encoder_type in ["mlp", "attention"]:
        print(f"\n{'=' * 80}")
        print(f"{encoder_type.upper()} Encoder".center(80))
        print("=" * 80)

        for config_name in ["small", "medium", "large"]:
            model = create_splendor_model(
                encoder_type=encoder_type, config=config_name
            )
            param_counts = model.count_parameters()

            print(f"\n{config_name.upper()} 配置:")
            print(f"  配置参数: {MODEL_CONFIGS[config_name]}")
            print(f"  编码器参数:   {param_counts['encoder']:>10,}")
            print(f"  策略头参数:   {param_counts['policy_head']:>10,}")
            print(f"  价值头参数:   {param_counts['value_head']:>10,}")
            print(f"  {'─' * 30}")
            print(f"  总参数量:     {param_counts['total']:>10,}")

            # 检查是否在目标范围内
            in_range = 100_000 <= param_counts["total"] <= 1_000_000
            status = "✓" if in_range else "✗"
            print(f"  目标范围 (100K-1M): {status}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_parameters()
