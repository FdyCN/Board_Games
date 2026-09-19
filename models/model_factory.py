"""
Model Factory

提供便捷的模型创建接口，支持预设配置和自定义配置。
"""

from typing import Literal

from models.actor_critic import ActorCritic


# 预设配置
# 所有配置都确保总参数量在 100K-1M 范围内
# 配置针对两种编码器类型优化：
# - MLP 编码器参数量较少，可以使用较大的维度
# - Attention 编码器参数量较多（MultiheadAttention），需要控制 intermediate_dim
MODEL_CONFIGS = {
    "small": {
        "hidden_dim": 128,
        "encoder_intermediate_dim": 256,
        "head_intermediate_dim": 64,
        "num_attention_heads": 4,
        "dropout": 0.0,
    },
    "medium": {
        "hidden_dim": 256,
        "encoder_intermediate_dim": 320,  # 减小以适配 attention encoder
        "head_intermediate_dim": 128,
        "num_attention_heads": 4,
        "dropout": 0.0,
    },
    "large": {
        "hidden_dim": 256,
        "encoder_intermediate_dim": 368,  # 减小以适配 attention encoder (<1M params)
        "head_intermediate_dim": 128,
        "num_attention_heads": 4,
        "dropout": 0.1,
    },
}


def create_model(
    obs_dim: int,
    action_size: int,
    encoder_type: Literal["mlp", "attention"] = "mlp",
    config: Literal["small", "medium", "large"] | dict | None = None,
) -> ActorCritic:
    """
    创建 Actor-Critic 模型

    Args:
        obs_dim: 观察空间维度
        action_size: 动作空间大小
        encoder_type: 编码器类型 ("mlp" 或 "attention")
        config: 配置名称（"small"、"medium"、"large"）或自定义配置字典
                如果为 None，使用 "medium" 配置

    Returns:
        配置好的 ActorCritic 模型

    Examples:
        >>> # 使用预设配置
        >>> model = create_model(384, 50, encoder_type="mlp", config="medium")

        >>> # 使用自定义配置
        >>> custom_config = {
        ...     "hidden_dim": 256,
        ...     "encoder_intermediate_dim": 512,
        ...     "head_intermediate_dim": 128,
        ...     "num_attention_heads": 4,
        ...     "dropout": 0.05,
        ... }
        >>> model = create_model(384, 50, encoder_type="attention", config=custom_config)
    """
    # 获取配置
    if config is None:
        config = "medium"

    if isinstance(config, str):
        if config not in MODEL_CONFIGS:
            raise ValueError(
                f"未知的配置名称: {config}. "
                f"可用配置: {list(MODEL_CONFIGS.keys())}"
            )
        model_config = MODEL_CONFIGS[config].copy()
    else:
        model_config = config.copy()

    # 创建模型
    model = ActorCritic(
        obs_dim=obs_dim,
        action_size=action_size,
        encoder_type=encoder_type,
        **model_config,
    )

    return model


def create_splendor_model(
    encoder_type: Literal["mlp", "attention"] = "mlp",
    config: Literal["small", "medium", "large"] | dict | None = None,
) -> ActorCritic:
    """
    为 Splendor 游戏创建模型（固定 obs_dim=384, action_size=50）

    Args:
        encoder_type: 编码器类型 ("mlp" 或 "attention")
        config: 配置名称或自定义配置字典

    Returns:
        配置好的 ActorCritic 模型

    Examples:
        >>> # 创建 MLP 编码器的中等规模模型
        >>> model = create_splendor_model(encoder_type="mlp", config="medium")

        >>> # 创建 Attention 编码器的大规模模型
        >>> model = create_splendor_model(encoder_type="attention", config="large")
    """
    return create_model(
        obs_dim=384,
        action_size=46,  # 规范动作空间大小（见 games/splendor/game.py 的 ACTION_SPACE_SIZE）
        encoder_type=encoder_type,
        config=config,
    )


def get_model_info(model: ActorCritic) -> dict:
    """
    获取模型信息

    Args:
        model: ActorCritic 模型

    Returns:
        包含模型配置和参数量的字典

    Examples:
        >>> model = create_splendor_model()
        >>> info = get_model_info(model)
        >>> print(f"Total parameters: {info['parameters']['total']:,}")
    """
    param_counts = model.count_parameters()

    info = {
        "encoder_type": model.encoder_type,
        "obs_dim": model.obs_dim,
        "action_size": model.action_size,
        "hidden_dim": model.hidden_dim,
        "parameters": param_counts,
    }

    return info


def print_model_summary(model: ActorCritic) -> None:
    """
    打印模型摘要信息

    Args:
        model: ActorCritic 模型

    Examples:
        >>> model = create_splendor_model(encoder_type="attention", config="medium")
        >>> print_model_summary(model)
        ========== Model Summary ==========
        Encoder Type:        attention
        Observation Dim:     384
        Action Size:         50
        Hidden Dim:          256

        Parameters:
          Encoder:           591,872
          Policy Head:       39,090
          Value Head:        32,897
          Total:             663,859
        ===================================
    """
    info = get_model_info(model)

    print("=" * 35)
    print("Model Summary".center(35))
    print("=" * 35)
    print(f"Encoder Type:        {info['encoder_type']}")
    print(f"Observation Dim:     {info['obs_dim']}")
    print(f"Action Size:         {info['action_size']}")
    print(f"Hidden Dim:          {info['hidden_dim']}")
    print()
    print("Parameters:")
    print(f"  Encoder:           {info['parameters']['encoder']:,}")
    print(f"  Policy Head:       {info['parameters']['policy_head']:,}")
    print(f"  Value Head:        {info['parameters']['value_head']:,}")
    print(f"  Total:             {info['parameters']['total']:,}")
    print("=" * 35)


# ===== 导出 =====
__all__ = [
    "create_model",
    "create_splendor_model",
    "get_model_info",
    "print_model_summary",
    "MODEL_CONFIGS",
]
