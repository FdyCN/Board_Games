"""
配置加载器

用于加载和解析 YAML 配置文件。
"""

import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class GameConfig:
    """游戏配置"""

    name: str
    num_players: int
    seed: int | None = None


@dataclass
class ModelConfig:
    """模型配置"""

    encoder_type: str = "mlp"
    config: str = "medium"
    hidden_dim: int | None = None
    intermediate_dim: int | None = None
    num_attention_heads: int | None = None
    dropout: float = 0.0


@dataclass
class DenseRewardsConfig:
    """稠密奖励配置（PPO 专用）"""

    take_gem: float = 0.01           # 每个拿取的宝石
    discard_gem: float = -0.05       # 每个丢弃的宝石（惩罚）
    reserve_card: float = 0.02       # 保留卡牌
    get_gold: float = 0.03           # 获得金宝石
    buy_card_points: float = 0.15    # 购买卡牌（每分）
    buy_card_bonus: float = 0.05     # 购买卡牌（获得永久宝石加成）
    noble_visit: float = 0.3         # 获得贵族
    win: float = 1.0                 # 游戏胜利
    step_penalty: float = 0.0        # 每步小惩罚（鼓励尽快结束游戏）


@dataclass
class MCTSSchedulerConfig:
    """MCTS 调度器配置"""

    enabled: bool = False
    schedule: list = field(default_factory=list)  # [(iteration, simulations), ...]


@dataclass
class MCTSConfig:
    """MCTS 配置"""

    enabled: bool = False           # 是否启用 MCTS
    simulations: int = 100          # 固定模拟次数
    c_puct: float = 1.5             # UCB 探索常数
    add_noise: bool = True          # 是否添加 Dirichlet 噪声
    temperature: float = 1.0        # 采样温度
    scheduler: MCTSSchedulerConfig = field(default_factory=MCTSSchedulerConfig)


@dataclass
class AlgorithmConfig:
    """算法配置"""

    name: str = "ppo"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.4  # 价值损失裁剪参数（通常比 clip_epsilon 更大）
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    use_value_clip: bool = True  # 是否使用价值损失裁剪（推荐启用）
    outcome_coef: float = 1.0  # 终局胜负辅助任务损失系数
    dense_rewards: DenseRewardsConfig = field(default_factory=DenseRewardsConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)


@dataclass
class TrainingConfig:
    """训练配置"""

    num_iterations: int = 1000
    episodes_per_iteration: int = 50
    update_epochs: int = 4
    minibatch_size: int = 256
    device: str = "cpu"
    num_workers: int = 1  # 并行进程数（1=单进程，>1=多进程）
    checkpoint_dir: str = "data/checkpoints"
    checkpoint_interval: int = 10
    log_interval: int = 1
    verbose: bool = True
    use_position_augmentation: bool = True  # 启用位置旋转数据增强（减少位置偏差）


@dataclass
class EvaluationConfig:
    """评估配置"""

    eval_interval: int = 50
    eval_episodes: int = 100
    deterministic: bool = True


@dataclass
class ExperimentConfig:
    """实验配置"""

    name: str = "experiment"
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Config:
    """完整配置"""

    game: GameConfig
    model: ModelConfig
    algorithm: AlgorithmConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    experiment: ExperimentConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """从字典创建配置"""
        # 解析 algorithm 配置，处理嵌套的 dense_rewards 和 mcts
        algorithm_dict = config_dict.get("algorithm", {})

        # 处理 dense_rewards
        dense_rewards_dict = algorithm_dict.pop("dense_rewards", {})

        # 处理 mcts
        mcts_dict = algorithm_dict.pop("mcts", {})
        mcts_scheduler_dict = mcts_dict.pop("scheduler", {})

        algorithm_config = AlgorithmConfig(**algorithm_dict)

        if dense_rewards_dict:
            algorithm_config.dense_rewards = DenseRewardsConfig(**dense_rewards_dict)

        if mcts_dict or mcts_scheduler_dict:
            mcts_config = MCTSConfig(**mcts_dict)
            if mcts_scheduler_dict:
                # 转换 schedule 格式
                schedule_list = mcts_scheduler_dict.get("schedule", [])
                # YAML 中 schedule 是列表的列表，需要转换为元组列表
                if schedule_list and isinstance(schedule_list[0], list):
                    mcts_scheduler_dict["schedule"] = [tuple(item) for item in schedule_list]
                mcts_config.scheduler = MCTSSchedulerConfig(**mcts_scheduler_dict)
            algorithm_config.mcts = mcts_config

        return cls(
            game=GameConfig(**config_dict.get("game", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            algorithm=algorithm_config,
            training=TrainingConfig(**config_dict.get("training", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {})),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """从 YAML 文件加载配置"""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        algorithm_dict = self.algorithm.__dict__.copy()
        algorithm_dict["dense_rewards"] = self.algorithm.dense_rewards.__dict__

        return {
            "game": self.game.__dict__,
            "model": self.model.__dict__,
            "algorithm": algorithm_dict,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "experiment": self.experiment.__dict__,
        }


# ===== 导出 =====
__all__ = [
    "Config",
    "GameConfig",
    "ModelConfig",
    "AlgorithmConfig",
    "DenseRewardsConfig",
    "MCTSConfig",
    "MCTSSchedulerConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ExperimentConfig",
]
