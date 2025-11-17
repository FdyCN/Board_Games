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
class AlgorithmConfig:
    """算法配置"""

    name: str = "ppo"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


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
        return cls(
            game=GameConfig(**config_dict.get("game", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            algorithm=AlgorithmConfig(**config_dict.get("algorithm", {})),
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
        return {
            "game": self.game.__dict__,
            "model": self.model.__dict__,
            "algorithm": self.algorithm.__dict__,
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
    "TrainingConfig",
    "EvaluationConfig",
    "ExperimentConfig",
]
