"""
按游戏组织的产物路径辅助函数。

约定（见 docs/ADDING_A_GAME.md）：

    data/
      <game_name>/                # 例如 splendor
        checkpoints/              # 模型检查点
          <run_name>/             # 例如 mlp_medium_3p_v1
            latest.pth
            checkpoint_iter_*.pth
            tensorboard/          # TensorBoard 事件（Trainer 默认放这里）
        logs/                     # JSONL 训练日志 + stdout 日志
          <run_name>.jsonl
          <run_name>.stdout.log
        reports/                  # 评估 / 基准测试报告（可选）

所有路径都锚定在项目根目录（依据本文件位置），不依赖运行时的工作目录。
"""

from __future__ import annotations

from pathlib import Path

# 项目根目录 = core/ 的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def game_dir(game_name: str) -> Path:
    """data/<game_name>/"""
    return DATA_DIR / game_name


def checkpoints_dir(game_name: str) -> Path:
    """data/<game_name>/checkpoints/"""
    return game_dir(game_name) / "checkpoints"


def logs_dir(game_name: str) -> Path:
    """data/<game_name>/logs/"""
    return game_dir(game_name) / "logs"


def reports_dir(game_name: str) -> Path:
    """data/<game_name>/reports/"""
    return game_dir(game_name) / "reports"


def run_checkpoint_dir(game_name: str, run_name: str) -> Path:
    """data/<game_name>/checkpoints/<run_name>/"""
    return checkpoints_dir(game_name) / run_name


def default_log_path(game_name: str, run_name: str) -> Path:
    """data/<game_name>/logs/<run_name>.jsonl"""
    return logs_dir(game_name) / f"{run_name}.jsonl"


def default_stdout_log_path(game_name: str, run_name: str) -> Path:
    """data/<game_name>/logs/<run_name>.stdout.log"""
    return logs_dir(game_name) / f"{run_name}.stdout.log"


def ensure_run_dirs(game_name: str, run_name: str) -> dict[str, Path]:
    """
    创建并返回一个 run 所需的全部目录（checkpoints / logs）。

    Args:
        game_name: 游戏名（与注册表一致）。
        run_name: 本次训练 / 实验名。

    Returns:
        {"checkpoints": Path, "logs": Path}
    """
    dirs = {
        "checkpoints": run_checkpoint_dir(game_name, run_name),
        "logs": logs_dir(game_name),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "game_dir",
    "checkpoints_dir",
    "logs_dir",
    "reports_dir",
    "run_checkpoint_dir",
    "default_log_path",
    "default_stdout_log_path",
    "ensure_run_dirs",
]
