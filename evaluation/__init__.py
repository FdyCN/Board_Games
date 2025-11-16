"""
评估模块

提供模型评估和对战功能。
"""

from evaluation.metrics import (
    GameResult,
    PlayerStats,
    MetricsCollector,
    compute_win_matrix,
)
from evaluation.elo_system import EloRating, EloSystem
from evaluation.arena import Arena, MatchConfig

__all__ = [
    # Metrics
    "GameResult",
    "PlayerStats",
    "MetricsCollector",
    "compute_win_matrix",
    # ELO
    "EloRating",
    "EloSystem",
    # Arena
    "Arena",
    "MatchConfig",
]
