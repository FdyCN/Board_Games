"""
Heads 模块

提供策略头和价值头实现
"""

from models.heads.policy_head import PolicyHead
from models.heads.value_head import ValueHead
from models.heads.outcome_head import OutcomeHead

__all__ = ["PolicyHead", "ValueHead", "OutcomeHead"]
