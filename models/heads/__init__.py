"""
Heads 模块

提供策略头和价值头实现
"""

from models.heads.policy_head import PolicyHead
from models.heads.value_head import ValueHead

__all__ = ["PolicyHead", "ValueHead"]
