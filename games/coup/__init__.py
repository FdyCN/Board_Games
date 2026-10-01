"""政变疑云（Coup）游戏模块。"""

from games.coup.game import CoupGame
from games.coup.state import CoupState
from games.coup.actions import CoupAction
from games.coup.encoder import CoupEncoder

__all__ = [
    "CoupGame",
    "CoupState",
    "CoupAction",
    "CoupEncoder",
]
