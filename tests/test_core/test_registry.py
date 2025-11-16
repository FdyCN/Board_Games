"""
测试游戏注册系统
"""


import numpy as np
import pytest

from core.exceptions import DuplicateRegistrationError, GameNotFoundError
from core.game_interface import GameInterface
from games.registry import (
    GAME_REGISTRY,
    clear_registry,
    create_game,
    get_game_class,
    is_registered,
    list_games,
    register_game,
    unregister_game,
)

# ===== 测试用的简单游戏 =====


class SimpleTestGame(GameInterface):
    """最简单的测试游戏：硬币翻转"""

    def __init__(self, num_players: int = 2):
        self._num_players = num_players
        self._state = {"done": False, "current_player": 0}

    def reset(self):
        self._state = {"done": False, "current_player": 0}
        return self._state

    def step(self, action):
        # 简单实现：轮流行动
        current = self._state["current_player"]
        next_player = (current + 1) % self._num_players

        # 模拟游戏结束
        if action == "end":
            self._state["done"] = True
            rewards = [1.0 if i == current else 0.0 for i in range(self._num_players)]
        else:
            self._state["current_player"] = next_player
            rewards = [0.0] * self._num_players

        return self._state, rewards, self._state["done"], {}

    def get_legal_actions(self, state):
        return ["action1", "action2", "end"]

    def get_current_player(self, state):
        return state["current_player"]

    def is_terminal(self, state):
        return state["done"]

    def get_final_rewards(self, state):
        if not state["done"]:
            return [0.0] * self._num_players
        return [1.0, 0.0]  # 简化：玩家0获胜

    def state_to_observation(self, state, player_id):
        return np.array([float(state["current_player"]), float(player_id)])

    def action_to_index(self, action):
        actions = {"action1": 0, "action2": 1, "end": 2}
        return actions[action]

    def index_to_action(self, index):
        actions = ["action1", "action2", "end"]
        return actions[index]

    @property
    def num_players(self):
        return self._num_players

    @property
    def observation_shape(self):
        return (2,)

    @property
    def action_space_size(self):
        return 3


# ===== 测试 Fixture =====


@pytest.fixture
def clean_registry():
    """清空注册表的 fixture（仅用于 registry 测试）"""
    clear_registry()
    yield
    clear_registry()


# ===== 测试用例 =====


def test_register_game(clean_registry):
    """测试游戏注册"""

    @register_game("test_game")
    class TestGame(SimpleTestGame):
        pass

    assert "test_game" in GAME_REGISTRY
    assert GAME_REGISTRY["test_game"] == TestGame


def test_register_game_duplicate(clean_registry):
    """测试重复注册抛出异常"""

    @register_game("test_game")
    class TestGame1(SimpleTestGame):
        pass

    with pytest.raises(DuplicateRegistrationError):

        @register_game("test_game")
        class TestGame2(SimpleTestGame):
            pass


def test_register_non_game_interface(clean_registry):
    """测试注册非 GameInterface 子类抛出异常"""

    with pytest.raises(TypeError):

        @register_game("invalid")
        class NotAGame:
            pass


def test_create_game(clean_registry):
    """测试创建游戏实例"""

    @register_game("test_game")
    class TestGame(SimpleTestGame):
        pass

    game = create_game("test_game", num_players=4)
    assert isinstance(game, TestGame)
    assert game.num_players == 4


def test_create_game_not_found(clean_registry):
    """测试创建未注册游戏抛出异常"""
    with pytest.raises(GameNotFoundError) as exc_info:
        create_game("nonexistent_game")

    assert "nonexistent_game" in str(exc_info.value)


def test_list_games(clean_registry):
    """测试列出所有游戏"""

    @register_game("game1")
    class Game1(SimpleTestGame):
        pass

    @register_game("game2")
    class Game2(SimpleTestGame):
        pass

    games = list_games()
    assert games == ["game1", "game2"]  # 应该是排序的


def test_get_game_class(clean_registry):
    """测试获取游戏类"""

    @register_game("test_game")
    class TestGame(SimpleTestGame):
        pass

    game_class = get_game_class("test_game")
    assert game_class == TestGame


def test_is_registered(clean_registry):
    """测试检查游戏是否注册"""

    @register_game("test_game")
    class TestGame(SimpleTestGame):
        pass

    assert is_registered("test_game")
    assert not is_registered("nonexistent")


def test_unregister_game(clean_registry):
    """测试取消注册"""

    @register_game("test_game")
    class TestGame(SimpleTestGame):
        pass

    assert is_registered("test_game")

    unregister_game("test_game")
    assert not is_registered("test_game")


def test_clear_registry(clean_registry):
    """测试清空注册表"""

    @register_game("game1")
    class Game1(SimpleTestGame):
        pass

    @register_game("game2")
    class Game2(SimpleTestGame):
        pass

    assert len(list_games()) == 2

    clear_registry()
    assert len(list_games()) == 0
