"""
Evaluation 测试配置
"""

import pytest
import importlib


# Session 级别 autouse fixture - 在这个目录的所有测试开始前运行一次
@pytest.fixture(scope="session", autouse=True)
def setup_splendor_game_session():
    """
    在测试会话开始时确保 Splendor 游戏已注册（session 级别）
    """
    import games.splendor
    importlib.reload(games.splendor)
    yield


# Function 级别 autouse fixture - 在每个测试前都重新加载
@pytest.fixture(scope="function", autouse=True)
def setup_splendor_game_function():
    """
    在每个测试前确保 Splendor 游戏已注册（function 级别）

    这个 fixture 处理测试隔离问题 - 当 test_core/test_registry.py 的测试清空注册表后，
    需要重新注册 Splendor 游戏。
    """
    import games.splendor
    importlib.reload(games.splendor)
    yield
