"""
游戏注册系统

本模块提供游戏的注册和创建功能，使得添加新游戏无需修改核心代码。

使用方式：
    1. 定义游戏类并继承 GameInterface
    2. 使用 @register_game 装饰器注册
    3. 通过 create_game 工厂函数创建实例

Examples:
    >>> from games.registry import register_game, create_game
    >>> from core.game_interface import GameInterface
    >>>
    >>> @register_game("my_game")
    >>> class MyGame(GameInterface):
    ...     def __init__(self, num_players=4):
    ...         self.num_players_val = num_players
    ...     # ... 实现所有抽象方法
    >>>
    >>> # 创建游戏实例
    >>> game = create_game("my_game", num_players=4)
    >>> print(game)  # MyGame(num_players=4)
"""

from typing import Any

from core.exceptions import DuplicateRegistrationError, GameNotFoundError
from core.game_interface import GameInterface

# 全局游戏注册表
GAME_REGISTRY: dict[str, type[GameInterface]] = {}


def register_game(name: str):
    """
    装饰器：将游戏类注册到全局注册表

    Args:
        name: 游戏名称（全局唯一，用于配置文件和命令行）

    Returns:
        装饰器函数

    Raises:
        DuplicateRegistrationError: 如果游戏名称已被注册（可以通过设置覆盖）
        TypeError: 如果被装饰的类不是 GameInterface 的子类

    Examples:
        >>> @register_game("splendor")
        >>> class SplendorGame(GameInterface):
        ...     pass
    """

    def decorator(cls: type[GameInterface]):
        # 验证类型
        if not issubclass(cls, GameInterface):
            raise TypeError(f"{cls.__name__} 必须继承自 GameInterface")

        # 检查重复注册（允许覆盖，用于模块 reload）
        if name in GAME_REGISTRY:
            # 如果是同一个类，忽略重复注册
            if GAME_REGISTRY[name] is not cls:
                # 不同的类，仅在非 reload 情况下报错
                import sys
                module_name = cls.__module__
                # 如果模块在 sys.modules 中且正在 reload，允许覆盖
                if module_name not in sys.modules or not hasattr(sys.modules[module_name], '__reload_in_progress__'):
                    # 为了向后兼容，改为警告而不是错误
                    import warnings
                    warnings.warn(
                        f"游戏 '{name}' 已被 {GAME_REGISTRY[name].__name__} 注册，"
                        f"现在被 {cls.__name__} 覆盖",
                        UserWarning
                    )

        # 注册
        GAME_REGISTRY[name] = cls

        # 添加元数据
        cls._registry_name = name

        return cls

    return decorator


def create_game(name: str, **kwargs: Any) -> GameInterface:
    """
    工厂函数：根据名称创建游戏实例

    Args:
        name: 游戏名称（已注册的名称）
        **kwargs: 传递给游戏构造函数的参数

    Returns:
        游戏实例

    Raises:
        GameNotFoundError: 如果游戏未注册

    Examples:
        >>> game = create_game("splendor", num_players=4)
        >>> game = create_game("uno", num_players=3)
    """
    if name not in GAME_REGISTRY:
        available_games = list_games()
        raise GameNotFoundError(name, available_games)

    game_class = GAME_REGISTRY[name]
    return game_class(**kwargs)


def list_games() -> list[str]:
    """
    获取所有已注册的游戏名称

    Returns:
        游戏名称列表（按字母顺序排序）

    Examples:
        >>> games = list_games()
        >>> print(f"可用游戏: {', '.join(games)}")
    """
    return sorted(GAME_REGISTRY.keys())


def get_game_class(name: str) -> type[GameInterface]:
    """
    获取游戏类（不创建实例）

    Args:
        name: 游戏名称

    Returns:
        游戏类

    Raises:
        GameNotFoundError: 如果游戏未注册

    Examples:
        >>> SplendorGame = get_game_class("splendor")
        >>> print(SplendorGame.num_players)  # 访问类属性
    """
    if name not in GAME_REGISTRY:
        available_games = list_games()
        raise GameNotFoundError(name, available_games)

    return GAME_REGISTRY[name]


def is_registered(name: str) -> bool:
    """
    检查游戏是否已注册

    Args:
        name: 游戏名称

    Returns:
        True 如果已注册，False 否则

    Examples:
        >>> if is_registered("splendor"):
        ...     game = create_game("splendor")
    """
    return name in GAME_REGISTRY


def unregister_game(name: str) -> None:
    """
    取消注册游戏（主要用于测试）

    Args:
        name: 游戏名称

    Examples:
        >>> unregister_game("test_game")
    """
    if name in GAME_REGISTRY:
        del GAME_REGISTRY[name]


def clear_registry() -> None:
    """
    清空注册表（主要用于测试）

    Examples:
        >>> clear_registry()
        >>> assert len(list_games()) == 0
    """
    GAME_REGISTRY.clear()


def print_registry() -> None:
    """
    打印注册表信息（用于调试）

    Examples:
        >>> print_registry()
        Registered Games:
          - splendor: SplendorGame
          - uno: UNOGame
    """
    if not GAME_REGISTRY:
        print("注册表为空")
        return

    print("已注册游戏:")
    for name in sorted(GAME_REGISTRY.keys()):
        game_class = GAME_REGISTRY[name]
        print(f"  - {name}: {game_class.__name__}")


# ===== 自动导入已注册游戏 =====


def auto_import_games() -> None:
    """
    自动导入 games 目录下的所有游戏

    遍历 games/ 目录，导入所有游戏模块，触发 @register_game 装饰器

    Note:
        游戏模块必须在 games/<game_name>/game.py 中定义
    """
    import importlib
    import os
    import sys

    # 获取 games 目录路径
    games_dir = os.path.dirname(__file__)

    # 遍历所有子目录
    for entry in os.listdir(games_dir):
        game_path = os.path.join(games_dir, entry)

        # 跳过非目录和特殊目录
        if not os.path.isdir(game_path):
            continue
        if entry.startswith("_") or entry.startswith("."):
            continue

        # 尝试导入 game.py
        try:
            module_name = f"games.{entry}.game"
            # 如果模块已经导入过，使用 reload 强制重新执行
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
        except ModuleNotFoundError:
            # 该目录下没有 game.py，跳过
            pass
        except Exception as e:
            # 导入失败，打印警告但不中断
            print(f"警告: 导入游戏 '{entry}' 失败: {e}")


def import_specific_games(game_names: list[str]) -> None:
    """
    导入指定的游戏列表

    Args:
        game_names: 要导入的游戏名称列表

    Examples:
        >>> import_specific_games(["splendor", "uno"])
    """
    import importlib
    import sys

    for game_name in game_names:
        try:
            module_name = f"games.{game_name}.game"
            # 如果模块已经导入过，使用 reload 强制重新执行
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
        except ModuleNotFoundError:
            print(f"警告: 找不到游戏模块 '{game_name}'")
        except Exception as e:
            print(f"警告: 导入游戏 '{game_name}' 失败: {e}")


def import_games_except(excluded_games: list[str]) -> None:
    """
    导入除了指定列表外的所有游戏

    Args:
        excluded_games: 要排除的游戏名称列表

    Examples:
        >>> import_games_except(["test_game"])
    """
    import importlib
    import os
    import sys

    games_dir = os.path.dirname(__file__)

    for entry in os.listdir(games_dir):
        game_path = os.path.join(games_dir, entry)

        if not os.path.isdir(game_path):
            continue
        if entry.startswith("_") or entry.startswith("."):
            continue
        if entry in excluded_games:
            continue

        try:
            module_name = f"games.{entry}.game"
            # 如果模块已经导入过，使用 reload 强制重新执行
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
        except ModuleNotFoundError:
            pass
        except Exception as e:
            print(f"警告: 导入游戏 '{entry}' 失败: {e}")


# 在模块导入时自动注册所有游戏（可选）
# 如果不希望自动导入，可以注释掉下面这行
auto_import_games()
