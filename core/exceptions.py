"""
自定义异常类

本模块定义了框架中使用的所有自定义异常。
"""


class BoardGameError(Exception):
    """所有桌游框架异常的基类"""

    pass


# ===== 游戏相关异常 =====


class IllegalActionError(BoardGameError):
    """动作非法异常"""

    def __init__(self, action, message: str = ""):
        self.action = action
        self.message = message or f"非法动作: {action}"
        super().__init__(self.message)


class GameNotTerminalError(BoardGameError):
    """游戏未结束异常（当需要终局状态时抛出）"""

    pass


class GameAlreadyTerminalError(BoardGameError):
    """游戏已结束异常（尝试继续游戏时抛出）"""

    pass


class InvalidStateError(BoardGameError):
    """无效状态异常"""

    pass


# ===== Agent 相关异常 =====


class AgentError(BoardGameError):
    """Agent 相关异常的基类"""

    pass


class NoLegalActionsError(AgentError):
    """没有合法动作异常"""

    pass


class ModelNotLoadedError(AgentError):
    """模型未加载异常"""

    pass


# ===== 训练相关异常 =====


class TrainingError(BoardGameError):
    """训练相关异常的基类"""

    pass


class ConfigurationError(TrainingError):
    """配置错误异常"""

    pass


class CheckpointError(TrainingError):
    """检查点保存/加载错误"""

    pass


# ===== 注册相关异常 =====


class RegistryError(BoardGameError):
    """注册系统相关异常"""

    pass


class GameNotFoundError(RegistryError):
    """游戏未找到异常"""

    def __init__(self, game_name: str, available_games: list):
        self.game_name = game_name
        self.available_games = available_games
        message = (
            f"游戏 '{game_name}' 未找到。"
            f"可用游戏: {', '.join(available_games) if available_games else '无'}"
        )
        super().__init__(message)


class DuplicateRegistrationError(RegistryError):
    """重复注册异常"""

    def __init__(self, name: str):
        super().__init__(f"'{name}' 已经注册")
