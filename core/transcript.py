"""
对局记录（Transcript）模块 —— 游戏无关。

用途：记录「真人 + AI」整局对局的完整过程，落成 JSONL（一局一行），供后续
行为克隆 / 离线 RL 微调。

设计要点：
- 只记录「决策者视角的 obs」+ 动作索引（action_idx）。这正是训练 π(obs)->action
  所需的数据，且与推理输入严格一致；不序列化完整上帝状态（隐藏信息游戏里会泄露
  对手手牌，且对训练无用）。
- 提供 step 级接口（EpisodeRecorder），第三方 Web 产品可自己驱动游戏循环、逐步
  记录，无需把游戏循环交给本框架。
- 附带一个「整局自动跑」的便捷函数 record_episode()，供本仓库 Web/脚本使用。

示例（第三方 Web 产品，submodule 接入）：
    from core.transcript import EpisodeRecorder
    rec = EpisodeRecorder(game, human_seats=[0, 1], ai_labels={2: "coup_3p_gru_v1"}, seed=42)
    # ... 游戏循环中，每步：
    rec.record_step(state, player, action, reward, done)
    # ... 终局：
    rec.finish(final_state)
    rec.save("backend/transcripts/coup.jsonl")
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _asdict_best_effort(action: Any) -> Any:
    """把动作对象转成可 JSON 序列化的形式；失败则退回 str()。"""
    if dataclasses.is_dataclass(action):
        try:
            return dataclasses.asdict(action)
        except Exception:
            return str(action)
    return str(action)


def _game_name(game: Any) -> str:
    """从注册元数据取游戏名（register_game 会写 cls._registry_name）。"""
    return getattr(type(game), "_registry_name", type(game).__name__)


class EpisodeRecorder:
    """逐步记录一局对局（游戏无关）。"""

    def __init__(
        self,
        game: Any,
        human_seats: list[int] | tuple[int, ...] = (),
        ai_labels: dict[int, str] | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            game: 已构造的游戏实例（含 state_to_observation / action_to_index）。
            human_seats: 哪些座位由真人控制（其余为 AI）。
            ai_labels: {座位: 模型名}，记录 AI 用了哪个模型（可选，纯标注）。
            seed: 对局种子（可选，纯标注，便于复现）。
        """
        self.game = game
        self.human_seats = list(human_seats)
        self.ai_labels = dict(ai_labels or {})
        self.seed = seed
        self.game_name = _game_name(game)
        self.num_players = game.num_players
        self.steps: list[dict] = []
        self.result: dict = {}
        self.started_at = time.time()

    def record_step(
        self,
        state: Any,
        player: int,
        action: Any,
        reward: float = 0.0,
        done: bool = False,
        phase: int | None = None,
    ) -> None:
        """记录一个决策步。state 是该玩家决策前看到的状态。

        Args:
            state: 决策前状态（用于算 obs）。
            player: 决策者座位。
            action: 所选动作对象。
            reward: 该玩家此步的奖励（离线 RL 用；行为克隆可忽略）。
            done: 是否终局。
            phase: 可选，多阶段游戏（如 Coup）的阶段号，便于人类阅读。
        """
        obs = self.game.state_to_observation(state, player)
        if isinstance(obs, np.ndarray):
            obs = obs.tolist()
        action_idx = self.game.action_to_index(action, state)
        if phase is None:
            phase = getattr(state, "phase", None)
        self.steps.append({
            "player": player,
            "is_human": player in self.human_seats,
            "phase": phase,
            "action_idx": action_idx,
            "action": _asdict_best_effort(action),
            "obs": obs,
            "reward": float(reward),
            "done": bool(done),
        })

    def finish(self, final_state: Any) -> None:
        """终局：记录胜者、最终奖励与富事件日志（如有）。"""
        game = self.game
        winner = game.get_winner(final_state) if game.is_terminal(final_state) else -1
        final_rewards = list(game.get_final_rewards(final_state))
        self.result = {"winner": int(winner), "final_rewards": final_rewards}
        log = getattr(final_state, "log", None)
        if log:
            self.result["event_log"] = log

    def to_dict(self) -> dict:
        """返回整局记录（可 JSON 序列化）。"""
        return {
            "type": "episode",
            "game": self.game_name,
            "num_players": self.num_players,
            "seed": self.seed,
            "human_seats": self.human_seats,
            "ai_labels": self.ai_labels,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "steps": self.steps,
            "result": self.result,
        }

    def save(self, path: str | Path) -> Path:
        """把整局记录追加写入 JSONL（一行一局），返回路径。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
        return path


def record_episode(
    game: Any,
    controllers: dict[int, Callable[[Any, int, list[Any]], Any]],
    human_seats: list[int] | tuple[int, ...] = (),
    ai_labels: dict[int, str] | None = None,
    seed: int | None = None,
) -> dict:
    """整局自动跑并记录（便捷函数）。

    Args:
        game: 游戏实例。
        controllers: {座位: 函数(game, state, player, legal_actions) -> action}，
            为每个座位提供动作（真人座位可抛异常或由外部阻塞）。
        human_seats / ai_labels / seed: 同 EpisodeRecorder。

    Returns:
        整局记录 dict（未写盘）。
    """
    rec = EpisodeRecorder(game, human_seats, ai_labels, seed)
    state = game.reset()
    while not game.is_terminal(state):
        player = game.get_current_player(state)
        legal = game.get_legal_actions(state)
        action = controllers[player](game, state, player, legal)
        next_state, rewards, done, info = game.step(action)
        rec.record_step(state, player, action, rewards[player], done)
        state = next_state
        if done:
            break
    rec.finish(state)
    return rec.to_dict()


__all__ = ["EpisodeRecorder", "record_episode"]
