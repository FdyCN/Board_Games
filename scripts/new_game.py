#!/usr/bin/env python3
"""
新游戏脚手架生成器。

用法:
    python scripts/new_game.py love_letter --players 4
    python scripts/new_game.py coup --players 4

会生成：
    games/<name>/             # 引擎骨架（game/state/actions/encoder/RULES）
    configs/<name>/           # 训练配置骨架
    tests/test_games/         # 测试骨架
    data/<name>/              # 产物目录（checkpoints/logs/reports + .gitkeep）

生成后按 TODO 提示逐个实现即可，详见 docs/ADDING_A_GAME.md。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _slug(name: str) -> str:
    """游戏名 → 合法 Python 模块名（snake_case）。"""
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    return re.sub(r"[^a-z0-9_]", "", name)


def _class_name(slug: str) -> str:
    """模块名 → PascalCase 类名。"""
    return "".join(part.capitalize() for part in slug.split("_"))


GAME_TEMPLATE = '''"""
{game_name} 游戏引擎。

TODO: 实现完整规则。参考 games/splendor/game.py 与 docs/ADDING_A_GAME.md。
"""

from __future__ import annotations

from core.game_interface import GameInterface
from games.registry import register_game


@register_game("{game_name}")
class {class_name}(GameInterface):
    """{game_name} 游戏（{num_players} 人）。"""

    ACTION_SPACE_SIZE = 16  # TODO: 改为实际的固定动作空间大小

    def __init__(self, num_players: int = {num_players}, seed: int | None = None,
                 reward_config: dict | None = None):
        if not 2 <= num_players <= {num_players}:
            raise ValueError(f"玩家数量必须在 2-{num_players} 之间，当前: {{num_players}}")
        self._num_players = num_players
        self._seed = seed
        # TODO: 定义默认奖励；PPO 训练时可被配置覆盖
        self._rewards = {{"win": 1.0, "step_penalty": -0.01}}
        if reward_config:
            self._rewards.update(reward_config)
        self._state = None

    # ===== 核心游戏循环 =====

    def reset(self):
        raise NotImplementedError("TODO: 实现 reset()")

    def step(self, action):
        raise NotImplementedError("TODO: 实现 step()，返回 (state, rewards, done, info)")

    # ===== 状态查询 =====

    def get_legal_actions(self, state):
        raise NotImplementedError("TODO: 返回合法动作对象列表")

    def get_current_player(self, state):
        raise NotImplementedError("TODO: 返回当前玩家 ID")

    def is_terminal(self, state):
        raise NotImplementedError("TODO: 判断游戏是否结束")

    def get_final_rewards(self, state):
        raise NotImplementedError("TODO: 返回所有玩家的终局奖励")

    # ===== 观察与动作编码 =====

    def state_to_observation(self, state, player_id):
        raise NotImplementedError("TODO: 返回该玩家视角的观察向量")

    def action_to_index(self, action, state=None):
        raise NotImplementedError("TODO: 动作对象 → 固定槽位索引")

    def index_to_action(self, index, state=None):
        raise NotImplementedError("TODO: 索引 → 动作对象")

    # ===== 属性 =====

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (128,)  # TODO: 改为实际观察维度

    @property
    def action_space_size(self) -> int:
        return self.ACTION_SPACE_SIZE

    def game_kwargs(self) -> dict:
        """多进程重建所需的构造参数。"""
        kwargs = super().game_kwargs()
        if self._seed is not None:
            kwargs["seed"] = self._seed
        kwargs["reward_config"] = dict(self._rewards)
        return kwargs
'''

STATE_TEMPLATE = '''"""游戏状态数据结构。

TODO: 定义不可变（或提供快速 clone 的）状态对象。
参考 games/splendor/state.py：状态应支持浅拷贝 clone()，供 MCTS 使用。
"""

from dataclasses import dataclass


@dataclass
class PlayerState:
    pass


@dataclass
class GameState:
    players: list
    current_player: int = 0

    def clone(self):
        import copy
        return copy.deepcopy(self)
'''

ACTIONS_TEMPLATE = '''"""动作定义。

TODO: 定义动作类型。为让策略网络学到稳定语义，建议使用「固定动作空间」
（每个索引语义固定），参考 games/splendor/actions.py 与 game.py 的
action_to_index / index_to_action。
"""
'''

ENCODER_TEMPLATE = '''"""状态编码器：state + player_id → 固定维度观察向量。

TODO: 实现 encode(state, player_id)。
注意：观察必须是固定维度、按玩家视角（只含该玩家可见信息），数值归一化。
"""
'''

RULES_TEMPLATE = '''# {game_name} 规则

TODO: 整理官方规则：组件、回合流程、动作、结束条件、计分。
'''

CONFIG_TEMPLATE = '''# {game_name} PPO 训练配置

game:
  name: "{game_name}"
  num_players: {num_players}
  seed: null

model:
  encoder_type: "mlp"   # 选项: "mlp", "attention"
  config: "medium"      # 选项: "small", "medium", "large"

algorithm:
  name: "ppo"
  learning_rate: 0.0001
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.1
  value_clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  use_value_clip: true
  outcome_coef: 1.0

  # 稠密奖励（自由字典，键由本游戏定义）
  dense_rewards:
    win: 1.0
    step_penalty: -0.01

training:
  num_iterations: 500
  episodes_per_iteration: 128
  update_epochs: 4
  minibatch_size: 256
  device: "cpu"
  use_position_augmentation: false
  num_workers: 2
  checkpoint_dir: "data/{game_name}/checkpoints/{game_name}_v1"
  checkpoint_interval: 50
  log_interval: 1
  verbose: true

evaluation:
  eval_interval: 50
  eval_episodes: 100
  deterministic: true

experiment:
  name: "{game_name}_ppo_v1"
  tags: ["{game_name}", "ppo"]
  notes: ""
'''

TEST_TEMPLATE = '''"""基础测试骨架。

TODO: 实现引擎后补全测试。参考 tests/test_games/test_splendor.py。
"""

import pytest

from games.registry import create_game


def test_game_registered():
    from games.registry import is_registered
    assert is_registered("{game_name}")


def test_initial_state():
    game = create_game("{game_name}", num_players={num_players})
    state = game.reset()
    assert game.num_players == {num_players}
'''

INIT_TEMPLATE = '''"""游戏包：{game_name}。"""
'''


def main() -> int:
    ap = argparse.ArgumentParser(description="生成新游戏脚手架")
    ap.add_argument("name", help="游戏名（snake_case，如 love_letter）")
    ap.add_argument("--players", type=int, default=4, help="默认玩家数")
    args = ap.parse_args()

    slug = _slug(args.name)
    if not slug:
        print("错误: 游戏名不能为空", file=sys.stderr)
        return 1

    cls = _class_name(slug)

    files = {
        PROJECT_ROOT / f"games/{slug}/__init__.py": INIT_TEMPLATE.format(game_name=slug),
        PROJECT_ROOT / f"games/{slug}/game.py": GAME_TEMPLATE.format(
            game_name=slug, class_name=cls, num_players=args.players,
        ),
        PROJECT_ROOT / f"games/{slug}/state.py": STATE_TEMPLATE,
        PROJECT_ROOT / f"games/{slug}/actions.py": ACTIONS_TEMPLATE,
        PROJECT_ROOT / f"games/{slug}/encoder.py": ENCODER_TEMPLATE,
        PROJECT_ROOT / f"games/{slug}/RULES.md": RULES_TEMPLATE.format(game_name=slug),
        PROJECT_ROOT / f"configs/{slug}/{slug}_ppo.yaml": CONFIG_TEMPLATE.format(
            game_name=slug, num_players=args.players,
        ),
        PROJECT_ROOT / f"tests/test_games/test_{slug}.py": TEST_TEMPLATE.format(
            game_name=slug, num_players=args.players,
        ),
    }

    for path, content in files.items():
        if path.exists():
            print(f"跳过（已存在）: {path.relative_to(PROJECT_ROOT)}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"创建: {path.relative_to(PROJECT_ROOT)}")

    # 产物目录 + .gitkeep
    for sub in ("checkpoints", "logs", "reports"):
        d = PROJECT_ROOT / f"data/{slug}/{sub}"
        d.mkdir(parents=True, exist_ok=True)
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")
            print(f"创建: {gitkeep.relative_to(PROJECT_ROOT)}")

    print("\n完成！下一步（详见 docs/ADDING_A_GAME.md）：")
    print(f"  1. 实现 games/{slug}/game.py 的抽象方法")
    print(f"  2. 实现 games/{slug}/encoder.py 的观察编码")
    print(f"  3. 补全 tests/test_games/test_{slug}.py")
    print(f"  4. 用 configs/{slug}/{slug}_ppo.yaml 训练：")
    print(f"       python scripts/train_convergence.py --config configs/{slug}/{slug}_ppo.yaml \\")
    print(f"           --iterations 100 --episodes 64 --log-file data/{slug}/logs/{slug}_v1.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
