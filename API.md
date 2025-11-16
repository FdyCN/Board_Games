# API 接口文档

> **最后更新**: 2025-11-16
> **适用版本**: v0.1.0-dev

本文档提供核心 API 接口的详细说明和使用示例。

---

## 目录

- [核心抽象层 API](#核心抽象层-api)
  - [GameInterface](#gameinterface)
  - [AgentInterface](#agentinterface)
- [游戏层 API](#游戏层-api)
  - [游戏注册系统](#游戏注册系统)
  - [Splendor 游戏](#splendor-游戏)
- [Agent 层 API](#agent-层-api)
  - [NeuralAgent](#neuralagent)
  - [RandomAgent](#randomagent)
- [模型层 API](#模型层-api)
  - [模型工厂](#模型工厂)
  - [编码器](#编码器)
- [训练层 API](#训练层-api)
  - [PPO Trainer](#ppo-trainer)
  - [自对弈引擎](#自对弈引擎)
- [评估层 API](#评估层-api)
  - [Arena](#arena)
  - [ELO 系统](#elo-系统)
- [配置系统](#配置系统)
- [完整示例](#完整示例)

---

## 核心抽象层 API

### GameInterface

所有游戏必须实现的抽象基类。

#### 类签名

```python
from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import numpy as np

class GameInterface(ABC):
    """桌游抽象基类"""
```

#### 核心方法

##### `reset() -> GameState`

重置游戏到初始状态。

**参数**: 无

**返回**:
- `GameState`: 游戏初始状态对象

**示例**:
```python
game = SplendorGame(num_players=4)
state = game.reset()
print(f"游戏开始，当前玩家: {game.get_current_player(state)}")
```

---

##### `step(action) -> Tuple[GameState, List[float], bool, dict]`

执行一个动作，推进游戏状态。

**参数**:
- `action` (Any): 动作对象（类型由具体游戏定义）

**返回**:
- `new_state` (GameState): 执行动作后的新状态
- `rewards` (List[float]): 所有玩家的即时奖励，长度为 `num_players`
- `done` (bool): 游戏是否结束
- `info` (dict): 额外信息（如合法动作列表、游戏事件等）

**异常**:
- `ValueError`: 如果动作非法

**示例**:
```python
# 拿取宝石动作
action = TakeGemsAction(gems=[2, 2, 0, 0, 0])
new_state, rewards, done, info = game.step(action)

if done:
    print(f"游戏结束！最终奖励: {rewards}")
else:
    print(f"即时奖励: {rewards}, 下一个玩家: {game.get_current_player(new_state)}")
```

---

##### `get_legal_actions(state) -> List[Any]`

获取当前状态下的所有合法动作。

**参数**:
- `state` (GameState): 当前游戏状态

**返回**:
- `List[Any]`: 合法动作对象列表

**示例**:
```python
legal_actions = game.get_legal_actions(state)
print(f"当前有 {len(legal_actions)} 个合法动作")

# 随机选择一个动作
import random
action = random.choice(legal_actions)
```

---

##### `state_to_observation(state, player_id) -> np.ndarray`

将游戏状态转换为指定玩家的观察向量（神经网络输入）。

**参数**:
- `state` (GameState): 游戏状态
- `player_id` (int): 观察者玩家 ID（0, 1, 2, ...）

**返回**:
- `np.ndarray`: 观察向量/矩阵，形状为 `observation_shape`

**设计要点**:
- 观察应该是玩家视角（只包含该玩家可见的信息）
- 归一化到合理范围（通常 [0, 1] 或 [-1, 1]）
- 保持固定维度

**示例**:
```python
# 获取玩家 0 的观察
obs = game.state_to_observation(state, player_id=0)
print(f"观察形状: {obs.shape}")  # 例如 (256,)

# 输入神经网络
with torch.no_grad():
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    policy, value = model(obs_tensor)
```

---

##### `action_to_index(action) -> int`

将动作对象转换为动作索引（用于神经网络输出层）。

**参数**:
- `action` (Any): 动作对象

**返回**:
- `int`: 动作索引（0 到 `action_space_size - 1`）

**示例**:
```python
action = BuyCardAction(card_id=5, tier=2)
action_idx = game.action_to_index(action)
print(f"动作索引: {action_idx}")  # 例如 42
```

---

##### `index_to_action(index) -> Any`

将动作索引转换为动作对象。

**参数**:
- `index` (int): 动作索引

**返回**:
- `Any`: 动作对象

**示例**:
```python
# 从神经网络输出采样动作索引
action_idx = torch.multinomial(policy_probs, 1).item()

# 转换为动作对象
action = game.index_to_action(action_idx)
new_state, rewards, done, info = game.step(action)
```

---

#### 属性

##### `num_players: int`

游戏玩家数量。

```python
print(f"这是一个 {game.num_players} 人游戏")
```

---

##### `observation_shape: Tuple[int, ...]`

观察空间的形状。

```python
print(f"观察形状: {game.observation_shape}")
# 输出: (256,) 或 (8, 8, 12) 等
```

---

##### `action_space_size: int`

离散动作空间大小（所有可能动作的数量）。

```python
print(f"动作空间大小: {game.action_space_size}")
# 输出: 例如 82（Splendor 约 80+ 个动作）
```

---

#### 可选方法

##### `render(state, mode='human') -> None`

可视化游戏状态。

**参数**:
- `state` (GameState): 当前状态
- `mode` (str): 渲染模式
  - `'human'`: 终端输出（ASCII 艺术）
  - `'rgb_array'`: 返回图像数组
  - `'ansi'`: 返回 ANSI 字符串

**示例**:
```python
game.render(state, mode='human')
```

输出示例:
```
┌────────────── Splendor ──────────────┐
│ 当前玩家: Player 1                    │
│ 回合数: 15                            │
├──────────────────────────────────────┤
│ 宝石堆:                               │
│   红 ♦: 5  绿 ♣: 4  蓝 ♠: 3          │
│   白 ○: 6  黑 ●: 4  金 ★: 5          │
├──────────────────────────────────────┤
│ 公开卡牌 (Tier 1):                    │
│  [1] 红♦ 成本:3白 分数:0              │
│  [2] 绿♣ 成本:2蓝1白 分数:0           │
│  ...                                 │
└──────────────────────────────────────┘
```

---

### AgentInterface

所有 Agent 的抽象基类。

#### 类签名

```python
from abc import ABC, abstractmethod

class AgentInterface(ABC):
    """Agent 抽象基类"""
```

#### 核心方法

##### `select_action(observation, legal_actions, deterministic=False) -> Tuple[int, dict]`

根据观察选择动作。

**参数**:
- `observation` (np.ndarray): 游戏观察（已编码）
- `legal_actions` (List[int]): 合法动作索引列表
- `deterministic` (bool): 是否使用确定性策略
  - `True`: 选择概率最高的动作（评估时用）
  - `False`: 按概率分布采样（训练时用）

**返回**:
- `action_index` (int): 选择的动作索引
- `info` (dict): 额外信息
  - `'log_prob'` (float): 动作的对数概率
  - `'value'` (float): 状态价值估计
  - `'policy'` (np.ndarray): 完整策略分布（可选）

**示例**:
```python
agent = NeuralAgent(model)
obs = game.state_to_observation(state, player_id=0)
legal_actions = [game.action_to_index(a) for a in game.get_legal_actions(state)]

# 训练时：采样动作
action_idx, info = agent.select_action(obs, legal_actions, deterministic=False)
print(f"采样动作: {action_idx}, log_prob: {info['log_prob']:.3f}")

# 评估时：选择最优动作
action_idx, info = agent.select_action(obs, legal_actions, deterministic=True)
print(f"最优动作: {action_idx}, value: {info['value']:.3f}")
```

---

##### `save(path) -> None`

保存 Agent（模型权重、配置等）。

**参数**:
- `path` (str): 保存路径

**示例**:
```python
agent.save("data/checkpoints/agent_v1.pth")
```

---

##### `load(path) -> None`

加载 Agent。

**参数**:
- `path` (str): 加载路径

**示例**:
```python
agent = NeuralAgent(model)
agent.load("data/checkpoints/agent_v1.pth")
```

---

## 游戏层 API

### 游戏注册系统

#### `register_game(name: str)`

装饰器：将游戏类注册到全局注册表。

**参数**:
- `name` (str): 游戏名称（用于配置文件引用）

**示例**:
```python
from games.registry import register_game
from core.game_interface import GameInterface

@register_game("my_game")
class MyGame(GameInterface):
    def __init__(self, num_players: int = 4):
        self.num_players_val = num_players

    # 实现所有抽象方法...
```

---

#### `create_game(name: str, **kwargs) -> GameInterface`

工厂函数：根据名称创建游戏实例。

**参数**:
- `name` (str): 游戏名称
- `**kwargs`: 游戏构造参数

**返回**:
- `GameInterface`: 游戏实例

**异常**:
- `ValueError`: 如果游戏未注册

**示例**:
```python
from games.registry import create_game

# 创建 Splendor 游戏
game = create_game("splendor", num_players=4)

# 创建 UNO 游戏
game = create_game("uno", num_players=3)

# 查看所有可用游戏
from games.registry import GAME_REGISTRY
print(f"可用游戏: {list(GAME_REGISTRY.keys())}")
```

---

### Splendor 游戏

#### 动作类型

Splendor 支持以下动作类型：

```python
from games.splendor.actions import (
    TakeGemsAction,      # 拿宝石
    BuyCardAction,       # 购买卡牌
    ReserveCardAction,   # 保留卡牌
    PassAction           # 跳过（极少使用）
)
```

##### TakeGemsAction

拿取宝石动作。

```python
# 拿 3 种不同颜色的宝石（每种1个）
action = TakeGemsAction(gems=[1, 1, 1, 0, 0])

# 拿 2 个同色宝石（需要宝石堆 >= 4）
action = TakeGemsAction(gems=[2, 0, 0, 0, 0])
```

**约束**:
- 最多拿 3 颗宝石
- 拿 2 个同色需要该颜色宝石堆 >= 4
- 手牌上限 10 颗宝石

---

##### BuyCardAction

购买公开或保留的卡牌。

```python
# 购买公开卡牌
action = BuyCardAction(
    card_id=5,          # 卡牌 ID
    tier=1,             # 卡牌等级（1, 2, 3）
    from_reserved=False # 是否从保留区购买
)

# 购买保留卡牌
action = BuyCardAction(
    card_id=12,
    tier=2,
    from_reserved=True
)
```

---

##### ReserveCardAction

保留卡牌（获得 1 个金宝石）。

```python
action = ReserveCardAction(
    card_id=8,
    tier=2
)
```

**约束**:
- 最多保留 3 张卡牌
- 获得 1 个金宝石（如果有）

---

#### 游戏状态结构

```python
@dataclass
class SplendorState:
    players: List[PlayerState]          # 玩家状态列表
    gem_bank: List[int]                 # 宝石堆 [红,绿,蓝,白,黑,金]
    noble_tiles: List[NobleTile]        # 贵族卡
    open_cards: Dict[int, List[Card]]   # 公开卡牌 {tier: [cards]}
    card_decks: Dict[int, List[Card]]   # 卡牌堆
    current_player: int                 # 当前玩家索引
    turn_number: int                    # 回合数
```

```python
@dataclass
class PlayerState:
    gems: List[int]                     # 持有宝石 [红,绿,蓝,白,黑,金]
    cards: List[Card]                   # 已购买卡牌
    reserved_cards: List[Card]          # 保留卡牌
    nobles: List[NobleTile]             # 拥有的贵族

    @property
    def score(self) -> int:
        """计算总分"""
        return sum(card.points for card in self.cards) + \
               sum(noble.points for noble in self.nobles)

    @property
    def bonuses(self) -> List[int]:
        """计算各颜色加成"""
        bonus = [0] * 5
        for card in self.cards:
            bonus[card.color] += 1
        return bonus
```

---

## Agent 层 API

### NeuralAgent

基于神经网络的 Agent。

#### 构造函数

```python
class NeuralAgent(AgentInterface):
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu"
    ):
        """
        Args:
            model: PyTorch 模型（Actor-Critic）
            device: 设备 ('cpu', 'cuda', 'mps')
        """
```

#### 使用示例

```python
import torch
from agents.neural_agent import NeuralAgent
from models.model_factory import build_model

# 创建模型
model = build_model(
    encoder_type="attention",
    obs_shape=(256,),
    action_size=82,
    hidden_size=256
)

# 创建 Agent
device = "mps" if torch.backends.mps.is_available() else "cpu"
agent = NeuralAgent(model, device=device)

# 使用 Agent
obs = game.state_to_observation(state, player_id=0)
legal_actions = [0, 1, 5, 10, 15]  # 合法动作索引
action_idx, info = agent.select_action(obs, legal_actions)

print(f"选择动作: {action_idx}")
print(f"状态价值: {info['value']:.3f}")
```

---

### RandomAgent

随机策略 Agent（Baseline）。

#### 使用示例

```python
from agents.random_agent import RandomAgent

agent = RandomAgent()

# 从合法动作中随机选择
obs = game.state_to_observation(state, 0)
legal_actions = [0, 1, 2, 3]
action_idx, info = agent.select_action(obs, legal_actions)

print(f"随机选择: {action_idx}")
# info 为空字典
```

---

## 模型层 API

### 模型工厂

#### `build_model(**config) -> nn.Module`

根据配置构建神经网络模型。

**参数**:
- `encoder_type` (str): 编码器类型 ('mlp', 'attention', 'cnn')
- `obs_shape` (Tuple[int, ...]): 观察空间形状
- `action_size` (int): 动作空间大小
- `hidden_size` (int): 隐藏层维度
- `num_layers` (int): 层数（可选）
- `dropout` (float): Dropout 比例（可选）

**返回**:
- `nn.Module`: Actor-Critic 模型

**示例**:
```python
from models.model_factory import build_model

# MLP 编码器（最简单）
model = build_model(
    encoder_type="mlp",
    obs_shape=(256,),
    action_size=82,
    hidden_size=256,
    num_layers=3
)

# Attention 编码器（适合卡牌游戏）
model = build_model(
    encoder_type="attention",
    obs_shape=(256,),
    action_size=82,
    hidden_size=256,
    num_heads=4,
    num_layers=2
)

# CNN 编码器（适合棋盘游戏）
model = build_model(
    encoder_type="cnn",
    obs_shape=(8, 8, 12),  # 棋盘大小 × 通道数
    action_size=64,
    hidden_size=256
)

# 检查参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")  # 例如 856,834
```

---

### 编码器

#### MLPEncoder

多层感知机编码器。

```python
from models.encoders.mlp_encoder import MLPEncoder

encoder = MLPEncoder(
    input_dim=256,
    hidden_dim=256,
    num_layers=3,
    activation='relu'
)

# 前向传播
import torch
x = torch.randn(32, 256)  # batch_size=32
features = encoder(x)     # (32, 256)
```

---

#### AttentionEncoder

自注意力编码器（适合卡牌游戏）。

```python
from models.encoders.attention_encoder import AttentionEncoder

encoder = AttentionEncoder(
    input_dim=256,
    hidden_dim=256,
    num_heads=4,
    num_layers=2,
    dropout=0.1
)

x = torch.randn(32, 256)
features = encoder(x)  # (32, 256)
```

---

## 训练层 API

### PPO Trainer

#### 构造函数

```python
from training.ppo_trainer import PPOTrainer

trainer = PPOTrainer(
    game=game,
    model=model,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    num_workers=4,
    episodes_per_update=100,
    epochs_per_update=4,
    batch_size=256,
    device="mps"
)
```

#### 训练方法

```python
# 开始训练
trainer.train(
    total_episodes=10000,
    eval_interval=100,        # 每 100 episodes 评估一次
    checkpoint_interval=500,  # 每 500 episodes 保存一次
    save_dir="data/checkpoints"
)
```

#### 回调函数

```python
def on_episode_end(episode_num, metrics):
    print(f"Episode {episode_num}: reward={metrics['mean_reward']:.2f}")

def on_eval_end(eval_results):
    print(f"Eval: win_rate={eval_results['win_rate']:.2%}")

trainer.train(
    total_episodes=10000,
    callbacks={
        'on_episode_end': on_episode_end,
        'on_eval_end': on_eval_end
    }
)
```

---

### 自对弈引擎

#### SelfPlayEngine

```python
from training.self_play_engine import SelfPlayEngine

engine = SelfPlayEngine(
    game=game,
    agents=[agent1, agent2, agent3, agent4],
    num_workers=4
)

# 收集经验
experiences = engine.collect_episodes(num_episodes=100)

# 经验格式
for exp in experiences[:3]:
    print(f"Player: {exp.player_id}, Reward: {exp.reward:.2f}")
```

---

## 评估层 API

### Arena

多模型对战竞技场。

#### 使用示例

```python
from evaluation.arena import Arena
from games.registry import create_game

# 创建竞技场
arena = Arena(
    game=create_game("splendor", num_players=4),
    num_games=100
)

# 运行锦标赛
agents = [agent_v1, agent_v2, agent_v3, agent_v4]
results = arena.run_tournament(agents)

print(f"胜率统计:")
for agent_id, win_rate in results.win_rates.items():
    print(f"  Agent {agent_id}: {win_rate:.2%}")
```

---

### ELO 系统

#### 使用示例

```python
from evaluation.elo_system import ELOSystem

elo = ELOSystem(k_factor=32)

# 初始化模型评分
elo.add_model("v1", initial_rating=1500)
elo.add_model("v2", initial_rating=1500)

# 记录对局结果
elo.update_rating("v1", "v2", result=1.0)  # v1 赢
elo.update_rating("v2", "v1", result=0.0)  # v2 输

# 查看评分
print(f"Model v1: {elo.get_rating('v1'):.0f}")
print(f"Model v2: {elo.get_rating('v2'):.0f}")

# 获取排行榜
leaderboard = elo.get_leaderboard()
for rank, (model_id, rating) in enumerate(leaderboard, 1):
    print(f"{rank}. {model_id}: {rating:.0f}")
```

---

## 配置系统

### 配置文件结构

#### 游戏配置

```yaml
# configs/games/splendor.yaml
game:
  name: "splendor"
  num_players: 4

  # 游戏特定参数
  initial_gems: 7        # 初始宝石数（2人局）
  noble_count: 3         # 贵族卡数量
  win_score: 15          # 获胜分数
```

#### 模型配置

```yaml
# configs/models/attention.yaml
model:
  encoder_type: "attention"
  hidden_size: 256
  num_heads: 4
  num_layers: 2
  dropout: 0.1

  # 策略头
  policy_head:
    hidden_sizes: [128]
    activation: "relu"

  # 价值头
  value_head:
    hidden_sizes: [128]
    activation: "relu"
```

#### 训练配置

```yaml
# configs/experiments/splendor_ppo.yaml
defaults:
  - game: splendor
  - model: attention

training:
  algorithm: "ppo"

  # PPO 超参数
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01

  # 训练流程
  total_episodes: 10000
  episodes_per_update: 100
  epochs_per_update: 4
  batch_size: 256

  # 分布式
  num_workers: 4

  # 评估
  eval_interval: 100
  eval_episodes: 50

  # 保存
  checkpoint_interval: 500
  save_dir: "data/checkpoints"

# 覆盖特定参数
model:
  hidden_size: 512  # 使用更大的模型
```

### 加载配置

```python
from utils.config_loader import load_config

# 从 YAML 加载
config = load_config("configs/experiments/splendor_ppo.yaml")

# 访问配置
print(config.game.name)              # "splendor"
print(config.model.hidden_size)      # 512
print(config.training.learning_rate) # 3e-4

# 命令行覆盖
config = load_config(
    "configs/experiments/splendor_ppo.yaml",
    overrides={
        "training.learning_rate": 1e-4,
        "training.num_workers": 8
    }
)
```

---

## 完整示例

### 示例 1: 简单对局

```python
from games.registry import create_game
from agents.random_agent import RandomAgent

# 创建游戏和 Agent
game = create_game("splendor", num_players=4)
agents = [RandomAgent() for _ in range(4)]

# 运行一局游戏
state = game.reset()
episode_rewards = [0.0] * 4

while not game.is_terminal(state):
    # 当前玩家
    player_id = game.get_current_player(state)

    # 获取观察和合法动作
    obs = game.state_to_observation(state, player_id)
    legal_actions = [game.action_to_index(a) for a in game.get_legal_actions(state)]

    # Agent 选择动作
    action_idx, _ = agents[player_id].select_action(obs, legal_actions)
    action = game.index_to_action(action_idx)

    # 执行动作
    state, rewards, done, info = game.step(action)

    # 累积奖励
    for i, r in enumerate(rewards):
        episode_rewards[i] += r

# 打印结果
print(f"游戏结束！最终奖励: {episode_rewards}")
winner = episode_rewards.index(max(episode_rewards))
print(f"获胜者: Player {winner}")
```

---

### 示例 2: 完整训练流程

```python
import torch
from games.registry import create_game
from models.model_factory import build_model
from training.ppo_trainer import PPOTrainer
from utils.config_loader import load_config

# 1. 加载配置
config = load_config("configs/experiments/splendor_ppo.yaml")

# 2. 创建游戏
game = create_game(config.game.name, num_players=config.game.num_players)

# 3. 创建模型
model = build_model(
    encoder_type=config.model.encoder_type,
    obs_shape=game.observation_shape,
    action_size=game.action_space_size,
    **config.model
)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 4. 创建训练器
device = "mps" if torch.backends.mps.is_available() else "cpu"
trainer = PPOTrainer(
    game=game,
    model=model,
    device=device,
    **config.training
)

# 5. 开始训练
trainer.train(
    total_episodes=config.training.total_episodes,
    eval_interval=config.training.eval_interval,
    checkpoint_interval=config.training.checkpoint_interval
)
```

---

### 示例 3: 模型评估

```python
from evaluation.arena import Arena
from evaluation.elo_system import ELOSystem
from agents.neural_agent import NeuralAgent
from agents.random_agent import RandomAgent

# 加载多个版本的模型
models = []
for version in [100, 200, 300, 400, 500]:
    model = build_model(...)
    model.load_state_dict(torch.load(f"checkpoints/model_v{version}.pth"))
    models.append(model)

# 创建 Agents
agents = [NeuralAgent(m) for m in models]
agents.append(RandomAgent())  # 添加随机 baseline

# 创建竞技场
arena = Arena(game, num_games=200)

# 运行锦标赛
results = arena.run_tournament(agents)

# 计算 ELO 分数
elo = ELOSystem()
for i, agent in enumerate(agents):
    elo.add_model(f"v{i}", initial_rating=1500)

# 根据对局结果更新 ELO
for game_result in results.game_results:
    elo.update_from_game(game_result)

# 打印排行榜
print("\n📊 ELO 排行榜:")
for rank, (model_id, rating) in enumerate(elo.get_leaderboard(), 1):
    print(f"{rank}. {model_id}: {rating:.0f}")
```

---

### 示例 4: 人机对战

```python
from agents.human_agent import HumanAgent
from visualization.game_renderer import SplendorRenderer

# 创建 Agents
human = HumanAgent()
ai_agents = [NeuralAgent(model) for _ in range(3)]
agents = [human] + ai_agents

# 创建渲染器
renderer = SplendorRenderer()

# 运行游戏
state = game.reset()

while not game.is_terminal(state):
    # 渲染当前状态
    renderer.render(state)

    player_id = game.get_current_player(state)
    obs = game.state_to_observation(state, player_id)
    legal_actions = [game.action_to_index(a) for a in game.get_legal_actions(state)]

    # 选择动作
    if player_id == 0:  # 人类玩家
        print("\n你的回合！")
        action_idx, _ = agents[player_id].select_action(obs, legal_actions)
    else:  # AI 玩家
        action_idx, info = agents[player_id].select_action(
            obs, legal_actions, deterministic=True
        )
        print(f"\nAI {player_id} 思考中... (价值估计: {info['value']:.2f})")

    # 执行动作
    action = game.index_to_action(action_idx)
    state, rewards, done, info = game.step(action)

# 游戏结束
renderer.render(state)
print("\n🎉 游戏结束！")
```

---

## API 版本兼容性

| 版本 | 发布日期 | 主要变更 | 向后兼容 |
|------|---------|---------|---------|
| v0.1.0-dev | 2025-11-16 | 初始版本 | N/A |

---

## 常见错误处理

### 错误 1: 动作非法

```python
try:
    state, rewards, done, info = game.step(action)
except ValueError as e:
    print(f"非法动作: {e}")
    # 回退或选择其他动作
```

### 错误 2: 模型输入维度不匹配

```python
# 检查观察形状
assert obs.shape == game.observation_shape, \
    f"观察形状错误: {obs.shape} vs {game.observation_shape}"

# 检查动作索引范围
assert 0 <= action_idx < game.action_space_size, \
    f"动作索引越界: {action_idx}"
```

### 错误 3: 设备不匹配

```python
# 确保数据和模型在同一设备
obs_tensor = torch.FloatTensor(obs).to(device)
model = model.to(device)
```

---

## 性能优化技巧

### 1. 批量推理

```python
# ❌ 慢：逐个推理
for obs in observations:
    action, _ = agent.select_action(obs, legal_actions)

# ✅ 快：批量推理
obs_batch = np.stack(observations)
obs_tensor = torch.FloatTensor(obs_batch).to(device)
with torch.no_grad():
    policy, value = model(obs_tensor)
```

### 2. 禁用梯度计算（推理时）

```python
with torch.no_grad():
    action_idx, info = agent.select_action(obs, legal_actions)
```

### 3. 使用 JIT 编译

```python
model = torch.jit.script(model)  # 加速推理 10-30%
```

---

## 扩展 API

### 添加自定义游戏

1. 继承 `GameInterface`
2. 实现所有抽象方法
3. 注册游戏

```python
@register_game("my_custom_game")
class MyCustomGame(GameInterface):
    # 实现...
```

### 添加自定义 Agent

```python
class MyCustomAgent(AgentInterface):
    def select_action(self, obs, legal_actions, deterministic=False):
        # 自定义策略逻辑
        pass
```

---

**文档完成！** 🎉

如需更多帮助，请查看：
- [README.md](./README.md) - 项目概览
- [ARCHITECTURE.md](./ARCHITECTURE.md) - 架构设计
- [DEVELOPMENT.md](./DEVELOPMENT.md) - 开发指南
