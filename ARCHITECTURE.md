# 架构设计文档

## 目录

- [设计理念](#设计理念)
- [系统架构](#系统架构)
- [核心抽象层](#核心抽象层)
- [游戏层](#游戏层)
- [模型层](#模型层)
- [训练层](#训练层)
- [评估层](#评估层)
- [数据流](#数据流)
- [设计模式](#设计模式)

---

## 设计理念

本框架遵循以下核心设计原则：

### 1. **依赖倒置原则 (DIP)**
高层模块（训练器、评估器）不依赖低层模块（具体游戏），双方都依赖抽象接口。

```python
# ✅ 正确：依赖抽象
class PPOTrainer:
    def __init__(self, game: GameInterface):
        self.game = game  # 可以是任何实现了 GameInterface 的游戏

# ❌ 错误：依赖具体实现
class PPOTrainer:
    def __init__(self):
        self.game = SplendorGame()  # 硬编码，无法扩展
```

### 2. **开闭原则 (OCP)**
对扩展开放，对修改封闭。添加新游戏不需要修改核心训练代码。

### 3. **单一职责原则 (SRP)**
每个模块只负责一个明确的功能：
- 游戏层：只负责游戏规则
- 模型层：只负责神经网络结构
- 训练层：只负责训练逻辑

### 4. **配置驱动**
所有可变参数通过 YAML 配置，避免硬编码。

---

## 系统架构

### 整体分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Scripts)                          │
│              train.py, evaluate.py, play_human.py            │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                 业务逻辑层 (Training/Evaluation)              │
│         PPO, 自对弈引擎, Arena, ELO, 可视化                   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                   模型层 (Models/Agents)                      │
│          神经网络架构, Agent 封装, 模型工厂                    │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                  核心抽象层 (Core Interfaces)                 │
│              GameInterface, AgentInterface                   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                   游戏层 (Game Implementations)               │
│                Splendor, UNO, 狼人杀...                       │
└─────────────────────────────────────────────────────────────┘
```

### 模块依赖关系

```
┌──────────────┐
│   Scripts    │
└──────┬───────┘
       │
       ├─────────────┬─────────────┬─────────────┐
       ▼             ▼             ▼             ▼
 ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
 │ Training │ │Evaluation│ │  Models  │ │  Agents  │
 └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘
       │            │            │            │
       └────────────┴────────────┴────────────┘
                     ▼
              ┌──────────────┐
              │  Core (ABC)  │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │    Games     │
              └──────────────┘
```

---

## 核心抽象层

### GameInterface - 游戏抽象接口

所有游戏必须实现的标准接口：

```python
from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import numpy as np

class GameInterface(ABC):
    """所有桌游的抽象基类"""

    @abstractmethod
    def reset(self) -> 'GameState':
        """
        重置游戏到初始状态

        Returns:
            GameState: 初始游戏状态
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple['GameState', List[float], bool, dict]:
        """
        执行一个动作，推进游戏状态

        Args:
            action: 动作（类型由具体游戏定义）

        Returns:
            new_state: 新的游戏状态
            rewards: 所有玩家的即时奖励 [r1, r2, r3, r4]
            done: 游戏是否结束
            info: 额外信息字典
        """
        pass

    @abstractmethod
    def get_legal_actions(self, state: 'GameState') -> List[Any]:
        """
        获取当前状态下的所有合法动作

        Args:
            state: 当前游戏状态

        Returns:
            合法动作列表
        """
        pass

    @abstractmethod
    def get_current_player(self, state: 'GameState') -> int:
        """
        获取当前应该行动的玩家 ID

        Args:
            state: 当前游戏状态

        Returns:
            玩家 ID (0, 1, 2, ...)
        """
        pass

    @abstractmethod
    def is_terminal(self, state: 'GameState') -> bool:
        """
        判断游戏是否结束

        Args:
            state: 当前游戏状态

        Returns:
            True 如果游戏结束
        """
        pass

    @abstractmethod
    def get_final_rewards(self, state: 'GameState') -> List[float]:
        """
        获取游戏结束时的最终奖励

        Args:
            state: 终局状态

        Returns:
            所有玩家的最终奖励 [1, 0, 0, 0] (胜者为1，其他为0)
        """
        pass

    @abstractmethod
    def state_to_observation(self, state: 'GameState', player_id: int) -> np.ndarray:
        """
        将游戏状态转换为指定玩家的观察（神经网络输入）

        Args:
            state: 游戏状态
            player_id: 观察者玩家 ID

        Returns:
            观察向量/矩阵（numpy array）
        """
        pass

    @abstractmethod
    def action_to_index(self, action: Any) -> int:
        """
        将动作对象转换为动作索引（用于神经网络输出）

        Args:
            action: 动作对象

        Returns:
            动作索引 (0 到 action_space_size-1)
        """
        pass

    @abstractmethod
    def index_to_action(self, index: int) -> Any:
        """
        将动作索引转换为动作对象

        Args:
            index: 动作索引

        Returns:
            动作对象
        """
        pass

    @property
    @abstractmethod
    def num_players(self) -> int:
        """游戏玩家数量"""
        pass

    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        """观察空间的形状，例如 (256,) 或 (8, 8, 12)"""
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """动作空间大小（离散动作总数）"""
        pass

    def clone_state(self, state: 'GameState') -> 'GameState':
        """
        深拷贝游戏状态（用于 MCTS 等搜索算法）

        Args:
            state: 原始状态

        Returns:
            状态的深拷贝
        """
        import copy
        return copy.deepcopy(state)

    def render(self, state: 'GameState', mode: str = 'human') -> None:
        """
        可视化游戏状态（可选实现）

        Args:
            state: 当前状态
            mode: 渲染模式 ('human', 'ascii', 'rgb_array')
        """
        pass
```

### AgentInterface - Agent 抽象接口

```python
from abc import ABC, abstractmethod
import numpy as np

class AgentInterface(ABC):
    """所有 Agent 的抽象基类"""

    @abstractmethod
    def select_action(
        self,
        observation: np.ndarray,
        legal_actions: List[int],
        deterministic: bool = False
    ) -> Tuple[int, dict]:
        """
        根据观察选择动作

        Args:
            observation: 游戏观察（已编码）
            legal_actions: 合法动作索引列表
            deterministic: 是否使用确定性策略（评估时用）

        Returns:
            action_index: 选择的动作索引
            info: 额外信息 (如概率分布、价值估计等)
        """
        pass

    def reset(self) -> None:
        """重置 Agent 内部状态（如 RNN 隐状态）"""
        pass

    def learn(self, experiences: List[dict]) -> dict:
        """
        从经验中学习（可选，用于在线学习）

        Args:
            experiences: 经验列表

        Returns:
            训练指标字典
        """
        return {}

    def save(self, path: str) -> None:
        """保存 Agent（模型权重等）"""
        pass

    def load(self, path: str) -> None:
        """加载 Agent"""
        pass
```

---

## 游戏层

### 游戏注册机制

```python
# games/registry.py
GAME_REGISTRY = {}

def register_game(name: str):
    """装饰器：注册游戏到全局注册表"""
    def decorator(cls):
        if not issubclass(cls, GameInterface):
            raise TypeError(f"{cls} must inherit from GameInterface")
        GAME_REGISTRY[name] = cls
        return cls
    return decorator

def create_game(name: str, **kwargs) -> GameInterface:
    """工厂函数：根据名称创建游戏实例"""
    if name not in GAME_REGISTRY:
        raise ValueError(f"Game '{name}' not found. Available: {list(GAME_REGISTRY.keys())}")
    return GAME_REGISTRY[name](**kwargs)
```

### Splendor 游戏实现示例

```python
# games/splendor/game.py
from core.game_interface import GameInterface
from games.registry import register_game

@register_game("splendor")
class SplendorGame(GameInterface):
    def __init__(self, num_players: int = 4):
        self.num_players_val = num_players
        # 初始化卡牌、宝石等游戏元素

    def reset(self):
        # 洗牌、发卡、放置宝石
        return SplendorState(...)

    def step(self, action):
        # 实现游戏规则：拿宝石、买卡、保留卡
        # 返回 (new_state, rewards, done, info)
        pass

    # ... 实现所有抽象方法
```

---

## 模型层

### 模型架构组成

```
输入观察 (observation)
       ▼
┌─────────────┐
│   Encoder   │  ← 可选: MLP, CNN, Attention
│  (共享表征)  │
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Policy  │    │  Value  │    │ Auxiliary│
│  Head   │    │  Head   │    │  Heads  │
└────┬────┘    └────┬────┘    └─────────┘
     │              │
     ▼              ▼
动作概率分布      状态价值
π(a|s)           V(s)
```

### Encoder 选择策略

| 游戏类型 | 推荐 Encoder | 原因 |
|---------|-------------|------|
| **卡牌游戏** (Splendor, UNO) | Attention / Transformer | 卡牌之间有关联，需要建模交互 |
| **棋盘游戏** (围棋, 象棋) | CNN | 空间局部性特征明显 |
| **简单状态** (小型游戏) | MLP | 计算高效，参数少 |

### 模型工厂

```python
# models/model_factory.py
def build_model(
    encoder_type: str,
    obs_shape: Tuple[int, ...],
    action_size: int,
    hidden_size: int = 256,
    **kwargs
) -> nn.Module:
    """
    根据配置构建模型

    Args:
        encoder_type: 'mlp', 'cnn', 'attention'
        obs_shape: 观察空间形状
        action_size: 动作空间大小
        hidden_size: 隐藏层大小

    Returns:
        PyTorch 模型
    """
    # 构建编码器
    if encoder_type == 'mlp':
        encoder = MLPEncoder(obs_shape, hidden_size)
    elif encoder_type == 'attention':
        encoder = AttentionEncoder(obs_shape, hidden_size)
    elif encoder_type == 'cnn':
        encoder = CNNEncoder(obs_shape, hidden_size)
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")

    # 构建策略头和价值头
    policy_head = PolicyHead(hidden_size, action_size)
    value_head = ValueHead(hidden_size)

    return ActorCriticModel(encoder, policy_head, value_head)
```

---

## 训练层

### PPO 训练流程

```
1. 并行自对弈阶段
   ┌─────────────┐
   │  Worker 1   │ ──┐
   │  (4 agents) │   │
   └─────────────┘   │
   ┌─────────────┐   │    ┌──────────────┐
   │  Worker 2   │ ──┼───▶│ Experience   │
   │  (4 agents) │   │    │    Buffer    │
   └─────────────┘   │    └──────┬───────┘
   ┌─────────────┐   │           │
   │  Worker N   │ ──┘           │
   │  (4 agents) │               │
   └─────────────┘               │
                                 ▼
2. 策略更新阶段
   ┌──────────────────────────────────┐
   │  PPO Update                      │
   │  - 计算优势函数 A(s,a)            │
   │  - Clip 策略梯度                 │
   │  - 更新价值网络                   │
   │  - KL 散度约束                   │
   └──────────────┬───────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────┐
   │  同步新策略到所有 Workers         │
   └──────────────────────────────────┘
```

### 自对弈 Worker

```python
# training/self_play_worker.py
class SelfPlayWorker:
    def __init__(
        self,
        game: GameInterface,
        agents: List[AgentInterface],
        worker_id: int
    ):
        self.game = game
        self.agents = agents
        self.worker_id = worker_id

    def collect_episode(self) -> List[Experience]:
        """运行一局游戏，收集经验"""
        experiences = []
        state = self.game.reset()

        while not self.game.is_terminal(state):
            player_id = self.game.get_current_player(state)
            obs = self.game.state_to_observation(state, player_id)
            legal_actions = self.game.get_legal_actions(state)

            # Agent 选择动作
            action_idx, info = self.agents[player_id].select_action(
                obs, legal_actions
            )

            # 执行动作
            next_state, rewards, done, _ = self.game.step(
                self.game.index_to_action(action_idx)
            )

            # 记录经验
            experiences.append(Experience(
                player_id=player_id,
                observation=obs,
                action=action_idx,
                reward=rewards[player_id],
                log_prob=info['log_prob'],
                value=info['value']
            ))

            state = next_state

        return experiences
```

---

## 评估层

### ELO 评分系统

```python
# evaluation/elo_system.py
class ELOSystem:
    """ELO 评分系统，追踪模型强度变化"""

    def __init__(self, k_factor: float = 32):
        self.k_factor = k_factor
        self.ratings = {}  # model_id -> rating

    def update_ratings(
        self,
        game_results: List[Tuple[str, str, float]]
    ):
        """
        更新 ELO 分数

        Args:
            game_results: [(model1_id, model2_id, score)]
                         score: 1.0 (model1赢), 0.0 (model2赢), 0.5 (平局)
        """
        for model1, model2, score in game_results:
            r1 = self.ratings.get(model1, 1500)
            r2 = self.ratings.get(model2, 1500)

            # 计算期望胜率
            e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
            e2 = 1 - e1

            # 更新分数
            self.ratings[model1] = r1 + self.k_factor * (score - e1)
            self.ratings[model2] = r2 + self.k_factor * ((1-score) - e2)
```

### Arena 竞技场

```python
# evaluation/arena.py
class Arena:
    """多模型对战竞技场"""

    def run_tournament(
        self,
        game: GameInterface,
        agents: List[AgentInterface],
        num_games: int
    ) -> TournamentResults:
        """
        运行循环赛

        Args:
            game: 游戏实例
            agents: 参赛 Agent 列表
            num_games: 对局数量

        Returns:
            比赛结果（胜率、ELO 分数等）
        """
        results = defaultdict(list)

        for _ in range(num_games):
            # 随机打乱座位顺序
            shuffled_agents = random.sample(agents, len(agents))

            # 运行一局
            winner = self.play_game(game, shuffled_agents)
            results[winner].append(1)

        return TournamentResults(results)
```

---

## 数据流

### 训练数据流

```
游戏状态 → 观察编码 → 神经网络 → 动作采样 → 执行
   ↓                                              ↓
   └──────────── 经验存储 ←────────────────────────┘
                   ↓
              [批量采样]
                   ↓
           ┌──────────────┐
           │ PPO 训练步骤 │
           │ - 优势估计    │
           │ - 策略梯度    │
           │ - 价值损失    │
           └──────┬───────┘
                  ↓
          更新神经网络参数
```

### 评估数据流

```
checkpoint_v1.pth ──┐
checkpoint_v2.pth ──┤
checkpoint_v3.pth ──┼──▶ Arena ──▶ ELO Rating ──▶ 排行榜
checkpoint_v4.pth ──┘
```

---

## 设计模式

### 1. 策略模式 (Strategy Pattern)
不同的 Encoder 可互换：

```python
class ActorCriticModel:
    def __init__(self, encoder: nn.Module):
        self.encoder = encoder  # 可以是 MLP、CNN、Attention
```

### 2. 工厂模式 (Factory Pattern)
游戏和模型的创建：

```python
game = create_game("splendor", num_players=4)
model = build_model("attention", obs_shape, action_size)
```

### 3. 观察者模式 (Observer Pattern)
训练监控和日志：

```python
class TrainingMonitor:
    def on_episode_end(self, metrics):
        self.tensorboard.log(metrics)
        self.wandb.log(metrics)
```

### 4. 模板方法模式 (Template Method)
通用训练循环：

```python
class BaseTrainer(ABC):
    def train(self):
        for epoch in range(self.num_epochs):
            data = self.collect_data()      # 子类实现
            metrics = self.update_policy(data)  # 子类实现
            self.log_metrics(metrics)       # 通用逻辑
```

---

## 性能优化策略

### 1. 多进程并行自对弈
```python
# 使用 torch.multiprocessing
with mp.Pool(processes=num_workers) as pool:
    experiences = pool.map(worker.collect_episode, range(num_episodes))
```

### 2. 经验重放批处理
```python
# 高效的 batch 采样
batch = buffer.sample(batch_size=256)
obs_batch = torch.stack([e.obs for e in batch])
```

### 3. MPS 加速（Apple Silicon）
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
```

### 4. JIT 编译（可选）
```python
model = torch.jit.script(model)  # 加速推理
```

---

## 可扩展性

### 添加新游戏的步骤

1. 继承 `GameInterface`
2. 实现所有抽象方法
3. 用 `@register_game` 注册
4. 创建配置文件
5. 直接使用现有训练/评估框架

### 添加新算法的步骤

1. 继承 `BaseTrainer`
2. 实现 `collect_data()` 和 `update_policy()`
3. 注册到算法工厂

### 添加新模型架构

1. 实现新的 `Encoder` 类
2. 注册到 `model_factory.py`
3. 通过配置文件选择

---

## 配置系统

### 配置优先级

```
命令行参数 > 实验配置 > 游戏配置 > 默认配置
```

### 配置组合示例

```yaml
# configs/experiments/splendor_ppo_v1.yaml
defaults:
  - game: splendor          # 继承 configs/games/splendor.yaml
  - model: attention        # 继承 configs/models/attention.yaml
  - algorithm: ppo          # 继承 configs/algorithms/ppo.yaml

# 覆盖特定参数
model:
  hidden_size: 512          # 覆盖默认的 256

training:
  num_workers: 8
  episodes_per_update: 100
```

---

## 总结

本架构的核心优势：

1. **高度解耦**：游戏、模型、训练算法完全独立
2. **易于测试**：每个模块可单独测试
3. **配置灵活**：通过 YAML 快速实验
4. **性能优化**：支持多进程、GPU/MPS 加速
5. **可扩展**：添加新游戏/算法/模型无需修改核心代码

这个架构既能支持当前的 Splendor 训练需求，也能轻松扩展到 UNO、狼人杀等其他桌游。
