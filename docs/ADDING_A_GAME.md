# 接入一个新桌游

本文档说明如何把一个新的桌游接入本框架，并给出**隐藏信息 / 淘汰制 / 回合制**游戏的适配要点（附「情书 vs 政变疑云」的评估）。

## 目录

- [最小步骤](#最小步骤)
- [必须实现的接口](#必须实现的接口)
- [核心设计要点](#核心设计要点)
- [产物目录与路径辅助](#产物目录与路径辅助)
- [配置与训练](#配置与训练)
- [测试](#测试)
- [适配评估：情书 vs 政变疑云](#适配评估情书-vs-政变疑云)

---

## 最小步骤

```bash
# 1. 生成骨架（游戏名用 snake_case）
python scripts/new_game.py love_letter --players 4
```

会生成：

```
games/love_letter/        # game.py / state.py / actions.py / encoder.py / RULES.md
configs/love_letter/      # love_letter_ppo.yaml
tests/test_games/         # test_love_letter.py
data/love_letter/         # checkpoints / logs / reports（含 .gitkeep）
```

然后按顺序实现：

1. `games/love_letter/game.py`：继承 `GameInterface`，用 `@register_game("love_letter")` 注册
2. `games/love_letter/state.py`：状态对象（支持 `clone()`，供 MCTS 使用）
3. `games/love_letter/encoder.py`：`state + player_id → 固定维度观察向量`
4. 补全 `tests/test_games/test_love_letter.py`
5. 用 `configs/love_letter/love_letter_ppo.yaml` 训练

训练脚本**无需任何改动**：它们从配置读 `game.name` 与构造参数。

---

## 必须实现的接口

`GameInterface`（`core/game_interface.py`）的核心抽象方法：

| 方法 | 作用 | 关键约束 |
|------|------|----------|
| `reset()` | 返回初始状态 | 状态对象 |
| `step(action)` | 执行动作，返回 `(state, rewards, done, info)` | 不修改原状态 |
| `get_legal_actions(state)` | 返回当前合法动作对象列表 | 必须与 `action_to_index` 对应 |
| `get_current_player(state)` | 当前行动玩家 ID | 淘汰玩家跳过 |
| `is_terminal(state)` | 是否结束 | 含回合/轮次上限 |
| `get_final_rewards(state)` | 每个玩家的终局奖励 | 通常胜者 1、其余 0，平局 0.5 |
| `state_to_observation(state, player_id)` | 该玩家视角观察 | **固定维度**，只含可见信息 |
| `action_to_index(action, state=None)` | 动作 → 固定槽位索引 | 槽位语义稳定 |
| `index_to_action(index, state=None)` | 索引 → 动作 | 与上互逆 |
| `num_players` / `observation_shape` / `action_space_size` | 三个属性 | 必须与实现一致 |

可选但推荐：

- `game_kwargs()`：返回多进程重建所需的构造参数（含 `seed` / `reward_config`）。基类默认只返回 `num_players`，带奖励的游戏应覆盖（参考 `games/splendor/game.py`）
- `clone_state(state)`：给 MCTS 提供快速浅拷贝（Splendor 用冻结 dataclass + 共享引用提速）
- `get_winner(state)`：基类默认「终局奖励最高者」，多人平局返回 -1

---

## 核心设计要点

### 1. 固定动作空间（推荐）

把动作映射到**语义稳定的固定槽位**，而不是「第 i 个合法动作」这类随状态漂移的相对索引。这样策略网络学到的偏好才有意义。

- 大动作空间：固定槽位 + 合法掩码（`get_legal_action_mask`），非法槽位概率置 0
- 小动作空间：可让 `action_to_index` 直接等于动作枚举值

### 2. 观察编码（隐藏信息的关键）

`state_to_observation(state, player_id)` 天然支持「按玩家视角」编码：

- **只编码该玩家可见的信息**：自己手牌 + 公开信息 + （可选）可推断信息
- **固定维度**：无论对局进行到哪一步，维度必须不变（用固定槽位 + 掩码）
- **归一化**：数值归一到 [0,1] 或 [-1,1]
- **位置编码**：对手信息按「相对自己」排列，减少位置偏差

对于隐藏信息游戏，共享模型自对弈时每个玩家调用 `state_to_observation(state, p)` 拿到的是**自己的**视角——这是正确的做法，不要让任何玩家看到别人的私密信息。

### 3. 淘汰制与回合制

- **淘汰玩家**：`get_current_player` 应跳过已淘汰玩家
- **回合制（如情书多局积分）**：决定「一个 episode」是单局还是完整一局到 N 分。推荐以**完整一局**为 episode，终局奖励 = 最终胜者，训练更稳定
- **回合上限**：像 Splendor 一样加 `max_total_turns`，防止 pass/僵局导致无限对局

### 4. 稠密奖励（PPO 可选）

PPO 支持稠密奖励（`algorithm.dense_rewards`），现在是**自由字典**，键由游戏自己定义。`step()` 通过 `info` 返回 `dense_reward`，终局胜负由框架按 `get_final_rewards` 注入零和信号。不需要稠密奖励的游戏可以不配置，`build_game_kwargs` 会跳过空字典。

### 5. 随机性

对局内的随机性（洗牌、发牌）用游戏自己的 RNG（`random.Random(seed)`），保证可复现；`seed` 通过 `game_kwargs()` 传入多进程子进程。

---

## 产物目录与路径辅助

每个游戏独立存放训练产物（框架已约定并迁移完毕）：

```
data/<game_name>/
  checkpoints/<run_name>/    # latest.pth + checkpoint_iter_*.pth + tensorboard/
  logs/<run_name>.jsonl      # JSONL 指标日志
  reports/                   # 评估/基准报告
```

路径辅助函数在 `core/paths.py`（`run_checkpoint_dir(game, run)`、`default_log_path(game, run)`、`ensure_run_dirs(game, run)`），路径锚定项目根目录，不依赖运行时 CWD。

---

## 配置与训练

配置为 YAML（见 `configs/splendor/` 作参考）。新游戏最小配置要点：

```yaml
game:
  name: "love_letter"
  num_players: 4
  seed: null

algorithm:
  name: "ppo"
  dense_rewards:        # 自由字典，键由本游戏定义
    win: 1.0
    step_penalty: -0.01

training:
  checkpoint_dir: "data/love_letter/checkpoints/love_letter_v1"
  # ...
```

训练：

```bash
python scripts/train_convergence.py \
    --config configs/love_letter/love_letter_ppo.yaml \
    --iterations 200 --episodes 128 \
    --log-file data/love_letter/logs/love_letter_v1.jsonl
```

---

## 测试

参考 `tests/test_games/test_splendor.py`：

1. 初始状态 / 重置
2. 每种动作的执行与合法性
3. `action_to_index` / `index_to_action` 互逆
4. 结束条件与 `get_final_rewards`
5. 观察维度固定 + 掩码正确
6. 随机 agent 能完整玩完一局（不卡死）
7. 状态 `clone()` 独立

运行：

```bash
pytest tests/test_games/test_love_letter.py -v
```

---

## 适配评估：情书 vs 政变疑云

> **更新**：情书（Love Letter）已按本文档实现，见 `games/love_letter/` 与
> `configs/love_letter/`。政变疑云（Coup）作为下一个候选，评估如下。

两者都适合作为「第二个游戏」，但难度不同：

| 维度 | 情书（Love Letter） | 政变疑云（Coup） |
|------|--------------------|------------------|
| 隐藏信息 | 1 张手牌 | 2 张身份牌 |
| 交互复杂度 | 低（卡牌效果确定） | 高（挑战 / 反制 / 诈唬） |
| 动作空间 | 小（8 种卡，部分带目标/猜测） | 中（7 种行动 + 目标 + 反制） |
| 淘汰制 | 单轮内淘汰，多局积分 | 整局淘汰 |
| 首次实现难度 | **低** | 中-高 |

**建议：先做情书，再做政变。**

- **情书**是更温和的第二步：主要新挑战是「隐藏手牌」的按视角编码 + 回合积分 + 淘汰，交互是确定性的，容易验证引擎正确性。
- **政变**的核心难点是「挑战 / 反制 / 诈唬」这类博弈决策，需要更强的对手建模与更多训练样本；在框架尚未验证隐藏信息编码之前，直接上政变风险较大。

**对两者都适用、且需要提前决策的点**：

1. **Episode 边界**：情书建议以「完整一局到 4 分」为 episode（终局奖励 = 最终胜者）；政变以「整局淘汰」为 episode。
2. **隐藏信息编码**：`state_to_observation(state, player_id)` 只给该玩家自己的手牌/身份，公开区（弃牌堆、牌库余量、金币）对所有人可见。
3. **终局零和奖励**：两者都是单一胜者，符合当前框架「胜者 +1、其余 0」的零和注入逻辑，无需改动训练代码。

> 决定做哪一个后，直接 `python scripts/new_game.py <game> --players N` 起步即可。
