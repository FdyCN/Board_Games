# 政变疑云（Coup）设计文档

> 状态：**设计稿，待评审**。评审通过后再进入 `scripts/new_game.py coup` 脚手架 + 实现。
> 本文目标是：规则精确化、状态机/动作空间/观察编码逐字段定死，避免实现时规则理解偏差返工。

---

## 0. 结论摘要

- **可训练性**：可以。但 Coup 是「隐藏身份 + 说谎」游戏，纯 PPO 自对弈会像情书那样塌缩（甚至更糟）。**首选算法是 NFSP**（我们已有 `scripts/train_nfsp.py`），模型仍是轻量 MLP/GRU，CPU 可训。
- **现实目标**：训练出「能打赢手写规则 bot」的模型；「说谎博弈论均衡」在 CPU 算力下只能逼近，不做承诺。
- **与现有框架契合度**：高。唯一大改动是「一回合拆成多步 `step()`」——把每个决策点（宣告/质疑/反制/质疑反制/揭示/换牌）映射成一次 `step()`，每步动作空间很小。

---

## 1. 游戏规则（我们要实现的精确版本）

### 1.1 组件

- 5 种角色 × 各 3 张 = **15 张牌**：公爵(Duke)、刺客(Assassin)、队长(Captain)、大使(Ambassador)、女伯爵(Contessa)。
- 每名玩家 **2 张隐藏手牌**（= 2 点影响力，生命值）、**2 枚金币**。
- 剩余牌组成**宫廷牌堆**（面朝下）。
- 支持 2-6 人；**本实现先做 3 人局**（2 人有特殊规则、4+ 人游戏变长，后续再加）。

### 1.2 角色能力

| 角色 | 主动行动 | 反制(Block) |
|---|---|---|
| 公爵 Duke | **征税 Tax**：拿 3 金币 | 反制「外援 Foreign Aid」 |
| 刺客 Assassin | **刺杀 Assassinate**：付 3 金币，令目标掉 1 点影响力 | —（被女伯爵反制） |
| 队长 Captain | **偷窃 Steal**：从目标拿 2 金币 | 反制「偷窃 Steal」 |
| 大使 Ambassador | **换牌 Exchange**：抽 2 张，从 4 张中保留 2 张 | 反制「偷窃 Steal」 |
| 女伯爵 Contessa | — | 反制「刺杀 Assassinate」 |

### 1.3 基础行动（无需卡牌）

| 行动 | 效果 | 可被质疑？ | 可被反制？ |
|---|---|---|---|
| 收入 Income | 拿 1 金币 | 否 | 否 |
| 外援 Foreign Aid | 拿 2 金币 | 否 | 是（任意玩家可宣「公爵」反制） |
| 政变 Coup | 付 7 金币，令目标掉 1 点影响力 | **否** | **否** |
| 征税/刺杀/偷窃/换牌 | 见上表（需**宣称**持有对应角色，可说谎） | **是** | 刺杀/偷窃可被反制 |

### 1.4 质疑（Challenge）与反制（Block）

- **宣称角色行动时**：其他玩家**顺时针依次**获得一次质疑机会，**第一个质疑生效**。
  - **质疑成功**（宣称者说谎）→ 宣称者**扣血：翻开并弃掉一张手牌**（掉 1 影响力），行动取消。
  - **质疑失败**（宣称者真有该角色）→ 质疑者扣血；宣称者把该角色牌**洗回牌堆并重抽一张**（这**不是**扣血，影响力不变，只是「证明后换牌」）。
- **反制**：刺杀/偷窃的目标可宣「女伯爵/队长或大使」反制；外援可由任意玩家宣「公爵」反制。
- **质疑反制**：反制本身是一次宣称，同样可被质疑（顺时针、第一个生效）。
  - 反制者说谎 → 反制者掉 1 影响力，反制取消，原行动继续。
  - 反制者真有该角色 → 质疑者掉 1 影响力，反制者洗回重抽，反制成立（原行动被挡）。

### 1.5 影响力（生命）与淘汰

- 掉 1 影响力 = 玩家**自选一张手牌公开揭示并弃掉**（单张时强制）。
- 两张都掉光 → **淘汰**，立即退出（其金币作废）。
- 金币 ≥ 10 时，轮到该玩家**只能执行 Coup**。

### 1.6 结束

- 最后一名仍有影响力者获胜。

### 1.7 边界情况清单（实现时必须覆盖）

1. Coup 不能被质疑/反制（它不是角色宣称）。
2. 外援不能被质疑，只能被「公爵」反制，且**任意玩家**都可反制（非目标）。
3. 偷窃可被「队长」或「大使」反制；刺杀只能被「女伯爵」反制。
4. 质疑/反制后「洗回重抽」：被质疑的牌回到牌堆并洗牌，再抽一张（保持 2 张手牌）。
5. 偷窃时若目标金币 < 2，只拿其全部金币（不足 2 也成立）。
6. 牌堆空时的换牌/重抽：若牌堆不足，尽可能抽；换牌仍执行「保留 2 弃 2」。
7. 玩家被淘汰后，其金币保留但不可用，也不再参与质疑/反制。

---

## 2. 状态机设计（多阶段 step）

核心：把 Coup 的一「回合」拆成多条 `step()`。每次 `step()` 的**当前玩家 = 下一个要决策的人**，动作空间随 `phase` 用掩码切换。这完全兼容现有 `GameInterface`（`step/get_legal_actions/get_current_player/action_to_index`）。

### 2.1 阶段（phase）状态机

```
            ┌───────────────────────────────────────────────────┐
            │                      下一回合                      │
            ▼                                                   │
ACTION ──(角色宣称? 且有人质疑)──▶ CHALLENGE ──▶ RESOLVE_CHALLENGE ─┘
   │                                                          
   │(未被质疑 / 非宣称 / 质疑后行动仍继续)                      
   ▼                                                          
(可被反制? 刺杀/偷窃/外援)──是──▶ BLOCK ──▶ BLOCK_CHALLENGE ──▶ RESOLVE_BLOCK ─┘
   │                              │(无)                          
   否                             ▼(无人反制)                    
   │                           EXECUTE ──▶ (可能 REVEAL/EXCHANGE) ──▶ TURN_END
   └──────────────────────────────────────────────────────────────────┘
```

### 2.2 各 phase 的决策者与合法动作

| phase | 当前决策者 | 合法动作（概念） |
|---|---|---|
| `ACTION` | 回合玩家 | Income / Foreign Aid / Coup(目标) / Tax / Assassinate(目标) / Steal(目标) / Exchange |
| `CHALLENGE` | 顺时针下一个存活的非行动者 | Pass / Challenge（第一个 Challenge 生效） |
| `RESOLVE_CHALLENGE` | 掉影响力者（若手牌 2 张） | Reveal 槽0 / Reveal 槽1（单张强制） |
| `BLOCK` | 被反制目标（或外援的任意玩家） | Pass / Block(Contessa) / Block(Duke) / Block(Captain) / Block(Ambassador)（掩码按行动类型限合法角色） |
| `BLOCK_CHALLENGE` | 顺时针下一个存活玩家 | Pass / Challenge |
| `RESOLVE_BLOCK` | 掉影响力者 | Reveal 槽0 / Reveal 槽1 |
| `REVEAL` | 掉影响力者 | Reveal 槽0 / Reveal 槽1 |
| `EXCHANGE` | 大使 | 6 种「保留 2 / 弃 2」组合 |
| `EXECUTE` / `TURN_END` | （无决策，引擎自动结算） | — |

> 说明：`RESOLVE_CHALLENGE` / `RESOLVE_BLOCK` 中的「掉影响力者」若手牌 2 张需选揭示哪张；这一步的揭示选择也走 `REVEAL` 槽位。为统一，所有「掉影响力」都进入 `REVEAL` 子阶段。

### 2.3 `step()` 返回信息

- `rewards`：当前决策玩家的稠密奖励（见 §5）。
- `info`：`phase`、`pending_action`、`eliminated`、事件时间线（供日志 + GRU 序列编码）。
- 终局 `winner`。

### 2.4 与现有游戏 / 训练循环的兼容性

多阶段 `step()` 是 **Coup 引擎内部（`games/coup/`）** 的行为，**不会改动 Splendor / 情书的引擎**（它们仍是"一回合一步"）。共享训练循环（`self_play_worker`、`train_league`、`train_nfsp`）只依赖 `get_current_player / get_legal_actions / step` 这套通用接口，不假设"每玩家每回合只行动一次"，因此天然兼容多阶段。

唯一需在 P1 显式验证的集成点：**经验收集 + GAE 按 `player_id` 分组**在多阶段下，奖励要正确归属到"触发该奖励的那个决策者"（例如扣血奖励归被打掉牌的玩家，而非当前 phase 无关者）。这是风险最高的点，会在 P1 用单测覆盖。

---

## 3. 动作空间（固定槽位）

统一槽位，跨 phase 复用，靠合法掩码切换。总槽位 `3n + 16`（**3 人 = 25 个**）。

### 槽位表（n 人，相对目标 r=1..n-1）

| 槽位 | 含义 | 适用 phase |
|---|---|---|
| 0 | Income | ACTION |
| 1 | Foreign Aid | ACTION |
| 2 | Tax（宣 Duke） | ACTION |
| 3 | Exchange（宣 Ambassador） | ACTION |
| 4 … 3+(n-1) | Coup 目标 r | ACTION |
| (3+n) … | Assassinate 目标 r | ACTION |
| (3+2n) … | Steal 目标 r | ACTION |
| 3n+1 | Pass | CHALLENGE / BLOCK_CHALLENGE / BLOCK |
| 3n+2 | Challenge | CHALLENGE / BLOCK_CHALLENGE |
| 3n+3 | Pass（不反制） | BLOCK |
| 3n+4 | Block as Contessa | BLOCK |
| 3n+5 | Block as Duke | BLOCK |
| 3n+6 | Block as Captain | BLOCK |
| 3n+7 | Block as Ambassador | BLOCK |
| 3n+8 | Reveal 手牌槽 0 | REVEAL / RESOLVE_* |
| 3n+9 | Reveal 手牌槽 1 | REVEAL / RESOLVE_* |
| 3n+10 … 3n+15 | Exchange 6 组合（4 选 2） | EXCHANGE |

> `index_to_action` 依据当前 `phase` 把槽位解释成 `CoupAction(claim=role, target=rel, ...)` 等具体对象；`action_to_index` 反向映射。掩码保证只暴露当前 phase 的合法槽位。

---

## 4. 观察编码

完全复用情书的「**静态段 + 序列段**」GRU 混合编码思路——**序列段让模型从「谁宣称了什么、谁质疑、谁揭晓」的事件时间线里自己学出信念**，而非手工堆信念特征。

### 4.1 静态段（当前状态快照，`12 + 5n` 维左右，实现时定稿）

1. 自己 2 张手牌：2 × 5 one-hot = 10 维
2. 自己金币：1 维（/10 归一化）
3. 牌堆剩余：1 维（/15）
4. 当前 phase：8 维 one-hot
5. 每个相对玩家（相对顺序）：金币 /10、存活、已公开揭示卡数 —— 3(n-1) 维
6. 已公开弃牌计数：5 角色各几张 = 5 维（信念先验）
7. 当前玩家 ID：n 维 one-hot（绝对，对齐时间线）
8. pending 声明：行动类型 one-hot(7) + 目标相对 one-hot(n) + 宣称角色 one-hot(5)

### 4.2 序列段（事件时间线，`seq_len × (n + 14)`）

每个事件 = 「谁」one-hot(n) + 「做了什么」one-hot(14)：

```
income, foreign_aid, coup, claim_tax, claim_assassinate, claim_steal, claim_exchange,
challenge, block_contessa, block_duke, block_captain, block_ambassador,
reveal_role, exchange
```

`seq_len` 取 40（Coup 一回合事件比情书多）。空槽全 0。

> 这比手工维护「对手可能是哪 5 个角色」的概率矩阵更简单且已被情书验证有效；若后续发现不够，再加显式信念段。

### 4.3 私密知识 / 上帝视角辅助标签

`auxiliary_shape = (n-1) × 5`，标签 = 每个相对对手**真实**持有的 2 张牌（5 角色 one-hot 或 -1 出局）。训练时用于「预测对手身份」辅助任务（情书同款 `get_auxiliary_labels`），帮模型在公开信息里学推断。

---

## 5. 奖励设计（稠密 + 终局，参照情书）

| 事件 | 奖励 | 说明 |
|---|---|---|
| 存活到终局且获胜 | +1 | 终局稀疏 |
| 被淘汰 | 0（或 -1 零和） | |
| 令对手掉 1 影响力 | +0.4 | 主动淘汰推进 |
| 自己掉 1 影响力 | -0.4 | |
| 质疑成功（揭穿谎言） | +0.3 | 鼓励信息利用 |
| 质疑失败 | -0.2 | 惩罚乱质疑 |
| 偷窃/征税/外援成功 | +0.02/金币 | 弱引导 |
| 每步 | -0.01 | 时间压力 |

> 具体系数训练时再调；方向是「生存 + 揭穿谎言 + 金币管理」，与情书 `round_win/eliminate/step_penalty` 同一套路。

---

## 6. 训练方案

### 6.1 算法

- **首选 NFSP**（`train_nfsp.py`）：Q 最佳响应网络 + π 平均策略网络，专为非完全信息说谎游戏设计，逼近纳什均衡。
- **对手池**：随机 + 手写规则 bot + 冻结快照（复用 `train_league.py` 的思路）。
- **编码器**：GRU 混合（静态 + 序列），与情书一致。

### 6.2 手写规则 bot（校准天花板，仿情书 strong_bot）

一个「会玩」的 Coup bot，只用手牌 + 公开信息：
- 金币管理：≥7 且能 Coup 就 Coup；缺金币先征税/外援；被威胁时留钱反杀。
- 按「对手宣称角色的先验概率」决定质疑（利用角色剩余张数算后验）。
- 有女伯爵就反刺杀、有公爵就反外援、有队长/大使就反偷窃。
- 换牌：弃掉「低价值/已暴露」角色，留高价值。

用它跑「bot vs 随机」得天花板，再拿 NFSP 模型对打看是否反超。

### 6.3 评估口径

沿用 `scripts/direct_head_to_head.py`（直接计数、随机座位），spec 增加 `coup` 的 `heuristic`/`strong`。**不信任 Arena 胜率**（历史教训）。

---

## 7. 任务分解（4 阶段，每阶段可验证）

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **P1 引擎** | `scripts/new_game.py coup` 脚手架 → 实现 state/actions/encoder/game（多阶段 step + 固定槽位 + 观察编码） | 随机 rollout 跑通、终局正确、`state_to_observation` 形状稳定、27+ 单测通过 |
| **P2 规则 bot** | `heuristic.py`（弱）+ `strong_bot.py`（强，含贝叶斯质疑/反制） | bot vs 随机胜率显著 > 基线，得天花板 |
| **P3 NFSP 训练** | 复用 `train_nfsp.py` + 对手池，产出 3 人模型 | 训练曲线收敛，模型 vs 规则 bot 对打 |
| **P4 评测 + 文档** | `direct_head_to_head.py` 支持 coup、CHANGELOG/MODELS 更新、Web 前端（可选） | 模型 vs 随机 / vs 规则 bot 报告 |

---

## 8. 参考资料

- Coup 规则与 POMDP 建模：[Solving Coup as an MDP/POMDP (Stanford)](https://web.stanford.edu/class/aa228/reports/2018/final81.pdf)
- Coup 深度强化学习 / 挑战决策：[Deep RL in Imperfect-Information Game Coup](https://ml.cmu.edu/research/phd-dissertation-pdfs/ssokota_ml_phd_thesis_2026-copy.pdf)、[CoupVisor](https://www.semanticscholar.org/reader/9730e4fe994e973176c6254f4edb62670f2a5ede)
- Coup 子博弈求解器：[OYASHI777/coup](https://github.com/OYASHI777/coup)
- 非完全信息博弈算法（NFSP/CFR）：[OpenSpiel 教程 (Marc Lanctot)](https://mlanctot.info/files/open_spiel_tutorial-mar2021-comarl.pdf)、[OpenSpiel](https://github.com/google-deepmind/open_spiel)
