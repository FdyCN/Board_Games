# 变更日志（Changelog）

> 本文件记录 **Bug 修复**、**训练结果** 与 **版本历史**。
> 使用指南见 [README.md](./README.md)，架构见 [ARCHITECTURE.md](./ARCHITECTURE.md)，
> 早期 MCTS 阶段的修复细节见 [docs/BUGFIXES.md](./docs/BUGFIXES.md)。

## 目录

- [当前状态](#当前状态)
- [训练结果与基准](#训练结果与基准)
- [核心 Bug 修复](#核心-bug-修复)
- [功能演进](#功能演进)
- [版本历史](#版本历史)

---

## 当前状态

- 最佳可用模型：**PPO（结构化模型，mlp medium，3 人）**
  - 检查点：`data/splendor/checkpoints/mlp_medium_3p_v1/latest.pth`
- **情书（Love Letter）**：引擎已实现（隐藏信息 + 回合制 + 淘汰制）。3 人 vs 2 随机对手胜率：**PPO 系列稳定 75~77%**（v1 绝对视角 77%、v2 相对视角 75.5%、对手池+信念+上帝视角 77%）；**NFSP 约 59%**（1000 迭代，低于 PPO）。上帝视角辅助头能学会推断对手手牌（aux_loss 2.03→1.63）但未转化为胜率；对手池与 NFSP 均未让价值函数学到有效信号（explained_variance ≈ 0）。
- AlphaZero 训练循环已搭好（价值函数可学），但策略网络在有限算力下尚未反超 PPO（见下）。
- 目录结构已完成「游戏无关化 + 按游戏分组产物」重构，可直接接入新桌游。

---

## 训练结果与基准

### PPO（3 人对局，结构化模型）

| 指标 | 数值 |
|---|---|
| 平均回合数 | ~87（自对弈）/ ~90（vs 随机） |
| tier1 买卡占比 | ~61%（修复前 ~69%） |
| 对随机 agent 胜率 | **98%**（100 局） |

### 与开源模型对比

引入 [cestpasphoto/alpha-zero-general](https://github.com/cestpasphoto/alpha-zero-general)（git submodule）的 3 人预训练模型做基准。

**1. 分别 vs 2 个随机对手（确定性策略）**

| 模型 | vs 2 随机 胜率 | 平均回合 |
|---|---|---|
| 开源 AlphaZero 3 人预训练 | 100% | 97.7 |
| 本项目 PPO（结构化模型） | 98% | 90.5 |

**2. 直接 head-to-head（本地 + 开源 + 1 随机，1000 局，随机座位）**

| 模型 | 胜局 | 胜率 |
|---|---|---|
| 开源 AlphaZero | 509 | 50.9% |
| 本项目 PPO | 491 | 49.1% |
| 随机 | 0 | 0% |

**模型大小对比**

| 模型 | 参数量 | 结构 |
|---|---|---|
| 本项目 PPO（mlp medium） | 307,672 | 扁平 MLP |
| 开源 AlphaZero（v80） | 219,911 | MobileNetV3 风格 1D 残差块 |

**结论**：本地 PPO 与开源 AlphaZero 直接对打基本打平（49.1% vs 50.9%，统计不显著），vs 随机也接近（98% vs 100%）。开源模型参数量更小（22 万 vs 31 万），差距来自「结构化架构 + AlphaZero」而非模型大小。复现脚本：`scripts/compare_with_open_source.py`（vs 随机）、`scripts/head_to_head.py`（直接对打）。

### AlphaZero 实验结论

40 / 60 / 500 / 700 迭代、最多 800 次模拟的实验中，AlphaZero 的价值函数能学到「谁赢」（value_loss ~0.12-0.17），但**策略网络在有限算力下退化为「不买卡」**（score 0）。根因：训练全程 `temp=1`，策略从未被温度退火锐化；参考实现需要在训练充分后使用温度调度 + 数千次迭代才能反超。因此当前 **PPO 仍是可用效果最好的模型**。

---

## 核心 Bug 修复

以下为 Splendor 训练从「崩坏」到「收敛」过程中修复的关键问题。

| # | 问题 | 修复 | 影响 |
|---|------|------|------|
| 1 | 多人自对弈把交错轨迹当单一轨迹算 GAE | 按 `player_id` 分组、逐玩家计算 GAE | advantage/value 由噪声变为有效信号 |
| 2 | 终局「胜利 +1」只落在最后行动玩家 | 终局零和奖励注入到每个玩家自己的最后一步 | 胜负信号不再丢失 |
| 3 | 要求「最后一位玩家也到 15 分」才结束 | 改为「有人到 15 分后走完当前一轮即结束」+ 回合上限 | 对局不再无限拖长 |
| 4 | 多进程子进程退回默认奖励系数 | 用 `game_kwargs()` 把 `reward_config` 传进子进程 | 自定义稠密奖励生效 |
| 5 | `PassAction` + 无回合上限 → 无限对局 | 加 `max_total_turns = 62 * num_players` | 消除死锁 |
| 6 | `outcome=None` 触发 `0 * -inf` → NaN | 终局无胜者时 outcome 置 0，并加 `all(... not None)` 守卫 | 消除 NaN 损失 |
| 7 | MCTS 多人局 `value = -value` 符号错误 | 移除该取负 | 多人搜索价值正确 |
| 8 | MCTS 的 `log_prob` 误用访问概率 | 改用策略网络的 `log_prob` | PPO importance ratio 正确 |
| 9 | AlphaZero `temp=0` 锁死弱模型 | 诊断：需温度调度 + 数千迭代 | 见上方实验结论 |
| 10 | 开源 numba `make_move` 不接受 kwargs | 改位置参数 `board.make_move(a, current, 0)` | 桥接可运行 |
| 11 | 开源 `DevelopmentCard` 断言 points ≤5 | 把总分分摊到多张 dummy 卡 | 桥接可运行 |
| 12 | 开源 `NobleTile` 断言需求 ≥8 | 用 `(8,0,0,0,0)` 占位 | 桥接可运行 |

早期 MCTS 阶段的修复（Experience 参数名、Episode 构造、终局状态搜索等）见 [docs/BUGFIXES.md](./docs/BUGFIXES.md)。

---

## 功能演进

- **固定动作空间（46 槽位）**：把「相对索引」改为固定槽位，槽位语义稳定
  - `0-4` 拿 2 同色 · `5-14` 拿 3 不同色 · `15-26` 保留明牌 · `27-29` 保留牌堆 · `30-41` 买明牌 · `42-44` 买保留 · `45` pass
- **结构化策略头**：参考开源实现，「买明牌 / 保留明牌 / 买保留牌」使用**共享卡评估器**，学到与位置无关的「这张卡值不值得买」
- **终局胜负辅助头**：新增 `outcome_head`，给共享编码器强监督
- **AlphaZero 训练循环**：`scripts/train_alphazero.py`（自对弈 MCTS + 策略 CE + 终局价值 MSE）
- **批量 MCTS**：虚拟损失 + 批量叶节点评估，CPU 提速约 25 倍
- **多进程自对弈**：spawn 进程池，每 worker 独立模型/游戏/MCTS
- **对手池训练**：`scripts/train_opponent_pool.py`（混合历史 checkpoint 对手，降低非平稳性）
- **游戏无关重构**（本版）：训练脚本读 `game.name`，`dense_rewards` 改为自由字典，`data/` 按游戏分组，`scripts/new_game.py` 脚手架
- **情书（Love Letter）引擎**：第二个完整游戏，固定动作空间（7+11n 槽位）、按玩家视角观察编码（只暴露自己手牌）、侍女保护/女伯爵强制/王子重抽等规则全覆盖
- **模型逐卡编码解耦**：`ActorCritic` 的 Splendor 逐卡评估器改为仅在 `action_size==46` 时启用，其他游戏走 flat 策略头
- **评估脚本游戏无关化**：`evaluate.py` 增加 `--encoder/--model-config`（此前硬编码 attention + `NeuralAgent(player_id=...)` 已失效）；`Arena` 改用它 `get_final_rewards` 判定胜负（此前硬编码 Splendor 的 `state.players[i].get_score()`）

---

## 版本历史

- `d20e0fa` benchmark：head-to-head 扩到 1000 局，两模型基本打平（49.1% vs 50.9%）
- `91022b4` benchmark：本地 PPO vs 开源 AlphaZero 直接 head-to-head
- `9a7864e` benchmark：引入开源 alpha-zero-general 作为 submodule 并做基准
- `1074a8b` perf：AlphaZero 自对弈多进程并行
- `79f2089` perf：MCTS 批量叶节点评估（CPU 提速约 25 倍）
- `cfd569b` docs：AlphaZero 500 迭代对比结果
- `d8d1153` Splendor 训练重构：固定动作空间 + 结构化模型 + AlphaZero
- （更早）`bdb85cf`…：AlphaZero 初探、TensorBoard、PPO 奖励配置、位置增强、评估系统、PPO 训练、模型架构等
