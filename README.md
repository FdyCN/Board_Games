# Board Games AI Training Framework

一个通用的、可扩展的桌游自对弈强化学习训练框架。支持多智能体竞技、分布式训练、ELO评分系统和人机对战。

## 项目特点

- **通用性设计**：基于抽象接口，轻松适配不同桌游（Splendor、UNO等）
- **轻量化模型**：针对桌游场景优化，模型参数量 100K-1M，适合本地训练和部署
- **多进程训练**：支持多核CPU并行自对弈，显著提升训练效率
- **完整评估体系**：ELO 评分系统、锦标赛管理、性能指标追踪
- **实时可视化**：【TBD】训练监控、对局回放、Web 人机对战界面
- **灵活配置**：YAML 配置驱动，支持快速实验迭代

## 训练进展（近期更新）

Splendor AI 训练经过一轮深入重构，主要改进如下：

### 关键 Bug 修复

- **按玩家分组计算 GAE**：多人自对弈中，之前把 4 个玩家交错的轨迹当单一轨迹算优势函数，导致 advantage/value 全部变成噪声。现已改为按 `player_id` 分组、每个玩家独立计算 GAE。
- **终局奖励正确注入**：之前"胜利 +1"只落在最后一位行动玩家身上，基本丢失。现改为终局零和奖励注入到每个玩家自己的最后一步。
- **游戏结束条件修复**：之前要求"最后一位玩家也必须达到 15 分"才结束，导致对局在最后一轮无限拖长。现改为"有人达到 15 分后走完当前一轮即结束"，并加了回合上限防止死锁。
- **多进程 reward_config 传递**：多进程收集数据时，子进程之前会退回默认奖励系数、忽略自定义稠密奖励。已修复。

### 固定动作空间（46 个规范槽位）

把之前"相对索引"（动作含义每步都在变）改成固定槽位，每个槽位语义稳定：

```
0-4 拿2同色 | 5-14 拿3不同色 | 15-26 保留明牌 | 27-29 保留牌堆
30-41 买明牌 | 42-44 买保留 | 45 pass
```

### 结构化模型（逐卡共享评估器）

参考 [alpha-zero-general](https://github.com/cestpasphoto/alpha-zero-general) 的 Splendor 实现，策略头对"买明牌/保留明牌/买保留牌"使用**共享卡评估器**（同一个线性层施加到每张卡），让模型学到"这张卡值不值得买"这个与位置无关的概念。

### AlphaZero 训练循环

新增 `scripts/train_alphazero.py`：自对弈用 MCTS 搜索，策略损失 = CE(policy, MCTS 目标)，价值损失 = MSE(value, 终局胜负)。AlphaZero 的价值函数能学到"谁赢"（value_loss 从 ~1.0 降到 ~0.17），而 PPO 的 GAE 价值在对称自对弈中几乎学不会。

> ⚠️ 注：AlphaZero 的策略网络需要**数百次模拟 + 上千次迭代**才能反超 PPO。本仓库默认配置（50 sims、16 局/迭代）跑 500 迭代后，价值函数学到了，但策略网络仍未学会高效买卡，对弈会打到回合上限；如需完整 AlphaZero，需要更强的算力（数百 sims + 数千迭代，参考实现用 numba 加速）。当前**PPO 仍是可用效果最好的模型**。

### 当前结果（3 人对局，PPO）

| 指标 | 数值 |
|---|---|
| 平均回合数 | ~87（随机策略 ~105） |
| tier1 买卡占比 | ~61%（修复前 ~69%） |
| 对随机 agent 胜率 | ~78% |

## 快速开始

### 开发环境

- Python 3.10+
- PyTorch 2.0+ (支持 MPS 加速)
- Apple M4 Max 64GB

### 安装

```bash
# 克隆仓库
git clone <repo_url>
cd Board_Games

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

### 训练你的第一个 Agent

```bash
# 训练 Splendor 3人对弈模型（PPO，带 JSONL 收敛日志）
python scripts/train_convergence.py \
    --config configs/splendor_ppo_mlp_medium_3p.yaml \
    --iterations 300 --episodes 128 \
    --log-file data/logs/convergence_3p.jsonl

# 使用 AlphaZero 训练（MCTS + 终局胜负价值）
python scripts/train_alphazero.py \
    --config configs/splendor_ppo_mlp_medium_3p.yaml \
    --iterations 500 --episodes 16 --simulations 50 \
    --log-file data/logs/alphazero_3p.jsonl

# 对手池训练（混合历史 checkpoint 做对手）
python scripts/train_opponent_pool.py \
    --config configs/splendor_ppo_mlp_medium_3p.yaml \
    --iterations 500 --episodes 128 \
    --log-file data/logs/pool_3p.jsonl

# 评估模型性能
python scripts/evaluate.py --checkpoint data/checkpoints/splendor_ppo/latest.pth

# 人机对战
python scripts/play_human.py --game splendor
```

#### 多进程训练配置

在配置文件中调整 `num_workers` 参数以启用多核并行训练：

```yaml
training:
  num_workers: 4  # 并行进程数（1=单进程，>1=多进程）
  episodes_per_iteration: 50
  device: "cpu"  # 多进程训练推荐使用CPU
```

**性能建议**：
- 设置 `num_workers` 为 CPU 核心数的 50-75%
- 例如 8 核 CPU 推荐设置为 4-6 workers
- 多进程可将训练速度提升 2-4 倍

#### TensorBoard 可视化

训练过程自动记录到 TensorBoard，可以实时监控训练进度：

```bash
# 启动 TensorBoard
tensorboard --logdir data/checkpoints/splendor_ppo/mlp_medium/tensorboard

# 在浏览器打开 http://localhost:6006
```

**监控指标**：
- **Loss**: policy_loss, value_loss, entropy
- **Performance**: mean_reward, mean_episode_length
- **PPO**: kl_divergence, clip_fraction, learning_rate
- **WinRate**: 各位置胜率 (position_0, position_1, ...)
- **PositionBias**: win_rate_std, win_rate_range

## 项目结构

```
Board_Games/
├── core/                   # 核心抽象层（游戏/Agent 接口）
├── games/                  # 游戏实现（Splendor、UNO 等）
├── agents/                 # Agent 实现（随机、神经网络、人类）
├── models/                 # 神经网络架构（编码器、策略头、价值头）
├── training/               # 训练框架（PPO、自对弈、分布式）
├── evaluation/             # 评估系统（Arena、ELO、锦标赛）
├── visualization/          # 可视化（训练监控、游戏渲染、Web 界面）
├── configs/                # 配置文件
├── scripts/                # 命令行工具
└── tests/                  # 单元测试
```

## 支持的游戏

| 游戏 | 状态 | 玩家数 | 模型类型 | 备注 |
|------|------|--------|----------|------|
| **Splendor** | 🚧 开发中 | 2-4 | Attention/MLP | 首个实现游戏 |
| **UNO** | 📋 计划中 | 2-4 | Attention/MLP | - |

## 架构文档

- [架构设计](./ARCHITECTURE.md) - 详细的系统架构和设计模式
- [开发指南](./DEVELOPMENT.md) - 开发进度和任务追踪
- [API 文档](./API.md) - 核心接口和使用示例

## 技术栈

- **深度学习**：PyTorch 2.x (MPS 加速)
- **强化学习**：PPO (Proximal Policy Optimization) + MCTS Enhancement
- **并行计算**：torch.multiprocessing
- **可视化**：TensorBoard / Weights & Biases
- **Web 框架**：FastAPI + React
- **配置管理**：YAML + Hydra

## 开发路线图

- [x] 项目架构设计
- [x] 核心抽象层实现
- [x] Splendor 游戏引擎
- [x] 神经网络模型
- [x] PPO 训练框架
- [x] **MCTS-Enhanced PPO** (v1.0 已完成)
- [x] TensorBoard 可视化
- [x] ELO 评分系统
- [x] 固定动作空间（46 个规范槽位）
- [x] 结构化模型（逐卡共享评估器）
- [x] AlphaZero 训练循环
- [ ] 分布式自对弈
- [ ] Web 人机对战界面

详细进度请查看 [DEVELOPMENT.md](./DEVELOPMENT.md)

## MCTS-Enhanced PPO Training 🆕

本框架现已支持 **MCTS-Enhanced PPO**，一种混合训练算法，结合了 PPO 的快速学习能力和 MCTS 的策略改进能力。

### 核心特性

- ✅ **渐进式 MCTS 调度**: 从纯 PPO 平滑过渡到 MCTS-增强训练
- ✅ **灵活配置**: 支持动态调整 MCTS 模拟次数
- ✅ **完整测试**: 包含单元测试和集成测试
- ✅ **生产就绪**: 所有已知 bug 已修复，系统稳定

### 快速开始 MCTS 训练

```bash
# 使用 MCTS 增强的 PPO 训练
python scripts/train.py --config configs/splendor_mcts_ppo.yaml

# 或使用快捷脚本
./scripts/train_mcts.sh
```

### MCTS 训练阶段

| 迭代范围 | MCTS 模拟次数 | 说明 |
|---------|--------------|------|
| 0-199   | 0 (纯 PPO)   | 快速探索，建立基础策略 |
| 200-499 | 50           | 引入 MCTS，开始改进决策 |
| 500-799 | 100          | 增强阶段，深化策略质量 |
| 800+    | 200          | 精炼阶段，高质量策略训练 |

### 文档

- 📘 [MCTS 训练指南](./docs/MCTS_TRAINING.md) - 完整使用说明
- 📗 [MCTS 实现总结](./docs/MCTS_IMPLEMENTATION_SUMMARY.md) - 技术细节
- 📙 [MCTS 待改进项](./docs/MCTS_TODO.md) - 优化建议
- 📕 [Bug 修复记录](./docs/BUGFIXES.md) - 问题解决历史
- 🚀 [准备就绪指南](./docs/MCTS_READY.md) - 开始训练前必读

### 性能预期

MCTS-Enhanced PPO 相比纯 PPO 预期改进：
- **平均步数**: 从 ~150 降到 100-120
- **位置偏差**: 胜率标准差从 >0.2 降到 <0.15
- **策略质量**: 更准确的价值估计和决策

## 示例：添加新游戏

```python
# games/my_game/game.py
from core.game_interface import GameInterface

@register_game("my_game")
class MyGame(GameInterface):
    def reset(self) -> GameState:
        # 实现游戏重置逻辑
        pass

    def step(self, action):
        # 实现游戏步进逻辑
        pass

    # 实现其他抽象方法...
```

## 贡献指南

欢迎贡献！请遵循以下流程：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

MIT License

## 致谢

- AlphaZero 论文启发了整体架构
- OpenAI Gym 提供了环境接口设计参考
- PettingZoo 提供了多智能体环境设计思路

## 联系方式

- Issue Tracker: [GitHub Issues](./issues)
- 讨论区: [GitHub Discussions](./discussions)

---

**当前版本**: v0.6.0-dev (固定动作空间 + 结构化模型 + AlphaZero)
**最后更新**: 2025-12
