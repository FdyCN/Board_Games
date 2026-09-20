# Board Games AI Training Framework

一个通用的、可扩展的桌游自对弈强化学习训练框架。支持多智能体自对弈、PPO / AlphaZero（MCTS）、ELO 评估，并按游戏组织训练产物（模型、日志、报告）。

> **文档导航**
> - 本文档：**使用指南**（安装、训练、评估、接入新游戏）
> - [CHANGELOG.md](./CHANGELOG.md)：Bug 修复与训练结果记录
> - [ARCHITECTURE.md](./ARCHITECTURE.md)：架构设计与设计模式
> - [DEVELOPMENT.md](./DEVELOPMENT.md)：开发规范、环境、测试
> - [docs/ADDING_A_GAME.md](./docs/ADDING_A_GAME.md)：如何接入一个新桌游

## 特性

- **游戏无关**：基于 `GameInterface` 抽象接口，训练/评估脚本从配置读取游戏名，接入新游戏零脚本改动
- **多智能体自对弈**：支持 2-4 人对称自对弈，多进程并行数据收集
- **两种训练范式**：PPO（GAE + clipped surrogate，稠密奖励）与 AlphaZero（MCTS + 策略 CE + 终局价值 MSE）
- **固定动作空间**：动作映射到语义稳定的固定槽位，策略网络学到的偏好不会因状态变化而漂移
- **轻量模型**：MLP / Attention 编码器 + 结构化策略头，参数量 100K-1M，适合本地 CPU 训练
- **完整评估体系**：Arena 竞技场、ELO 评分、胜率/位置偏差统计
- **按游戏组织的产物**：`data/<game>/checkpoints|logs|reports`，每个游戏独立存放

## 支持的游戏

| 游戏 | 状态 | 玩家数 | 模型 | 备注 |
|------|------|--------|------|------|
| **Splendor** | ✅ 已实现 | 2-4 | MLP / Attention | 首个完整实现，PPO 已收敛（vs 随机 98%） |
| **情书（Love Letter）** | ✅ 已实现 | 2-4 | MLP | 隐藏信息 + 回合制 + 淘汰制，3 人 PPO 训练就绪 |
| 政变疑云（Coup） | 📋 计划 | 3-6 | — | 见 [docs/ADDING_A_GAME.md](./docs/ADDING_A_GAME.md) |

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+（可选 MPS 加速；本项目 CPU 训练已足够快）

### 安装

```bash
git clone <repo_url>
cd Board_Games

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 训练一个模型

训练脚本是**游戏无关**的：游戏名、构造参数、奖励配置都从 YAML 配置读取。

```bash
# PPO 训练（JSONL 收敛日志）
python scripts/train_convergence.py \
    --config configs/splendor/splendor_ppo_mlp_medium_3p.yaml \
    --iterations 300 --episodes 128 \
    --log-file data/splendor/logs/convergence_3p.jsonl

# AlphaZero 训练（MCTS + 终局胜负价值）
python scripts/train_alphazero.py \
    --config configs/splendor/splendor_ppo_mlp_medium_3p.yaml \
    --iterations 500 --episodes 16 --simulations 50 \
    --log-file data/splendor/logs/alphazero_3p.jsonl

# 对手池训练（混合历史 checkpoint 做对手）
python scripts/train_opponent_pool.py \
    --config configs/splendor/splendor_ppo_mlp_medium_3p.yaml \
    --iterations 500 --episodes 128 \
    --log-file data/splendor/logs/convergence_3p_pool.jsonl
```

训练过程中，模型检查点保存到配置里的 `training.checkpoint_dir`（例如 `data/splendor/checkpoints/mlp_medium_3p_v1/`），TensorBoard 事件保存在该目录的 `tensorboard/` 子目录。

### 评估模型

```bash
# 单个模型 vs 随机 Agent
python scripts/evaluate.py \
    --model data/splendor/checkpoints/mlp_medium_3p_v1/latest.pth \
    --mode vs_random --games 100

# 多个模型锦标赛
python scripts/evaluate.py \
    --models model1.pth model2.pth model3.pth \
    --mode tournament --games 200

# 检查点进化曲线
python scripts/evaluate.py \
    --checkpoints data/splendor/checkpoints/mlp_medium_3p_v1/ \
    --mode evolution --games 50
```

### 与开源模型对比（Splendor 专用）

仓库引入了 [cestpasphoto/alpha-zero-general](https://github.com/cestpasphoto/alpha-zero-general) 作为 git submodule 做基准：

```bash
# vs 随机（本地 PPO vs 开源 AlphaZero）
python scripts/compare_with_open_source.py

# 直接 head-to-head（本地 PPO vs 开源 AlphaZero + 随机，同场对打）
python scripts/head_to_head.py
```

## 数据与产物目录

所有训练产物按游戏分组，互不干扰：

```
data/
  <game_name>/                 # 例如 splendor
    checkpoints/               # 模型检查点
      <run_name>/              # 例如 mlp_medium_3p_v1
        latest.pth
        checkpoint_iter_*.pth
        tensorboard/           # TensorBoard 事件
    logs/                      # JSONL 训练日志 + stdout 日志
      <run_name>.jsonl
    reports/                   # 评估 / 基准报告（可选）
```

路径辅助函数见 `core/paths.py`（`run_checkpoint_dir`、`default_log_path` 等）。新增游戏时用 `scripts/new_game.py` 会自动建好这套目录。

## 项目结构

```
Board_Games/
├── core/                   # 抽象接口（GameInterface/AgentInterface）+ 路径辅助
├── games/                  # 游戏实现（每个游戏一个子目录）
│   └── splendor/           #   Splendor 引擎（game/state/actions/encoder/RULES）
├── agents/                 # Agent（随机、神经网络）
├── models/                 # 神经网络（编码器、策略/价值/胜负头、模型工厂）
├── training/               # 训练框架（PPO、自对弈 worker、MCTS、经验缓冲）
├── evaluation/             # 评估（Arena、ELO、指标）
├── configs/                # 配置文件（按游戏分目录，如 configs/splendor/）
├── scripts/                # 命令行工具（train_*、evaluate、new_game 脚手架）
├── docs/                   # 主题文档（接入新游戏、MCTS 等）
├── tests/                  # 单元测试
└── third_party/            # git submodule（开源基准）
```

## 接入一个新游戏

最小步骤（详见 [docs/ADDING_A_GAME.md](./docs/ADDING_A_GAME.md)）：

1. `python scripts/new_game.py <game_name> --players N` 生成骨架
2. 实现 `games/<game_name>/game.py`（继承 `GameInterface`，`@register_game` 注册）
3. 实现 `games/<game_name>/encoder.py`（状态 → 固定维度观察）
4. 补全测试，用 `configs/<game_name>/<game_name>_ppo.yaml` 训练

训练脚本无需修改——它们从配置读取 `game.name` 与构造参数。

## 配置说明

配置为 YAML，分六段：`game` / `model` / `algorithm` / `training` / `evaluation` / `experiment`。示例见 `configs/splendor/`。要点：

- `game.name`：游戏注册名，训练脚本据此调用 `create_game`
- `algorithm.dense_rewards`：PPO 稠密奖励，是**自由字典**，键由每个游戏自己定义
- `training.checkpoint_dir`：检查点目录（建议按 `data/<game>/checkpoints/<run>` 组织）

## 技术栈

- **深度学习**：PyTorch 2.x
- **强化学习**：PPO（GAE + clipped surrogate）、AlphaZero（MCTS）
- **并行**：torch.multiprocessing
- **可视化**：TensorBoard
- **配置**：YAML

## 贡献

欢迎贡献！流程：Fork → 特性分支 → 提交（Conventional Commits）→ Pull Request。

## 许可证

MIT License
