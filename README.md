# Board Games AI Training Framework

一个通用的、可扩展的桌游自对弈强化学习训练框架。支持多智能体竞技、分布式训练、ELO评分系统和人机对战。

## 项目特点

- **通用性设计**：基于抽象接口，轻松适配不同桌游（Splendor、UNO等）
- **轻量化模型**：针对桌游场景优化，模型参数量 100K-1M，适合本地训练和部署
- **多进程训练**：支持多核CPU并行自对弈，显著提升训练效率
- **完整评估体系**：ELO 评分系统、锦标赛管理、性能指标追踪
- **实时可视化**：【TBD】训练监控、对局回放、Web 人机对战界面
- **灵活配置**：YAML 配置驱动，支持快速实验迭代

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
# 训练 Splendor 4人对弈模型（单进程）
python scripts/train.py --config configs/splendor_ppo_mlp_medium.yaml

# 使用多进程加速训练（推荐CPU训练时使用）
# 在配置文件中设置 training.num_workers: 4

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
- **强化学习**：PPO (Proximal Policy Optimization)
- **并行计算**：torch.multiprocessing
- **可视化**：TensorBoard / Weights & Biases
- **Web 框架**：FastAPI + React
- **配置管理**：YAML + Hydra

## 开发路线图

- [x] 项目架构设计
- [x]  核心抽象层实现
- [x] Splendor 游戏引擎
- [x] 神经网络模型
- [x] PPO 训练框架
- [ ] 分布式自对弈
- [x] ELO 评分系统
- [ ] 训练可视化
- [ ] Web 人机对战界面

详细进度请查看 [DEVELOPMENT.md](./DEVELOPMENT.md)

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

**当前版本**: v0.3.0-dev
**最后更新**: 2025-11-17
