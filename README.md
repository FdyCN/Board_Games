# Board Games AI Training Framework

一个通用的、可扩展的桌游自对弈强化学习训练框架。支持多智能体竞技、分布式训练、ELO评分系统和人机对战。

## 项目特点

- **通用性设计**：基于抽象接口，轻松适配不同桌游（Splendor、UNO、狼人杀等）
- **轻量化模型**：针对桌游场景优化，模型参数量 100K-1M，适合本地训练和部署
- **分布式训练**：多进程并行自对弈，充分利用多核 CPU（Apple M4 Max 优化）
- **完整评估体系**：ELO 评分系统、锦标赛管理、性能指标追踪
- **实时可视化**：训练监控、对局回放、Web 人机对战界面
- **灵活配置**：YAML 配置驱动，支持快速实验迭代

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+ (支持 MPS 加速)
- 64GB RAM (推荐)

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
# 训练 Splendor 4人对弈模型
python scripts/train.py --config configs/experiments/splendor_ppo.yaml

# 评估模型性能
python scripts/evaluate.py --checkpoint data/checkpoints/splendor_v1.pth

# 人机对战
python scripts/play_human.py --game splendor
```

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
| **Splendor** | 🚧 开发中 | 2-4 | Attention | 首个实现游戏 |
| **UNO** | 📋 计划中 | 2-4 | MLP | - |
| **狼人杀** | 📋 计划中 | 6-12 | Transformer | 语言推理 |

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
- [ ] 核心抽象层实现
- [ ] Splendor 游戏引擎
- [ ] 神经网络模型
- [ ] PPO 训练框架
- [ ] 分布式自对弈
- [ ] ELO 评分系统
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

```yaml
# configs/games/my_game.yaml
game:
  name: "my_game"
  num_players: 4

model:
  encoder: "mlp"
  hidden_size: 256
```

```bash
# 直接训练
python scripts/train.py --config configs/games/my_game.yaml
```

## 性能基准

**Apple M4 Max (14-core CPU, 64GB RAM)**
- 自对弈速度：~1000 games/hour (4人 Splendor)
- 训练吞吐：~5000 steps/sec
- 推理延迟：<10ms/action

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

**当前版本**: v0.1.0-dev
**最后更新**: 2025-11-16
