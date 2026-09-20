# 开发指南

本文件是**开发规范与贡献指南**。项目历史 / Bug 修复 / 训练结果见 [CHANGELOG.md](./CHANGELOG.md)，使用指南见 [README.md](./README.md)，架构见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

## 目录

- [开发环境搭建](#开发环境搭建)
- [代码规范](#代码规范)
- [测试策略](#测试策略)
- [Git 工作流](#git-工作流)
- [FAQ](#faq)

---

## 开发环境搭建

```bash
# 1. 克隆
git clone <repo_url>
cd Board_Games

# 2. 虚拟环境
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. 依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, black, ruff 等
pip install -e .
```

```bash
# 验证
pytest tests/                 # 运行全部测试
ruff check .                  # 静态检查
black --check .               # 格式检查
```

### IDE（VSCode 推荐）

启用 pytest、black（formatOnSave）、ruff。配置文件见 `.vscode/`。

---

## 代码规范

### 风格

- **格式化**：Black（line-length=100）
- **Lint**：Ruff
- **类型检查**：MyPy

### 命名

```python
game_interface.py          # 文件: snake_case
class GameInterface: ...   # 类: PascalCase
def reset_game(): ...      # 函数/变量: snake_case
MAX_PLAYERS = 4            # 常量: UPPER_SNAKE_CASE
def _internal_helper(): ...# 私有: _leading_underscore
```

### Docstring

Google 风格（`Args:` / `Returns:` / `Raises:` / `Examples:`），见现有模块。

### 架构约定

- 游戏、模型、训练算法相互解耦，通过 `core/` 抽象接口连接（依赖倒置）
- 新游戏：继承 `GameInterface` + `@register_game`，放入 `games/<name>/`（见 [docs/ADDING_A_GAME.md](./docs/ADDING_A_GAME.md)）
- 训练产物按游戏分组：`data/<game>/checkpoints|logs|reports`（路径辅助见 `core/paths.py`）

---

## 测试策略

### 测试金字塔

```
      /\        E2E（少量）：完整训练流程
     /──\       Integration：引擎 + Agent 集成
    /────\      Unit（大量）：每个函数独立测试
```

### 覆盖目标

- `core/`、`games/`：> 90%
- `training/`：> 80%

### 新增游戏的最小测试

初始状态、动作合法性、`action_to_index`/`index_to_action` 互逆、结束条件、观察维度固定、随机 agent 完整对局不卡死、状态 `clone()` 独立。

---

## Git 工作流

```bash
git checkout -b feature/splendor-engine
git add games/splendor/state.py
git commit -m "feat(splendor): implement game state"
git push origin feature/splendor-engine
```

### Commit 规范（Conventional Commits）

```
feat(scope):    新功能
fix(scope):     修复 Bug
docs:           文档
test:           测试
refactor:       重构
perf:           性能
chore:          构建/工具链
```

---

## FAQ

### Q1：为什么选 PPO，而不是只用 AlphaZero？

PPO 实现更简单、训练更稳定，无需 MCTS 的大计算开销，对多人游戏已够用。AlphaZero 作为可选范式已实现（`scripts/train_alphazero.py`），但策略网络需要温度调度 + 大量迭代才能在算力上反超 PPO（见 [CHANGELOG.md](./CHANGELOG.md)）。

### Q2：如何保证游戏规则正确？

参考官方规则书，编写覆盖边界情况的单元测试，并用随机 agent 跑大量完整对局验证不卡死；必要时对比 BoardGameArena 等在线实现。

### Q3：为什么不直接用 Stable-Baselines3？

SB3 不直接支持多人自对弈；本框架需要自定义自对弈逻辑，但可参考其实现细节。

### Q4：训练不收敛怎么排查？

1. 检查奖励信号是否合理（终局零和是否注入到每个玩家）
2. 降低学习率、增大 `entropy_coef` 鼓励探索
3. 检查数值稳定性（NaN / 梯度爆炸）
4. 可视化策略分布，判断是否过早坍缩
5. 看 `data/<game>/logs/*.jsonl` 的 `explained_variance` / `kl_div` 曲线
