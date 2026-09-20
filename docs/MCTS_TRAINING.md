# MCTS-Enhanced PPO Training

混合 MCTS + PPO 训练系统,用于提升 Splendor 4人对弈的策略质量。

## 核心特性

✅ **渐进式 MCTS 增强**: 从纯 PPO 逐步过渡到 MCTS 增强
✅ **保留 PPO 基础设施**: 无需重写训练框架
✅ **灵活配置**: 支持固定模拟次数或调度器
✅ **完整监控**: TensorBoard 实时追踪 MCTS 模拟次数

## 快速开始

### 1. 运行测试

首先验证 MCTS 模块是否正常工作:

```bash
python tests/test_mcts.py
```

预期输出:
```
==============================================================
MCTS 单元测试
==============================================================

测试 MCTS 节点
...
✅ MCTS 节点测试通过

测试 MCTS 调度器
...
✅ MCTS 调度器测试通过

测试 MCTS 搜索
...
✅ MCTS 搜索测试通过

✅ 所有测试通过!
```

### 2. 开始训练

使用提供的配置文件开始训练:

```bash
# 方式 1: 使用快速启动脚本
./scripts/train_mcts.sh

# 方式 2: 直接运行训练脚本
python scripts/train.py --config configs/splendor_mcts_ppo.yaml
```

### 3. 监控训练

启动 TensorBoard 查看训练进度:

```bash
tensorboard --logdir data/checkpoints/splendor_mcts_ppo/mlp_medium_v1/tensorboard
```

访问 http://localhost:6006 查看:
- **MCTS/simulations**: 当前 MCTS 模拟次数
- **Performance/mean_reward**: 平均奖励
- **WinRate/position_X**: 各位置胜率
- **PositionBias/win_rate_std**: 位置偏差标准差

## 配置说明

### 渐进式 MCTS 调度

默认配置使用渐进式调度:

```yaml
algorithm:
  mcts:
    enabled: true
    scheduler:
      enabled: true
      schedule:
        - [0, 0]          # 0-199 次: 纯 PPO (快速学习基础策略)
        - [200, 50]       # 200-499 次: 50 次模拟 (开始引入搜索)
        - [500, 100]      # 500-799 次: 100 次模拟 (中等搜索深度)
        - [800, 200]      # 800+ 次: 200 次模拟 (接近最优)
```

**优势**:
- 前期训练快速,快速积累基础经验
- 后期逐步增强,提升策略质量
- 训练速度比纯 AlphaZero 快 3-5 倍

### 固定 MCTS 模拟

如果想全程使用固定模拟次数:

```yaml
algorithm:
  mcts:
    enabled: true
    simulations: 100        # 固定 100 次模拟
    scheduler:
      enabled: false        # 禁用调度器
```

### 禁用 MCTS (纯 PPO)

对比实验时可以禁用 MCTS:

```yaml
algorithm:
  mcts:
    enabled: false
```

## 重要配置修改

### ⚠️ 奖励配置已修改

为了让模型学习"赢游戏"而非"拿高分",**所有稠密奖励已设为 0**:

```yaml
algorithm:
  dense_rewards:
    take_gem: 0.0          # ❌ 去掉拿宝石奖励
    buy_card_points: 0.0   # ❌ 去掉买卡分数奖励
    win: 1.0               # ✅ 只保留胜利奖励!
```

**原因**: 之前的稠密奖励导致模型疯狂买高分卡,但不在乎是否能赢。

## 架构概览

```
训练流程:
┌─────────────────────────────────────────────────────┐
│  迭代 N                                             │
│                                                     │
│  1. 获取 MCTS 模拟次数 (from scheduler)              │
│     ↓                                               │
│  2. 数据收集:                                        │
│     - 如果 simulations > 0:                         │
│       → MCTS 搜索 → 改进的动作概率 → 采样           │
│     - 如果 simulations = 0:                         │
│       → 直接从策略网络采样 (原始 PPO)                │
│     ↓                                               │
│  3. PPO 更新 (保持不变):                            │
│     - GAE 优势估计                                  │
│     - Clipped Surrogate Objective                  │
│     - 价值函数更新                                  │
│     ↓                                               │
│  4. 保存检查点 & TensorBoard 记录                   │
└─────────────────────────────────────────────────────┘
```

## 性能建议

### 计算资源

- **CPU 训练**: 推荐 (MCTS 主要是 CPU 密集型)
- **GPU 训练**: 可选 (神经网络推理可加速)
- **多进程**: 暂不支持 (MCTS 与多进程存在兼容问题,设置 `num_workers: 1`)

### 训练速度估算

以 M4 Max CPU 为例:

| MCTS 模拟次数 | 每次迭代时间 | 100 次迭代总时间 |
|--------------|-------------|----------------|
| 0 (纯 PPO)   | ~30 秒      | ~50 分钟       |
| 50 次模拟    | ~2 分钟     | ~3.5 小时      |
| 100 次模拟   | ~4 分钟     | ~7 小时        |
| 200 次模拟   | ~8 分钟     | ~14 小时       |

### 超参数调优

如果训练效果不理想,可以调整:

1. **MCTS 探索常数 `c_puct`**:
   - 增大 → 更多探索 (推荐范围: 1.0-2.5)
   - 减小 → 更多利用

2. **采样温度 `temperature`**:
   - 1.0 → 随机采样 (前期推荐)
   - 0.5 → 适度随机
   - 0.0 → 确定性 (后期可尝试)

3. **Dirichlet 噪声**:
   - `add_noise: true` → 增加探索 (推荐)
   - `add_noise: false` → 减少随机性

## 故障排查

### 问题 1: 内存不足

**现象**: OOM (Out of Memory)

**解决方案**:
```yaml
training:
  episodes_per_iteration: 64  # 从 128 减少到 64
  minibatch_size: 128         # 从 256 减少到 128
```

### 问题 2: MCTS 搜索太慢

**现象**: 每次迭代超过 10 分钟

**解决方案**:
```yaml
algorithm:
  mcts:
    simulations: 50  # 减少模拟次数
    # 或使用更小的模型
model:
  config: "small"    # 从 "medium" 改为 "small"
```

### 问题 3: 位置偏差仍然明显

**现象**: `PositionBias/win_rate_range` > 0.3

**解决方案**:
```yaml
training:
  use_position_augmentation: true  # 确保启用
algorithm:
  mcts:
    add_noise: true  # 确保启用噪声
```

## 文件结构

```
Board_Games/
├── training/
│   └── mcts/
│       ├── __init__.py
│       ├── node.py           # MCTS 节点实现
│       ├── search.py         # MCTS 搜索引擎
│       └── scheduler.py      # 渐进式调度器
├── configs/
│   └── splendor_mcts_ppo.yaml  # MCTS 训练配置
├── scripts/
│   ├── train.py              # 训练脚本 (已更新支持 MCTS)
│   └── train_mcts.sh         # 快速启动脚本
└── tests/
    └── test_mcts.py          # MCTS 单元测试
```

## 下一步

1. **验证效果**: 训练 200-300 次迭代,观察 `mean_episode_length` 是否下降
2. **对比实验**: 运行纯 PPO 配置作为基线对比
3. **调优超参数**: 根据 TensorBoard 曲线调整 MCTS 参数
4. **扩展应用**: 成功后可应用到其他多人博弈游戏

## 贡献者

实现参考:
- AlphaZero 论文 (Silver et al., 2017)
- PPO 论文 (Schulman et al., 2017)

---

**版本**: v1.0
**最后更新**: 2025-11-19
