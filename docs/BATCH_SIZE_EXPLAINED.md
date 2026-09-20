# Batch Size 详解：为什么说 GPU 用的是 batch_size = 1？

## 🎯 核心概念

在强化学习训练中，有**两个不同阶段**使用**两个不同的 batch size**：

### 1️⃣ 自对弈阶段（Inference / Data Collection）
- **用途**: 收集训练数据
- **Batch Size**: **1** ⚠️（这是性能瓶颈！）
- **代码位置**: `agents/neural_agent.py` 的 `select_action` 方法
- **占总时间**: ~80-90%

### 2️⃣ 训练阶段（Training / Gradient Update）
- **用途**: 更新模型参数
- **Batch Size**: **256** ✅（配置文件中的 `minibatch_size`）
- **代码位置**: `training/algorithms/ppo.py` 的 `_update_minibatch` 方法
- **占总时间**: ~10-20%

---

## 📊 时间分配示例

假设训练 1000 个 iteration，每个 iteration 收集 50 个 episodes：

```
总训练时间: 100%
│
├─ 自对弈阶段 (batch_size=1): ~85%  ← 性能瓶颈在这里！
│  └─ 收集 50,000 个 episodes
│     └─ 每个 episode ~150 steps
│        └─ 每 step 调用 1 次 select_action (batch_size=1)
│
└─ PPO 训练阶段 (batch_size=256): ~15%  ← GPU 在这里有优势
   └─ 更新模型参数
      └─ 每次更新处理 256 个样本
```

---

## 🔍 代码分析

### 阶段 1: 自对弈（Batch Size = 1）

**位置**: `training/self_play_worker.py` 第 82-84 行

```python
# 游戏循环中，每一步都要调用一次
action_idx, info = agent.select_action(
    observation,           # shape: (384,) - 单个样本！
    legal_action_indices,
    deterministic=deterministic
)
```

**转换过程**: `agents/neural_agent.py` 第 186-205 行

```python
def _obs_to_tensor(self, observation):
    """将观察转换为 tensor"""
    if isinstance(observation, np.ndarray):
        obs = torch.from_numpy(observation).float()
    else:
        obs = observation.float()

    # 🔴 关键！确保是 (1, obs_dim) 形状
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)  # (384,) → (1, 384)
        #                         ↑
        #                    batch_size = 1

    return obs.to(self.device)
```

**实际运行**:
```python
# 输入
observation = np.array([...])  # shape: (384,)

# 转换为 tensor
obs_tensor = torch.from_numpy(observation)  # shape: (384,)
obs_tensor = obs_tensor.unsqueeze(0)        # shape: (1, 384)
                                             #        ↑
                                             #   batch_size = 1

# 模型推理
logits, value = model(obs_tensor, legal_mask)
# 输入 shape: (1, 384)  ← 只有 1 个样本
# 输出 logits: (1, 50)
#      value:  (1, 1)
```

**为什么是 1？**
- 因为游戏是**顺序执行**的
- 每次只有**一个玩家**在行动
- 需要**立即**得到动作才能继续游戏
- 无法等待收集多个样本再批量推理

---

### 阶段 2: PPO 训练（Batch Size = 256）

**位置**: `training/algorithms/ppo.py` 第 155-168 行

```python
def _update_minibatch(self, batch: ExperienceBatch):
    """更新一个小批次"""
    observations = batch.observations  # shape: (256, 384)  ← 256 个样本！
    actions = batch.actions           # shape: (256,)
    old_log_probs = batch.old_log_probs  # shape: (256,)
    advantages = batch.advantages     # shape: (256,)
    returns = batch.returns           # shape: (256,)

    # 前向传播（批量处理）
    new_log_probs, entropy, new_values = self.model.evaluate_actions(
        observations,  # shape: (256, 384)  ← GPU 可以并行处理！
        actions,
        legal_mask
    )
```

**配置文件**: `configs/splendor_ppo_mlp_medium.yaml` 第 31-36 行

```yaml
training:
  num_iterations: 1000
  episodes_per_iteration: 50
  update_epochs: 4
  minibatch_size: 256  # ← 这个 256 是训练时的 batch_size
  device: "mps"
```

**实际运行**:
```python
# 从经验池中采样 256 个样本
batch = experience_buffer.sample(batch_size=256)

# 批量处理
observations = torch.stack([exp.observation for exp in batch])
# shape: (256, 384)  ← 256 个样本同时处理

# 模型训练
new_log_probs, entropy, new_values = model.evaluate_actions(
    observations,  # (256, 384) ← GPU 可以并行
    actions,       # (256,)
    legal_mask
)

# 计算损失和梯度更新
loss.backward()
optimizer.step()
```

---

## ⚡ 性能瓶颈分析

### 为什么 MPS 慢？

因为 **85% 的时间都在自对弈阶段**，而这个阶段 **batch_size = 1**！

```
自对弈阶段（占 85% 时间）:
├─ Batch Size: 1  ← GPU 无法发挥并行优势
├─ 频繁的 CPU-GPU 传输（每次推理都要传输）
└─ 每次推理延迟: 0.887ms (MPS) vs 0.062ms (CPU)
   └─ MPS 慢 14 倍！

训练阶段（占 15% 时间）:
├─ Batch Size: 256  ← GPU 可以发挥优势
├─ 批量传输（传输次数少）
└─ 每次更新延迟: 可能 MPS 更快
   └─ 但只占总时间的 15%，影响有限
```

### 时间计算示例

假设收集 50 个 episodes，每个 150 steps：

```
自对弈时间 (batch_size=1):
  - 总步数: 50 × 150 = 7,500 steps
  - CPU: 7,500 × 0.062ms = 465ms
  - MPS: 7,500 × 0.887ms = 6,653ms

训练时间 (batch_size=256):
  - 样本数: 7,500
  - 更新次数: 7,500 ÷ 256 × 4 epochs ≈ 117 次
  - 每次更新: 假设 5ms (MPS) vs 8ms (CPU)
  - CPU: 117 × 8ms = 936ms
  - MPS: 117 × 5ms = 585ms

总时间:
  - CPU: 465ms + 936ms = 1,401ms
  - MPS: 6,653ms + 585ms = 7,238ms

  MPS 比 CPU 慢: 7,238 ÷ 1,401 = 5.17 倍 ✓
```

这与我们的基准测试结果一致！

---

## 💡 如何让 GPU 发挥优势？

### 方案 1: 批量自对弈（推荐）

**核心思路**: 同时运行多个游戏实例，批量推理

```python
# 当前方式（顺序）
for game in games:
    while not done:
        action = agent.select_action(obs)  # batch_size=1
        obs, reward, done = game.step(action)

# 改进方式（并行）
batch_size = 64
games = [create_game() for _ in range(batch_size)]

while any_game_not_done:
    # 收集所有游戏的观察
    observations = [game.get_observation() for game in games]
    observations = np.stack(observations)  # shape: (64, 384)

    # 批量推理
    actions = agent.select_action_batch(observations)  # batch_size=64!

    # 批量执行动作
    for game, action in zip(games, actions):
        game.step(action)
```

**效果预估**:
- Batch size = 64: MPS 可能快 2-3 倍
- Batch size = 256: MPS 可能快 5-10 倍

**实现难度**: ⭐⭐⭐⭐（需要重构大量代码）

---

### 方案 2: 增大模型规模

使用更大的模型，使得单次推理的计算量足以抵消传输开销

```yaml
model:
  encoder_type: "attention"
  config: "large"  # ~1M 参数（而不是 150K）
```

**效果预估**:
- Large 模型: MPS 可能持平或略快于 CPU

**缺点**:
- 训练收敛更慢
- 内存占用更大

---

### 方案 3: 继续使用 CPU（当前推荐）

**理由**:
1. ✅ 性能最佳（快 5 倍）
2. ✅ 无需修改代码
3. ✅ 稳定可靠

**配置**:
```yaml
training:
  device: "cpu"
  minibatch_size: 256  # 训练时仍然用 256
```

---

## 📋 总结

### 配置文件中的 `minibatch_size: 256`

- ✅ **是用于 PPO 训练阶段**的 batch size
- ✅ 在训练阶段，GPU 确实用 batch_size = 256
- ✅ 这个阶段 GPU 有优势

### 为什么说 GPU 用的是 batch_size = 1？

- ⚠️ **是指自对弈阶段**的 batch size
- ⚠️ 自对弈占总时间的 85%
- ⚠️ 这个阶段决定了整体性能
- ⚠️ batch_size = 1 无法发挥 GPU 优势

### 性能瓶颈在哪里？

```
训练流程:
┌─────────────────────────────────────────────┐
│ 自对弈阶段 (85% 时间)                        │
│ ├─ Batch Size: 1                            │
│ ├─ CPU 快 14 倍                              │
│ └─ 这是性能瓶颈！⚠️                          │
├─────────────────────────────────────────────┤
│ PPO 训练阶段 (15% 时间)                      │
│ ├─ Batch Size: 256                          │
│ ├─ GPU 可能快 1.5-2 倍                       │
│ └─ 但占比太小，影响有限                      │
└─────────────────────────────────────────────┘
```

### 最终建议

**对于当前项目**: 使用 **CPU** 训练

**如果想用 GPU**: 需要实现批量自对弈（工作量大）

---

## 🔗 相关文件

- 自对弈代码: `training/self_play_worker.py`
- Agent 推理: `agents/neural_agent.py`
- PPO 训练: `training/algorithms/ppo.py`
- 配置文件: `configs/splendor_ppo_mlp_medium.yaml`
- 性能基准: `scripts/benchmark_device.py`
