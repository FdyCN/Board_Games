# 位置偏差数据增强方案实现

## 问题回顾

您的模型出现严重的位置过拟合：
- **最强位置 3：55% 胜率**
- **最弱位置 0：5% 胜率**
- **差异：50%！**
- **平均胜率：18.75% （低于随机基线 25%）**

## 解决方案：位置ID随机化数据增强

### 方案说明

由于 Splendor 观察编码是不对称的：
- 当前玩家：60 维（包含详细信息如保留卡）
- 其他玩家：15 维（只有摘要信息）

完整的位置旋转会导致信息丢失。因此我们采用**简化但有效的方案**：

**只随机化"当前玩家 ID"字段（321-325 维）**

这样：
1. ✅ 模型无法学习到"我总是在位置 X"
2. ✅ 保持所有其他信息完整
3. ✅ 避免不对称编码带来的问题

### 实现细节

#### 1. 数据增强函数 (`training/data_augmentation.py`)

```python
def rotate_splendor_observation(obs, rotation, num_players=4):
    """
    随机化"当前玩家 ID"字段
    - 输入：384 维观察
    - 操作：修改索引 321-325 的 one-hot 编码
    - 输出：修改后的观察
    """
```

#### 2. 批次处理 (`training/experience.py`)

```python
ExperienceBatch.from_experiences(
    experiences,
    device="cpu",
    use_position_augmentation=True,  # ← 启用数据增强
    num_players=4,
)
```

#### 3. 训练器集成 (`training/trainer.py`)

```python
trainer.train(
    ...
    use_position_augmentation=True,  # ← 传递给训练循环
)
```

#### 4. 配置文件 (`configs/splendor_ppo_mlp_medium.yaml`)

```yaml
training:
  use_position_augmentation: true  # ← 在配置中启用
```

## 使用方法

### 快速开始

**选项 1：使用配置文件（推荐）**

```bash
# 编辑配置文件
vim configs/splendor_ppo_mlp_medium.yaml

# 设置
training:
  use_position_augmentation: true

# 开始训练
python scripts/train.py --config configs/splendor_ppo_mlp_medium.yaml
```

**选项 2：命令行参数（如果支持）**

```bash
python scripts/train.py \
    --config configs/splendor_ppo_mlp_medium.yaml \
    --use-position-augmentation
```

### 验证数据增强工作

训练开始时，日志会显示：

```
============================================================
📊 训练配置
============================================================
起始迭代: 1
目标迭代: 1000
每次迭代 episodes: 50
PPO 更新轮数: 4
小批次大小: 256
并行进程数: 4
位置增强: 启用  ← **确认这一行显示"启用"**
设备: cpu
============================================================
```

### 运行测试

在训练前运行测试确保功能正常：

```bash
python scripts/test_augmentation.py
```

预期输出：
```
============================================================
✅ 所有测试通过！
============================================================
```

## 预期效果

### 训练过程中

1. **胜率分布更均衡**
   - 之前：[15.62%, 18.75%, 12.50%, 34.38%]
   - 期望：[~23%, ~25%, ~27%, ~25%] (差异 <10%)

2. **整体性能提升**
   - 之前：平均胜率 18.75% (低于随机)
   - 期望：平均胜率 >30% (超过随机)

3. **训练稳定性改善**
   - 减少过拟合到特定位置
   - 更快收敛到通用策略

### 如何验证改善

训练 100-200 迭代后，运行位置偏差测试：

```bash
python scripts/test_position_bias.py \
    --checkpoint data/checkpoints/splendor_ppo/mlp_medium/checkpoint_iter_100.pt \
    --num-games 100 \
    --device cpu
```

查看输出：

```
偏差指标:
  胜率标准差: 0.0245  ← 应该 <0.05
  胜率范围: 0.0600 (22.00% - 28.00%)  ← 应该 <10%

偏差评估:
  ✅ 低偏差 (<10% 差异) - 模型在各位置表现均衡
```

## 技术细节

### 为什么这个方案有效？

1. **阻断位置信息流**
   - 模型通过"当前玩家 ID"字段推断自己的位置
   - 随机化这个字段后，模型无法建立可靠的位置依赖
   - 迫使模型学习位置无关的策略

2. **保持信息完整性**
   - 不改变玩家状态、卡牌、贵族等关键信息
   - 避免不对称编码带来的信息丢失
   - 模型仍然能学习游戏机制

3. **50% 旋转概率**
   - 不是所有经验都被旋转
   - 50% 保持原样，50% 随机化
   - 平衡原始信息和增强信息

### 与其他方案的对比

| 方案 | 优点 | 缺点 | 适用性 |
|------|------|------|--------|
| **位置ID随机化** | 简单、无信息丢失 | 仅针对位置ID | ✅ 推荐 |
| 完整状态旋转 | 理论上最完整 | 信息丢失（不对称编码） | ❌ 不可行 |
| 移除位置编码 | 彻底消除偏差 | 丢失有用信息 | ⚠️ 备选 |
| 位置归一化训练 | 均衡位置经验 | 实现复杂 | ⚠️ 未来考虑 |

## 故障排除

### 问题 1：训练日志显示"位置增强: 禁用"

**原因**：配置文件未正确设置

**解决**：
```yaml
# configs/splendor_ppo_mlp_medium.yaml
training:
  use_position_augmentation: true  # 确保这一行存在且为 true
```

### 问题 2：测试失败

**原因**：代码版本不匹配

**解决**：
```bash
# 确保所有文件都已更新
git pull
python scripts/test_augmentation.py
```

### 问题 3：位置偏差仍然存在

**可能原因**：
1. 训练迭代不够（需要 200+ 迭代）
2. 模型已经过拟合（需要从头训练）
3. 其他因素导致位置偏差（如游戏设计）

**解决**：
```bash
# 从头开始训练新模型
python scripts/train.py --config configs/splendor_ppo_mlp_medium.yaml

# 训练至少 200 迭代后再测试
python scripts/test_position_bias.py \
    --checkpoint data/checkpoints/.../checkpoint_iter_200.pt \
    --num-games 100
```

## 下一步行动

1. **立即开始训练**
   ```bash
   python scripts/train.py --config configs/splendor_ppo_mlp_medium.yaml
   ```

2. **定期监控**
   - 每 50 迭代检查胜率分布
   - 观察是否更均衡

3. **中期评估 (迭代 100-200)**
   ```bash
   python scripts/test_position_bias.py \
       --checkpoint .../checkpoint_iter_100.pt \
       --num-games 100
   ```

4. **最终评估 (迭代 500+)**
   - 与随机 agent 对比
   - 确认位置偏差 <10%
   - 整体胜率 >30%

## 参考

- **位置偏差文档**: `docs/position_bias.md`
- **数据增强代码**: `training/data_augmentation.py`
- **测试脚本**: `scripts/test_augmentation.py`
- **位置偏差测试**: `scripts/test_position_bias.py`

---

**实现状态**: ✅ 已完成并测试
**上次更新**: 2025-11-17
**作者**: Claude Code
