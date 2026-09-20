# MCTS Training Bug Fixes

## Bug #1: Experience 参数名错误 ✅ 已修复

**错误信息**:
```
TypeError: Experience.__init__() got an unexpected keyword argument 'old_log_prob'
```

**原因**: `Experience` 类使用 `log_prob` 参数,不是 `old_log_prob`

**修复位置**: `training/trainer.py:602`
```python
# 修复前
exp = Experience(old_log_prob=step['log_prob'], ...)

# 修复后
exp = Experience(log_prob=step['log_prob'], ...)
```

---

## Bug #2: Episode 构造参数错误 ✅ 已修复

**错误信息**: Episode 初始化失败

**原因**: `Episode` 需要 `experiences` 列表,不是 `player_experiences`

**修复位置**: `training/trainer.py:631-638`
```python
# 修复前
episode = Episode(player_experiences=player_experiences, ...)

# 修复后
episode = Episode(experiences=all_experiences, ...)
```

---

## Bug #3: MCTS 搜索遇到终局状态 ✅ 已修复

**错误信息**:
```
ValueError: No legal actions available
```

**原因**: 游戏进入终局但训练循环没有提前检测

**修复位置**: `training/trainer.py:493-496`
```python
# 添加提前检查
legal_actions = self.game.get_legal_actions(state)
if not legal_actions:
    # 没有合法动作,游戏应该结束
    break
```

同时优化了 `training/mcts/search.py:88-93` 的错误信息

---

## 验证结果

### 单元测试
```bash
python tests/test_mcts.py
# ✅ 所有测试通过
```

### 集成测试
```bash
python tests/test_mcts_training.py
# ✅ 训练成功完成
# - Policy Loss: 0.2107
# - Value Loss: 0.0134
# - Mean Reward: 3.7050
```

### 压力测试
```bash
# 收集 10 个完整 episodes
# ✅ 成功收集并训练
# - 平均步数: 155.8
# - 平均奖励: 2.8720
```

---

## 现在可以安全训练了!

```bash
# 从检查点恢复训练
python scripts/train.py \
  --config configs/splendor_mcts_ppo.yaml \
  --resume data/checkpoints/splendor_mcts_ppo/mlp_medium_v1/latest.pth
```

或使用快速脚本:
```bash
./scripts/train_mcts.sh
# 选择 "y" 从检查点恢复
```

---

**修复时间**: 2025-11-19
**状态**: 所有已知 bug 已修复
**验证**: 通过完整测试套件
