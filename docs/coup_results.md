# 政变疑云（Coup）训练结果

> 引擎设计见 [coup_design.md](./coup_design.md)。评测口径：`scripts/direct_head_to_head.py` 直接计数（随机座位，400 局/seed）。

## 结论速览

- **Coup 可以训练**：纯 league 自对弈（对手池 = 随机 + 冻结快照）的模型 vs 随机 **~74%**、vs 两个手写强 bot 直接对打 **~49%**（≈ 每个强 bot 的 1.9 倍），**显著反超手写规则天花板（强 bot 69.2%）**。
- **与情书结论相反**：情书里「强 bot 对手池」是决定性因素；但 Coup 里**强 bot 对手池反而拖后腿**（64.2% < 74%）。纯自对弈 + 冻结快照对「说谎」游戏更有效——手写贝叶斯 bot 是窄而可被利用的对手。

## 天花板校准（bot vs 随机，400 局）

| 对手 | 胜率 vs 2 随机 |
|---|---|
| 随机 | 33.0% |
| 弱 heuristic bot | 61.0% |
| **强 Bayesian bot** | **69.2%** |

## 模型评测（直接计数）

| 模型（500 迭代 GRU） | vs 2 随机 | vs 2 强 bot |
|---|---|---|
| league（随机 + **强 bot** + 快照） | 64.2% | 35.2%（强 bot 各 ~32%） |
| **league（随机 + 快照，无强 bot）** | **72.2% ~ 76.0%** | **48.0% ~ 50.5%**（强 bot 各 ~25%） |

> 两 seed（7 / 42）复测，无强 bot 版 vs 随机 ~74%、vs 强 bot ~49%，结果稳健。

## 解读

1. **可训练性确认**：Coup 的零和「掉血=翻牌」结构 + 金币管理给了价值函数足够信号，纯 league 自对弈不再像情书那样卡在随机（价值函数 EV≈0）。
2. **强 bot 对手池有害**：手写强 bot 的「诚实反制 + 贝叶斯质疑」策略是固定的、可被模型专门针对的；而冻结快照（模型自己的历史版本）让策略多样性更丰富，说谎学得更好。
3. **vs 强 bot 反超**：最佳模型 vs 两个强 bot 直接对打 ~49%（每局三选一，均值 25%，模型拿 49%），即模型 ≈ 每个强 bot 的近 2 倍。

## 产物

- 检查点：`data/coup/checkpoints/coup_3p_league_nostrong_v1/checkpoint_iter_500.pth`（**最佳**）
- 对照：`data/coup/checkpoints/coup_3p_league_strong_v1/checkpoint_iter_500.pth`
- 配置：`configs/coup/coup_ppo.yaml`、`coup_ppo_nostrong.yaml`
- 引擎 + bot：`games/coup/`（game / encoder / heuristic / strong_bot）
- 训练脚本：`scripts/train_league.py`（已支持 coup + `--no-strong`）

## 后续可选

- NFSP（`train_nfsp.py` 已支持 coup）作为说谎博弈的进阶算法再比一轮。
- 把最佳模型接入 Web 人机对战前端。
