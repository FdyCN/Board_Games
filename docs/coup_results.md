# 政变疑云（Coup）训练结果

> 引擎设计见 [coup_design.md](./coup_design.md)。评测口径：`scripts/direct_head_to_head.py` 直接计数（随机座位，400 局/seed）。

## 结论速览

- **Coup 可以训练**：三种算法训练的模型都显著强于随机（63%~76% vs 33%），且都打赢或打平手写强 bot。
- **最佳「剥削型」模型 = league 无强 bot**：vs 随机 ~74%、vs 两个强 bot 直接对打 ~49%（≈ 强 bot 的 2 倍）。
- **NFSP 是更「均衡」的对手**：虽然 vs 随机只有 63%，但**与 league 模型直接对打打平**（46.5% vs 45.2%）。这是「剥削 vs 纳什均衡」的经典权衡——league 更会虐菜，NFSP 更不可被利用。

## 天花板校准（bot vs 随机，400 局）

| 对手 | 胜率 vs 2 随机 |
|---|---|
| 随机 | 33.0% |
| 弱 heuristic bot | 61.0% |
| 强 Bayesian bot | 69.2% |

## 模型评测（直接计数）

| 模型 | vs 2 随机 | vs 2 强 bot | 与 league 无强bot 直接对打 |
|---|---|---|---|
| league（随机+**强bot**+快照） | 64.2% | 35.2% | — |
| **league（随机+快照，无强bot）** | **72~76%** | **48~50%** | — |
| NFSP（200 迭代） | 63.2% | 34.2% | 46.5% vs 45.2%（**打平**） |

> league 无强bot 用两 seed（7/42）复测 vs 随机 ~74%、vs 强 bot ~49%，结果稳健。

## 关键解读

1. **可训练性确认**：Coup 的零和「掉血=翻牌」结构 + 金币管理给了价值函数足够信号，纯 league 自对弈不再像情书那样卡在随机。
2. **强 bot 对手池有害**（与情书相反）：league 无强 bot（74%）> league 有强 bot（64.2%）。手写贝叶斯 bot 是固定、可被专门针对的窄对手；冻结快照自对弈在说谎游戏里演化出更丰富策略。
3. **NFSP 不占优但更均衡**：NFSP 逼近纳什，vs 弱对手胜率更低（63%），但 league 模型无法直接剥削它（打平）。说明「vs 随机胜率」**确实不是衡量说谎游戏 AI 的充分指标**——真正的强度要看 vs 均衡对手的对打。

## 关于「76% 是否好量化」

用户质疑「76% vs 随机」无法量化模型好坏——**这个质疑是对的**。现在有了 NFSP 这个 principled 基线后，结论更清楚：

- league 模型（74%）在「虐菜」维度明确强于 NFSP（63%）和手写强 bot（69%）。
- 但 league 与 NFSP 直接对打只有 45% vs 46%（打平），说明 league 的相对优势主要来自「更会打弱对手」，而非「更强」。
- **要真正量化到纳什距离，需要接外部求解器**（如 [Stanford POMDP 求解](https://web.stanford.edu/class/aa228/reports/2018/final81.pdf) 或 [OYASHI777/coup](https://github.com/OYASHI777/coup)），这是后续可选的严谨化方向。

## 产物

- 最佳（剥削型）：`data/coup/checkpoints/coup_3p_league_nostrong_v1/checkpoint_iter_500.pth`
- 均衡型：`data/coup/checkpoints/coup_3p_nfsp_v1/nfsp_iter_200.pth`
- 对照：`data/coup/checkpoints/coup_3p_league_strong_v1/checkpoint_iter_500.pth`
- 配置：`configs/coup/coup_ppo.yaml`、`coup_ppo_nostrong.yaml`、`coup_nfsp.yaml`
- 引擎 + bot：`games/coup/`（game / encoder / heuristic / strong_bot）
- 训练脚本：`scripts/train_league.py`（支持 coup + `--no-strong`）、`scripts/train_nfsp.py`

## 修复记录

训练中暴露并修复了两个引擎边界 bug（随机 rollout 撞不到、NFSP 的长对局才触发）：
1. 换牌「1 张手牌 + 空牌堆」→ 退化为无操作（避免空合法动作）。
2. 换牌把 `-1` 占位牌归还牌堆 → 污染牌堆 → 后续换牌空合法动作崩溃（过滤哨兵牌）。
