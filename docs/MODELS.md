# 预训练模型（Git LFS）

本项目的最佳模型（`checkpoint_iter_*.pth`）放在仓库的 `weights/` 目录下，用 **Git LFS** 跟踪分发——模型跟代码一起版本化，clone 仓库即得。`weights/MANIFEST.json` 记录每个模型的 sha256 供校验。

> 需要本地装 `git-lfs`（`brew install git-lfs && git lfs install`），否则 clone 下来的 `.pth` 是指针文件而非真模型。

## 模型清单（9 个，共约 76MB）

| 游戏 | 人数 | 文件 | 编码器 | 胜率（直接计数） |
|---|---|---|---|---|
| Splendor | 2 | `weights/splendor_2p_mlp_medium_v1.pth` | mlp medium | vs 随机 97.0% |
| Splendor | 3 | `weights/splendor_3p_mlp_medium_v1.pth` | mlp medium | vs 随机 98.5%，vs 开源 AlphaZero 打平 |
| Splendor | 4 | `weights/splendor_4p_mlp_medium_v1.pth` | mlp medium | vs 随机 96.5% |
| 情书 | 2 | `weights/love_letter_2p_gru_v1.pth` | gru medium | vs 随机 89.2%，vs 强 bot 53.7% |
| 情书 | 3 | `weights/love_letter_3p_gru_v1.pth` | gru medium | vs 随机 83.0%，vs 强 bot 55% |
| 情书 | 4 | `weights/love_letter_4p_gru_v1.pth` | gru medium | vs 随机 70.8%，vs 强 bot 35.3% |
| 政变疑云 | 2 | `weights/coup_2p_gru_v1.pth` | gru medium | vs 随机 84.5%，vs 强 bot 63.5% |
| 政变疑云 | 3 | `weights/coup_3p_gru_v1.pth` | gru medium | vs 随机 77.5%，vs 强 bot 36% |
| 政变疑云 | 4 | `weights/coup_4p_gru_v1.pth` | gru medium | vs 随机 73.2%，vs 强 bot 48% |

> 胜率口径：`scripts/direct_head_to_head.py` 直接计数（随机座位）。「vs 强 bot」是模型与手写强贝叶斯规则 bot 同场对打。完整 sha256 见 `weights/MANIFEST.json`。

## 获取模型

**方式一：clone 仓库（含 LFS 对象）**

```bash
git lfs install                 # 首次使用需装并启用 LFS
git clone git@github.com:FdyCN/Board_Games.git
cd Board_Games
git lfs pull                    # 拉取 weights/*.pth 真实内容
```

**方式二：raw URL 下载单个文件**（无需 clone，GitHub 会自动重定向 LFS 对象）

```
https://github.com/FdyCN/Board_Games/raw/main/weights/splendor_3p_mlp_medium_v1.pth
```

下载后校验完整性：

```bash
shasum -a 256 splendor_3p_mlp_medium_v1.pth   # 对照 weights/MANIFEST.json
```

## 加载方式

每个模型都有 `metadata.iteration`，加载前可用 `core.checkpoints.load_model_state_dict(min_iteration=...)` 校验（防止拿到被短跑覆盖的 `latest.pth` 这种垃圾文件）。

### Splendor（mlp medium，obs 384，action 46）

```python
from games.registry import create_game
from models.model_factory import create_model
from core.checkpoints import load_model_state_dict

game = create_game("splendor", num_players=3)
model = create_model(obs_dim=384, action_size=46, encoder_type="mlp", config="medium")
model.load_state_dict(load_model_state_dict("weights/splendor_3p_mlp_medium_v1.pth", min_iteration=300))
model.eval()
```

### 情书（gru medium）

```python
from games.registry import create_game
from models.model_factory import create_model
from core.checkpoints import load_model_state_dict

game = create_game("love_letter", num_players=3)   # 2/3/4 按需改
model = create_model(
    obs_dim=game.observation_shape[0],
    action_size=game.action_space_size,
    encoder_type="gru",
    config="medium",
    aux_dim=game.auxiliary_shape[0],
    encoder_params=game.encoder_params,
)
model.load_state_dict(load_model_state_dict("weights/love_letter_3p_gru_v1.pth", min_iteration=450))
model.eval()
```

> 情书的 `obs_dim`/`action_size` 随人数变化（2p=275/19，3p=304/30，4p=333/41），务必用对应人数的模型与 `num_players`。

### 政变疑云（gru medium + 显式信念编码）

```python
from games.registry import create_game
from models.model_factory import create_model
from core.checkpoints import load_model_state_dict

game = create_game("coup", num_players=3)   # 2/3/4 按需改
model = create_model(
    obs_dim=game.observation_shape[0],
    action_size=game.action_space_size,
    encoder_type="gru",
    config="medium",
    aux_dim=game.auxiliary_shape[0],
    encoder_params=game.encoder_params,
)
model.load_state_dict(load_model_state_dict("weights/coup_3p_gru_v1.pth", min_iteration=450))
model.eval()
```

> 政变疑云的 `obs_dim`/`action_size` 随人数变化（2p=1071/22，3p=1145/25，4p=1219/28）。编码器含显式信念状态（每对手持有各角色的概率）。

完整「加载 + 对战」示例见 `scripts/direct_head_to_head.py`（`neural:<ckpt>:<enc>` spec）。

## 如何更新模型（维护流程）

1. 训练出新模型后，把它填进 `scripts/export_models.py` 的 `MODELS` 清单（改 `source` / `name` / `benchmark`）。
2. 跑 `python scripts/export_models.py` —— 逐个校验（迭代数守卫 + 实际加载验证）并导出到 `weights/`，重新生成 `weights/MANIFEST.json`。
3. `git add weights/ && git commit && git push`（`*.pth` 自动走 LFS，只把最佳模型留在工作区）。
4. 更新本文档的清单表。

> 原则：**只导出 `checkpoint_iter_*.pth`（带迭代号），绝不导出 `latest.pth`**（它是易被覆盖的指针，本仓库已多次被短跑覆盖成 iter=1/2 的垃圾）。
