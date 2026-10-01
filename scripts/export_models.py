#!/usr/bin/env python3
"""
把「最佳模型」导出到 `weights/` 目录，并生成 MANIFEST.json（含 sha256 + 元数据 + 胜率）。

`weights/*.pth` 由 Git LFS 跟踪（见 .gitattributes），随仓库一起分发；MANIFEST.json 是
普通文本，记录每个模型的 sha256 供下载后校验。

设计要点：
- 只导出 `checkpoint_iter_*.pth`（带明确迭代号），绝不导出 `latest.pth`（易被短跑覆盖）。
- 每个模型都用 `core.checkpoints.load_model_state_dict(min_iteration=...)` 校验迭代数，
  再实际 `create_model()` + `load_state_dict()` 验证能加载，杜绝垃圾/架构不匹配的模型。

用法:
    python scripts/export_models.py --outdir weights
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from games.registry import create_game
from models.model_factory import create_model
from core.checkpoints import load_model_state_dict

# ===== 最佳模型清单（人工维护：换更好的模型时改这里）=====
# source: 相对项目根的 checkpoint 路径（用 checkpoint_iter_*.pth，不用 latest.pth）
# name:   发布文件名（干净、带版本号）
MODELS = [
    {
        "game": "splendor", "players": 2, "encoder": "mlp", "config": "medium",
        "source": "data/splendor/checkpoints/mlp_medium_2p_v1/checkpoint_iter_500.pth",
        "name": "splendor_2p_mlp_medium_v1.pth", "min_iteration": 450,
        "benchmark": {"vs_random": "97.0%", "random_baseline": "50%"},
    },
    {
        "game": "splendor", "players": 3, "encoder": "mlp", "config": "medium",
        "source": "data/splendor/checkpoints/mlp_medium_3p_v1/checkpoint_iter_300.pth",
        "name": "splendor_3p_mlp_medium_v1.pth", "min_iteration": 300,
        "benchmark": {"vs_random": "98.5%", "random_baseline": "33%", "vs_open_source_alphazero": "49% vs 51% (打平)"},
    },
    {
        "game": "splendor", "players": 4, "encoder": "mlp", "config": "medium",
        "source": "data/splendor/checkpoints/mlp_medium_4p_v1/checkpoint_iter_500.pth",
        "name": "splendor_4p_mlp_medium_v1.pth", "min_iteration": 450,
        "benchmark": {"vs_random": "96.5%", "random_baseline": "25%"},
    },
    {
        "game": "love_letter", "players": 2, "encoder": "gru", "config": "medium",
        "source": "data/love_letter/checkpoints/love_letter_2p_league_strong_v1/checkpoint_iter_500.pth",
        "name": "love_letter_2p_gru_v1.pth", "min_iteration": 450,
        "benchmark": {"vs_random": "89.2%", "random_baseline": "50%",
                      "vs_strong_bot": "53.7% vs 46.3%", "strong_bot_vs_random": "94.0%"},
    },
    {
        "game": "love_letter", "players": 3, "encoder": "gru", "config": "medium",
        "source": "data/love_letter/checkpoints/love_letter_3p_league_strong_v1/checkpoint_iter_500.pth",
        "name": "love_letter_3p_gru_v1.pth", "min_iteration": 450,
        "benchmark": {"vs_random": "83.0%", "random_baseline": "33%",
                      "vs_strong_bot": "55% vs 41%", "strong_bot_vs_random": "84.5%"},
    },
    {
        "game": "love_letter", "players": 4, "encoder": "gru", "config": "medium",
        "source": "data/love_letter/checkpoints/love_letter_4p_league_strong_v1/checkpoint_iter_500.pth",
        "name": "love_letter_4p_gru_v1.pth", "min_iteration": 450,
        "benchmark": {"vs_random": "70.8%", "random_baseline": "25%",
                      "vs_strong_bot": "35.3% vs ~21% (各 bot)", "strong_bot_vs_random": "59.3%"},
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_and_export(model: dict, outdir: Path, device: str) -> dict:
    """校验并导出一个模型，返回填充好 sha256/size/形状 的记录。"""
    src = project_root / model["source"]
    if not src.exists():
        raise FileNotFoundError(f"找不到 {src}")

    # 1. 迭代数守卫（防止被短跑覆盖的 latest.pth 混进来）
    state_dict = load_model_state_dict(src, device=device, min_iteration=model["min_iteration"])

    # 2. 实际建模型 + 加载权重，验证架构匹配
    game = create_game(model["game"], num_players=model["players"])
    m = create_model(
        obs_dim=game.observation_shape[0],
        action_size=game.action_space_size,
        encoder_type=model["encoder"],
        config=model["config"],
        aux_dim=game.auxiliary_shape[0] if game.auxiliary_shape else None,
        encoder_params=game.encoder_params,
    )
    m.load_state_dict(state_dict)
    m.eval()

    # 3. 拷贝到发布目录
    dst = outdir / model["name"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    record = dict(model)
    record["sha256"] = sha256(dst)
    record["size_bytes"] = dst.stat().st_size
    record["obs_dim"] = game.observation_shape[0]
    record["action_size"] = game.action_space_size
    print(f"  ✓ {model['name']}  ({model['game']} {model['players']}p, "
          f"obs={record['obs_dim']}, action={record['action_size']}, "
          f"{record['size_bytes']//1024}KB, iter>= {model['min_iteration']})")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="weights")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    outdir = project_root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {"models": [], "generated_by": "scripts/export_models.py"}
    for model in MODELS:
        try:
            manifest["models"].append(verify_and_export(model, outdir, args.device))
        except Exception as e:
            print(f"  ✗ {model['name']}: {type(e).__name__}: {e}")
            sys.exit(1)

    manifest_path = outdir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已导出 {len(manifest['models'])} 个模型到 {outdir}/")
    print(f"清单: {manifest_path}")


if __name__ == "__main__":
    main()
