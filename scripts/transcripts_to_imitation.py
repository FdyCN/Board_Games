#!/usr/bin/env python3
"""
把对局记录（transcripts JSONL）转成行为克隆数据集。

只抽取「真人决策步」的 (obs, action_idx)，存成 .npz，供后续监督式微调策略头。

用法:
    python scripts/transcripts_to_imitation.py \
        --transcripts data/coup/transcripts/*.jsonl \
        --out data/coup/transcripts/imitation.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_human_pairs(transcript_files: list[Path]) -> tuple[list[list[float]], list[int]]:
    """从多个 transcript 文件抽取真人决策步 (obs, action_idx)。"""
    obs_list: list[list[float]] = []
    actions: list[int] = []
    n_human = 0
    n_total = 0
    for fp in transcript_files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                for step in ep.get("steps", []):
                    n_total += 1
                    if step.get("is_human"):
                        obs_list.append(step["obs"])
                        actions.append(step["action_idx"])
                        n_human += 1
    print(f"总决策步 {n_total}，真人决策步 {n_human}")
    return obs_list, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", nargs="+", required=True, help="transcript JSONL 文件（可多个/通配）")
    ap.add_argument("--out", required=True, help="输出 .npz 路径")
    args = ap.parse_args()

    files = [Path(p) for p in args.transcripts]
    for fp in files:
        if not fp.exists():
            print(f"找不到文件: {fp}")
            sys.exit(1)

    obs_list, actions = load_human_pairs(files)
    if not obs_list:
        print("没有真人决策步，未生成数据集")
        sys.exit(1)

    obs = np.array(obs_list, dtype=np.float32)
    acts = np.array(actions, dtype=np.int64)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=obs, actions=acts)
    print(f"已保存 {obs.shape[0]} 条样本到 {out}（obs={obs.shape[1]} 维）")


if __name__ == "__main__":
    main()
