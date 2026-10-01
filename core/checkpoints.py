"""
Checkpoint 加载与校验工具。

背景：`latest.pth` 是一个「脆弱的指针」——任何往同一个 checkpoint 目录写入的
运行都会覆盖它。历史上 `data/splendor/checkpoints/mlp_medium_3p_v1/latest.pth`
就被一次只跑 1 个迭代的快速运行覆盖，导致把一个「几乎没训练」的模型当成最佳
模型来评测（元数据 `metadata.iteration == 1`，只收集了 4 局）。

本模块提供的 `load_model_state_dict` 在加载前校验 `metadata.iteration`，
默认要求传入 `min_iteration`（期望的训练轮数下限）。此外还提供读取元数据的
辅助函数，方便评测脚本在加载 `latest.pth` 前先人工确认。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def read_metadata(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """读取 checkpoint 的 metadata（不存在则返回空字典），不加载权重。"""
    ck = torch.load(str(path), map_location=device, weights_only=False)
    meta = ck.get("metadata")
    return dict(meta) if meta else {}


def load_model_state_dict(
    path: str | Path,
    device: str = "cpu",
    min_iteration: int | None = None,
) -> dict[str, Any]:
    """
    加载 checkpoint 的 `model_state_dict`，并在加载前校验训练迭代数。

    Args:
        path: checkpoint 路径。
        device: 加载设备。
        min_iteration: 期望的 `metadata.iteration` 下限。若 checkpoint 的迭代数
            小于该值，说明它很可能是被「短跑」覆盖的 `latest.pth`（或未训练完），
            直接抛错，避免把垃圾模型当成品。

    Returns:
        模型的 state_dict。

    Raises:
        ValueError: 缺 `metadata.iteration` 或 `iteration < min_iteration`。
    """
    ck = torch.load(str(path), map_location=device, weights_only=False)
    if min_iteration is not None:
        meta = ck.get("metadata") or {}
        it = meta.get("iteration")
        if it is None:
            raise ValueError(
                f"checkpoint {path} 缺少 metadata.iteration，无法校验是否训练完成；"
                "若确认无误，可显式传入 min_iteration=None 跳过校验"
            )
        if it < min_iteration:
            raise ValueError(
                f"checkpoint {path} 的 iteration={it} 低于期望下限 {min_iteration}，"
                "很可能是被短跑覆盖的 latest.pth；请改用同目录下带明确迭代号的 "
                "checkpoint_iter_*.pth"
            )
    if "model_state_dict" not in ck:
        raise ValueError(f"checkpoint {path} 缺少 model_state_dict 字段")
    return ck["model_state_dict"]


__all__ = ["read_metadata", "load_model_state_dict"]
