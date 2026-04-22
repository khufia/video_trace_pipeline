from __future__ import annotations

import os
from typing import Optional


def _env_visible_cuda_count() -> int:
    raw = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw:
        return 0
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return 0
    if len(values) == 1 and values[0] in {"-1", "none", "None", "NoDevFiles"}:
        return 0
    return len(values)


def available_cuda_device_count() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return _env_visible_cuda_count()


def resolve_device_label(device_label: Optional[str], default: str = "cpu") -> str:
    label = str(device_label or "").strip()
    if not label:
        return default
    if not label.startswith("cuda"):
        return label

    device_count = max(0, int(available_cuda_device_count()))
    if device_count <= 0:
        return default

    if ":" not in label:
        return "cuda:0"
    try:
        requested_index = int(label.split(":", 1)[1])
    except Exception:
        requested_index = 0
    resolved_index = requested_index % device_count
    return "cuda:%d" % resolved_index
