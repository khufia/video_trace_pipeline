from __future__ import annotations

import os
from typing import List, Optional


def _env_visible_cuda_devices() -> Optional[List[str]]:
    raw = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return []
    if len(values) == 1 and values[0] in {"-1", "none", "None", "NoDevFiles"}:
        return []
    return values


def _env_visible_cuda_count() -> int:
    values = _env_visible_cuda_devices()
    if values is None:
        return 0
    return len(values)


def _requested_cuda_index(label: str) -> int:
    if ":" not in label:
        return 0
    try:
        return int(label.split(":", 1)[1])
    except Exception:
        return 0


def _validate_cuda_index(label: str, requested_index: int, device_count: int, *, source: str) -> str:
    if requested_index < 0 or requested_index >= device_count:
        raise ValueError(
            "Requested CUDA device %r is outside the visible %s range [0, %d]."
            % (label, source, max(0, device_count - 1))
        )
    return "cuda:%d" % requested_index


def available_cuda_device_count() -> int:
    explicit_visible = _env_visible_cuda_count()
    if explicit_visible > 0:
        return explicit_visible
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return explicit_visible


def resolve_device_label(device_label: Optional[str], default: str = "cpu") -> str:
    label = str(device_label or "").strip()
    if not label:
        return default
    if not label.startswith("cuda"):
        return label

    visible_devices = _env_visible_cuda_devices()
    if visible_devices is not None:
        device_count = len(visible_devices)
        if device_count <= 0:
            return default
        requested_index = _requested_cuda_index(label)
        return _validate_cuda_index(
            label,
            requested_index,
            device_count,
            source="CUDA_VISIBLE_DEVICES=%s" % ",".join(visible_devices),
        )

    device_count = max(0, int(available_cuda_device_count()))
    if device_count <= 0:
        return default

    requested_index = _requested_cuda_index(label)
    return _validate_cuda_index(label, requested_index, device_count, source="CUDA devices")


def describe_device_mapping(device_label: Optional[str], default: str = "cpu") -> dict:
    resolved = resolve_device_label(device_label, default=default)
    mapping = {
        "requested_label": str(device_label or "").strip() or default,
        "resolved_label": resolved,
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip() or None,
        "cuda_device_order": str(os.environ.get("CUDA_DEVICE_ORDER") or "").strip() or None,
        "local_index": None,
        "physical_index_hint": None,
        "mapping_source": None,
    }
    if not resolved.startswith("cuda"):
        return mapping

    local_index = _requested_cuda_index(resolved)
    mapping["local_index"] = int(local_index)

    visible_devices = _env_visible_cuda_devices()
    if visible_devices is not None:
        mapping["mapping_source"] = "CUDA_VISIBLE_DEVICES"
        if 0 <= local_index < len(visible_devices):
            mapping["physical_index_hint"] = str(visible_devices[local_index])
        return mapping

    mapping["mapping_source"] = "CUDA runtime"
    return mapping
