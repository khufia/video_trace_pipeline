import sys
import types

import pytest

from video_trace_pipeline import runtime_devices


def test_resolve_device_label_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(runtime_devices, "available_cuda_device_count", lambda: 0)
    assert runtime_devices.resolve_device_label("cuda:3") == "cpu"


def test_resolve_device_label_uses_visible_device_namespace(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    assert runtime_devices.resolve_device_label("cuda:1") == "cuda:1"


def test_resolve_device_label_rejects_out_of_range_visible_index(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES=1,3"):
        runtime_devices.resolve_device_label("cuda:2")


def test_available_cuda_device_count_prefers_explicit_visible_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert runtime_devices.available_cuda_device_count() == 4


def test_describe_device_mapping_uses_visible_device_hint(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    mapping = runtime_devices.describe_device_mapping("cuda:1")

    assert mapping["resolved_label"] == "cuda:1"
    assert mapping["local_index"] == 1
    assert mapping["physical_index_hint"] == "3"
    assert mapping["mapping_source"] == "CUDA_VISIBLE_DEVICES"
