from video_trace_pipeline import runtime_devices


def test_resolve_device_label_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(runtime_devices, "available_cuda_device_count", lambda: 0)
    assert runtime_devices.resolve_device_label("cuda:3") == "cpu"


def test_resolve_device_label_wraps_visible_index(monkeypatch):
    monkeypatch.setattr(runtime_devices, "available_cuda_device_count", lambda: 2)
    assert runtime_devices.resolve_device_label("cuda:3") == "cuda:1"
