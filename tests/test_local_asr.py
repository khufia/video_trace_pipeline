import sys
import types
import wave

import torch

from video_trace_pipeline.schemas import ASRRequest, ClipRef
from video_trace_pipeline.tools import local_asr


class UnsafeCheckpointPayload(object):
    pass


def test_transcribe_with_whisperx_loads_trusted_checkpoint_with_weights_only_disabled(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "unsafe-checkpoint.pt"
    torch.save({"payload": UnsafeCheckpointPayload()}, checkpoint_path)

    class FakeModel(object):
        def transcribe(self, audio_path, batch_size=8, **kwargs):
            assert audio_path == "/tmp/audio.wav"
            assert batch_size == 8
            assert kwargs == {"language": "en"}
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

    fake_whisperx = types.ModuleType("whisperx")

    def _load_model(model_name, device, compute_type="int8", **kwargs):
        assert model_name == "large-v3"
        assert device == "cpu"
        assert "device_index" not in kwargs
        assert compute_type == "int8"
        loaded = torch.load(str(checkpoint_path))
        assert isinstance(loaded["payload"], UnsafeCheckpointPayload)
        return FakeModel()

    fake_whisperx.load_model = _load_model
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    result, runtime_warning = local_asr._transcribe_with_whisperx(
        "/tmp/audio.wav",
        model_name="large-v3",
        device_label="cpu",
        language="en",
    )

    assert runtime_warning is None
    assert result["segments"][0]["text"] == "hello"


def test_transcribe_with_whisperx_overrides_explicit_none_weights_only(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "unsafe-checkpoint.pt"
    torch.save({"payload": UnsafeCheckpointPayload()}, checkpoint_path)

    class FakeModel(object):
        def transcribe(self, audio_path, batch_size=8, **kwargs):
            assert audio_path == "/tmp/audio.wav"
            assert batch_size == 8
            assert kwargs == {"language": "en"}
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

    fake_whisperx = types.ModuleType("whisperx")

    def _load_model(model_name, device, compute_type="int8", **kwargs):
        assert model_name == "large-v3"
        assert device == "cpu"
        assert "device_index" not in kwargs
        assert compute_type == "int8"
        loaded = torch.load(str(checkpoint_path), weights_only=None)
        assert isinstance(loaded["payload"], UnsafeCheckpointPayload)
        return FakeModel()

    fake_whisperx.load_model = _load_model
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    result, runtime_warning = local_asr._transcribe_with_whisperx(
        "/tmp/audio.wav",
        model_name="large-v3",
        device_label="cpu",
        language="en",
    )

    assert runtime_warning is None
    assert result["segments"][0]["text"] == "hello"


def test_transcribe_with_whisperx_falls_back_to_cpu_when_gpu_runtime_is_missing(monkeypatch):
    class FakeModel(object):
        def transcribe(self, audio_path, batch_size=8, **kwargs):
            assert audio_path == "/tmp/audio.wav"
            assert batch_size == 8
            assert kwargs == {"language": "en"}
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

    fake_whisperx = types.ModuleType("whisperx")

    def _load_model(model_name, device, compute_type="int8", **kwargs):
        assert model_name == "large-v3"
        assert device == "cpu"
        assert compute_type == "int8"
        assert "device_index" not in kwargs
        return FakeModel()

    fake_whisperx.load_model = _load_model
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setattr(
        local_asr,
        "_resolve_whisperx_runtime",
        lambda device_label: ("cpu", "WhisperX GPU runtime unavailable on cuda:2"),
    )

    result, runtime_warning = local_asr._transcribe_with_whisperx(
        "/tmp/audio.wav",
        model_name="large-v3",
        device_label="cuda:2",
        language="en",
    )

    assert runtime_warning == "WhisperX GPU runtime unavailable on cuda:2"
    assert result["segments"][0]["text"] == "hello"


def test_local_asr_failure_summary_does_not_expose_loader_error_text(monkeypatch):
    adapter = local_asr.LocalASRAdapter(name="asr", extra={})
    request = ASRRequest(
        tool_name="asr",
        clip=ClipRef(video_id="video-1", start_s=0.0, end_s=2.0),
        speaker_attribution=False,
    )
    context = types.SimpleNamespace(
        task=types.SimpleNamespace(video_path="/tmp/video.mp4"),
        workspace=types.SimpleNamespace(profile=types.SimpleNamespace(ffmpeg_bin=None, gpu_assignments={})),
    )

    monkeypatch.setattr(local_asr, "normalize_clip_bounds", lambda video_path, start_s, end_s: (start_s, end_s))
    monkeypatch.setattr(local_asr, "extract_audio_clip", lambda *args, **kwargs: "/tmp/audio.wav")
    monkeypatch.setattr(local_asr, "cleanup_temp_path", lambda path: None)
    monkeypatch.setattr(
        local_asr,
        "_transcribe_with_whisperx",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Weights only load failed for local checkpoint")),
    )

    result = adapter.execute(request, context)

    assert result.ok is False
    assert result.summary == "ASR unavailable."
    assert "Weights only load failed" in result.data["error"]
    assert "Weights only load failed" in result.metadata["error"]


def test_transcribe_with_whisperx_returns_empty_segments_on_index_error(monkeypatch):
    calls = []

    class FakeModel(object):
        def transcribe(self, audio_path, batch_size=8, **kwargs):
            calls.append({"audio_path": audio_path, "batch_size": batch_size, "kwargs": dict(kwargs)})
            raise IndexError("list index out of range")

    fake_whisperx = types.ModuleType("whisperx")
    fake_whisperx.load_model = lambda *args, **kwargs: FakeModel()
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    result, runtime_warning = local_asr._transcribe_with_whisperx(
        "/tmp/audio.wav",
        model_name="large-v3",
        device_label="cpu",
        language=None,
    )

    assert runtime_warning is None
    assert result == {"segments": [], "language": None}
    assert calls == [{"audio_path": "/tmp/audio.wav", "batch_size": 8, "kwargs": {}}]


def test_transcribe_with_whisperx_returns_empty_segments_for_empty_wav(tmp_path, monkeypatch):
    calls = []
    audio_path = tmp_path / "empty.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"")

    class FakeModel(object):
        def transcribe(self, audio_path, batch_size=8, **kwargs):
            calls.append({"audio_path": audio_path, "batch_size": batch_size, "kwargs": dict(kwargs)})
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

    fake_whisperx = types.ModuleType("whisperx")
    fake_whisperx.load_model = lambda *args, **kwargs: FakeModel()
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    result, runtime_warning = local_asr._transcribe_with_whisperx(
        str(audio_path),
        model_name="large-v3",
        device_label="cpu",
        language=None,
    )

    assert runtime_warning is None
    assert result == {"segments": [], "language": None}
    assert calls == []
