from types import SimpleNamespace

import pytest

from video_trace_pipeline.tools import media


def test_extract_audio_clip_uses_duration_window(monkeypatch):
    recorded = {}

    def _fake_run(cmd, **kwargs):
        recorded["cmd"] = list(cmd)
        recorded["kwargs"] = dict(kwargs)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(media.subprocess, "run", _fake_run)
    monkeypatch.setattr(media.tempfile, "mkdtemp", lambda prefix="": "/tmp/vtp_audio_test")

    audio_path = media.extract_audio_clip("/tmp/video.mp4", "ffmpeg", 70.0, 74.0)

    assert audio_path == "/tmp/vtp_audio_test/clip.wav"
    assert "-to" not in recorded["cmd"]
    assert recorded["cmd"][recorded["cmd"].index("-ss") + 1] == "70.000"
    assert recorded["cmd"][recorded["cmd"].index("-t") + 1] == "4.000"


def test_extract_audio_clip_rejects_non_positive_duration():
    with pytest.raises(ValueError, match="audio clip duration must be positive"):
        media.extract_audio_clip("/tmp/video.mp4", "ffmpeg", 70.0, 70.0)
