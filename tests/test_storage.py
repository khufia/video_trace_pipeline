from pathlib import Path

from video_trace_pipeline.common import sanitize_for_persistence
from video_trace_pipeline.schemas import MachineProfile, TaskSpec
from video_trace_pipeline.storage import WorkspaceManager


def test_workspace_creates_unique_run_dirs(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path))
    workspace = WorkspaceManager(profile)
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="What is shown?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    run_a = workspace.create_run(task)
    run_b = workspace.create_run(task)
    assert run_a.run_id != run_b.run_id
    assert run_a.run_dir != run_b.run_dir


def test_store_file_artifact_copies_small_file(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    source = tmp_path / "frame.png"
    source.write_bytes(b"fake image bytes")
    artifact = workspace.store_file_artifact(str(source), kind="frame", source_tool="frame_retriever")
    assert artifact.relpath is not None
    copied = workspace.workspace_root / artifact.relpath
    assert copied.exists()


def test_sanitize_for_persistence_removes_absolute_paths():
    payload = {
        "frame": {"source_path": "/tmp/frame.png", "relpath": "cache/artifacts/frame.png"},
        "video_path": "/tmp/video.mp4",
        "note": "keep me",
    }
    cleaned = sanitize_for_persistence(payload)
    assert "video_path" not in cleaned
    assert "source_path" not in cleaned["frame"]
    assert cleaned["note"] == "keep me"
