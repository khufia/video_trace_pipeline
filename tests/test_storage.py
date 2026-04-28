from pathlib import Path

from video_trace_pipeline.common import sanitize_for_persistence
from video_trace_pipeline.schemas import ArtifactRef, AtomicObservation, EvidenceEntry, MachineProfile, TaskSpec
from video_trace_pipeline.storage import EvidenceLedger, WorkspaceManager


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


def test_store_file_artifact_uses_custom_cache_root(tmp_path):
    profile = MachineProfile(
        workspace_root=str(tmp_path / "workspace"),
        cache_root=str(tmp_path / "repo_cache"),
    )
    workspace = WorkspaceManager(profile)
    source = tmp_path / "frame.png"
    source.write_bytes(b"fake image bytes")
    artifact = workspace.store_file_artifact(str(source), kind="frame", source_tool="frame_retriever")
    assert artifact.relpath is not None
    copied = workspace.workspace_root / artifact.relpath
    assert copied.exists()
    assert str(copied).startswith(str(workspace.artifacts_root / "unknown_video" / "frames"))


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


def test_sanitize_for_persistence_relativizes_repo_paths():
    repo_root = Path(__file__).resolve().parents[1]
    repo_path = repo_root / "workspace" / "cache" / "preprocess" / "demo"
    payload = {
        "cache_dir": str(repo_path),
        "note": "bundle stored at %s" % repo_path,
        "relative_note": "cache/preprocess/demo",
    }
    cleaned = sanitize_for_persistence(payload)
    assert cleaned["cache_dir"] == "workspace/cache/preprocess/demo"
    assert cleaned["note"] == "bundle stored at workspace/cache/preprocess/demo"
    assert cleaned["relative_note"] == "cache/preprocess/demo"


def test_evidence_ledger_persists_sqlite_index(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="What is shown?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    (tmp_path / "video.mp4").write_bytes(b"video")
    run = workspace.create_run(task)
    ledger = EvidenceLedger(run)

    entry = EvidenceEntry(
        evidence_id="ev_01_demo",
        tool_name="asr",
        evidence_text='speaker_1 said "hello"',
        observation_ids=["obs_demo"],
    )
    observation = AtomicObservation(
        observation_id="obs_demo",
        subject="speaker_1",
        subject_type="speaker",
        predicate="said",
        object_text="hello",
        object_type="utterance",
        source_tool="asr",
        atomic_text='speaker_1 said "hello" from 1.00s to 2.00s.',
        time_start_s=1.0,
        time_end_s=2.0,
    )
    ledger.append(entry, [observation])

    assert ledger.sqlite_path.exists()
    retrieved = ledger.retrieve(query_terms=["hello"], subject="speaker_1")
    assert len(retrieved) == 1
    assert retrieved[0]["observation_id"] == "obs_demo"


def test_evidence_ledger_lookup_records_resolves_evidence_and_observation_ids(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="What is shown?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    (tmp_path / "video.mp4").write_bytes(b"video")
    run = workspace.create_run(task)
    ledger = EvidenceLedger(run)

    entry = EvidenceEntry(
        evidence_id="ev_01_demo",
        tool_name="asr",
        evidence_text='speaker_1 said "hello"',
        artifact_refs=[
            ArtifactRef(
                artifact_id="art_01",
                kind="frame",
                relpath="cache/artifacts/demo/frame.png",
                metadata={"timestamp_s": 1.5, "video_id": "sample1"},
            )
        ],
        observation_ids=["obs_demo"],
        time_start_s=1.0,
        time_end_s=2.0,
    )
    observation = AtomicObservation(
        observation_id="obs_demo",
        subject="speaker_1",
        subject_type="speaker",
        predicate="said",
        object_text="hello",
        object_type="utterance",
        source_tool="asr",
        atomic_text='speaker_1 said "hello" from 1.00s to 2.00s.',
        time_start_s=1.0,
        time_end_s=2.0,
    )
    ledger.append(entry, [observation])

    by_evidence = ledger.lookup_records(["ev_01_demo"])
    assert by_evidence[0]["evidence_id"] == "ev_01_demo"
    assert by_evidence[0]["atomic_text"] == 'speaker_1 said "hello"'
    assert by_evidence[0]["artifact_refs"][0]["relpath"] == "cache/artifacts/demo/frame.png"
    assert any(item.get("observation_id") == "obs_demo" for item in by_evidence)

    by_observation = ledger.lookup_records(["obs_demo"])
    assert len(by_observation) == 1
    assert by_observation[0]["observation_id"] == "obs_demo"
