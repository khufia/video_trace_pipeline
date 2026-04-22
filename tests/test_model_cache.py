from pathlib import Path

from video_trace_pipeline.model_cache import describe_model_resolution, resolve_model_snapshot


def _write_snapshot(repo_dir: Path, snapshot_id: str = "abc123") -> Path:
    snapshot_path = repo_dir / "snapshots" / snapshot_id
    snapshot_path.mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(snapshot_id, encoding="utf-8")
    return snapshot_path


def test_resolve_model_snapshot_from_hf_cache(tmp_path):
    cache_root = tmp_path / "huggingface"
    repo_dir = cache_root / "hub" / "models--Qwen--Qwen3-VL-Embedding-8B"
    snapshot_path = _write_snapshot(repo_dir)
    resolved = resolve_model_snapshot("Qwen/Qwen3-VL-Embedding-8B", hf_cache=str(cache_root))
    assert resolved == snapshot_path.resolve()


def test_describe_model_resolution_marks_missing(tmp_path):
    report = describe_model_resolution("Penguin-VL-8B", hf_cache=str(tmp_path / "hf"))
    assert report["status"] == "missing"
    assert report["resolved_path"] is None


def test_resolve_model_snapshot_supports_qwen35_vl_alias(tmp_path):
    cache_root = tmp_path / "huggingface"
    repo_dir = cache_root / "models--Qwen--Qwen3.5-9B"
    snapshot_path = _write_snapshot(repo_dir, snapshot_id="qwen35snapshot")

    resolved = resolve_model_snapshot("Qwen/Qwen3.5-VL-9B-Instruct", hf_cache=str(cache_root))

    assert resolved == snapshot_path.resolve()
