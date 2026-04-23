from pathlib import Path

from video_trace_pipeline.model_cache import (
    describe_model_resolution,
    download_model_snapshot,
    ensure_hf_cache_symlink,
    resolve_model_snapshot,
)


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


def test_ensure_hf_cache_symlink_links_root_repo_to_hub_repo(tmp_path):
    cache_root = tmp_path / "huggingface"
    hub_repo_dir = cache_root / "hub" / "models--Loie--SpotSound"
    snapshot_path = _write_snapshot(hub_repo_dir, snapshot_id="spotsound-snapshot")

    report = ensure_hf_cache_symlink("Loie/SpotSound", str(cache_root))

    root_repo_dir = cache_root / "models--Loie--SpotSound"
    assert report["status"] == "ok"
    assert root_repo_dir.is_symlink()
    assert root_repo_dir.resolve() == hub_repo_dir.resolve()
    assert resolve_model_snapshot("Loie/SpotSound", hf_cache=str(cache_root)) == snapshot_path.resolve()


def test_download_model_snapshot_uses_cache_root_and_creates_root_symlink(tmp_path, monkeypatch):
    cache_root = tmp_path / "huggingface"
    hub_repo_dir = cache_root / "hub" / "models--Loie--SpotSound"
    captured = {}

    def _fake_snapshot_download(*, repo_id, cache_dir, revision=None, allow_patterns=None, ignore_patterns=None):
        captured["repo_id"] = repo_id
        captured["cache_dir"] = cache_dir
        captured["revision"] = revision
        captured["allow_patterns"] = allow_patterns
        captured["ignore_patterns"] = ignore_patterns
        snapshot_path = _write_snapshot(hub_repo_dir, snapshot_id="downloaded")
        return str(snapshot_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot_download)

    resolved = download_model_snapshot(
        "Loie/SpotSound",
        hf_cache=str(cache_root),
        allow_patterns=["*.json"],
    )

    assert captured["repo_id"] == "Loie/SpotSound"
    assert captured["cache_dir"] == str(cache_root)
    assert captured["allow_patterns"] == ["*.json"]
    assert resolved == (hub_repo_dir / "snapshots" / "downloaded").resolve()
    assert (cache_root / "models--Loie--SpotSound").is_symlink()
