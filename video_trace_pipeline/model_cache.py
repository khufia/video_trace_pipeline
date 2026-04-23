from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


_MODEL_ALIASES = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "small": "Systran/faster-whisper-small",
    "Qwen/Qwen3.5-VL-9B-Instruct": "Qwen/Qwen3.5-9B",
}


def normalize_model_reference(model_name: Optional[str]) -> str:
    raw = str(model_name or "").strip()
    return _MODEL_ALIASES.get(raw, raw)


def normalize_hf_cache_root(hf_cache: Optional[str]) -> Optional[Path]:
    raw = str(hf_cache or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.name == "hub":
        return path.parent
    return path


def hf_cache_roots(explicit_hf_cache: Optional[str] = None) -> List[Path]:
    roots: List[Path] = []

    def _add(path_value: Optional[str]) -> None:
        path = normalize_hf_cache_root(path_value)
        if path is None:
            return
        candidates = [path]
        if path.name != "hub":
            candidates.append(path / "hub")
        for candidate in candidates:
            normalized = str(candidate)
            if all(str(existing) != normalized for existing in roots):
                roots.append(candidate)

    _add(explicit_hf_cache)
    _add(os.environ.get("HUGGINGFACE_HUB_CACHE"))
    _add(os.environ.get("HF_HUB_CACHE"))
    _add(os.environ.get("HF_HOME"))
    return roots


def _find_snapshot_dir(repo_dir: Path) -> Optional[Path]:
    snapshots_dir = repo_dir / "snapshots"
    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        snapshot_id = ref_main.read_text(encoding="utf-8").strip()
        if snapshot_id:
            candidate = snapshots_dir / snapshot_id
            if candidate.is_dir():
                return candidate.resolve()
    if snapshots_dir.is_dir():
        snapshot_dirs = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshot_dirs:
            return snapshot_dirs[-1].resolve()
    return None


def resolve_model_snapshot(model_name: Optional[str], hf_cache: Optional[str] = None) -> Optional[Path]:
    normalized = normalize_model_reference(model_name)
    if not normalized:
        return None

    direct = Path(normalized).expanduser()
    if direct.exists():
        return direct.resolve()

    repo_suffixes: List[str] = []
    if "/" in normalized:
        repo_suffixes.append(normalized.replace("/", "--"))
    else:
        repo_suffixes.append(normalized)

    for root in hf_cache_roots(hf_cache):
        if not root.exists():
            continue
        for repo_suffix in repo_suffixes:
            candidate = root / ("models--%s" % repo_suffix)
            if candidate.is_dir():
                snapshot = _find_snapshot_dir(candidate)
                if snapshot is not None:
                    return snapshot
        if "/" not in normalized:
            for candidate in sorted(root.glob("models--*--%s" % normalized)):
                snapshot = _find_snapshot_dir(candidate)
                if snapshot is not None:
                    return snapshot
    return None


def describe_model_resolution(model_name: Optional[str], hf_cache: Optional[str] = None) -> Dict[str, Optional[str]]:
    normalized = normalize_model_reference(model_name)
    resolved = resolve_model_snapshot(normalized, hf_cache=hf_cache)
    return {
        "requested_model": str(model_name or "").strip() or None,
        "normalized_model": normalized or None,
        "resolved_path": str(resolved) if resolved is not None else None,
        "hf_cache": str(Path(hf_cache).expanduser()) if str(hf_cache or "").strip() else None,
        "status": "ok" if resolved is not None else "missing",
    }


def _repo_cache_paths(model_name: Optional[str], hf_cache: Optional[str]) -> Optional[Dict[str, Path]]:
    normalized = normalize_model_reference(model_name)
    cache_root = normalize_hf_cache_root(hf_cache)
    if not normalized or "/" not in normalized or cache_root is None:
        return None
    repo_suffix = normalized.replace("/", "--")
    return {
        "cache_root": cache_root,
        "hub_root": cache_root / "hub",
        "root_repo_dir": cache_root / ("models--%s" % repo_suffix),
        "hub_repo_dir": cache_root / "hub" / ("models--%s" % repo_suffix),
    }


def ensure_hf_cache_symlink(model_name: Optional[str], hf_cache: Optional[str]) -> Dict[str, Optional[str]]:
    layout = _repo_cache_paths(model_name, hf_cache)
    if layout is None:
        return {"status": "not_hf_repo", "created": None, "source": None, "target": None}

    root_repo_dir = layout["root_repo_dir"]
    hub_repo_dir = layout["hub_repo_dir"]
    hub_root = layout["hub_root"]

    root_exists = root_repo_dir.exists() or root_repo_dir.is_symlink()
    hub_exists = hub_repo_dir.exists() or hub_repo_dir.is_symlink()
    if root_exists and hub_exists:
        return {"status": "ok", "created": None, "source": str(root_repo_dir), "target": str(hub_repo_dir)}

    if hub_exists:
        source = hub_repo_dir
        target = root_repo_dir
    elif root_exists:
        source = root_repo_dir
        target = hub_repo_dir
    else:
        return {"status": "missing_repo_dir", "created": None, "source": None, "target": None}

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and not target.exists():
        target.unlink()
    if not target.exists():
        target.symlink_to(source)
        return {"status": "ok", "created": str(target), "source": str(source), "target": str(target)}
    return {"status": "ok", "created": None, "source": str(source), "target": str(target)}


def download_model_snapshot(
    model_name: Optional[str],
    *,
    hf_cache: Optional[str],
    revision: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
    ignore_patterns: Optional[Iterable[str]] = None,
) -> Path:
    normalized = normalize_model_reference(model_name)
    if not normalized or "/" not in normalized:
        raise ValueError("download_model_snapshot requires a Hugging Face repo id, got %r" % (model_name,))

    resolved = resolve_model_snapshot(normalized, hf_cache=hf_cache)
    if resolved is not None:
        ensure_hf_cache_symlink(normalized, hf_cache)
        return resolved

    cache_root = normalize_hf_cache_root(hf_cache)
    if cache_root is None:
        raise RuntimeError("HF cache root is not configured for %r" % normalized)
    cache_root.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_path = Path(
        snapshot_download(
            repo_id=normalized,
            cache_dir=str(cache_root),
            revision=revision,
            allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
            ignore_patterns=list(ignore_patterns) if ignore_patterns is not None else None,
        )
    ).resolve()
    ensure_hf_cache_symlink(normalized, hf_cache)
    return snapshot_path
