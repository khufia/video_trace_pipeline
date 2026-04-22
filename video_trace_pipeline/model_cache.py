from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


_MODEL_ALIASES = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "small": "Systran/faster-whisper-small",
}


def normalize_model_reference(model_name: Optional[str]) -> str:
    raw = str(model_name or "").strip()
    return _MODEL_ALIASES.get(raw, raw)


def hf_cache_roots(explicit_hf_cache: Optional[str] = None) -> List[Path]:
    roots: List[Path] = []

    def _add(path_value: Optional[str]) -> None:
        text = str(path_value or "").strip()
        if not text:
            return
        path = Path(text).expanduser()
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
