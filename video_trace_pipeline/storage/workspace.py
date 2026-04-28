from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from filelock import FileLock

from ..common import (
    ensure_dir,
    fingerprint_file,
    guess_media_type,
    hash_payload,
    make_run_id,
    relative_to_root,
    sanitize_path_component,
    write_json,
)
from ..schemas import ArtifactRef, MachineProfile, TaskSpec


class RunContext(object):
    def __init__(self, workspace_root: Path, video_id: str, run_id: str):
        self.workspace_root = workspace_root
        self.video_id = video_id
        self.run_id = run_id
        self.run_dir = ensure_dir(workspace_root / "runs" / sanitize_path_component(video_id) / run_id)
        self.evidence_dir = ensure_dir(self.run_dir / "evidence")
        self.debug_dir = ensure_dir(self.run_dir / "debug")
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.snapshot_path = self.run_dir / "runtime_snapshot.json"
        self.trace_package_path = self.run_dir / "trace_package.json"
        self.benchmark_export_path = self.run_dir / "benchmark_export.json"
        self.final_result_path = self.run_dir / "final_result.json"

    def round_dir(self, round_index: int) -> Path:
        return ensure_dir(self.run_dir / ("round_%02d" % int(round_index)))

    def tool_step_dir(self, step_id: int, tool_name: str, round_index: Optional[int] = None) -> Path:
        effective_round = 1 if round_index is None else int(round_index)
        return ensure_dir(
            self.round_dir(effective_round) / "tools" / ("%02d_%s" % (int(step_id), sanitize_path_component(tool_name)))
        )


class WorkspaceManager(object):
    def __init__(self, profile: MachineProfile):
        self.profile = profile
        self.repo_root = Path(__file__).resolve().parents[2]
        self.workspace_root = ensure_dir(Path(profile.workspace_root).expanduser().resolve())
        self.package_root = Path(__file__).resolve().parents[1]
        cache_root_value = str(profile.cache_root or "").strip()
        self.cache_root = ensure_dir(
            Path(cache_root_value).expanduser().resolve() if cache_root_value else (self.workspace_root / "cache")
        )
        self.preprocess_root = ensure_dir(self.workspace_root / "preprocess")
        self.evidence_cache_root = ensure_dir(self.workspace_root / "evidence_cache")
        self.artifacts_root = ensure_dir(self.workspace_root / "artifacts")
        self.runs_root = ensure_dir(self.workspace_root / "runs")

    def create_run(self, task: TaskSpec) -> RunContext:
        run_id = make_run_id()
        resolved_video_id = sanitize_path_component(str(task.video_id or task.sample_key or "video"))
        return RunContext(self.workspace_root, resolved_video_id, run_id)

    def video_fingerprint(self, video_path: str) -> str:
        return fingerprint_file(video_path)

    def preprocess_dir(
        self,
        video_fingerprint_value: str,
        model_id: str,
        clip_duration_s: float,
        prompt_version: str,
        settings_signature: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> Path:
        del video_fingerprint_value, model_id, clip_duration_s, prompt_version, settings_signature
        return ensure_dir(self.preprocess_root / sanitize_path_component(video_id or "video"))

    def evidence_cache_dir(self, tool_name: str, request_hash: str) -> Path:
        return ensure_dir(
            self.evidence_cache_root / sanitize_path_component(tool_name) / sanitize_path_component(request_hash)
        )

    def artifact_path_for_file(self, source_path: str, *, kind: str, video_id: Optional[str] = None) -> Path:
        source = Path(source_path)
        video_dir = self.artifacts_root / sanitize_path_component(video_id or "unknown_video")
        subdir_name = {
            "frame": "frames",
            "clip": "clips",
            "region": "regions",
        }.get(str(kind or "").strip(), "misc")
        return ensure_dir(video_dir / subdir_name)

    def logical_clip_artifact(self, video_id: str, start_s: float, end_s: float, source_tool: Optional[str] = None) -> ArtifactRef:
        safe_video_id = sanitize_path_component(video_id or "video")
        artifact_id = "clip_%06.2f_%06.2f" % (float(start_s or 0.0), float(end_s or 0.0))
        artifact_id = artifact_id.replace(".", "_")
        relpath = self.relative_path(
            ensure_dir(self.artifacts_root / safe_video_id / "clips") / ("%s.mp4" % artifact_id)
        )
        return ArtifactRef(
            artifact_id=artifact_id,
            kind="clip",
            relpath=relpath,
            media_type="video",
            source_tool=source_tool,
            metadata={
                "video_id": video_id,
                "start_s": float(start_s or 0.0),
                "end_s": float(end_s or 0.0),
                "copied": False,
                "logical_artifact": True,
            },
        )

    def store_file_artifact(
        self,
        source_path: str,
        kind: str,
        source_tool: Optional[str] = None,
        copy_file: Optional[bool] = None,
        video_id: Optional[str] = None,
    ) -> ArtifactRef:
        source = Path(source_path)
        artifact_dir = self.artifact_path_for_file(str(source), kind=kind, video_id=video_id)
        media_type = guess_media_type(str(source))
        should_copy = copy_file
        if should_copy is None:
            should_copy = source.is_file() and source.stat().st_size <= 50 * 1024 * 1024 and media_type != "video"
        artifact_id = source.stem or sanitize_path_component(kind or "artifact")
        stored_relpath = None
        if should_copy and source.is_file():
            dest = artifact_dir / sanitize_path_component(source.name)
            lock = FileLock(str(artifact_dir / ".lock"))
            with lock:
                if not dest.exists():
                    shutil.copy2(str(source), str(dest))
            stored_relpath = self.relative_path(dest)
        metadata = {
            "source_name": source.name,
            "copied": bool(stored_relpath),
            "media_type": media_type,
        }
        if source.is_file():
            metadata["size_bytes"] = source.stat().st_size
        return ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            relpath=stored_relpath,
            media_type=media_type,
            source_tool=source_tool,
            metadata=metadata,
        )

    def write_run_manifest(self, run: RunContext, payload: Dict[str, Any]) -> None:
        write_json(run.manifest_path, payload)

    def relative_path(self, path: Path) -> str:
        resolved = path.resolve()
        for root in (self.workspace_root, self.cache_root.parent, self.repo_root, self.package_root):
            try:
                return relative_to_root(resolved, root.resolve())
            except Exception:
                continue
        return str(resolved)
