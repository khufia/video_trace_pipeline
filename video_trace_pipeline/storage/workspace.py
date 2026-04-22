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
    read_json,
    relative_to_root,
    sanitize_path_component,
    short_hash,
    write_json,
)
from ..schemas import ArtifactRef, MachineProfile, TaskSpec


class RunContext(object):
    def __init__(self, workspace_root: Path, benchmark: str, sample_key: str, run_id: str):
        self.workspace_root = workspace_root
        self.benchmark = benchmark
        self.sample_key = sample_key
        self.run_id = run_id
        self.run_dir = ensure_dir(
            workspace_root / "runs" / sanitize_path_component(benchmark) / sanitize_path_component(sample_key) / run_id
        )
        self.planner_dir = ensure_dir(self.run_dir / "planner")
        self.synthesizer_dir = ensure_dir(self.run_dir / "synthesizer")
        self.auditor_dir = ensure_dir(self.run_dir / "auditor")
        self.tools_dir = ensure_dir(self.run_dir / "tools")
        self.evidence_dir = ensure_dir(self.run_dir / "evidence")
        self.trace_dir = ensure_dir(self.run_dir / "trace")
        self.results_dir = ensure_dir(self.run_dir / "results")
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.snapshot_path = self.run_dir / "runtime_snapshot.yaml"

    def tool_step_dir(self, step_id: int, tool_name: str) -> Path:
        return ensure_dir(self.tools_dir / ("%02d_%s" % (int(step_id), sanitize_path_component(tool_name))))


class WorkspaceManager(object):
    def __init__(self, profile: MachineProfile):
        self.profile = profile
        self.repo_root = Path(__file__).resolve().parents[2]
        self.workspace_root = ensure_dir(Path(profile.workspace_root).expanduser().resolve())
        self.package_root = Path(__file__).resolve().parents[1]
        self.package_results_root = ensure_dir(self.workspace_root / "results")
        cache_root_value = str(profile.cache_root or "").strip()
        self.cache_root = ensure_dir(
            Path(cache_root_value).expanduser().resolve() if cache_root_value else (self.workspace_root / "cache")
        )
        self.preprocess_root = ensure_dir(self.cache_root / "preprocess")
        self.evidence_cache_root = ensure_dir(self.cache_root / "evidence")
        self.artifacts_root = ensure_dir(self.cache_root / "artifacts")
        self.runs_root = ensure_dir(self.workspace_root / "runs")

    def create_run(self, task: TaskSpec) -> RunContext:
        run_id = make_run_id()
        return RunContext(self.workspace_root, task.benchmark, task.sample_key, run_id)

    def video_fingerprint(self, video_path: str) -> str:
        return fingerprint_file(video_path)

    def preprocess_dir(self, video_fingerprint_value: str, model_id: str, clip_duration_s: float, prompt_version: str) -> Path:
        return ensure_dir(
            self.preprocess_root
            / video_fingerprint_value
            / "dense_caption"
            / sanitize_path_component(model_id)
            / sanitize_path_component(str(int(clip_duration_s)))
            / sanitize_path_component(prompt_version)
        )

    def evidence_cache_dir(self, tool_name: str, request_hash: str) -> Path:
        return ensure_dir(
            self.evidence_cache_root / sanitize_path_component(tool_name) / sanitize_path_component(request_hash)
        )

    def artifact_path_for_file(self, source_path: str) -> Path:
        source = Path(source_path)
        fingerprint = hash_payload(
            {
                "name": source.name,
                "size": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns,
            }
        )
        return ensure_dir(self.artifacts_root / fingerprint)

    def store_file_artifact(
        self,
        source_path: str,
        kind: str,
        source_tool: Optional[str] = None,
        copy_file: Optional[bool] = None,
    ) -> ArtifactRef:
        source = Path(source_path)
        artifact_dir = self.artifact_path_for_file(str(source))
        media_type = guess_media_type(str(source))
        should_copy = copy_file
        if should_copy is None:
            should_copy = source.is_file() and source.stat().st_size <= 50 * 1024 * 1024 and media_type != "video"
        artifact_id = artifact_dir.name
        stored_relpath = None
        if should_copy and source.is_file():
            dest = artifact_dir / source.name
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

    def export_run_target(self, run: RunContext, task: TaskSpec, results_name: Optional[str]) -> Optional[Path]:
        export_name = sanitize_path_component(str(results_name or "").strip())
        if not export_name:
            return None
        video_id = sanitize_path_component(str(task.video_id or task.sample_key or "video"))
        return self.package_results_root / export_name / sanitize_path_component(run.run_id) / video_id

    def export_run_bundle(self, run: RunContext, task: TaskSpec, results_name: Optional[str]) -> Optional[Path]:
        target = self.export_run_target(run, task, results_name)
        if target is None:
            return None
        ensure_dir(target.parent)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(run.run_dir, target)
        return target
