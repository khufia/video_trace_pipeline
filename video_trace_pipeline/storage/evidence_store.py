from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from filelock import FileLock

from ..common import append_jsonl, ensure_dir, read_json, read_jsonl, write_json, write_text
from ..schemas import AtomicObservation, EvidenceEntry, ToolResult
from .workspace import RunContext, WorkspaceManager


class SharedEvidenceCache(object):
    def __init__(self, workspace: WorkspaceManager):
        self.workspace = workspace

    def load(self, tool_name: str, request_hash: str) -> Optional[Dict[str, Any]]:
        cache_dir = self.workspace.evidence_cache_dir(tool_name, request_hash)
        manifest_path = cache_dir / "manifest.json"
        result_path = cache_dir / "result.json"
        observations_path = cache_dir / "observations.jsonl"
        if not manifest_path.exists() or not result_path.exists():
            return None
        return {
            "manifest": read_json(manifest_path),
            "result": read_json(result_path),
            "observations": read_jsonl(observations_path),
            "summary_markdown": (cache_dir / "summary.md").read_text(encoding="utf-8")
            if (cache_dir / "summary.md").exists()
            else "",
        }

    def store(
        self,
        tool_name: str,
        request_hash: str,
        manifest: Dict[str, Any],
        result: Dict[str, Any],
        observations: Iterable[Dict[str, Any]],
        summary_markdown: str,
    ) -> Path:
        cache_dir = self.workspace.evidence_cache_dir(tool_name, request_hash)
        lock = FileLock(str(cache_dir / ".lock"))
        with lock:
            write_json(cache_dir / "manifest.json", manifest)
            write_json(cache_dir / "result.json", result)
            write_text(cache_dir / "summary.md", summary_markdown)
            obs_path = cache_dir / "observations.jsonl"
            obs_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
            append_jsonl(obs_path, observations)
        return cache_dir


class EvidenceLedger(object):
    def __init__(self, run: RunContext):
        self.run = run
        self.index_path = run.evidence_dir / "evidence_index.jsonl"
        self.observations_path = run.evidence_dir / "atomic_observations.jsonl"
        self.readable_path = run.evidence_dir / "evidence_readable.md"

    def append(
        self,
        entry: EvidenceEntry,
        observations: List[AtomicObservation],
    ) -> None:
        append_jsonl(self.index_path, [entry.dict()])
        append_jsonl(self.observations_path, [item.dict() for item in observations])

    def entries(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.index_path)

    def observations(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.observations_path)

    def render_readable_markdown(self) -> str:
        observations = self.observations()
        grouped_by_subject = defaultdict(list)
        grouped_by_time = defaultdict(list)
        timeless = []

        for item in observations:
            subject = str(item.get("subject") or "unknown")
            grouped_by_subject[subject].append(item)
            if item.get("time_start_s") is not None:
                key = "%.3f-%.3f" % (
                    float(item.get("time_start_s") or 0.0),
                    float(item.get("time_end_s") or item.get("time_start_s") or 0.0),
                )
                grouped_by_time[key].append(item)
            else:
                timeless.append(item)

        lines = ["# Evidence Ledger", ""]
        lines.append("## By Subject")
        lines.append("")
        for subject in sorted(grouped_by_subject):
            lines.append("### %s" % subject)
            for item in grouped_by_subject[subject]:
                lines.append("- %s" % item.get("atomic_text", ""))
            lines.append("")

        lines.append("## By Time")
        lines.append("")
        for key in sorted(grouped_by_time):
            lines.append("### %s" % key)
            for item in grouped_by_time[key]:
                lines.append("- %s" % item.get("atomic_text", ""))
            lines.append("")

        if timeless:
            lines.append("## Without Time")
            lines.append("")
            for item in timeless:
                lines.append("- %s" % item.get("atomic_text", ""))
            lines.append("")

        markdown = "\n".join(lines).rstrip() + "\n"
        write_text(self.readable_path, markdown)
        return markdown

    def retrieve(
        self,
        query_terms: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        observations = self.observations()
        if not query_terms:
            return observations[:limit]
        lowered = [term.lower() for term in query_terms if term]
        scored = []
        for item in observations:
            haystack = json.dumps(item, ensure_ascii=False).lower()
            score = sum(1 for term in lowered if term in haystack)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]
