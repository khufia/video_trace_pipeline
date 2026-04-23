from __future__ import annotations

import json
import sqlite3
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

    def lock(self, tool_name: str, request_hash: str) -> FileLock:
        cache_dir = self.workspace.evidence_cache_dir(tool_name, request_hash)
        return FileLock(str(cache_dir / ".lock"))

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

    def store_unlocked(
        self,
        tool_name: str,
        request_hash: str,
        manifest: Dict[str, Any],
        result: Dict[str, Any],
        observations: Iterable[Dict[str, Any]],
        summary_markdown: str,
    ) -> Path:
        cache_dir = self.workspace.evidence_cache_dir(tool_name, request_hash)
        write_json(cache_dir / "manifest.json", manifest)
        write_json(cache_dir / "result.json", result)
        write_text(cache_dir / "summary.md", summary_markdown)
        obs_path = cache_dir / "observations.jsonl"
        obs_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
        append_jsonl(obs_path, observations)
        return cache_dir

    def store(
        self,
        tool_name: str,
        request_hash: str,
        manifest: Dict[str, Any],
        result: Dict[str, Any],
        observations: Iterable[Dict[str, Any]],
        summary_markdown: str,
    ) -> Path:
        lock = self.lock(tool_name, request_hash)
        with lock:
            return self.store_unlocked(
                tool_name=tool_name,
                request_hash=request_hash,
                manifest=manifest,
                result=result,
                observations=observations,
                summary_markdown=summary_markdown,
            )


class EvidenceLedger(object):
    def __init__(self, run: RunContext):
        self.run = run
        self.index_path = run.evidence_dir / "evidence_index.jsonl"
        self.observations_path = run.evidence_dir / "atomic_observations.jsonl"
        self.readable_path = run.evidence_dir / "evidence_readable.md"
        self.sqlite_path = run.evidence_dir / "evidence.sqlite3"
        self._init_sqlite()

    def _connect(self):
        connection = sqlite3.connect(str(self.sqlite_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _init_sqlite(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence_entries (
                    evidence_id TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    evidence_text TEXT NOT NULL,
                    inference_hint TEXT,
                    confidence REAL,
                    observation_ids_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS atomic_observations (
                    observation_id TEXT PRIMARY KEY,
                    evidence_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_text TEXT,
                    object_type TEXT,
                    numeric_value REAL,
                    unit TEXT,
                    time_start_s REAL,
                    time_end_s REAL,
                    frame_ts_s REAL,
                    bbox_json TEXT,
                    speaker_id TEXT,
                    confidence REAL,
                    source_tool TEXT NOT NULL,
                    direct_or_derived TEXT,
                    atomic_text TEXT NOT NULL,
                    source_artifact_refs_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(evidence_id) REFERENCES evidence_entries(evidence_id)
                );

                CREATE INDEX IF NOT EXISTS idx_atomic_subject ON atomic_observations(subject);
                CREATE INDEX IF NOT EXISTS idx_atomic_predicate ON atomic_observations(predicate);
                CREATE INDEX IF NOT EXISTS idx_atomic_source_tool ON atomic_observations(source_tool);
                CREATE INDEX IF NOT EXISTS idx_atomic_time_start ON atomic_observations(time_start_s);
                CREATE INDEX IF NOT EXISTS idx_atomic_frame_ts ON atomic_observations(frame_ts_s);
                """
            )
            connection.commit()

    def append(
        self,
        entry: EvidenceEntry,
        observations: List[AtomicObservation],
    ) -> None:
        append_jsonl(self.index_path, [entry.dict()])
        append_jsonl(self.observations_path, [item.dict() for item in observations])
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO evidence_entries (
                    evidence_id, tool_name, evidence_text, inference_hint, confidence,
                    observation_ids_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.evidence_id,
                    entry.tool_name,
                    entry.evidence_text,
                    entry.inference_hint,
                    entry.confidence,
                    json.dumps(list(entry.observation_ids or []), ensure_ascii=False),
                    json.dumps(dict(entry.metadata or {}), ensure_ascii=False),
                ),
            )
            for item in observations:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO atomic_observations (
                        observation_id, evidence_id, subject, subject_type, predicate, object_text,
                        object_type, numeric_value, unit, time_start_s, time_end_s, frame_ts_s,
                        bbox_json, speaker_id, confidence, source_tool, direct_or_derived,
                        atomic_text, source_artifact_refs_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.observation_id,
                        entry.evidence_id,
                        item.subject,
                        item.subject_type,
                        item.predicate,
                        item.object_text,
                        item.object_type,
                        item.numeric_value,
                        item.unit,
                        item.time_start_s,
                        item.time_end_s,
                        item.frame_ts_s,
                        json.dumps(item.bbox, ensure_ascii=False) if item.bbox is not None else None,
                        item.speaker_id,
                        item.confidence,
                        item.source_tool,
                        item.direct_or_derived,
                        item.atomic_text,
                        json.dumps(list(item.source_artifact_refs or []), ensure_ascii=False),
                        json.dumps(dict(item.metadata or {}), ensure_ascii=False),
                    ),
                )
            connection.commit()

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
        subject: Optional[str] = None,
        source_tool: Optional[str] = None,
        time_start_s: Optional[float] = None,
        time_end_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        lowered = [term.lower() for term in (query_terms or []) if term]
        if not self.sqlite_path.exists():
            observations = self.observations()
            if not query_terms:
                return observations[:limit]
            scored = []
            for item in observations:
                haystack = json.dumps(item, ensure_ascii=False).lower()
                score = sum(1 for term in lowered if term in haystack)
                if score:
                    scored.append((score, item))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            return [item for _, item in scored[:limit]]

        clauses = []
        params: List[Any] = []
        if subject:
            clauses.append("LOWER(subject) LIKE ?")
            params.append("%%%s%%" % str(subject).lower())
        if source_tool:
            clauses.append("source_tool = ?")
            params.append(str(source_tool))
        if time_start_s is not None:
            clauses.append("(time_end_s IS NULL OR time_end_s >= ?)")
            params.append(float(time_start_s))
        if time_end_s is not None:
            clauses.append("(time_start_s IS NULL OR time_start_s <= ?)")
            params.append(float(time_end_s))

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT observation_id, subject, subject_type, predicate, object_text, object_type,
                       numeric_value, unit, time_start_s, time_end_s, frame_ts_s, bbox_json,
                       speaker_id, confidence, source_tool, direct_or_derived, atomic_text,
                       source_artifact_refs_json, metadata_json
                FROM atomic_observations
                %s
                LIMIT 1000
                """
                % (("WHERE " + " AND ".join(clauses)) if clauses else ""),
                params,
            ).fetchall()

        parsed = []
        for row in rows:
            item = dict(row)
            item["bbox"] = json.loads(item["bbox_json"]) if item.get("bbox_json") else None
            item["source_artifact_refs"] = json.loads(item["source_artifact_refs_json"] or "[]")
            item["metadata"] = json.loads(item["metadata_json"] or "{}")
            item.pop("bbox_json", None)
            item.pop("source_artifact_refs_json", None)
            item.pop("metadata_json", None)
            parsed.append(item)

        if not lowered:
            return parsed[:limit]

        scored = []
        for item in parsed:
            haystack = json.dumps(item, ensure_ascii=False).lower()
            score = sum(1 for term in lowered if term in haystack)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]
