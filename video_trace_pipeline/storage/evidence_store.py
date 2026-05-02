from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from ..common import append_jsonl, read_jsonl
from ..schemas import AtomicObservation, EvidenceEntry, ToolResult
from .workspace import RunContext


class EvidenceLedger(object):
    def __init__(self, run: RunContext):
        self.run = run
        self.index_path = run.evidence_dir / "evidence_index.jsonl"
        self.observations_path = run.evidence_dir / "atomic_observations.jsonl"
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
                    status TEXT NOT NULL DEFAULT 'candidate',
                    time_start_s REAL,
                    time_end_s REAL,
                    frame_ts_s REAL,
                    time_intervals_json TEXT NOT NULL DEFAULT '[]',
                    artifact_refs_json TEXT NOT NULL DEFAULT '[]',
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
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(evidence_entries)").fetchall()
            }
            column_defaults = {
                "status": "TEXT NOT NULL DEFAULT 'candidate'",
                "time_start_s": "REAL",
                "time_end_s": "REAL",
                "frame_ts_s": "REAL",
                "time_intervals_json": "TEXT NOT NULL DEFAULT '[]'",
                "artifact_refs_json": "TEXT NOT NULL DEFAULT '[]'",
            }
            for column_name, ddl in column_defaults.items():
                if column_name not in columns:
                    connection.execute("ALTER TABLE evidence_entries ADD COLUMN %s %s" % (column_name, ddl))
            connection.commit()

    def _dedup_entry_payloads(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: Dict[str, Dict[str, Any]] = {}
        ordered_ids: List[str] = []
        for item in list(entries or []):
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("evidence_id") or "").strip()
            if not evidence_id:
                continue
            if evidence_id not in deduped:
                ordered_ids.append(evidence_id)
            deduped[evidence_id] = dict(item)
        return [deduped[evidence_id] for evidence_id in ordered_ids]

    def append(
        self,
        entry: EvidenceEntry,
        observations: List[AtomicObservation],
    ) -> None:
        append_jsonl(self.index_path, [entry.dict()])
        append_jsonl(
            self.observations_path,
            [
                {
                    **item.dict(),
                    "evidence_id": entry.evidence_id,
                }
                for item in observations
            ],
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO evidence_entries (
                    evidence_id, tool_name, evidence_text, inference_hint, confidence,
                    status, time_start_s, time_end_s, frame_ts_s, time_intervals_json,
                    artifact_refs_json, observation_ids_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.evidence_id,
                    entry.tool_name,
                    entry.evidence_text,
                    entry.inference_hint,
                    entry.confidence,
                    entry.status,
                    entry.time_start_s,
                    entry.time_end_s,
                    entry.frame_ts_s,
                    json.dumps([item.dict() if hasattr(item, "dict") else item for item in list(entry.time_intervals or [])], ensure_ascii=False),
                    json.dumps([item.dict() if hasattr(item, "dict") else item for item in list(entry.artifact_refs or [])], ensure_ascii=False),
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
        if not self.sqlite_path.exists():
            return self._dedup_entry_payloads(read_jsonl(self.index_path))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT evidence_id, tool_name, evidence_text, inference_hint, confidence,
                       status, time_start_s, time_end_s, frame_ts_s, time_intervals_json,
                       artifact_refs_json, observation_ids_json, metadata_json
                FROM evidence_entries
                ORDER BY rowid
                """
            ).fetchall()
        entries = []
        for row in rows:
            item = dict(row)
            item["time_intervals"] = json.loads(item.pop("time_intervals_json") or "[]")
            item["artifact_refs"] = json.loads(item.pop("artifact_refs_json") or "[]")
            item["observation_ids"] = json.loads(item.pop("observation_ids_json") or "[]")
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            entries.append(item)
        return self._dedup_entry_payloads(entries)

    def update_entry_statuses(self, status_by_id: Dict[str, str]) -> None:
        normalized_updates = {
            str(evidence_id).strip(): str(status).strip().lower()
            for evidence_id, status in dict(status_by_id or {}).items()
            if str(evidence_id).strip() and str(status).strip()
        }
        if not normalized_updates:
            return
        current_entries = self.entries()
        updated_entries = []
        changed_entries = []
        for entry in current_entries:
            evidence_id = str(entry.get("evidence_id") or "").strip()
            if evidence_id not in normalized_updates:
                updated_entries.append(entry)
                continue
            updated = dict(entry)
            updated["status"] = normalized_updates[evidence_id]
            updated_entries.append(updated)
            changed_entries.append(updated)
        if changed_entries:
            append_jsonl(self.index_path, changed_entries)
            with self._connect() as connection:
                for evidence_id, status in normalized_updates.items():
                    connection.execute(
                        "UPDATE evidence_entries SET status = ? WHERE evidence_id = ?",
                        (status, evidence_id),
                    )
                connection.commit()

    def observations(self) -> List[Dict[str, Any]]:
        if not self.sqlite_path.exists():
            return read_jsonl(self.observations_path)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT observation_id, evidence_id, subject, subject_type, predicate, object_text,
                       object_type, numeric_value, unit, time_start_s, time_end_s, frame_ts_s,
                       bbox_json, speaker_id, confidence, source_tool, direct_or_derived,
                       atomic_text, source_artifact_refs_json, metadata_json
                FROM atomic_observations
                ORDER BY rowid
                """
            ).fetchall()
        parsed = []
        for row in rows:
            item = dict(row)
            item["bbox"] = json.loads(item.pop("bbox_json") or "null")
            item["source_artifact_refs"] = json.loads(item.pop("source_artifact_refs_json") or "[]")
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            parsed.append(item)
        return parsed

    def lookup_records(self, ids: Iterable[str]) -> List[Dict[str, Any]]:
        requested_ids = [str(item).strip() for item in list(ids or []) if str(item).strip()]
        if not requested_ids:
            return []

        entries_by_id = {
            str(item.get("evidence_id") or "").strip(): dict(item)
            for item in self.entries()
            if str(item.get("evidence_id") or "").strip()
        }
        observations = [dict(item) for item in self.observations()]
        observations_by_id = {
            str(item.get("observation_id") or "").strip(): item
            for item in observations
            if str(item.get("observation_id") or "").strip()
        }
        observations_by_evidence_id = defaultdict(list)
        for item in observations:
            evidence_id = str(item.get("evidence_id") or "").strip()
            if evidence_id:
                observations_by_evidence_id[evidence_id].append(item)

        resolved: List[Dict[str, Any]] = []
        seen = set()

        def _append_record(record: Dict[str, Any]) -> None:
            observation_id = str(record.get("observation_id") or "").strip()
            evidence_id = str(record.get("evidence_id") or "").strip()
            text = str(
                record.get("atomic_text")
                or record.get("evidence_text")
                or record.get("text")
                or ""
            ).strip()
            key = (
                observation_id or None,
                evidence_id or None,
                text,
            )
            if key in seen:
                return
            seen.add(key)
            resolved.append(record)

        for requested_id in requested_ids:
            if requested_id in entries_by_id:
                entry = dict(entries_by_id[requested_id])
                _append_record(
                    {
                        "record_id": requested_id,
                        "evidence_id": requested_id,
                        "tool_name": entry.get("tool_name"),
                        "text": entry.get("evidence_text"),
                        "evidence_text": entry.get("evidence_text"),
                        "atomic_text": entry.get("evidence_text"),
                        "time_start_s": entry.get("time_start_s"),
                        "time_end_s": entry.get("time_end_s"),
                        "frame_ts_s": entry.get("frame_ts_s"),
                        "confidence": entry.get("confidence"),
                        "status": entry.get("status"),
                        "observation_ids": list(entry.get("observation_ids") or []),
                        "artifact_refs": list(entry.get("artifact_refs") or []),
                        "source_artifact_refs": list(entry.get("artifact_refs") or []),
                        "metadata": dict(entry.get("metadata") or {}),
                    }
                )
                for item in observations_by_evidence_id.get(requested_id, []):
                    _append_record(dict(item))
                continue
            if requested_id in observations_by_id:
                _append_record(dict(observations_by_id[requested_id]))

        return resolved

    def retrieve(
        self,
        query_terms: Optional[List[str]] = None,
        limit: int = 50,
        subject: Optional[str] = None,
        source_tool: Optional[str] = None,
        evidence_status: Optional[str] = None,
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
            clauses.append("LOWER(atomic_observations.subject) LIKE ?")
            params.append("%%%s%%" % str(subject).lower())
        if source_tool:
            clauses.append("atomic_observations.source_tool = ?")
            params.append(str(source_tool))
        if evidence_status:
            clauses.append("LOWER(evidence_entries.status) = ?")
            params.append(str(evidence_status).lower())
        if time_start_s is not None:
            clauses.append("(atomic_observations.time_end_s IS NULL OR atomic_observations.time_end_s >= ?)")
            params.append(float(time_start_s))
        if time_end_s is not None:
            clauses.append("(atomic_observations.time_start_s IS NULL OR atomic_observations.time_start_s <= ?)")
            params.append(float(time_end_s))

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT atomic_observations.observation_id, atomic_observations.subject, atomic_observations.subject_type,
                       atomic_observations.predicate, atomic_observations.object_text, atomic_observations.object_type,
                       atomic_observations.numeric_value, atomic_observations.unit, atomic_observations.time_start_s,
                       atomic_observations.time_end_s, atomic_observations.frame_ts_s, atomic_observations.bbox_json,
                       atomic_observations.speaker_id, atomic_observations.confidence, atomic_observations.source_tool,
                       atomic_observations.direct_or_derived, atomic_observations.atomic_text,
                       atomic_observations.source_artifact_refs_json, atomic_observations.metadata_json,
                       atomic_observations.evidence_id,
                       evidence_entries.status AS evidence_status
                FROM atomic_observations
                LEFT JOIN evidence_entries ON evidence_entries.evidence_id = atomic_observations.evidence_id
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
