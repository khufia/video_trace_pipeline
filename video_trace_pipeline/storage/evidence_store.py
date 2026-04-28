from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from filelock import FileLock

from ..common import append_jsonl, ensure_dir, read_json, read_jsonl, sanitize_path_component, write_json
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
        }

    def store_unlocked(
        self,
        tool_name: str,
        request_hash: str,
        manifest: Dict[str, Any],
        result: Dict[str, Any],
        observations: Iterable[Dict[str, Any]],
    ) -> Path:
        cache_dir = self.workspace.evidence_cache_dir(tool_name, request_hash)
        write_json(cache_dir / "manifest.json", manifest)
        write_json(cache_dir / "result.json", result)
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
    ) -> Path:
        lock = self.lock(tool_name, request_hash)
        with lock:
            return self.store_unlocked(
                tool_name=tool_name,
                request_hash=request_hash,
                manifest=manifest,
                result=result,
                observations=observations,
            )


class EvidenceLedger(object):
    def __init__(self, run: RunContext):
        self.run = run
        self.index_path = run.evidence_dir / "evidence_index.jsonl"
        self.observations_path = run.evidence_dir / "atomic_observations.jsonl"
        self.sqlite_path = run.evidence_dir / "evidence.sqlite3"
        self._init_sqlite()

    def _artifact_context_path(self) -> Path:
        video_id = sanitize_path_component(self.run.video_id or "video")
        return self.run.workspace_root / "artifacts" / video_id / "artifact_context.jsonl"

    def _artifact_context_lock(self) -> FileLock:
        path = self._artifact_context_path()
        ensure_dir(path.parent)
        return FileLock(str(path.parent / ".artifact_context.lock"))

    @staticmethod
    def _artifact_ref_payload(artifact_ref: Any) -> Dict[str, Any]:
        if hasattr(artifact_ref, "model_dump"):
            return artifact_ref.model_dump()
        if hasattr(artifact_ref, "dict"):
            return artifact_ref.dict()
        return dict(artifact_ref or {}) if isinstance(artifact_ref, dict) else {}

    @staticmethod
    def _compact_text(value: Any, limit: int = 360) -> str:
        text = " ".join(str(value or "").split()).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _timestamp_from_frame_artifact(artifact_ref: Dict[str, Any]) -> Optional[float]:
        candidates = [
            str(artifact_ref.get("artifact_id") or ""),
            Path(str(artifact_ref.get("relpath") or "")).stem,
        ]
        for candidate in candidates:
            match = re.search(r"frame_([0-9]+(?:[._][0-9]+)?)", candidate)
            if not match:
                continue
            value = match.group(1).replace("_", ".")
            try:
                return float(value)
            except Exception:
                continue
        return None

    @staticmethod
    def _merge_unique_records(existing: List[Dict[str, Any]], new_items: List[Dict[str, Any]], key_fields: List[str], limit: int) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()
        for item in list(existing or []) + list(new_items or []):
            if not isinstance(item, dict):
                continue
            key = tuple(str(item.get(field) or "").strip() for field in key_fields)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    @staticmethod
    def _merge_unique_text(existing: List[str], new_items: List[str], limit: int) -> List[str]:
        merged: List[str] = []
        seen = set()
        for item in list(existing or []) + list(new_items or []):
            text = " ".join(str(item or "").split()).strip()
            if not text or text in seen:
                continue
            if text.startswith("A candidate frame was retrieved at "):
                continue
            seen.add(text)
            merged.append(text)
            if len(merged) >= limit:
                break
        return merged

    @staticmethod
    def _observation_is_contains_text(observation: Dict[str, Any]) -> bool:
        source_tool = str(observation.get("source_tool") or "").strip()
        predicate = str(observation.get("predicate") or "").strip()
        subject = str(observation.get("subject") or "").strip().lower()
        subject_type = str(observation.get("subject_type") or "").strip().lower()
        atomic_text = str(observation.get("atomic_text") or "").strip()
        atomic_lower = atomic_text.lower()
        if source_tool == "frame_retriever" and predicate == "retrieved_frame_at":
            return False
        if predicate in {"present_in_interval"}:
            return False
        if subject_type in {"prompt", "question", "task"}:
            return False
        if subject in {"the prompt", "prompt", "the user", "user"}:
            return False
        if atomic_lower.startswith(("the prompt asks", "the user wants", "the question asks")):
            return False
        return bool(atomic_text)

    @classmethod
    def _seed_artifact_context_record(cls, artifact_ref: Dict[str, Any]) -> Dict[str, Any]:
        artifact_id = str(artifact_ref.get("artifact_id") or "").strip()
        relpath = str(artifact_ref.get("relpath") or "").strip()
        metadata = dict(artifact_ref.get("metadata") or {})
        record: Dict[str, Any] = {
            "artifact_id": artifact_id,
            "artifact_type": str(artifact_ref.get("kind") or "artifact").strip() or "artifact",
        }
        if relpath:
            record["relpath"] = relpath
        time_payload: Dict[str, Any] = {}
        timestamp_s = metadata.get("timestamp_s") or metadata.get("frame_ts_s")
        if timestamp_s is None and record["artifact_type"] == "frame":
            timestamp_s = cls._timestamp_from_frame_artifact(artifact_ref)
        if timestamp_s is not None:
            time_payload["timestamp_s"] = timestamp_s
        start_s = metadata.get("start_s")
        end_s = metadata.get("end_s")
        if start_s is not None or end_s is not None:
            time_payload["source_clip" if record["artifact_type"] != "clip" else "clip"] = {
                "start_s": start_s,
                "end_s": end_s,
            }
        if time_payload:
            record["time"] = time_payload
        record["contains"] = []
        record["linked_observations"] = []
        record["linked_evidence"] = []
        return record

    @classmethod
    def _merge_artifact_time(cls, record: Dict[str, Any], observations: List[Dict[str, Any]]) -> None:
        time_payload = dict(record.get("time") or {})
        for observation in observations:
            if time_payload.get("timestamp_s") is None and observation.get("frame_ts_s") is not None:
                time_payload["timestamp_s"] = observation.get("frame_ts_s")
            if "source_clip" not in time_payload and (
                observation.get("time_start_s") is not None or observation.get("time_end_s") is not None
            ):
                time_payload["source_clip"] = {
                    "start_s": observation.get("time_start_s"),
                    "end_s": observation.get("time_end_s"),
                }
        if time_payload:
            record["time"] = time_payload

    def _update_artifact_context(self, entry: EvidenceEntry, observations: List[AtomicObservation]) -> None:
        artifact_refs = [self._artifact_ref_payload(item) for item in list(entry.artifact_refs or [])]
        artifact_refs = [item for item in artifact_refs if str(item.get("artifact_id") or "").strip()]
        if not artifact_refs:
            return

        artifact_by_id = {
            str(item.get("artifact_id") or "").strip(): item
            for item in artifact_refs
            if str(item.get("artifact_id") or "").strip()
        }
        observation_payloads = [
            {
                **(item.model_dump() if hasattr(item, "model_dump") else item.dict()),
                "source_artifact_refs": [
                    str(artifact_id).strip()
                    for artifact_id in list(getattr(item, "source_artifact_refs", []) or [])
                    if str(artifact_id).strip()
                ],
            }
            for item in list(observations or [])
        ]
        path = self._artifact_context_path()
        with self._artifact_context_lock():
            existing_records = {
                str(item.get("artifact_id") or "").strip(): dict(item)
                for item in read_jsonl(path)
                if isinstance(item, dict) and str(item.get("artifact_id") or "").strip()
            }
            for artifact_id, artifact_ref in sorted(artifact_by_id.items()):
                linked_observations = [
                    item
                    for item in observation_payloads
                    if artifact_id in list(item.get("source_artifact_refs") or [])
                ]
                if not linked_observations and len(artifact_by_id) == 1:
                    linked_observations = list(observation_payloads)

                record = existing_records.get(artifact_id) or self._seed_artifact_context_record(artifact_ref)
                seeded = self._seed_artifact_context_record(artifact_ref)
                for key in ("artifact_type", "relpath"):
                    if seeded.get(key):
                        record[key] = seeded[key]
                self._merge_artifact_time(record, linked_observations)
                contains = [
                    self._compact_text(item.get("atomic_text"))
                    for item in linked_observations
                    if self._observation_is_contains_text(item)
                ]
                record["contains"] = self._merge_unique_text(list(record.get("contains") or []), contains, limit=40)
                observation_records = [
                    {
                        "observation_id": item.get("observation_id"),
                        "source_tool": item.get("source_tool"),
                        "text": self._compact_text(item.get("atomic_text")),
                    }
                    for item in linked_observations
                    if self._observation_is_contains_text(item)
                ]
                record["linked_observations"] = self._merge_unique_records(
                    list(record.get("linked_observations") or []),
                    observation_records,
                    ["observation_id", "text"],
                    limit=80,
                )
                linked_texts = [
                    self._compact_text(item.get("atomic_text"), limit=220)
                    for item in linked_observations
                    if self._observation_is_contains_text(item)
                ][:12]
                evidence_record = {
                    "evidence_id": entry.evidence_id,
                    "tool_name": entry.tool_name,
                    "summary": self._compact_text(entry.evidence_text),
                    "observation_texts": linked_texts,
                }
                record["linked_evidence"] = self._merge_unique_records(
                    list(record.get("linked_evidence") or []),
                    [evidence_record],
                    ["evidence_id", "summary"],
                    limit=40,
                )
                existing_records[artifact_id] = record

            ensure_dir(path.parent)
            with path.open("w", encoding="utf-8") as handle:
                for artifact_id in sorted(existing_records):
                    handle.write(json.dumps(existing_records[artifact_id], ensure_ascii=False))
                    handle.write("\n")

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
                    status TEXT NOT NULL DEFAULT 'provisional',
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
                "status": "TEXT NOT NULL DEFAULT 'provisional'",
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
        self._update_artifact_context(entry, observations)

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
