from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..common import read_jsonl, sanitize_path_component


_QUERY_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "and",
        "are",
        "answer",
        "appears",
        "audio",
        "because",
        "been",
        "before",
        "being",
        "between",
        "but",
        "can",
        "could",
        "does",
        "for",
        "frame",
        "frames",
        "from",
        "had",
        "has",
        "have",
        "how",
        "into",
        "its",
        "level",
        "line",
        "link",
        "message",
        "moment",
        "more",
        "needs",
        "non",
        "not",
        "now",
        "only",
        "over",
        "same",
        "should",
        "sound",
        "status",
        "that",
        "the",
        "their",
        "then",
        "there",
        "this",
        "through",
        "trace",
        "under",
        "use",
        "used",
        "using",
        "validate",
        "validated",
        "validation",
        "verdict",
        "video",
        "was",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
        "your",
        "coinciding",
        "each",
        "evidence",
        "firmer",
        "grounded",
        "grounding",
        "inspected",
        "existing",
        "queried",
        "retrieve",
        "retrieved",
        "source",
        "target",
        "textually",
    }
)


def _audit_search_text(audit_report: Any) -> str:
    if not isinstance(audit_report, dict):
        return json.dumps(audit_report, ensure_ascii=False) if audit_report is not None else ""
    pieces: List[str] = []
    for key in ("feedback", "missing_information"):
        value = audit_report.get(key)
        if isinstance(value, list):
            pieces.extend(str(item) for item in value)
        elif value:
            pieces.append(str(value))
    for finding in list(audit_report.get("findings") or []):
        if isinstance(finding, dict):
            message = str(finding.get("message") or "").strip()
            if message:
                pieces.append(message)
    diagnostics = audit_report.get("diagnostics")
    if isinstance(diagnostics, dict):
        pieces.extend(str(value) for value in diagnostics.values() if str(value).strip())
    return "\n".join(pieces)


def query_terms_from_task_and_audit(task, audit_report=None) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", "%s %s" % (task.question, " ".join(task.options)))
    if audit_report is not None:
        tokens.extend(re.findall(r"[A-Za-z0-9]+", _audit_search_text(audit_report)))
    ordered = []
    seen = set()
    for token in tokens:
        lowered = token.lower()
        if lowered in seen or lowered in _QUERY_STOPWORDS or len(lowered) < 3:
            continue
        seen.add(lowered)
        ordered.append(lowered)
    return ordered[:30]


def _score_record(record: Dict[str, Any], terms: List[str]) -> int:
    tokens = set(re.findall(r"[a-z0-9]+", json.dumps(record, ensure_ascii=False).lower()))

    def _matches(term: str) -> bool:
        if term in tokens:
            return True
        if term.endswith("s") and term[:-1] in tokens:
            return True
        if len(term) >= 6:
            prefix = term[:4]
            return any(token.startswith(prefix) for token in tokens if len(token) >= 4)
        return False

    return sum(1 for term in terms if _matches(term))


def _score_text(text: str, terms: List[str]) -> int:
    if not terms:
        return 0
    tokens = set(re.findall(r"[a-z0-9]+", str(text or "").lower()))
    score = 0
    for term in terms:
        if term in tokens:
            score += 1
            continue
        if term.endswith("s") and term[:-1] in tokens:
            score += 1
            continue
        if len(term) >= 6:
            prefix = term[:4]
            if any(token.startswith(prefix) for token in tokens if len(token) >= 4):
                score += 1
    return score


def _is_prompt_meta_text(text: str) -> bool:
    normalized = " ".join(str(text or "").split()).strip().lower()
    return normalized.startswith(("the prompt asks", "the user wants", "the question asks"))


def _top_matching_records(records: List[Dict[str, Any]], terms: List[str], *, limit: int) -> List[Dict[str, Any]]:
    if not terms:
        return [dict(item) for item in list(records or [])[:limit] if isinstance(item, dict)]
    scored = []
    for item in list(records or []):
        if not isinstance(item, dict):
            continue
        score = _score_record(item, terms)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [dict(item) for _, item in scored[:limit]]


def _top_text_items(items: List[Any], terms: List[str], *, limit: int) -> List[Any]:
    candidates = []
    for index, item in enumerate(list(items or [])):
        if isinstance(item, dict):
            text = " ".join(str(item.get(key) or "") for key in ("text", "summary", "evidence_text"))
        else:
            text = str(item or "")
        if _is_prompt_meta_text(text):
            continue
        score = _score_text(text, terms)
        candidates.append((score, index, item))
    if terms:
        positives = [item for item in candidates if item[0] > 0]
        source = positives if positives else candidates
    else:
        source = candidates
    source.sort(key=lambda pair: (-pair[0], pair[1]))
    return [item for _, _, item in source[:limit]]


def _compact_artifact_context_record(record: Dict[str, Any], terms: List[str]) -> Dict[str, Any]:
    compact = {
        key: record.get(key)
        for key in ("artifact_id", "artifact_type", "relpath", "time")
        if record.get(key) not in (None, "", [], {})
    }
    contains = _top_text_items(list(record.get("contains") or []), terms, limit=12)
    if contains:
        compact["contains"] = contains
    linked_observations = _top_text_items(list(record.get("linked_observations") or []), terms, limit=8)
    if linked_observations:
        compact["linked_observations"] = linked_observations
    linked_evidence = []
    for item in _top_text_items(list(record.get("linked_evidence") or []), terms, limit=5):
        if not isinstance(item, dict):
            continue
        evidence_item = {
            key: item.get(key)
            for key in ("evidence_id", "tool_name", "summary")
            if item.get(key) not in (None, "", [], {})
        }
        if _is_prompt_meta_text(evidence_item.get("summary", "")):
            evidence_item.pop("summary", None)
        observation_texts = _top_text_items(list(item.get("observation_texts") or []), terms, limit=4)
        observation_texts = [text for text in observation_texts if not _is_prompt_meta_text(str(text))]
        if observation_texts:
            evidence_item["observation_texts"] = observation_texts
        if evidence_item:
            linked_evidence.append(evidence_item)
    if linked_evidence:
        compact["linked_evidence"] = linked_evidence
    return compact


def _top_artifact_context_records(records: List[Dict[str, Any]], terms: List[str], *, limit: int) -> List[Dict[str, Any]]:
    return [_compact_artifact_context_record(item, terms) for item in _top_matching_records(records, terms, limit=limit)]


def _trace_claims(trace_package: Optional[Any]) -> List[Dict[str, Any]]:
    claims = []
    if trace_package is None:
        return claims
    steps = list(getattr(trace_package, "inference_steps", []) or [])
    for step in steps:
        text = str(getattr(step, "text", "") or "").strip()
        if not text:
            continue
        claims.append(
            {
                "step_id": getattr(step, "step_id", None),
                "text": text,
                "supporting_observation_ids": list(getattr(step, "supporting_observation_ids", []) or []),
            }
        )
    return claims


def _compact_text(value: Any, max_len: int = 180) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _terms_from_text(*values: Any, limit: int = 30) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        for token in re.findall(r"[A-Za-z0-9]+", str(value or "")):
            lowered = token.lower()
            if lowered in seen or lowered in _QUERY_STOPWORDS or len(lowered) < 3:
                continue
            seen.add(lowered)
            ordered.append(lowered)
            if len(ordered) >= limit:
                return ordered
    return ordered


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _request_payload(request: Any) -> Dict[str, Any]:
    if hasattr(request, "model_dump"):
        return dict(request.model_dump())
    if hasattr(request, "dict"):
        return dict(request.dict())
    return dict(request or {})


def _time_filter_from_request(request: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    time_range = request.get("time_range") if isinstance(request.get("time_range"), dict) else {}
    return _as_float(time_range.get("start_s")), _as_float(time_range.get("end_s"))


def _record_time_bounds(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    start = _as_float(record.get("time_start_s"))
    end = _as_float(record.get("time_end_s"))
    if start is None:
        start = _as_float(record.get("start_s"))
    if end is None:
        end = _as_float(record.get("end_s"))
    if start is None and end is None:
        timestamp = _as_float(record.get("frame_ts_s"))
        if timestamp is None:
            timestamp = _as_float(record.get("timestamp_s"))
        if timestamp is not None:
            start = timestamp
            end = timestamp
    if start is None and end is None and isinstance(record.get("time"), dict):
        time_payload = record.get("time") or {}
        start = _as_float(time_payload.get("start_s"))
        end = _as_float(time_payload.get("end_s"))
        timestamp = _as_float(time_payload.get("timestamp_s"))
        if start is None and end is None and timestamp is not None:
            start = timestamp
            end = timestamp
    if start is None and end is None and isinstance(record.get("clip"), dict):
        clip = record.get("clip") or {}
        start = _as_float(clip.get("start_s"))
        end = _as_float(clip.get("end_s"))
    if start is None and end is not None:
        start = end
    if end is None and start is not None:
        end = start
    return start, end


def _record_overlaps_time(record: Dict[str, Any], start_s: Optional[float], end_s: Optional[float]) -> bool:
    if start_s is None and end_s is None:
        return True
    record_start, record_end = _record_time_bounds(record)
    if record_start is None and record_end is None:
        return False
    if start_s is not None and (record_end is None or record_end < start_s):
        return False
    if end_s is not None and (record_start is None or record_start > end_s):
        return False
    return True


def _filter_by_time(records: List[Dict[str, Any]], start_s: Optional[float], end_s: Optional[float]) -> List[Dict[str, Any]]:
    return [item for item in list(records or []) if isinstance(item, dict) and _record_overlaps_time(item, start_s, end_s)]


def _filter_by_source_tools(records: List[Dict[str, Any]], source_tools: List[str]) -> List[Dict[str, Any]]:
    allowed = {str(item or "").strip() for item in list(source_tools or []) if str(item or "").strip()}
    if not allowed:
        return records
    return [
        item
        for item in list(records or [])
        if str(item.get("tool_name") or item.get("source_tool") or "").strip() in allowed
    ]


def _count_by(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in list(records or []):
        value = str(item.get(key) or "").strip() or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def _first_present_text(payload: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _compact_catalog_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    dense_caption = segment.get("dense_caption") if isinstance(segment.get("dense_caption"), dict) else {}
    asr = segment.get("asr") if isinstance(segment.get("asr"), dict) else {}
    captions = list(dense_caption.get("captions") or [])
    transcript_spans = list(asr.get("transcript_spans") or [])
    attributes: List[Any] = []
    for caption in captions[:6]:
        if isinstance(caption, dict):
            attributes.extend(list(caption.get("attributes") or []))
    compact = {
        "segment_id": segment.get("segment_id"),
        "start_s": segment.get("start_s"),
        "end_s": segment.get("end_s"),
        "overall_summary": _compact_text(dense_caption.get("overall_summary"), max_len=260),
        "caption_count": len(captions),
        "asr_span_count": len(transcript_spans),
        "attributes": [_compact_text(item, max_len=90) for item in attributes[:12] if str(item or "").strip()],
        "asr_preview": [
            {
                key: span.get(key)
                for key in ("start_s", "end_s", "text")
                if span.get(key) not in (None, "", [], {})
            }
            for span in transcript_spans[:4]
            if isinstance(span, dict)
        ],
    }
    clips = list(dense_caption.get("clips") or segment.get("clips") or [])
    if clips:
        compact["clips"] = clips[:3]
    return {
        key: value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }


def _compact_evidence_catalog_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "evidence_id": entry.get("evidence_id"),
        "tool_name": entry.get("tool_name"),
        "status": entry.get("status"),
        "time_start_s": entry.get("time_start_s"),
        "time_end_s": entry.get("time_end_s"),
        "frame_ts_s": entry.get("frame_ts_s"),
        "summary": _compact_text(entry.get("evidence_text"), max_len=220),
        "observation_ids": list(entry.get("observation_ids") or [])[:8],
    }
    return {
        key: value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }


def _compact_observation_catalog_entry(observation: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "observation_id": observation.get("observation_id"),
        "evidence_id": observation.get("evidence_id"),
        "source_tool": observation.get("source_tool"),
        "evidence_status": observation.get("evidence_status"),
        "time_start_s": observation.get("time_start_s"),
        "time_end_s": observation.get("time_end_s"),
        "frame_ts_s": observation.get("frame_ts_s"),
        "text": _compact_text(observation.get("atomic_text") or observation.get("text"), max_len=220),
    }
    return {
        key: value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }


def _compact_artifact_catalog_record(record: Dict[str, Any]) -> Dict[str, Any]:
    contains = list(record.get("contains") or [])
    linked_observations = list(record.get("linked_observations") or [])
    linked_evidence = list(record.get("linked_evidence") or [])
    compact = {
        "artifact_id": record.get("artifact_id"),
        "artifact_type": record.get("artifact_type"),
        "relpath": record.get("relpath"),
        "time": record.get("time"),
        "contains_preview": [_compact_text(item, max_len=160) for item in contains[:4] if str(item or "").strip()],
        "linked_observation_count": len(linked_observations),
        "linked_evidence_ids": [
            item.get("evidence_id")
            for item in linked_evidence[:8]
            if isinstance(item, dict) and item.get("evidence_id")
        ],
    }
    return {
        key: value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }


def _dedupe_dicts(records: List[Dict[str, Any]], key_fields: List[str], *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for record in list(records or []):
        if not isinstance(record, dict):
            continue
        signature = None
        for key in key_fields:
            value = str(record.get(key) or "").strip()
            if value:
                signature = (key, value)
                break
        if signature is None:
            signature = ("json", json.dumps(record, sort_keys=True, ensure_ascii=False, default=str))
        if signature in seen:
            continue
        seen.add(signature)
        ordered.append(dict(record))
        if limit is not None and len(ordered) >= limit:
            break
    return ordered


def merge_retrieved_contexts(*contexts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for context in contexts:
        if not isinstance(context, dict):
            continue
        for key, value in context.items():
            if value in (None, "", [], {}):
                continue
            if key == "query_terms":
                merged[key] = _dedupe_dicts(
                    [{"term": item} for item in list(merged.get(key) or []) + list(value or [])],
                    ["term"],
                )
                merged[key] = [item["term"] for item in merged[key]]
                continue
            if key == "preprocess_matches" and isinstance(value, dict):
                current = dict(merged.get(key) or {})
                for subkey, subvalue in value.items():
                    current[subkey] = _dedupe_dicts(
                        list(current.get(subkey) or []) + list(subvalue or []),
                        ["segment_id", "transcript_id", "caption_id"],
                        limit=30,
                    )
                merged[key] = current
                continue
            if isinstance(value, list):
                key_fields = {
                    "observations": ["observation_id", "evidence_id"],
                    "evidence": ["evidence_id"],
                    "artifact_context": ["artifact_id", "relpath"],
                    "retrieval_requests": ["request_id"],
                    "lookup_records": ["record_id", "observation_id", "evidence_id"],
                    "prior_trace_claims": ["step_id", "text"],
                }.get(key, ["id", "text"])
                merged[key] = _dedupe_dicts(list(merged.get(key) or []) + list(value or []), key_fields, limit=80)
                continue
            merged[key] = value
    return {
        key: value
        for key, value in merged.items()
        if value not in (None, "", [], {})
    }


class PlannerContextRetriever(object):
    """Deterministic text retriever used before planner calls.

    It does not call models or tools. The planner remains one LLM call, but its
    refine prompt receives a focused text package from canonical preprocess,
    evidence, and artifact-context stores.
    """

    def __init__(self, workspace):
        self.workspace = workspace

    def _artifact_context_path(self, video_id: str) -> Path:
        return self.workspace.artifacts_root / sanitize_path_component(video_id or "video") / "artifact_context.jsonl"

    def _artifact_context_records(self, task) -> List[Dict[str, Any]]:
        return read_jsonl(self._artifact_context_path(task.video_id or task.sample_key))

    def build_catalog(
        self,
        *,
        task,
        preprocess_bundle: Dict[str, Any],
        evidence_ledger,
        audit_report=None,
        current_trace=None,
    ) -> Dict[str, Any]:
        audit_payload = audit_report.dict() if hasattr(audit_report, "dict") else audit_report
        terms = query_terms_from_task_and_audit(task, audit_payload)
        entries = evidence_ledger.entries()
        observations = evidence_ledger.observations() if hasattr(evidence_ledger, "observations") else []
        raw_segments = list(preprocess_bundle.get("raw_segments") or preprocess_bundle.get("segments") or [])
        planner_segments = list(preprocess_bundle.get("planner_segments") or [])
        asr_transcripts = list(preprocess_bundle.get("asr_transcripts") or [])
        dense_caption_segments = list(preprocess_bundle.get("dense_caption_segments") or [])
        artifact_context = self._artifact_context_records(task)
        artifact_catalog_records = [_compact_artifact_catalog_record(item) for item in artifact_context if isinstance(item, dict)]

        claims = _trace_claims(current_trace)
        audit_missing = []
        if isinstance(audit_payload, dict):
            audit_missing = [
                str(item).strip()
                for item in list(audit_payload.get("missing_information") or [])
                if str(item).strip()
            ]

        catalog = {
            "retrieval_design": "deterministic lexical retrieval over SQLite evidence, artifact_context.jsonl, and canonical preprocess JSON",
            "query_terms": terms,
            "preprocess": {
                "planner_segment_count": len(planner_segments),
                "raw_segment_count": len(raw_segments),
                "asr_transcript_count": len(asr_transcripts),
                "dense_caption_segment_count": len(dense_caption_segments),
                "segments": [_compact_catalog_segment(item) for item in planner_segments],
            },
            "evidence_store": {
                "evidence_entry_count": len(entries),
                "observation_count": len(observations),
                "counts_by_status": _count_by(entries, "status"),
                "counts_by_tool": _count_by(entries, "tool_name"),
                "validated_evidence": [
                    _compact_evidence_catalog_entry(item)
                    for item in entries
                    if str(item.get("status") or "").strip().lower() == "validated"
                ][:30],
                "recent_evidence": [_compact_evidence_catalog_entry(item) for item in entries[-30:]],
                "recent_observations": [_compact_observation_catalog_entry(item) for item in observations[-40:]],
            },
            "artifact_context": {
                "record_count": len(artifact_context),
                "counts_by_type": _count_by(artifact_context, "artifact_type"),
                "records": artifact_catalog_records[:250],
                "omitted_record_count": max(0, len(artifact_catalog_records) - 250),
            },
            "prior_trace": {
                "claim_count": len(claims),
                "claims": claims[:40],
            },
        }
        if audit_missing:
            catalog["audit_gaps"] = audit_missing
        return {
            key: value
            for key, value in catalog.items()
            if value not in (None, "", [], {})
        }

    def retrieve_for_requests(
        self,
        *,
        task,
        preprocess_bundle: Dict[str, Any],
        evidence_ledger,
        requests: List[Any],
        audit_report=None,
        current_trace=None,
    ) -> Dict[str, Any]:
        audit_payload = audit_report.dict() if hasattr(audit_report, "dict") else audit_report
        raw_segments = list(preprocess_bundle.get("raw_segments") or preprocess_bundle.get("segments") or [])
        planner_segments = list(preprocess_bundle.get("planner_segments") or [])
        asr_transcripts = list(preprocess_bundle.get("asr_transcripts") or [])
        dense_caption_segments = list(preprocess_bundle.get("dense_caption_segments") or [])
        artifact_context = self._artifact_context_records(task)
        entries = evidence_ledger.entries()
        entries_by_id = {
            str(item.get("evidence_id") or "").strip(): dict(item)
            for item in entries
            if str(item.get("evidence_id") or "").strip()
        }
        claims = _trace_claims(current_trace)

        all_terms: List[str] = []
        collected_observations: List[Dict[str, Any]] = []
        collected_evidence: List[Dict[str, Any]] = []
        collected_artifacts: List[Dict[str, Any]] = []
        collected_lookup_records: List[Dict[str, Any]] = []
        collected_claims: List[Dict[str, Any]] = []
        preprocess_matches: Dict[str, List[Dict[str, Any]]] = {
            "planner_segments": [],
            "raw_segments": [],
            "asr_transcripts": [],
            "dense_caption_segments": [],
        }
        request_summaries: List[Dict[str, Any]] = []

        for request_item in list(requests or []):
            request = _request_payload(request_item)
            target = str(request.get("target") or "mixed").strip().lower()
            limit = int(request.get("limit") or 20)
            limit = max(1, min(50, limit))
            start_s, end_s = _time_filter_from_request(request)
            source_tools = [str(item).strip() for item in list(request.get("source_tools") or []) if str(item).strip()]
            evidence_status = str(request.get("evidence_status") or "").strip().lower()
            evidence_ids = [str(item).strip() for item in list(request.get("evidence_ids") or []) if str(item).strip()]
            observation_ids = [str(item).strip() for item in list(request.get("observation_ids") or []) if str(item).strip()]
            artifact_ids = {str(item).strip() for item in list(request.get("artifact_ids") or []) if str(item).strip()}
            terms = _terms_from_text(request.get("need"), request.get("query"), " ".join(request.get("modalities") or []))
            if not terms:
                terms = query_terms_from_task_and_audit(task, audit_payload)
            all_terms.extend(terms)
            request_summaries.append(
                {
                    key: value
                    for key, value in {
                        "request_id": request.get("request_id"),
                        "target": target,
                        "need": request.get("need"),
                        "query": request.get("query"),
                        "time_range": request.get("time_range"),
                        "source_tools": source_tools,
                        "evidence_status": evidence_status,
                        "artifact_ids": sorted(artifact_ids),
                        "evidence_ids": evidence_ids,
                        "observation_ids": observation_ids,
                        "limit": limit,
                    }.items()
                    if value not in (None, "", [], {})
                }
            )

            exact_ids = evidence_ids + observation_ids
            if exact_ids and hasattr(evidence_ledger, "lookup_records"):
                collected_lookup_records.extend(evidence_ledger.lookup_records(exact_ids))
                for evidence_id in evidence_ids:
                    entry = entries_by_id.get(evidence_id)
                    if entry:
                        collected_evidence.append(entry)

            wants_evidence = target in {"mixed", "existing_evidence", "evidence"}
            wants_observations = target in {"mixed", "existing_evidence", "observations"}
            wants_artifacts = target in {"mixed", "artifact_context"}
            wants_preprocess = target in {"mixed", "preprocess", "asr_transcripts", "dense_captions"}
            wants_trace = target in {"mixed", "prior_trace"}

            if wants_observations:
                if source_tools:
                    observation_candidates: List[Dict[str, Any]] = []
                    for source_tool in source_tools:
                        observation_candidates.extend(
                            evidence_ledger.retrieve(
                                query_terms=terms,
                                evidence_status=evidence_status or None,
                                source_tool=source_tool,
                                time_start_s=start_s,
                                time_end_s=end_s,
                                limit=limit,
                            )
                        )
                else:
                    observation_candidates = evidence_ledger.retrieve(
                        query_terms=terms,
                        evidence_status=evidence_status or None,
                        time_start_s=start_s,
                        time_end_s=end_s,
                        limit=limit,
                    )
                if observation_ids:
                    requested = set(observation_ids)
                    observation_candidates = [
                        item
                        for item in observation_candidates
                        if str(item.get("observation_id") or "").strip() in requested
                    ] or observation_candidates
                observation_candidates = _filter_by_source_tools(observation_candidates, source_tools)
                collected_observations.extend(observation_candidates[:limit])

            if wants_evidence:
                entry_candidates = list(entries)
                if evidence_ids:
                    requested = set(evidence_ids)
                    entry_candidates = [
                        item
                        for item in entry_candidates
                        if str(item.get("evidence_id") or "").strip() in requested
                    ]
                if evidence_status:
                    entry_candidates = [
                        item
                        for item in entry_candidates
                        if str(item.get("status") or "").strip().lower() == evidence_status
                    ]
                entry_candidates = _filter_by_source_tools(entry_candidates, source_tools)
                entry_candidates = _filter_by_time(entry_candidates, start_s, end_s)
                collected_evidence.extend(_top_matching_records(entry_candidates, terms, limit=limit))

            if wants_artifacts:
                artifact_candidates = [
                    item
                    for item in artifact_context
                    if isinstance(item, dict)
                    and (not artifact_ids or str(item.get("artifact_id") or "").strip() in artifact_ids)
                ]
                artifact_candidates = _filter_by_time(artifact_candidates, start_s, end_s)
                collected_artifacts.extend(_top_artifact_context_records(artifact_candidates, terms, limit=limit))

            if wants_preprocess:
                if target in {"mixed", "preprocess", "dense_captions"}:
                    planner_candidates = _filter_by_time(planner_segments, start_s, end_s)
                    preprocess_matches["planner_segments"].extend(_top_matching_records(planner_candidates, terms, limit=limit))
                    dense_candidates = _filter_by_time(dense_caption_segments, start_s, end_s)
                    preprocess_matches["dense_caption_segments"].extend(_top_matching_records(dense_candidates, terms, limit=limit))
                if target in {"mixed", "preprocess"}:
                    raw_candidates = _filter_by_time(raw_segments, start_s, end_s)
                    preprocess_matches["raw_segments"].extend(_top_matching_records(raw_candidates, terms, limit=limit))
                if target in {"mixed", "preprocess", "asr_transcripts"}:
                    asr_candidates = _filter_by_time(asr_transcripts, start_s, end_s)
                    preprocess_matches["asr_transcripts"].extend(_top_matching_records(asr_candidates, terms, limit=limit))

            if wants_trace:
                collected_claims.extend(_top_matching_records(claims, terms, limit=limit))

        payload = {
            "query_terms": _terms_from_text(" ".join(all_terms), limit=60),
            "retrieval_requests": request_summaries,
            "lookup_records": _dedupe_dicts(collected_lookup_records, ["record_id", "observation_id", "evidence_id"], limit=80),
            "observations": _dedupe_dicts(collected_observations, ["observation_id", "evidence_id"], limit=80),
            "evidence": _dedupe_dicts(collected_evidence, ["evidence_id"], limit=80),
            "artifact_context": _dedupe_dicts(collected_artifacts, ["artifact_id", "relpath"], limit=80),
            "preprocess_matches": {
                key: _dedupe_dicts(value, ["segment_id", "transcript_id", "caption_id"], limit=40)
                for key, value in preprocess_matches.items()
                if value
            },
            "prior_trace_claims": _dedupe_dicts(collected_claims, ["step_id", "text"], limit=40),
        }
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, "", [], {})
        }

    def retrieve(
        self,
        *,
        task,
        preprocess_bundle: Dict[str, Any],
        evidence_ledger,
        audit_report=None,
        current_trace=None,
        mode: str = "generate",
        limit: int = 50,
    ) -> Dict[str, Any]:
        audit_payload = audit_report.dict() if hasattr(audit_report, "dict") else audit_report
        terms = query_terms_from_task_and_audit(task, audit_payload)
        entries = evidence_ledger.entries()
        prefer_validated = any(str(item.get("status") or "").strip().lower() == "validated" for item in entries)
        if prefer_validated:
            observations = evidence_ledger.retrieve(query_terms=terms, evidence_status="validated", limit=limit)
            if len(observations) < limit:
                seen_observation_ids = {
                    str(item.get("observation_id") or "").strip()
                    for item in observations
                    if str(item.get("observation_id") or "").strip()
                }
                supplemental = []
                for item in evidence_ledger.retrieve(query_terms=terms, limit=limit):
                    observation_id = str(item.get("observation_id") or "").strip()
                    if observation_id and observation_id in seen_observation_ids:
                        continue
                    supplemental.append(item)
                    if observation_id:
                        seen_observation_ids.add(observation_id)
                    if len(observations) + len(supplemental) >= limit:
                        break
                observations = observations + supplemental
        else:
            observations = evidence_ledger.retrieve(query_terms=terms, limit=limit)

        raw_segments = list(preprocess_bundle.get("raw_segments") or preprocess_bundle.get("segments") or [])
        planner_segments = list(preprocess_bundle.get("planner_segments") or [])
        artifact_context = read_jsonl(self._artifact_context_path(task.video_id or task.sample_key))

        payload = {
            "query_terms": terms,
            "observations": observations,
            "evidence": _top_matching_records(entries, terms, limit=20),
            "artifact_context": _top_artifact_context_records(artifact_context, terms, limit=20),
        }
        if str(mode or "").strip() == "refine":
            payload["preprocess_matches"] = {
                "planner_segments": _top_matching_records(planner_segments, terms, limit=12),
                "raw_segments": _top_matching_records(raw_segments, terms, limit=8),
                "asr_transcripts": _top_matching_records(list(preprocess_bundle.get("asr_transcripts") or []), terms, limit=8),
            }
        audit_missing = []
        if isinstance(audit_payload, dict):
            audit_missing = [
                str(item).strip()
                for item in list(audit_payload.get("missing_information") or [])
                if str(item).strip()
            ]
        if audit_missing:
            payload["audit_gaps"] = audit_missing
        claims = _trace_claims(current_trace)
        if claims:
            payload["prior_trace_claims"] = _top_matching_records(claims, terms, limit=20)
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, "", [], {})
        }
