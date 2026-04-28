from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "firmer",
        "grounded",
        "grounding",
        "inspected",
        "queried",
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
            "artifact_context": _top_matching_records(artifact_context, terms, limit=20),
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
