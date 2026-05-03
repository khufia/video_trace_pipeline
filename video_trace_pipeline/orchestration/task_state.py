from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from ..common import hash_payload
from ..schemas import (
    AnswerCandidate,
    CounterRecord,
    CoverageRecord,
    EvidenceStatusUpdate,
    OCRValueOccurrence,
    ReferentSlot,
    TaskClaimResult,
    TaskState,
    TemporalEventRecord,
    RetrievalMemoryRecord,
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _dedupe_strings(values: Iterable[Any]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        item = _text(value)
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _claim_type(question: str, option: str = "") -> tuple[str, List[str]]:
    text = "%s %s" % (question.lower(), option.lower())
    if re.search(r"\b(transcript|said|says|mentioned|quote|phrase|not mentioned|dialogue|word)\b", text):
        return "transcript", ["asr"]
    if re.search(r"\b(scoreboard|score|ocr|read|text|label|price|chart|table|graph|percentage|percent|dollar|letter|sign|plate|blackboard|whiteboard)\b", text):
        return "ocr", ["visual", "ocr"]
    if re.search(r"\b(sound|bang|whistle|chirp|music|noise|gasp|crying|bark|audio)\b", text):
        return "audio_event", ["audio", "visual"]
    if re.search(r"\b(how many|count|times|number of|empty|full|open|closed|on the table|bottle|object|state)\b", text):
        return "count", ["visual"]
    if re.search(r"\b(first|second|third|last|before|after|order|sequence|earliest|latest|then)\b", text):
        return "temporal_order", ["visual"]
    if re.search(r"\b(relation|relationship|same|different|who|person|speaker|mother|father|woman|man|dog|cat)\b", text):
        return "referent_relation", ["visual"]
    if re.search(r"\b(tone|voice|emotion|angry|sad|happy|firm|loudest)\b", text):
        return "speaker_tone", ["asr", "visual"]
    return "option_mapping", []


def _transcript_coverage(preprocess_bundle: Dict[str, Any]) -> Dict[str, Any]:
    spans = []
    for segment in list(preprocess_bundle.get("planner_segments") or []):
        if not isinstance(segment, dict):
            continue
        asr = segment.get("asr") if isinstance(segment.get("asr"), dict) else {}
        for span in list(asr.get("transcript_spans") or []):
            if isinstance(span, dict) and _text(span.get("text")):
                spans.append(span)
    if not spans:
        return {"available": False, "span_count": 0}
    starts = [float(span.get("start_s", 0.0) or 0.0) for span in spans]
    ends = [float(span.get("end_s", span.get("start_s", 0.0)) or 0.0) for span in spans]
    return {
        "available": True,
        "span_count": len(spans),
        "start_s": min(starts),
        "end_s": max(ends),
    }


def _initial_counter_records(question: str, claim_results: List[TaskClaimResult]) -> List[CounterRecord]:
    del claim_results
    lowered = str(question or "").lower()
    has_counter_language = re.search(
        r"\b(how many|count|number of|times|occurrences?|distinct|different|deduplicate)\b",
        lowered,
    )
    if not has_counter_language:
        return []
    return [
        CounterRecord(
            counter_id="task_count",
            target=question,
            inclusion_rule="Count only answer-critical occurrences directly supported by tool evidence for this question.",
            exclusion_rule="Reject duplicates, unrelated candidates, ambiguous candidates, and candidates outside the question scope.",
            status="open",
        )
    ]


def build_initial_task_state(task, preprocess_bundle: Dict[str, Any]) -> TaskState:
    question = _text(getattr(task, "question", ""))
    options = [str(item) for item in list(getattr(task, "options", []) or []) if str(item).strip()]
    claim_results: List[TaskClaimResult] = []
    answer_candidates: List[AnswerCandidate] = []
    if options:
        for index, option in enumerate(options, start=1):
            claim_type, modalities = _claim_type(question, option)
            claim_id = "opt_%02d_claim" % index
            claim_results.append(
                TaskClaimResult(
                    claim_id=claim_id,
                    option=option,
                    text="%s Option: %s" % (question, option),
                    claim_type=claim_type,
                    required_modalities=modalities,
                    status="unverified",
                )
            )
            answer_candidates.append(AnswerCandidate(option=option, unknown_claim_ids=[claim_id], status="possible"))
    else:
        claim_type, modalities = _claim_type(question)
        claim_results.append(
            TaskClaimResult(
                claim_id="answer_claim",
                text=question,
                claim_type=claim_type,
                required_modalities=modalities,
                status="unverified",
            )
        )

    metadata = {
        "transcript_coverage": _transcript_coverage(preprocess_bundle),
        "preprocess_segment_count": len(list(preprocess_bundle.get("planner_segments") or [])),
    }
    return TaskState(
        task_key=getattr(task, "sample_key", "") or getattr(task, "video_id", "") or "task",
        claim_results=claim_results,
        answer_candidates=answer_candidates,
        counter_records=_initial_counter_records(question, claim_results),
        open_questions=[claim.text for claim in claim_results],
        metadata=metadata,
    )


def _state_payload(task_state: TaskState | Dict[str, Any] | None) -> Dict[str, Any]:
    if task_state is None:
        return {}
    if isinstance(task_state, TaskState):
        return task_state.dict()
    return dict(task_state or {})


def compact_task_state(task_state: TaskState | Dict[str, Any] | None) -> Dict[str, Any]:
    payload = _state_payload(task_state)
    if not payload:
        return {}
    return {
        "schema_version": payload.get("schema_version"),
        "task_key": payload.get("task_key"),
        "ready_for_synthesis": payload.get("ready_for_synthesis", False),
        "claim_results": [
            {
                key: item.get(key)
                for key in (
                    "claim_id",
                    "option",
                    "claim_type",
                    "required_modalities",
                    "status",
                    "supporting_evidence_ids",
                    "supporting_observation_ids",
                    "refuting_evidence_ids",
                    "refuting_observation_ids",
                    "notes",
                )
                if item.get(key) not in (None, "", [], {})
            }
            for item in list(payload.get("claim_results") or [])
            if isinstance(item, dict)
        ],
        "referent_slots": list(payload.get("referent_slots") or []),
        "counter_records": list(payload.get("counter_records") or []),
        "coverage_records": list(payload.get("coverage_records") or [])[-12:],
        "ocr_occurrences": list(payload.get("ocr_occurrences") or [])[-20:],
        "evidence_status_updates": list(payload.get("evidence_status_updates") or [])[-20:],
        "retired_evidence": list(payload.get("retired_evidence") or [])[-30:],
        "retrieval_memory": list(payload.get("retrieval_memory") or [])[-12:],
        "open_questions": list(payload.get("open_questions") or []),
        "metadata": dict(payload.get("metadata") or {}),
    }


def record_retrieval_memory(
    task_state: TaskState | Dict[str, Any],
    *,
    query: str,
    target: str,
    results: Dict[str, Any],
    used_result_ids: Optional[List[str]] = None,
) -> TaskState:
    state = task_state if isinstance(task_state, TaskState) else TaskState.model_validate(task_state)
    result_ids: List[str] = []
    for key in ("observations", "evidence", "lookup_records"):
        for item in list((results or {}).get(key) or []):
            if not isinstance(item, dict):
                continue
            identifier = _text(item.get("observation_id") or item.get("evidence_id") or item.get("record_id"))
            if identifier:
                result_ids.append(identifier)
    preprocess_matches = (results or {}).get("preprocess_matches")
    if isinstance(preprocess_matches, dict):
        for items in preprocess_matches.values():
            for item in list(items or []):
                if isinstance(item, dict):
                    identifier = _text(item.get("segment_id") or item.get("transcript_id") or item.get("caption_id"))
                    if identifier:
                        result_ids.append(identifier)
    task_state_matches = list((results or {}).get("task_state_matches") or [])
    for item in task_state_matches:
        if isinstance(item, dict):
            identifier = _text(item.get("claim_id") or item.get("coverage_id") or item.get("occurrence_id") or item.get("event_id"))
            if identifier:
                result_ids.append(identifier)
    if not result_ids:
        return state
    record = RetrievalMemoryRecord(
        retrieval_id="ret_%s" % hash_payload({"query": query, "target": target, "result_ids": result_ids}, 12),
        query=query,
        target=target,
        result_ids=_dedupe_strings(result_ids),
        used_result_ids=_dedupe_strings(used_result_ids or []),
    )
    _append_unique_model(state.retrieval_memory, record, "retrieval_id")
    return state


def _record_ids(record: Dict[str, Any]) -> tuple[str, List[str]]:
    evidence = dict(record.get("evidence_entry") or {})
    evidence_id = _text(evidence.get("evidence_id"))
    observation_ids = [
        _text(item.get("observation_id"))
        for item in list(record.get("observations") or [])
        if isinstance(item, dict) and _text(item.get("observation_id"))
    ]
    if not observation_ids:
        observation_ids = [_text(item) for item in list(evidence.get("observation_ids") or []) if _text(item)]
    return evidence_id, observation_ids


def _append_unique_model(items: List[Any], item: Any, key: str) -> List[Any]:
    value = getattr(item, key, None)
    existing = {getattr(old, key, None) for old in list(items or [])}
    if value not in existing:
        items.append(item)
    return items


def _counter_candidate_status_rank(status: str) -> int:
    return {
        "accepted": 4,
        "validated": 4,
        "supported": 4,
        "rejected": 3,
        "refuted": 3,
        "candidate": 2,
        "unknown": 1,
    }.get(_text(status).lower(), 0)


def _normalize_counter_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(candidate or {})
    canonical_label = _text(
        payload.get("canonical_label")
        or payload.get("dedupe_key")
        or payload.get("label")
        or payload.get("event_label")
        or payload.get("name")
        or payload.get("text")
    )
    if canonical_label:
        payload["canonical_label"] = canonical_label
    raw_mentions = list(payload.get("raw_mentions") or [])
    for key in ("raw_mention", "label", "event_label", "name", "text"):
        value = _text(payload.get(key))
        if value:
            raw_mentions.append(value)
    if raw_mentions:
        payload["raw_mentions"] = _dedupe_strings(raw_mentions)
    if payload.get("status") is not None:
        payload["status"] = _text(payload.get("status")).lower()
    for list_key in (
        "observation_ids",
        "accepted_observation_ids",
        "rejected_observation_ids",
        "evidence_ids",
        "source_ids",
    ):
        if list_key in payload:
            payload[list_key] = _dedupe_strings(payload.get(list_key) or [])
    for text_key in ("reason", "dedupe_rationale"):
        if text_key in payload:
            payload[text_key] = _text(payload.get(text_key))
    return payload


def _counter_candidate_key(candidate: Dict[str, Any]) -> str:
    canonical_label = _text(candidate.get("canonical_label")).lower()
    if canonical_label:
        return "canonical:%s" % canonical_label
    dedupe_key = _text(candidate.get("dedupe_key")).lower()
    if dedupe_key:
        return "dedupe:%s" % dedupe_key
    return "hash:%s" % hash_payload(candidate, 16)


def _merge_counter_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged_by_key: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for raw_candidate in list(candidates or []):
        if not isinstance(raw_candidate, dict):
            continue
        candidate = _normalize_counter_candidate(raw_candidate)
        key = _counter_candidate_key(candidate)
        if key not in merged_by_key:
            merged_by_key[key] = candidate
            order.append(key)
            continue
        existing = merged_by_key[key]
        for list_key in (
            "raw_mentions",
            "observation_ids",
            "accepted_observation_ids",
            "rejected_observation_ids",
            "evidence_ids",
            "source_ids",
        ):
            merged_values = _dedupe_strings(
                list(existing.get(list_key) or []) + list(candidate.get(list_key) or [])
            )
            if merged_values:
                existing[list_key] = merged_values
            else:
                existing.pop(list_key, None)
        for text_key in ("reason", "dedupe_rationale"):
            incoming = _text(candidate.get(text_key))
            current = _text(existing.get(text_key))
            if incoming and incoming not in current:
                existing[text_key] = "; ".join(_dedupe_strings([current, incoming]))
        if _counter_candidate_status_rank(candidate.get("status")) > _counter_candidate_status_rank(existing.get("status")):
            existing["status"] = candidate.get("status")
    return [merged_by_key[key] for key in order]


def _merge_counter_record(items: List[CounterRecord], update: CounterRecord) -> List[CounterRecord]:
    for index, existing in enumerate(list(items or [])):
        if existing.counter_id != update.counter_id:
            continue
        existing.target = update.target or existing.target
        existing.inclusion_rule = update.inclusion_rule or existing.inclusion_rule
        existing.exclusion_rule = update.exclusion_rule or existing.exclusion_rule
        existing.accepted_observation_ids = _dedupe_strings(
            list(existing.accepted_observation_ids or []) + list(update.accepted_observation_ids or [])
        )
        existing.rejected_observation_ids = _dedupe_strings(
            list(existing.rejected_observation_ids or []) + list(update.rejected_observation_ids or [])
        )
        existing.candidates = _merge_counter_candidates(list(existing.candidates or []) + list(update.candidates or []))
        if update.count is not None:
            existing.count = update.count
        if update.status:
            existing.status = update.status
        items[index] = existing
        return items
    update.candidates = _merge_counter_candidates(list(update.candidates or []))
    items.append(update)
    return items


def _numeric_from_text(value: Any) -> Optional[float]:
    text = str(value or "")
    match = re.search(r"[-+]?\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)\s*%?", text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except Exception:
        return None


def _update_answer_candidates(state: TaskState) -> None:
    claims_by_option: Dict[str, List[TaskClaimResult]] = {}
    for claim in list(state.claim_results or []):
        option = _text(claim.option)
        if not option:
            continue
        claims_by_option.setdefault(option, []).append(claim)
    for candidate in list(state.answer_candidates or []):
        option_claims = claims_by_option.get(_text(candidate.option), [])
        candidate.supporting_claim_ids = _dedupe_strings(
            claim.claim_id for claim in option_claims if claim.status == "validated"
        )
        candidate.refuting_claim_ids = _dedupe_strings(
            claim.claim_id for claim in option_claims if claim.status == "refuted"
        )
        candidate.unknown_claim_ids = _dedupe_strings(
            claim.claim_id
            for claim in option_claims
            if claim.status in {"unverified", "unknown", "partially_validated"}
        )
        if candidate.supporting_claim_ids and not candidate.unknown_claim_ids and not candidate.refuting_claim_ids:
            candidate.status = "supported"
        elif candidate.refuting_claim_ids and not candidate.supporting_claim_ids:
            candidate.status = "rejected"
        else:
            candidate.status = "possible"


def _prune_open_questions(state: TaskState, extra_open_questions: Optional[List[str]] = None) -> List[str]:
    unresolved_statuses = {"unverified", "unknown", "partially_validated"}
    unresolved_claims = [
        claim.text
        for claim in state.claim_results
        if claim.status in unresolved_statuses
    ]
    unresolved_texts = {_text(item) for item in unresolved_claims if _text(item)}
    resolved_claim_texts = {
        _text(claim.text)
        for claim in state.claim_results
        if claim.status not in unresolved_statuses
    }
    carried = []
    for item in list(extra_open_questions or []) + list(state.open_questions or []):
        text = _text(item)
        if not text:
            continue
        if text in resolved_claim_texts and text not in unresolved_texts:
            continue
        carried.append(text)
    return _dedupe_strings(unresolved_claims + carried)


def _append_evidence_update(
    state: TaskState,
    *,
    evidence_id: str,
    previous_status: str,
    new_status: str,
    reason: str,
    updated_by: str,
    round_index: int,
    claim_id: str = "",
) -> None:
    evidence_id = _text(evidence_id)
    if not evidence_id:
        return
    try:
        update = EvidenceStatusUpdate(
            evidence_id=evidence_id,
            previous_status=previous_status,
            new_status=new_status,
            claim_id=claim_id or None,
            reason=reason,
            updated_by=updated_by,
            round_index=round_index,
        )
    except ValueError:
        return
    state.evidence_status_updates.append(update)


def update_task_state_after_execution(
    task_state: TaskState | Dict[str, Any],
    execution_records: List[Dict[str, Any]],
    *,
    round_index: int,
) -> TaskState:
    state = task_state if isinstance(task_state, TaskState) else TaskState.model_validate(task_state)
    claim_by_id = {claim.claim_id: claim for claim in state.claim_results}
    open_questions = list(state.open_questions or [])

    for record in list(execution_records or []):
        tool_name = _text(record.get("tool_name"))
        evidence_id, observation_ids = _record_ids(record)
        result = dict(record.get("result") or {})
        data = dict(result.get("data") or {})
        status = "unknown" if not result.get("ok", True) else "candidate"
        if evidence_id:
            _append_evidence_update(
                state,
                evidence_id=evidence_id,
                previous_status="",
                new_status=status,
                reason="Tool execution recorded %s output." % (status if status != "candidate" else "candidate"),
                updated_by=tool_name or "executor",
                round_index=round_index,
            )
        if tool_name in {"frame_retriever", "ocr", "generic_purpose", "verifier", "asr", "dense_captioner", "spatial_grounder"} and evidence_id:
            time_range = {}
            evidence_entry = dict(record.get("evidence_entry") or {})
            if evidence_entry.get("time_start_s") is not None:
                time_range["start_s"] = evidence_entry.get("time_start_s")
            if evidence_entry.get("time_end_s") is not None:
                time_range["end_s"] = evidence_entry.get("time_end_s")
            _append_unique_model(
                state.coverage_records,
                CoverageRecord(
                    coverage_id="cov_%s" % hash_payload({"evidence_id": evidence_id, "tool": tool_name}, 12),
                    modality="text" if tool_name in {"asr", "ocr", "verifier"} else "visual",
                    time_range=time_range,
                    sampling=_text((record.get("request") or {}).get("query")),
                    checked_by=tool_name,
                    status=status,
                ),
                "coverage_id",
            )
        if tool_name == "ocr":
            for observation in list(record.get("observations") or []):
                if not isinstance(observation, dict):
                    continue
                raw_text = _text(observation.get("object_text") or observation.get("atomic_text"))
                if not raw_text:
                    continue
                occurrence_id = "ocr_%s" % hash_payload(
                    {
                        "text": raw_text,
                        "frame_ts_s": observation.get("frame_ts_s"),
                        "bbox": observation.get("bbox"),
                        "evidence_id": evidence_id,
                    },
                    12,
                )
                source_artifact_id = (list(observation.get("source_artifact_refs") or []) or [None])[0]
                _append_unique_model(
                    state.ocr_occurrences,
                    OCRValueOccurrence(
                        occurrence_id=occurrence_id,
                        kind="text",
                        raw_text=raw_text,
                        normalized_value=observation.get("numeric_value") if observation.get("numeric_value") is not None else _numeric_from_text(raw_text),
                        source_artifact_id=source_artifact_id,
                        bbox=observation.get("bbox"),
                        confidence=observation.get("confidence"),
                        status="candidate",
                    ),
                    "occurrence_id",
                )
        if tool_name in {"visual_temporal_grounder", "audio_temporal_grounder"}:
            for observation in list(record.get("observations") or []):
                if not isinstance(observation, dict):
                    continue
                event_id = "event_%s" % hash_payload(
                    {
                        "text": observation.get("atomic_text"),
                        "start": observation.get("time_start_s"),
                        "end": observation.get("time_end_s"),
                    },
                    12,
                )
                _append_unique_model(
                    state.temporal_events,
                    TemporalEventRecord(
                        event_id=event_id,
                        description=_text(observation.get("atomic_text")),
                        start_s=observation.get("time_start_s"),
                        end_s=observation.get("time_end_s"),
                        source=tool_name,
                        status="candidate",
                    ),
                    "event_id",
                )
        if tool_name == "verifier":
            for claim_result in list(data.get("claim_results") or []):
                if not isinstance(claim_result, dict):
                    continue
                claim_id = _text(claim_result.get("claim_id"))
                claim = claim_by_id.get(claim_id)
                verdict = _text(claim_result.get("verdict")).lower()
                mapped_status = {
                    "supported": "validated",
                    "refuted": "refuted",
                    "partially_supported": "partially_validated",
                    "unknown": "unknown",
                }.get(verdict, "unknown")
                if claim is not None:
                    claim.status = mapped_status
                    claim.supporting_evidence_ids = _dedupe_strings(
                        list(claim.supporting_evidence_ids or [])
                        + list(claim_result.get("supporting_evidence_ids") or [])
                        + ([evidence_id] if evidence_id and mapped_status in {"validated", "partially_validated"} else [])
                    )
                    claim.supporting_observation_ids = _dedupe_strings(
                        list(claim.supporting_observation_ids or [])
                        + list(claim_result.get("supporting_observation_ids") or [])
                        + (observation_ids if mapped_status in {"validated", "partially_validated"} else [])
                    )
                    claim.refuting_evidence_ids = _dedupe_strings(
                        list(claim.refuting_evidence_ids or [])
                        + list(claim_result.get("refuting_evidence_ids") or [])
                        + ([evidence_id] if evidence_id and mapped_status == "refuted" else [])
                    )
                    claim.refuting_observation_ids = _dedupe_strings(
                        list(claim.refuting_observation_ids or [])
                        + list(claim_result.get("refuting_observation_ids") or [])
                        + (observation_ids if mapped_status == "refuted" else [])
                    )
                    claim.notes = _text(claim_result.get("rationale"))
                if evidence_id:
                    _append_evidence_update(
                        state,
                        evidence_id=evidence_id,
                        previous_status="candidate",
                        new_status="validated" if mapped_status == "validated" else mapped_status if mapped_status in {"refuted", "unknown"} else "candidate",
                        claim_id=claim_id,
                        reason=_text(claim_result.get("rationale")) or "Verifier updated claim state.",
                        updated_by="verifier",
                        round_index=round_index,
                    )
            for update in list(data.get("evidence_updates") or []):
                if not isinstance(update, dict):
                    continue
                _append_evidence_update(
                    state,
                    evidence_id=update.get("evidence_id"),
                    previous_status=update.get("previous_status") or "candidate",
                    new_status=update.get("new_status") or update.get("status") or "unknown",
                    claim_id=_text(update.get("claim_id")),
                    reason=_text(update.get("reason")) or "Verifier emitted evidence status update.",
                    updated_by="verifier",
                    round_index=round_index,
                )
            for update in list(data.get("counter_updates") or []):
                if not isinstance(update, dict):
                    continue
                counter_id = _text(update.get("counter_id") or update.get("id") or update.get("target"))
                target = _text(update.get("target") or update.get("description"))
                if not counter_id or not target:
                    continue
                _merge_counter_record(
                    state.counter_records,
                    CounterRecord(
                        counter_id=counter_id,
                        target=target,
                        inclusion_rule=_text(update.get("inclusion_rule")),
                        exclusion_rule=_text(update.get("exclusion_rule")),
                        candidates=[item for item in list(update.get("candidates") or []) if isinstance(item, dict)],
                        accepted_observation_ids=update.get("accepted_observation_ids") or [],
                        rejected_observation_ids=update.get("rejected_observation_ids") or [],
                        count=update.get("count"),
                        status=_text(update.get("status")) or "candidate",
                    ),
                )
            for update in list(data.get("referent_updates") or []):
                if not isinstance(update, dict):
                    continue
                referent_id = _text(update.get("referent_id") or update.get("id") or update.get("description"))
                description = _text(update.get("description") or update.get("text"))
                if not referent_id or not description:
                    continue
                _append_unique_model(
                    state.referent_slots,
                    ReferentSlot(
                        referent_id=referent_id,
                        description=description,
                        scope=_text(update.get("scope")),
                        status=_text(update.get("status")) or "candidate",
                        linked_claim_ids=update.get("linked_claim_ids") or [],
                        evidence_ids=update.get("evidence_ids") or ([evidence_id] if evidence_id else []),
                        observation_ids=update.get("observation_ids") or observation_ids,
                    ),
                    "referent_id",
                )
            for update in list(data.get("ocr_occurrence_updates") or []):
                if not isinstance(update, dict):
                    continue
                raw_text = _text(update.get("raw_text") or update.get("text"))
                occurrence_id = _text(update.get("occurrence_id")) or "ocr_%s" % hash_payload(
                    {"raw_text": raw_text, "source_artifact_id": update.get("source_artifact_id")},
                    12,
                )
                if not raw_text:
                    continue
                _append_unique_model(
                    state.ocr_occurrences,
                    OCRValueOccurrence(
                        occurrence_id=occurrence_id,
                        kind=_text(update.get("kind")) or "text",
                        raw_text=raw_text,
                        normalized_value=update.get("normalized_value") if update.get("normalized_value") is not None else _numeric_from_text(raw_text),
                        nearby_label=_text(update.get("nearby_label")),
                        source_artifact_id=update.get("source_artifact_id"),
                        bbox=update.get("bbox"),
                        confidence=update.get("confidence"),
                        status=_text(update.get("status")) or "candidate",
                        dedupe_key=update.get("dedupe_key"),
                    ),
                    "occurrence_id",
                )
            open_questions.extend(_text(item) for item in list(data.get("unresolved_gaps") or []) if _text(item))

    state.open_questions = _prune_open_questions(state, open_questions)
    _update_answer_candidates(state)
    supported_candidates = [
        candidate
        for candidate in list(state.answer_candidates or [])
        if candidate.status == "supported"
    ]
    if state.answer_candidates:
        state.ready_for_synthesis = len(supported_candidates) == 1
    else:
        state.ready_for_synthesis = bool(state.claim_results) and all(claim.status == "validated" for claim in state.claim_results)
    state.retired_evidence = _dedupe_strings(
        list(state.retired_evidence or [])
        + [
            update.evidence_id
            for update in state.evidence_status_updates
            if update.new_status in {"refuted", "irrelevant", "stale", "superseded"}
        ]
    )
    return state


def update_task_state_after_audit(
    task_state: TaskState | Dict[str, Any],
    audit_report: Any,
    *,
    accepted: bool,
) -> TaskState:
    state = task_state if isinstance(task_state, TaskState) else TaskState.model_validate(task_state)
    missing = []
    if audit_report is not None:
        missing = [_text(item) for item in list(getattr(audit_report, "missing_information", []) or []) if _text(item)]
    if missing:
        state.open_questions = _prune_open_questions(state, missing)
        state.ready_for_synthesis = False
    elif accepted:
        state.open_questions = []
        state.ready_for_synthesis = True
    return state
