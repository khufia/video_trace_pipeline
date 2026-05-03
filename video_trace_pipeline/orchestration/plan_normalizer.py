from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..common import stable_json_dumps
from ..schemas import ExecutionPlan, InputRef, PlanStep
from .executor import request_model_field_names

_LIST_FIELDS = {
    "clips",
    "frames",
    "regions",
    "transcripts",
    "text_contexts",
    "evidence_ids",
    "time_hints",
}

_GENERIC_PURPOSE_CONTEXT_FIELDS = frozenset(
    {
        "clips",
        "frames",
        "transcripts",
        "text_contexts",
        "evidence_ids",
    }
)

_STRUCTURAL_INPUT_REF_FIELDS = {
    "clips",
    "frames",
    "regions",
    "transcripts",
}

_STRUCTURAL_INPUT_REF_PATTERNS = {
    "clips": (
        "clips",
        "clips[]",
        "frames[].clip",
        "regions[].frame.clip",
        "transcripts[].clip",
    ),
    "frames": (
        "frames",
        "frames[]",
        "regions[].frame",
    ),
    "regions": (
        "regions",
        "regions[]",
    ),
    "transcripts": (
        "transcripts",
        "transcripts[]",
    ),
}

_TIME_HINT_INPUT_REF_PATTERNS = (
    "phrase_matches[].time_hint",
)

_TEXT_CONTEXT_SOURCE_FIELDS = frozenset(
    {
        "text",
        "summary",
        "overall_summary",
        "analysis",
        "answer",
        "supporting_points",
        "spatial_description",
        "raw_output_text",
    }
)

_CLIP_CAPABLE_DOWNSTREAM_TOOLS = frozenset(
    {
        "generic_purpose",
        "ocr",
        "spatial_grounder",
        "asr",
        "dense_captioner",
        "audio_temporal_grounder",
    }
)

_CANONICAL_OUTPUT_FIELD_OVERRIDES = {
    "asr": {"clips", "transcripts", "phrase_matches"},
    "dense_captioner": {"clips", "captions", "overall_summary", "captioned_range", "sampled_frames"},
    "ocr": {"query", "text", "lines", "reads", "timestamp_s", "source_frame_path", "backend"},
    "spatial_grounder": {"query", "frames", "detections", "regions", "groundings", "spatial_description", "backend"},
}

def _retrieved_evidence_ids(retrieved_context: Dict[str, Any] | None) -> set[str]:
    ids = set()
    if not isinstance(retrieved_context, dict):
        return ids
    evidence_items = list(retrieved_context.get("evidence") or []) + list(retrieved_context.get("evidence_entries") or [])
    observation_items = list(retrieved_context.get("observations") or []) + list(retrieved_context.get("recent_observations") or [])
    for item in evidence_items:
        if isinstance(item, dict) and str(item.get("evidence_id") or "").strip():
            ids.add(str(item.get("evidence_id")).strip())
    for item in observation_items:
        if isinstance(item, dict) and str(item.get("evidence_id") or "").strip():
            ids.add(str(item.get("evidence_id")).strip())
    for item in list(retrieved_context.get("lookup_records") or []):
        if isinstance(item, dict) and str(item.get("evidence_id") or "").strip():
            ids.add(str(item.get("evidence_id")).strip())
    return ids


_EXACT_TIMESTAMP_RE = re.compile(
    r"\b\d{1,3}:\d{2}(?::\d{2})?(?:\.\d+)?\b"
    r"|\b(?:timestamp|time|at|around|near|second|sec)\s*(?:=|:)?\s*\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)?\b"
    r"|\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)\b",
    re.IGNORECASE,
)

_ANCHORABLE_TIME_HINT_RE = re.compile(
    r"\b\d{1,3}:\d{2}(?::\d{2})?(?:\.\d+)?\b"
    r"|\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)\b"
    r"|\b(?:timestamp|time|at|around|near|second|sec)\s*(?:=|:)?\s*\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)?\b"
    r"|\b(start|begin|opening|onset|first|end|finish|closing|last|middle|midpoint|center|centre)\b"
    r"|\b\d+(?:\.\d+)?\s*%",
    re.IGNORECASE,
)

_PLACEHOLDER_TIME_HINT_RE = re.compile(
    r"\buse\s+the\s+timestamp\b"
    r"|\btimestamp\s+of\s+the\b"
    r"|\bnearest\s+transcript\s+span\b"
    r"|\bphrase\s+match\b"
    r"|\basr\s+phrase\b"
    r"|\bprevious\s+step\b"
    r"|\btool\s+\d+\s+output\b",
    re.IGNORECASE,
)

_SPEECH_ANCHOR_QUERY_RE = re.compile(
    r"\b(asr|speech|spoken|utterance|dialogue|dialog|quote|quoted|phrase|transcript|line)\b",
    re.IGNORECASE,
)

_EXPLICIT_FRAME_ARTIFACT_NEED_RE = re.compile(
    r"\b("
    r"readable|legible|static|still|single|particular|specific|exact|timestamp|anchor|neighboring?|"
    r"high[- ]?res(?:olution)?|ocr|text|small|fine|subtle|detail|crop|"
    r"frame[- ]by[- ]frame|individual frames|per[- ]frame"
    r")\b",
    re.IGNORECASE,
)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _dedupe_list(values: List[Any]) -> List[Any]:
    ordered: List[Any] = []
    seen = set()
    for item in list(values or []):
        signature = stable_json_dumps(item)
        if signature in seen:
            continue
        seen.add(signature)
        ordered.append(item)
    return ordered


def _normalize_list_value(field_name: str, value: Any) -> List[Any]:
    values = _dedupe_list(_coerce_list(value))
    if field_name in {"text_contexts", "evidence_ids"}:
        return sorted(values, key=stable_json_dumps)
    return values


def _looks_like_transcript_payload(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in value for key in ("transcript_id", "segments", "clip"))
    if isinstance(value, list):
        return any(_looks_like_transcript_payload(item) for item in value)
    return False


def _merge_field_value(field_name: str, current: Any, new_value: Any) -> Any:
    if field_name in _LIST_FIELDS:
        return _normalize_list_value(field_name, _coerce_list(current) + _coerce_list(new_value))
    if current is None:
        return new_value
    return current


def _matches_structural_pattern(field_path: str, pattern: str) -> bool:
    normalized_path = str(field_path or "").strip()
    normalized_pattern = str(pattern or "").strip()
    if not normalized_path or not normalized_pattern:
        return False
    if normalized_path == normalized_pattern:
        return True
    regex = "^%s$" % r"\.".join(
        r"%s(?:\[\d+\]|\.\d+)" % re.escape(token[:-2]) if token.endswith("[]") else re.escape(token)
        for token in normalized_pattern.split(".")
    )
    return re.fullmatch(regex, normalized_path) is not None


def _is_allowed_structural_input_ref(target_field: str, field_path: str) -> bool:
    allowed_patterns = _STRUCTURAL_INPUT_REF_PATTERNS.get(str(target_field or "").strip(), ())
    return any(_matches_structural_pattern(field_path, pattern) for pattern in allowed_patterns)


def _is_allowed_time_hint_input_ref(field_path: str) -> bool:
    return any(_matches_structural_pattern(field_path, pattern) for pattern in _TIME_HINT_INPUT_REF_PATTERNS)


def _literal_time_hints(step: PlanStep) -> List[str]:
    return [str(item).strip() for item in list((step.inputs or {}).get("time_hints") or []) if str(item).strip()]


def _has_anchorable_time_hint(time_hints: List[str]) -> bool:
    return any(_ANCHORABLE_TIME_HINT_RE.search(str(item or "")) for item in list(time_hints or []))


def _has_time_hint_ref(step: PlanStep) -> bool:
    return bool(list(dict(step.input_refs or {}).get("time_hints") or []))


def _field_path_head(field_path: str) -> str:
    head = str(field_path or "").strip().split(".", 1)[0]
    return re.sub(r"\[.*?\]$", "", head).strip()


def _source_ids(step: PlanStep) -> List[int]:
    ordered: List[int] = []
    seen = set()
    for refs in dict(step.input_refs or {}).values():
        for ref in list(refs or []):
            step_id = int(ref.step_id)
            if step_id in seen:
                continue
            seen.add(step_id)
            ordered.append(step_id)
    return ordered


def _task_text(task: Any) -> str:
    question = getattr(task, "question", "")
    options = " ".join(str(item or "") for item in list(getattr(task, "options", []) or []))
    return _normalize_text("%s %s" % (question, options)).lower()


def _ref_signature(target_field: str, ref: InputRef) -> Tuple[str, int, str]:
    return (
        str(target_field or "").strip(),
        int(ref.step_id),
        str(ref.field_path or "").strip(),
    )


def _step_search_text(step: PlanStep) -> str:
    parts: List[str] = [str(step.purpose or "")]
    inputs = dict(step.inputs or {})
    for key in ("query", "sequence_mode", "sort_order"):
        if inputs.get(key) is not None:
            parts.append(str(inputs.get(key)))
    for item in list(inputs.get("time_hints") or []):
        parts.append(str(item))
    for key, value in sorted(dict(step.expected_outputs or {}).items()):
        parts.append(str(key))
        parts.append(stable_json_dumps(value) if isinstance(value, (dict, list)) else str(value))
    return _normalize_text(" ".join(parts))


def _frame_step_has_explicit_frame_artifact_need(step: PlanStep) -> bool:
    inputs = dict(step.inputs or {})
    if str(inputs.get("sequence_mode") or "").strip().lower() == "anchor_window":
        return True
    text = _step_search_text(step)
    return bool(_EXPLICIT_FRAME_ARTIFACT_NEED_RE.search(text) or _EXACT_TIMESTAMP_RE.search(text))


def _step_consumes_output_from(step: PlanStep, source_step_id: int) -> bool:
    for refs in dict(step.input_refs or {}).values():
        for ref in list(refs or []):
            if int(ref.step_id) == int(source_step_id):
                return True
    return False


def _frame_retriever_depends_on_visual_temporal_grounder(step: PlanStep, step_by_id: Dict[int, PlanStep]) -> bool:
    if str(step.tool_name or "").strip() != "frame_retriever":
        return False
    for ref in list(dict(step.input_refs or {}).get("clips") or []):
        source_step = step_by_id.get(int(ref.step_id))
        if source_step is not None and str(source_step.tool_name or "").strip() == "visual_temporal_grounder":
            return True
    return False


class ExecutionPlanNormalizer(object):
    VERSION = "field_keyed_v2"

    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def _allowed_fields(self, tool_name: str) -> List[str]:
        adapter = self.tool_registry.get_adapter(tool_name)
        return [name for name in request_model_field_names(getattr(adapter, "request_model", None)) if name != "tool_name"]

    def _allowed_output_fields(self, tool_name: str) -> List[str]:
        tool_name = _normalize_text(tool_name)
        adapter = self.tool_registry.get_adapter(tool_name)
        fields = set(request_model_field_names(getattr(adapter, "output_model", None)))
        fields.update(_CANONICAL_OUTPUT_FIELD_OVERRIDES.get(tool_name, set()))
        return sorted(field for field in fields if field)

    def _normalized_field_name(self, field_name: str) -> str:
        return str(field_name or "").strip()

    def _validate_asr_transcript_contract(self, *, step: PlanStep, target_field: str, ref: InputRef, source_tool_name: str) -> None:
        if str(step.tool_name or "").strip() != "generic_purpose":
            return
        if str(source_tool_name or "").strip() != "asr":
            return
        field_path = str(ref.field_path or "").strip()
        if target_field == "text_contexts":
            raise ValueError(
                "Plan step %s must bind ASR transcript content to generic_purpose via transcripts -> transcripts, not %s -> %s."
                % (int(step.step_id), field_path or "<empty>", target_field or "<empty>")
            )
        if target_field == "transcripts" and field_path != "transcripts":
            raise ValueError(
                "Plan step %s must bind ASR transcript content to generic_purpose via transcripts -> transcripts, not %s -> %s."
                % (int(step.step_id), field_path or "<empty>", target_field or "<empty>")
            )

    def _validate_input_ref_contract(self, *, step: PlanStep, target_field: str, ref: InputRef) -> None:
        step_id = int(step.step_id)
        field_path = str(ref.field_path or "").strip()
        if target_field == "evidence_ids":
            raise ValueError(
                "Plan step %s binds input_refs into evidence_ids. Current-plan steps do not emit bindable evidence ids; "
                "pass frames, clips, transcripts, or text_contexts instead."
                % step_id
            )
        if target_field == "time_hints":
            if _is_allowed_time_hint_input_ref(field_path):
                return
            raise ValueError(
                "Plan step %s binds %s into time_hints. time_hints input_refs may only bind explicit timestamp "
                "strings such as ASR phrase_matches[].time_hint; do not bind transcripts or broad clips into time_hints."
                % (step_id, field_path or "<empty>")
            )
        if target_field in _STRUCTURAL_INPUT_REF_FIELDS and not _is_allowed_structural_input_ref(target_field, field_path):
            raise ValueError(
                "Plan step %s must bind %s via a structural %s path (%s), not %s -> %s."
                % (
                    step_id,
                    target_field,
                    target_field,
                    ", ".join(_STRUCTURAL_INPUT_REF_PATTERNS.get(target_field, (target_field,))),
                    field_path or "<empty>",
                    target_field,
                )
            )
        if target_field == "text_contexts" and field_path not in _TEXT_CONTEXT_SOURCE_FIELDS:
            raise ValueError(
                "Plan step %s binds %s into text_contexts, but text_contexts only accepts textual outputs such as: %s."
                % (
                    step_id,
                    field_path or "<empty>",
                    ", ".join(sorted(_TEXT_CONTEXT_SOURCE_FIELDS)),
                )
            )
    def _validate_source_output_field(self, *, step: PlanStep, target_field: str, ref: InputRef, source_step: PlanStep) -> None:
        del target_field
        field_head = _field_path_head(ref.field_path)
        if not field_head:
            raise ValueError(
                "Plan step %s references an empty output field from step %s."
                % (int(step.step_id), int(source_step.step_id))
            )
        allowed_outputs = self._allowed_output_fields(source_step.tool_name)
        if field_head not in allowed_outputs:
            raise ValueError(
                "Plan step %s references output field %r from step %s (%s), but that tool emits only: %s."
                % (
                    int(step.step_id),
                    field_head,
                    int(source_step.step_id),
                    source_step.tool_name,
                    ", ".join(allowed_outputs),
                )
            )

    def _normalize_step(self, step: PlanStep) -> PlanStep:
        tool_name = _normalize_text(step.tool_name)
        allowed_fields = set(self._allowed_fields(tool_name))

        normalized_inputs: Dict[str, Any] = {}
        for key in sorted(dict(step.inputs or {})):
            canonical_key = self._normalized_field_name(key)
            raw_value = dict(step.inputs or {}).get(key)
            if canonical_key == "tool_name" or canonical_key not in allowed_fields:
                raise ValueError(
                    "Plan step %s uses unexpected input %r for %s. Use canonical request fields only: %s."
                    % (
                        int(step.step_id),
                        canonical_key,
                        tool_name,
                        ", ".join(sorted(allowed_fields)),
                    )
                )
            if tool_name == "generic_purpose" and canonical_key == "text_contexts" and _looks_like_transcript_payload(raw_value):
                raise ValueError(
                    "Plan step %s passes transcript payloads via text_contexts. Use transcripts instead."
                    % int(step.step_id)
                )
            normalized_inputs[canonical_key] = _merge_field_value(
                canonical_key,
                normalized_inputs.get(canonical_key),
                raw_value,
            )
        normalized_inputs = {
            key: normalized_inputs[key]
            for key in sorted(normalized_inputs)
        }
        if "query" in normalized_inputs:
            normalized_inputs["query"] = _normalize_text(normalized_inputs.get("query"))

        normalized_refs: Dict[str, List[InputRef]] = {}
        seen_refs = set()
        for target_field, refs in sorted(dict(step.input_refs or {}).items()):
            canonical_target = self._normalized_field_name(target_field)
            if canonical_target == "tool_name" or canonical_target not in allowed_fields:
                raise ValueError(
                    "Plan step %s uses unexpected input_ref target field %r for %s. Use canonical request fields only: %s."
                    % (
                        int(step.step_id),
                        canonical_target,
                        tool_name,
                        ", ".join(sorted(allowed_fields)),
                    )
                )
            for ref in list(refs or []):
                normalized_ref = InputRef(
                    step_id=ref.step_id,
                    field_path=_normalize_text(ref.field_path),
                )
                signature = _ref_signature(canonical_target, normalized_ref)
                if signature in seen_refs:
                    continue
                seen_refs.add(signature)
                normalized_refs.setdefault(canonical_target, []).append(normalized_ref)
        normalized_refs = {
            key: sorted(values, key=lambda ref: _ref_signature(key, ref))
            for key, values in sorted(normalized_refs.items())
        }

        return PlanStep(
            step_id=step.step_id,
            tool_name=tool_name,
            purpose=_normalize_text(step.purpose),
            inputs=normalized_inputs,
            input_refs=normalized_refs,
            expected_outputs=dict(step.expected_outputs or {}),
        )

    def _validate_references(self, steps: List[PlanStep]) -> None:
        ordered_step_ids = [int(step.step_id) for step in list(steps or [])]
        if len(set(ordered_step_ids)) != len(ordered_step_ids):
            raise ValueError("Execution plan contains duplicate step_ids: %s" % ordered_step_ids)
        step_ids = set(ordered_step_ids)
        step_by_id = {int(step.step_id): step for step in list(steps or [])}
        for step in list(steps or []):
            step_id = int(step.step_id)
            for target_field, refs in dict(step.input_refs or {}).items():
                for ref in list(refs or []):
                    source_id = int(ref.step_id)
                    if source_id not in step_ids:
                        raise ValueError(
                            "Plan step %s references missing input source step %s via %s."
                            % (step_id, source_id, target_field)
                        )
                    if source_id == step_id:
                        raise ValueError(
                            "Plan step %s has a self-referential input_ref for %s."
                            % (step_id, target_field)
                        )
                    source_step = step_by_id.get(source_id)
                    if source_step is not None:
                        self._validate_input_ref_contract(step=step, target_field=target_field, ref=ref)
                        self._validate_asr_transcript_contract(
                            step=step,
                            target_field=target_field,
                            ref=ref,
                            source_tool_name=source_step.tool_name,
                        )
                        self._validate_source_output_field(
                            step=step,
                            target_field=target_field,
                            ref=ref,
                            source_step=source_step,
                        )

    def _validate_generic_purpose_context_contract(self, steps: List[PlanStep]) -> None:
        for step in list(steps or []):
            if str(step.tool_name or "").strip() != "generic_purpose":
                continue
            has_input_context = any(
                _normalize_text((step.inputs or {}).get(field_name))
                if isinstance((step.inputs or {}).get(field_name), str)
                else bool((step.inputs or {}).get(field_name))
                for field_name in _GENERIC_PURPOSE_CONTEXT_FIELDS
            )
            has_bound_context = bool(step.input_refs or {})
            if has_input_context or has_bound_context:
                continue
            raise ValueError(
                "Plan step %s is a context-free generic_purpose request. "
                "Pass frames, clips, transcripts, text_contexts, evidence_ids, or an earlier input_ref explicitly."
                % int(step.step_id)
            )

    def _validate_grounded_clip_bridge_policy(self, steps: List[PlanStep]) -> None:
        step_by_id = {int(step.step_id): step for step in list(steps or [])}
        for frame_step in list(steps or []):
            frame_step_id = int(frame_step.step_id)
            if not _frame_retriever_depends_on_visual_temporal_grounder(frame_step, step_by_id):
                continue
            if _frame_step_has_explicit_frame_artifact_need(frame_step):
                continue
            for consumer_step in list(steps or []):
                consumer_tool = str(consumer_step.tool_name or "").strip()
                if consumer_tool not in _CLIP_CAPABLE_DOWNSTREAM_TOOLS:
                    continue
                if not _step_consumes_output_from(consumer_step, frame_step_id):
                    continue
                raise ValueError(
                    "Plan step %s uses frame_retriever as a bridge from visual_temporal_grounder to %s "
                    "without an explicit frame-specific need. Pass grounded clips directly to %s, or make "
                    "the frame_retriever purpose state an exact/readable/static/OCR/anchor-window frame need."
                    % (frame_step_id, consumer_tool, consumer_tool)
                )

    def _step_has_literal_or_bound_context(self, step: PlanStep, fields: set[str]) -> bool:
        fields = {str(field or "").strip() for field in fields}
        for field_name in fields:
            value = (step.inputs or {}).get(field_name)
            if isinstance(value, str):
                if _normalize_text(value):
                    return True
            elif bool(value):
                return True
        return any(str(field_name or "").strip() in fields for field_name in dict(step.input_refs or {}))

    def _validate_tool_context_contracts(self, steps: List[PlanStep]) -> None:
        requirements = {
            "frame_retriever": {"clips", "time_hints"},
            "audio_temporal_grounder": {"clips"},
            "asr": {"clips"},
            "dense_captioner": {"clips"},
            "spatial_grounder": {"clips", "frames"},
            "ocr": {"clips", "frames", "regions"},
        }
        descriptions = {
            "frame_retriever": "literal clips, literal time_hints, or clips input_refs",
            "audio_temporal_grounder": "clips",
            "asr": "clips",
            "dense_captioner": "clips",
            "spatial_grounder": "clips or frames",
            "ocr": "clips, frames, or regions",
        }
        for step in list(steps or []):
            tool_name = str(step.tool_name or "").strip()
            required_fields = requirements.get(tool_name)
            if not required_fields:
                continue
            if self._step_has_literal_or_bound_context(step, required_fields):
                continue
            raise ValueError(
                "Plan step %s (%s) lacks required context. Provide %s via inputs or input_refs before this tool runs."
                % (int(step.step_id), tool_name, descriptions.get(tool_name) or ", ".join(sorted(required_fields)))
            )

    def _validate_frame_retriever_time_contracts(self, steps: List[PlanStep]) -> None:
        step_by_id = {int(step.step_id): step for step in list(steps or [])}
        for step in list(steps or []):
            if str(step.tool_name or "").strip() != "frame_retriever":
                continue
            step_id = int(step.step_id)
            literal_hints = _literal_time_hints(step)
            placeholder_hints = [hint for hint in literal_hints if _PLACEHOLDER_TIME_HINT_RE.search(hint)]
            if placeholder_hints:
                raise ValueError(
                    "Plan step %s uses placeholder time_hints (%s). Resolve the timestamp structurally first, "
                    "for example ASR -> phrase_matches[].time_hint, or use an explicit literal such as '129.125s'."
                    % (step_id, ", ".join(placeholder_hints))
                )
            sequence_mode = str((step.inputs or {}).get("sequence_mode") or "ranked").strip().lower()
            if sequence_mode == "anchor_window" and literal_hints and not _has_anchorable_time_hint(literal_hints) and not _has_time_hint_ref(step):
                raise ValueError(
                    "Plan step %s requests frame_retriever anchor_window with non-anchorable time_hints. "
                    "Use explicit timestamps like '129.125s' or bind ASR phrase_matches[].time_hint."
                    % step_id
                )
            for ref in list(dict(step.input_refs or {}).get("time_hints") or []):
                source_step = step_by_id.get(int(ref.step_id))
                if source_step is None:
                    continue
                if str(source_step.tool_name or "").strip() != "asr":
                    raise ValueError(
                        "Plan step %s binds time_hints from step %s (%s). time_hints input_refs currently "
                        "must come from ASR phrase_matches[].time_hint."
                        % (step_id, int(ref.step_id), source_step.tool_name)
                    )
            if _has_time_hint_ref(step):
                continue
            search_text = _step_search_text(step)
            if not _SPEECH_ANCHOR_QUERY_RE.search(search_text):
                continue
            for ref in list(dict(step.input_refs or {}).get("clips") or []):
                source_step = step_by_id.get(int(ref.step_id))
                if source_step is None or str(source_step.tool_name or "").strip() != "asr":
                    continue
                raise ValueError(
                    "Plan step %s follows ASR output into frame_retriever for a speech/quote anchor without "
                    "time_hints. Bind time_hints from ASR phrase_matches[].time_hint so frame_retriever inspects "
                    "the quoted moment instead of the whole ASR clip."
                    % step_id
                )

    def _is_visual_conditioned_audio_task(self, task: Any) -> bool:
        text = _task_text(task)
        if not re.search(r"\b(sound|sounds|noise|noises|audio|heard|listen|bang|beep|chirp|music)\b", text):
            return False
        return re.search(
            r"\b(when|while|during)\b.{0,80}\b(using|use|uses|used|doing|holding|showing|squeezing|pouring|opening|closing|playing|touching|eating|drinking|wearing|carrying|handling)\b",
            text,
        ) is not None

    def _audio_step_has_prior_visual_scope(self, step: PlanStep, step_by_id: Dict[int, PlanStep]) -> bool:
        for ref in list((step.input_refs or {}).get("clips") or []):
            source_step = step_by_id.get(int(ref.step_id))
            if source_step is not None and str(source_step.tool_name or "").strip() == "visual_temporal_grounder":
                return True
        for clip in list((step.inputs or {}).get("clips") or []):
            if not isinstance(clip, dict):
                continue
            metadata = clip.get("metadata") if isinstance(clip.get("metadata"), dict) else {}
            source = _normalize_text(metadata.get("source"))
            if "visual" in source or "visible" in source:
                return True
        return False

    def _validate_visual_conditioned_audio_order(self, task: Any, steps: List[PlanStep]) -> None:
        if not self._is_visual_conditioned_audio_task(task):
            return
        step_by_id = {int(step.step_id): step for step in list(steps or [])}
        for step in list(steps or []):
            if str(step.tool_name or "").strip() != "audio_temporal_grounder":
                continue
            if self._audio_step_has_prior_visual_scope(step, step_by_id):
                continue
            raise ValueError(
                "Plan step %s (audio_temporal_grounder) is invalid for this visual-conditioned audio task. "
                "Ground the visible condition first with visual_temporal_grounder, then pass those clips to audio_temporal_grounder."
                % int(step.step_id)
            )

    def _validate_retrieved_evidence_ids(self, steps: List[PlanStep], retrieved_context: Dict[str, Any] | None) -> None:
        available_ids = _retrieved_evidence_ids(retrieved_context)
        for step in list(steps or []):
            evidence_ids = [str(item).strip() for item in list((step.inputs or {}).get("evidence_ids") or []) if str(item).strip()]
            if not evidence_ids:
                continue
            missing = sorted(evidence_id for evidence_id in evidence_ids if evidence_id not in available_ids)
            if missing:
                raise ValueError(
                    "Plan step %s uses evidence_ids that were not retrieved for this planning round: %s."
                    % (int(step.step_id), ", ".join(missing))
                )

    def _validate_ocr_full_frame_policy(self, steps: List[PlanStep]) -> None:
        step_by_id = {int(step.step_id): step for step in list(steps or [])}
        for step in list(steps or []):
            if str(step.tool_name or "").strip() != "ocr":
                continue
            for refs in dict(step.input_refs or {}).values():
                for ref in list(refs or []):
                    source_step = step_by_id.get(int(ref.step_id))
                    if source_step is None or str(source_step.tool_name or "").strip() != "spatial_grounder":
                        continue
                    raise ValueError(
                        "Plan step %s routes spatial_grounder output into OCR. Do not use spatial_grounder "
                        "as an OCR cropper; OCR must use complete frames or grounded clips directly."
                        % int(step.step_id)
                    )

    def _step_sort_key(self, step: PlanStep) -> Tuple[Any, ...]:
        return (
            tuple(_source_ids(step)),
            str(step.tool_name or "").strip(),
            str(step.purpose or "").strip().lower(),
            stable_json_dumps(dict(step.inputs or {})),
            stable_json_dumps(
                [
                    {
                        "target_field": target_field,
                        "step_id": ref.step_id,
                        "field_path": ref.field_path,
                    }
                    for target_field, refs in sorted(dict(step.input_refs or {}).items())
                    for ref in list(refs or [])
                ]
            ),
            int(step.step_id),
        )

    def _canonical_topological_order(self, steps: List[PlanStep]) -> List[PlanStep]:
        pending = list(steps or [])
        ordered: List[PlanStep] = []
        resolved_ids = set()

        while pending:
            pending_ids = {int(step.step_id) for step in pending}
            ready = [
                step
                for step in pending
                if all(source_id in resolved_ids or source_id not in pending_ids for source_id in _source_ids(step))
            ]
            if not ready:
                unresolved = {
                    int(step.step_id): sorted(source_id for source_id in _source_ids(step) if source_id in pending_ids)
                    for step in pending
                }
                raise ValueError(
                    "Execution plan contains a dependency cycle or invalid ordering: %s" % unresolved
                )
            ready = sorted(ready, key=self._step_sort_key)
            ready_ids = {int(step.step_id) for step in ready}
            for step in ready:
                ordered.append(step)
                resolved_ids.add(int(step.step_id))
            pending = [step for step in pending if int(step.step_id) not in ready_ids]

        return ordered

    def _resequence(self, steps: List[PlanStep]) -> List[PlanStep]:
        step_id_map = {step.step_id: index + 1 for index, step in enumerate(steps)}
        resequenced: List[PlanStep] = []
        for index, step in enumerate(steps, start=1):
            input_refs: Dict[str, List[InputRef]] = {}
            for target_field, refs in sorted(dict(step.input_refs or {}).items()):
                input_refs[target_field] = [
                    InputRef(
                        step_id=step_id_map.get(ref.step_id, ref.step_id),
                        field_path=ref.field_path,
                    )
                    for ref in list(refs or [])
                ]
            resequenced.append(
                PlanStep(
                    step_id=index,
                    tool_name=step.tool_name,
                    purpose=step.purpose,
                    inputs=dict(step.inputs or {}),
                    input_refs=input_refs,
                    expected_outputs=dict(step.expected_outputs or {}),
                )
            )
        return resequenced

    def _validate_resequenced_order(self, steps: List[PlanStep]) -> None:
        for step in list(steps or []):
            step_id = int(step.step_id)
            for refs in dict(step.input_refs or {}).values():
                for ref in list(refs or []):
                    if int(ref.step_id) >= step_id:
                        raise ValueError(
                            "Plan normalization left step %s with non-earlier input_ref source step %s."
                            % (step_id, ref.step_id)
                        )

    def normalize(self, task, plan: ExecutionPlan, retrieved_context: Dict[str, Any] | None = None) -> ExecutionPlan:
        normalized_steps = [self._normalize_step(step) for step in list(plan.steps or [])]
        self._validate_references(normalized_steps)
        self._validate_grounded_clip_bridge_policy(normalized_steps)
        self._validate_ocr_full_frame_policy(normalized_steps)
        self._validate_generic_purpose_context_contract(normalized_steps)
        self._validate_tool_context_contracts(normalized_steps)
        self._validate_frame_retriever_time_contracts(normalized_steps)
        self._validate_visual_conditioned_audio_order(task, normalized_steps)
        self._validate_retrieved_evidence_ids(normalized_steps, retrieved_context)
        ordered_steps = self._canonical_topological_order(normalized_steps)
        resequenced_steps = self._resequence(ordered_steps)
        self._validate_resequenced_order(resequenced_steps)
        return ExecutionPlan(
            strategy=_normalize_text(plan.strategy),
            steps=resequenced_steps,
            refinement_instructions=_normalize_text(plan.refinement_instructions),
        )
