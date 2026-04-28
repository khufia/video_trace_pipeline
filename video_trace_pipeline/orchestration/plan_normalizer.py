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


def _ref_signature(target_field: str, ref: InputRef) -> Tuple[str, int, str]:
    return (
        str(target_field or "").strip(),
        int(ref.step_id),
        str(ref.field_path or "").strip(),
    )


class ExecutionPlanNormalizer(object):
    VERSION = "field_keyed_v2"

    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def _allowed_fields(self, tool_name: str) -> List[str]:
        adapter = self.tool_registry.get_adapter(tool_name)
        return [name for name in request_model_field_names(getattr(adapter, "request_model", None)) if name != "tool_name"]

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
            raise ValueError(
                "Plan step %s binds input_refs into time_hints. Use literal strings in inputs.time_hints when the "
                "timestamps are known from preprocess or retrieved context; otherwise pass clips explicitly."
                % step_id
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
        del task
        del retrieved_context
        normalized_steps = [self._normalize_step(step) for step in list(plan.steps or [])]
        self._validate_references(normalized_steps)
        self._validate_generic_purpose_context_contract(normalized_steps)
        ordered_steps = self._canonical_topological_order(normalized_steps)
        resequenced_steps = self._resequence(ordered_steps)
        self._validate_resequenced_order(resequenced_steps)
        return ExecutionPlan(
            strategy=_normalize_text(plan.strategy),
            steps=resequenced_steps,
            refinement_instructions=_normalize_text(plan.refinement_instructions),
        )
