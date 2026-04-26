from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..common import stable_json_dumps
from ..schemas import ArgumentBinding, ExecutionPlan, InputRef, PlanStep
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


def _binding_signature(binding: ArgumentBinding) -> Tuple[str, int, str]:
    return (
        str(binding.target_field or "").strip(),
        int(binding.source.step_id),
        str(binding.source.field_path or "").strip(),
    )


class ExecutionPlanNormalizer(object):
    VERSION = "deterministic_v1"

    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def _allowed_fields(self, tool_name: str) -> List[str]:
        adapter = self.tool_registry.get_adapter(tool_name)
        return [name for name in request_model_field_names(getattr(adapter, "request_model", None)) if name != "tool_name"]

    def _normalized_field_name(self, field_name: str) -> str:
        return str(field_name or "").strip()

    def _validate_asr_transcript_contract(self, *, step: PlanStep, binding: ArgumentBinding, source_tool_name: str) -> None:
        if str(step.tool_name or "").strip() != "generic_purpose":
            return
        if str(source_tool_name or "").strip() != "asr":
            return
        target_field = str(binding.target_field or "").strip()
        field_path = str(binding.source.field_path or "").strip()
        if target_field != "transcripts" or field_path != "transcripts":
            raise ValueError(
                "Plan step %s must bind ASR output to generic_purpose via transcripts -> transcripts, not %s -> %s."
                % (int(step.step_id), field_path or "<empty>", target_field or "<empty>")
            )

    def _validate_input_ref_contract(self, *, step: PlanStep, binding: ArgumentBinding) -> None:
        step_id = int(step.step_id)
        target_field = str(binding.target_field or "").strip()
        field_path = str(binding.source.field_path or "").strip()
        if target_field == "evidence_ids":
            raise ValueError(
                "Plan step %s binds input_refs into evidence_ids. Current-plan steps do not emit bindable evidence ids; "
                "pass frames, clips, transcripts, or text_contexts instead."
                % step_id
            )
        if target_field in _STRUCTURAL_INPUT_REF_FIELDS and field_path != target_field:
            raise ValueError(
                "Plan step %s must bind %s via %s -> %s, not %s -> %s."
                % (step_id, target_field, target_field, target_field, field_path or "<empty>", target_field)
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

        normalized_arguments: Dict[str, Any] = {}
        for key in sorted(dict(step.arguments or {})):
            canonical_key = self._normalized_field_name(key)
            raw_value = dict(step.arguments or {}).get(key)
            if canonical_key == "tool_name":
                continue
            if tool_name == "generic_purpose" and canonical_key == "text_contexts" and _looks_like_transcript_payload(raw_value):
                raise ValueError(
                    "Plan step %s passes transcript payloads via text_contexts. Use transcripts instead."
                    % int(step.step_id)
                )
            if canonical_key not in allowed_fields:
                raise ValueError(
                    "Plan step %s uses unexpected argument %r for %s. Use canonical request fields only: %s."
                    % (
                        int(step.step_id),
                        canonical_key,
                        tool_name,
                        ", ".join(sorted(allowed_fields)),
                    )
                )
            normalized_arguments[canonical_key] = _merge_field_value(
                canonical_key,
                normalized_arguments.get(canonical_key),
                raw_value,
            )
        normalized_arguments = {
            key: normalized_arguments[key]
            for key in sorted(normalized_arguments)
        }
        if "query" in normalized_arguments:
            normalized_arguments["query"] = _normalize_text(normalized_arguments.get("query"))

        normalized_refs: List[ArgumentBinding] = []
        seen_refs = set()
        for binding in list(step.input_refs or []):
            target_field = self._normalized_field_name(binding.target_field)
            if target_field not in allowed_fields:
                raise ValueError(
                    "Plan step %s uses unexpected input_ref target_field %r for %s. Use canonical request fields only: %s."
                    % (
                        int(step.step_id),
                        target_field,
                        tool_name,
                        ", ".join(sorted(allowed_fields)),
                    )
                )
            normalized_binding = ArgumentBinding(
                target_field=target_field,
                source=InputRef(
                    step_id=binding.source.step_id,
                    field_path=_normalize_text(binding.source.field_path),
                ),
            )
            signature = _binding_signature(normalized_binding)
            if signature in seen_refs:
                continue
            seen_refs.add(signature)
            normalized_refs.append(normalized_binding)
        normalized_refs = sorted(normalized_refs, key=_binding_signature)

        depends_on = sorted(
            {int(item) for item in list(step.depends_on or [])}
            | {int(binding.source.step_id) for binding in normalized_refs}
        )

        return PlanStep(
            step_id=step.step_id,
            tool_name=tool_name,
            purpose=_normalize_text(step.purpose),
            arguments=normalized_arguments,
            input_refs=normalized_refs,
            depends_on=depends_on,
        )

    def _validate_references(self, steps: List[PlanStep]) -> None:
        ordered_step_ids = [int(step.step_id) for step in list(steps or [])]
        if len(set(ordered_step_ids)) != len(ordered_step_ids):
            raise ValueError("Execution plan contains duplicate step_ids: %s" % ordered_step_ids)
        step_ids = set(ordered_step_ids)
        step_by_id = {int(step.step_id): step for step in list(steps or [])}
        for step in list(steps or []):
            step_id = int(step.step_id)
            for binding in list(step.input_refs or []):
                source_id = int(binding.source.step_id)
                if source_id not in step_ids:
                    raise ValueError(
                        "Plan step %s references missing input source step %s via %s."
                        % (step_id, source_id, binding.target_field)
                    )
                if source_id == step_id:
                    raise ValueError(
                        "Plan step %s has a self-referential input_ref for %s."
                        % (step_id, binding.target_field)
                    )
                source_step = step_by_id.get(source_id)
                if source_step is not None:
                    self._validate_input_ref_contract(step=step, binding=binding)
                    self._validate_asr_transcript_contract(
                        step=step,
                        binding=binding,
                        source_tool_name=source_step.tool_name,
                    )
            for dep in list(step.depends_on or []):
                dep_id = int(dep)
                if dep_id not in step_ids:
                    raise ValueError("Plan step %s depends on missing step %s." % (step_id, dep_id))
                if dep_id == step_id:
                    raise ValueError("Plan step %s depends on itself." % step_id)

    def _validate_generic_purpose_context_contract(self, steps: List[PlanStep]) -> None:
        for step in list(steps or []):
            if str(step.tool_name or "").strip() != "generic_purpose":
                continue
            has_argument_context = any(
                _normalize_text((step.arguments or {}).get(field_name))
                if isinstance((step.arguments or {}).get(field_name), str)
                else bool((step.arguments or {}).get(field_name))
                for field_name in _GENERIC_PURPOSE_CONTEXT_FIELDS
            )
            has_bound_context = bool(step.input_refs or []) or bool(step.depends_on or [])
            if has_argument_context or has_bound_context:
                continue
            raise ValueError(
                "Plan step %s is a context-free generic_purpose request. "
                "Pass frames, clips, transcripts, text_contexts, evidence_ids, or an earlier dependency explicitly."
                % int(step.step_id)
            )

    def _step_sort_key(self, step: PlanStep) -> Tuple[Any, ...]:
        return (
            tuple(int(item) for item in list(step.depends_on or [])),
            str(step.tool_name or "").strip(),
            str(step.purpose or "").strip().lower(),
            stable_json_dumps(dict(step.arguments or {})),
            stable_json_dumps(
                [
                    {
                        "target_field": binding.target_field,
                        "step_id": binding.source.step_id,
                        "field_path": binding.source.field_path,
                    }
                    for binding in list(step.input_refs or [])
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
                if all(int(dep) in resolved_ids or int(dep) not in pending_ids for dep in list(step.depends_on or []))
            ]
            if not ready:
                unresolved = {
                    int(step.step_id): sorted(int(dep) for dep in list(step.depends_on or []) if int(dep) in pending_ids)
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
            resequenced.append(
                PlanStep(
                    step_id=index,
                    tool_name=step.tool_name,
                    purpose=step.purpose,
                    arguments=dict(step.arguments or {}),
                    input_refs=[
                        ArgumentBinding(
                            target_field=binding.target_field,
                            source=InputRef(
                                step_id=step_id_map.get(binding.source.step_id, binding.source.step_id),
                                field_path=binding.source.field_path,
                            ),
                        )
                        for binding in list(step.input_refs or [])
                    ],
                    depends_on=sorted(step_id_map.get(dep, dep) for dep in list(step.depends_on or [])),
                )
            )
        return resequenced

    def _validate_resequenced_order(self, steps: List[PlanStep]) -> None:
        for step in list(steps or []):
            step_id = int(step.step_id)
            for binding in list(step.input_refs or []):
                if int(binding.source.step_id) >= step_id:
                    raise ValueError(
                        "Plan normalization left step %s with non-earlier input_ref source step %s."
                        % (step_id, binding.source.step_id)
                    )
            for dep in list(step.depends_on or []):
                if int(dep) >= step_id:
                    raise ValueError(
                        "Plan normalization left step %s with non-earlier dependency step %s."
                        % (step_id, dep)
                    )

    def normalize(self, task, plan: ExecutionPlan, preprocess_planning_memory: Dict[str, Any] | None = None) -> ExecutionPlan:
        del task
        del preprocess_planning_memory
        normalized_steps = [self._normalize_step(step) for step in list(plan.steps or [])]
        self._validate_references(normalized_steps)
        self._validate_generic_purpose_context_contract(normalized_steps)
        ordered_steps = self._canonical_topological_order(normalized_steps)
        resequenced_steps = self._resequence(ordered_steps)
        self._validate_resequenced_order(resequenced_steps)
        return ExecutionPlan(
            strategy=_normalize_text(plan.strategy),
            use_summary=bool(plan.use_summary),
            steps=resequenced_steps,
            refinement_instructions=_normalize_text(plan.refinement_instructions),
        )
