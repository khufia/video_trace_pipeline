from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..common import stable_json_dumps
from ..schemas import ArgumentBinding, ExecutionPlan, InputRef, PlanStep
from .executor import request_model_field_names


_COMMON_ARGUMENT_ALIASES = {
    "prompt": "query",
    "instruction": "query",
    "question": "query",
}

_TOOL_ARGUMENT_ALIASES = {
    "visual_temporal_grounder": {
        "k": "top_k",
        "max_segments": "top_k",
        "num_segments": "top_k",
    },
    "frame_retriever": {
        "k": "num_frames",
        "top_k": "num_frames",
        "count": "num_frames",
        "max_frames": "num_frames",
    },
    "ocr": {
        "prompt": "query",
    },
    "generic_purpose": {
        "evidence": "text_contexts",
        "evidence_id": "evidence_ids",
        "text": "text_contexts",
        "text_context": "text_contexts",
        "texts": "text_contexts",
        "ocr_text": "text_contexts",
        "ocr_texts": "text_contexts",
    },
}

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
        "clip",
        "clips",
        "frame",
        "frames",
        "transcript",
        "transcripts",
        "text_contexts",
        "evidence_ids",
    }
)

_GENERIC_PURPOSE_REUSE_HINTS = (
    "already grounded",
    "earlier clip",
    "earlier frame",
    "grounded frame",
    "grounded frames",
    "previous frame",
    "previous frames",
    "previously grounded",
    "previously retrieved",
    "prior evidence",
    "provided evidence",
    "retrieved frame",
    "retrieved frames",
    "supplied evidence",
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

    def _canonical_field_name(self, tool_name: str, field_name: str) -> str:
        aliases = dict(_COMMON_ARGUMENT_ALIASES)
        aliases.update(_TOOL_ARGUMENT_ALIASES.get(tool_name, {}))
        return aliases.get(str(field_name or "").strip(), str(field_name or "").strip())

    def _preferred_target_field(self, tool_name: str, target_field: str, source_field_path: str) -> str:
        allowed_fields = set(self._allowed_fields(tool_name))
        field_name = self._canonical_field_name(tool_name, target_field)
        source_tail = str(source_field_path or "").split(".")[-1].strip().lower()
        if tool_name == "generic_purpose" and source_tail in {"transcript", "transcripts"} and "transcripts" in allowed_fields:
            return "transcripts"
        plural_map = {
            "clip": "clips",
            "frame": "frames",
            "region": "regions",
            "transcript": "transcripts",
            "time_hint": "time_hints",
            "text_context": "text_contexts",
            "evidence_id": "evidence_ids",
        }
        if field_name in plural_map and source_tail.endswith("s") and plural_map[field_name] in allowed_fields:
            return plural_map[field_name]
        return field_name

    def _normalize_step(self, step: PlanStep) -> PlanStep:
        tool_name = _normalize_text(step.tool_name)
        allowed_fields = set(self._allowed_fields(tool_name))

        normalized_arguments: Dict[str, Any] = {}
        for key in sorted(dict(step.arguments or {})):
            canonical_key = self._canonical_field_name(tool_name, key)
            raw_value = dict(step.arguments or {}).get(key)
            if (
                tool_name == "generic_purpose"
                and canonical_key == "text_contexts"
                and "transcripts" in allowed_fields
                and _looks_like_transcript_payload(raw_value)
            ):
                canonical_key = "transcripts"
            if canonical_key not in allowed_fields:
                continue
            normalized_arguments[canonical_key] = _merge_field_value(
                canonical_key,
                normalized_arguments.get(canonical_key),
                raw_value,
            )
        normalized_arguments = {
            key: normalized_arguments[key]
            for key in sorted(normalized_arguments)
        }

        normalized_refs: List[ArgumentBinding] = []
        seen_refs = set()
        for binding in list(step.input_refs or []):
            target_field = self._preferred_target_field(tool_name, binding.target_field, binding.source.field_path)
            if target_field not in allowed_fields:
                continue
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
        step_ids = {int(step.step_id) for step in list(steps or [])}
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
            haystack = " ".join(
                part
                for part in (
                    str(step.purpose or "").strip().lower(),
                    str((step.arguments or {}).get("query") or "").strip().lower(),
                )
                if part
            )
            if not any(hint in haystack for hint in _GENERIC_PURPOSE_REUSE_HINTS):
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
                "Plan step %s is a context-free generic_purpose reuse follow-up. "
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

    def normalize(self, task, plan: ExecutionPlan) -> ExecutionPlan:
        del task
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
