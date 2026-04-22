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
        "text": "text_contexts",
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
}


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
        plural_map = {
            "clip": "clips",
            "frame": "frames",
            "region": "regions",
            "transcript": "transcripts",
            "text_context": "text_contexts",
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
            if canonical_key not in allowed_fields:
                continue
            normalized_arguments[canonical_key] = _merge_field_value(
                canonical_key,
                normalized_arguments.get(canonical_key),
                dict(step.arguments or {}).get(key),
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
            {
                int(item)
                for item in list(step.depends_on or [])
                if int(item) != int(step.step_id)
            }
        )

        return PlanStep(
            step_id=step.step_id,
            tool_name=tool_name,
            purpose=_normalize_text(step.purpose),
            arguments=normalized_arguments,
            input_refs=normalized_refs,
            depends_on=depends_on,
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
                ready = list(pending)
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

    def normalize(self, task, plan: ExecutionPlan) -> ExecutionPlan:
        del task
        normalized_steps = [self._normalize_step(step) for step in list(plan.steps or [])]
        ordered_steps = self._canonical_topological_order(normalized_steps)
        return ExecutionPlan(
            strategy=_normalize_text(plan.strategy),
            use_summary=bool(plan.use_summary),
            steps=self._resequence(ordered_steps),
            refinement_instructions=_normalize_text(plan.refinement_instructions),
        )
