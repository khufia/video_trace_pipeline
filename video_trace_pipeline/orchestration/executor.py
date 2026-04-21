from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ValidationError

from ..common import assign_path, hash_payload, sanitize_for_persistence, traverse_path, write_json, write_text
from ..schemas import AtomicObservation, EvidenceEntry, ToolResult
from ..storage import EvidenceLedger, SharedEvidenceCache
from ..tools import ObservationExtractor
from ..tools.base import ToolExecutionContext


def render_tool_summary_markdown(step_id: int, tool_name: str, tool_result: ToolResult, observations: List[AtomicObservation]) -> str:
    lines = ["# Tool Step %02d - %s" % (int(step_id), tool_name), ""]
    lines.append("## Summary")
    lines.append("")
    lines.append(tool_result.summary or "")
    lines.append("")
    lines.append("## Observations")
    lines.append("")
    for item in observations:
        lines.append("- %s" % item.atomic_text)
    lines.append("")
    return "\n".join(lines)


def augment_dependency_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    augmented = dict(payload)
    list_aliases = [
        ("clips", "segments"),
        ("segments", "clips"),
    ]
    for source_key, alias_key in list_aliases:
        values = augmented.get(source_key)
        if alias_key not in augmented and isinstance(values, list):
            augmented[alias_key] = list(values)
    singular_aliases = [
        ("clips", "clip"),
        ("frames", "frame"),
        ("detections", "detection"),
        ("regions", "region"),
        ("segments", "segment"),
        ("events", "event"),
        ("lines", "line"),
    ]
    for plural_key, singular_key in singular_aliases:
        values = augmented.get(plural_key)
        if singular_key not in augmented and isinstance(values, list) and values:
            augmented[singular_key] = values[0]
    if "best_frame" not in augmented:
        if isinstance(augmented.get("frame"), dict):
            augmented["best_frame"] = augmented["frame"]
        elif isinstance(augmented.get("frames"), list) and augmented["frames"]:
            augmented["best_frame"] = augmented["frames"][0]
    if "puzzle_bbox" not in augmented:
        if isinstance(augmented.get("region"), dict):
            augmented["puzzle_bbox"] = augmented["region"]
        elif isinstance(augmented.get("regions"), list) and augmented["regions"]:
            augmented["puzzle_bbox"] = augmented["regions"][0]
    if "analysis" not in augmented:
        for candidate_key in ("analysis", "response", "answer", "spatial_description", "summary", "text"):
            candidate_value = augmented.get(candidate_key)
            if isinstance(candidate_value, str) and candidate_value.strip():
                augmented["analysis"] = candidate_value
                break
    return augmented


def _hydrate_media_refs(value: Any, video_id: str) -> Any:
    if isinstance(value, list):
        return [_hydrate_media_refs(item, video_id) for item in value]
    if not isinstance(value, dict):
        return value
    hydrated = {key: _hydrate_media_refs(item, video_id) for key, item in value.items()}
    has_clip_bounds = any(key in hydrated for key in ("start_s", "end_s", "time_start_s", "time_end_s"))
    if has_clip_bounds:
        if "start_s" not in hydrated and "time_start_s" in hydrated:
            hydrated["start_s"] = hydrated["time_start_s"]
        if "end_s" not in hydrated and "time_end_s" in hydrated:
            hydrated["end_s"] = hydrated["time_end_s"]
        if "start_s" in hydrated and "end_s" in hydrated and "video_id" not in hydrated:
            hydrated["video_id"] = video_id
    has_frame_timestamp = any(key in hydrated for key in ("timestamp_s", "timestamp"))
    if has_frame_timestamp and "timestamp_s" not in hydrated and "timestamp" in hydrated:
        hydrated["timestamp_s"] = hydrated["timestamp"]
    if has_frame_timestamp and "video_id" not in hydrated:
        hydrated["video_id"] = video_id
    return hydrated


def hydrate_arguments_with_task_context(arguments: Dict[str, Any], task) -> Dict[str, Any]:
    video_id = str(getattr(task, "video_id", None) or getattr(task, "sample_key", "")).strip()
    if not video_id:
        return dict(arguments or {})
    return _hydrate_media_refs(dict(arguments or {}), video_id)


def request_model_field_names(model_cls) -> List[str]:
    if hasattr(model_cls, "model_fields"):
        return list(getattr(model_cls, "model_fields").keys())
    if hasattr(model_cls, "__fields__"):
        return list(getattr(model_cls, "__fields__").keys())
    return []


class PlanExecutor(object):
    def __init__(self, tool_registry, evidence_cache: SharedEvidenceCache, extractor: ObservationExtractor, models_config):
        self.tool_registry = tool_registry
        self.evidence_cache = evidence_cache
        self.extractor = extractor
        self.models_config = models_config

    def _resolve_arguments(self, step, step_outputs: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        resolved = dict(step.arguments or {})
        for binding in step.input_refs:
            source_obj = step_outputs.get(binding.source.step_id)
            if source_obj is None:
                raise KeyError("Missing step output for step %s" % binding.source.step_id)
            value = traverse_path(source_obj, binding.source.field_path)
            if value is None:
                raise KeyError(
                    "Could not resolve %s from step %s" % (binding.source.field_path, binding.source.step_id)
                )
            assign_path(resolved, binding.target_field, value)
        return resolved

    def _autofill_dependency_inputs(self, step, adapter, resolved_arguments: Dict[str, Any], step_outputs: Dict[int, Dict[str, Any]]):
        payload = dict(resolved_arguments or {})
        model_fields = set(request_model_field_names(getattr(adapter, "request_model", None)))
        candidate_fields = [name for name in ("clip", "frame", "region", "transcript") if name in model_fields]
        for field_name in candidate_fields:
            if payload.get(field_name) is not None:
                continue
            for dependency_id in reversed(list(step.depends_on or [])):
                source_obj = step_outputs.get(dependency_id) or {}
                value = source_obj.get(field_name)
                if value is not None:
                    payload[field_name] = value
                    break
        return payload

    def execute_plan(
        self,
        plan,
        execution_context: ToolExecutionContext,
        evidence_ledger: EvidenceLedger,
        video_fingerprint: str,
    ) -> List[Dict[str, Any]]:
        results = []
        step_outputs = {}
        for step in sorted(plan.steps, key=lambda item: item.step_id):
            resolved_arguments = self._resolve_arguments(step, step_outputs)
            adapter = self.tool_registry.get_adapter(step.tool_name)
            resolved_arguments = self._autofill_dependency_inputs(step, adapter, resolved_arguments, step_outputs)
            resolved_arguments = hydrate_arguments_with_task_context(resolved_arguments, execution_context.task)
            try:
                request = adapter.parse_request(resolved_arguments)
            except ValidationError as exc:
                raise ValueError(
                    "Failed to parse request for step %s (%s) with argument keys %s: %s"
                    % (step.step_id, step.tool_name, sorted(resolved_arguments.keys()), exc)
                ) from exc
            request_hash = hash_payload(
                {
                    "tool": step.tool_name,
                    "request": request.dict(),
                    "video_fingerprint": video_fingerprint,
                    "prompt_version": self.models_config.tools[step.tool_name].prompt_version,
                }
            )
            cached = self.evidence_cache.load(step.tool_name, request_hash)
            if cached:
                tool_result = ToolResult.parse_obj(cached["result"])
                tool_result.cache_hit = True
                observations = [AtomicObservation.parse_obj(item) for item in cached["observations"]]
                summary_markdown = cached.get("summary_markdown") or render_tool_summary_markdown(
                    step.step_id, step.tool_name, tool_result, observations
                )
            else:
                tool_result = adapter.execute(request, execution_context)
                tool_result.request_hash = request_hash
                tool_result.cache_hit = False
                observations = self.extractor.extract(tool_result)
                summary_markdown = render_tool_summary_markdown(step.step_id, step.tool_name, tool_result, observations)
                self.evidence_cache.store(
                    tool_name=step.tool_name,
                    request_hash=request_hash,
                    manifest={
                        "tool_name": step.tool_name,
                        "request_hash": request_hash,
                        "prompt_version": self.models_config.tools[step.tool_name].prompt_version,
                        "request": sanitize_for_persistence(request.dict()),
                    },
                    result=sanitize_for_persistence(tool_result.dict()),
                    observations=[sanitize_for_persistence(item.dict()) for item in observations],
                    summary_markdown=summary_markdown,
                )

            evidence_entry = EvidenceEntry(
                evidence_id="ev_%02d_%s" % (step.step_id, request_hash[:8]),
                tool_name=step.tool_name,
                evidence_text=tool_result.summary or tool_result.raw_output_text or "",
                confidence=tool_result.metadata.get("confidence") if isinstance(tool_result.metadata, dict) else None,
                artifact_refs=tool_result.artifact_refs,
                observation_ids=[item.observation_id for item in observations],
                metadata={"request_hash": request_hash, "cache_hit": tool_result.cache_hit},
            )
            evidence_ledger.append(evidence_entry, observations)
            step_dir = execution_context.run.tool_step_dir(step.step_id, step.tool_name)
            write_json(step_dir / "request.json", sanitize_for_persistence(request.dict()))
            write_json(step_dir / "result.json", sanitize_for_persistence(tool_result.dict()))
            write_json(step_dir / "artifact_refs.json", [item.dict() for item in tool_result.artifact_refs])
            write_text(step_dir / "summary.md", summary_markdown)

            step_outputs[step.step_id] = augment_dependency_output(tool_result.data)
            results.append(
                {
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "purpose": step.purpose,
                    "request": sanitize_for_persistence(request.dict()),
                    "result": sanitize_for_persistence(tool_result.dict()),
                    "evidence_entry": evidence_entry.dict(),
                    "observations": [item.dict() for item in observations],
                }
            )
        evidence_ledger.render_readable_markdown()
        return results
