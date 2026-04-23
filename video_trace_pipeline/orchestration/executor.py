from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import ValidationError

from ..common import assign_path, hash_payload, sanitize_for_persistence, traverse_path, write_json, write_text
from ..schemas import AtomicObservation, EvidenceEntry, ToolResult
from ..storage import EvidenceLedger, SharedEvidenceCache
from ..tools import ObservationExtractor
from ..tools.extractors import evidence_temporal_bounds
from ..tools.base import ToolExecutionContext
from ..tools.specs import tool_implementation


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
    if "top_frame" not in augmented:
        if isinstance(augmented.get("best_frame"), dict):
            augmented["top_frame"] = augmented["best_frame"]
        elif isinstance(augmented.get("frame"), dict):
            augmented["top_frame"] = augmented["frame"]
        elif isinstance(augmented.get("frames"), list) and augmented["frames"]:
            augmented["top_frame"] = augmented["frames"][0]
    if "top_clip" not in augmented:
        if isinstance(augmented.get("clip"), dict):
            augmented["top_clip"] = augmented["clip"]
        elif isinstance(augmented.get("clips"), list) and augmented["clips"]:
            augmented["top_clip"] = augmented["clips"][0]
    if "best_clip" not in augmented:
        if isinstance(augmented.get("clip"), dict):
            augmented["best_clip"] = augmented["clip"]
        elif isinstance(augmented.get("top_clip"), dict):
            augmented["best_clip"] = augmented["top_clip"]
        elif isinstance(augmented.get("clips"), list) and augmented["clips"]:
            augmented["best_clip"] = augmented["clips"][0]
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


def fallback_dependency_value(payload: Dict[str, Any], field_path: str) -> Any:
    if not isinstance(payload, dict):
        return None
    lowered = str(field_path or "").strip().lower()
    if not lowered:
        return None
    clip_values = None
    for key in ("clips", "segments"):
        values = payload.get(key)
        if isinstance(values, list) and values:
            clip_values = values
            break
    if any(token in lowered for token in ("clip", "segment", "window")):
        for key in ("best_clip", "top_clip", "clip"):
            value = payload.get(key)
            if value is not None:
                return value
        if clip_values:
            return clip_values[0]
    if any(token in lowered for token in ("interval", "range", "timespan", "time_range")) and clip_values:
        semantic_index = None
        if any(token in lowered for token in ("cleanliness", "clean", "hygiene", "sanitary")):
            semantic_index = 0
        elif any(token in lowered for token in ("value", "dollar", "price", "pricing", "cost")):
            semantic_index = 1
        elif "first" in lowered:
            semantic_index = 0
        elif "second" in lowered:
            semantic_index = 1
        elif "third" in lowered:
            semantic_index = 2
        elif "fourth" in lowered:
            semantic_index = 3
        if semantic_index is not None and semantic_index < len(clip_values):
            return clip_values[semantic_index]
        return clip_values[0]
    if "frame" in lowered or "image" in lowered:
        for key in ("best_frame", "top_frame", "frame"):
            value = payload.get(key)
            if value is not None:
                return value
        values = payload.get("frames")
        if isinstance(values, list) and values:
            return values[0]
    if any(token in lowered for token in ("text", "ocr", "transcript", "analysis", "answer")):
        for key in ("text", "full_text", "analysis", "answer", "summary"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    if "artifact" in lowered:
        for key in ("source_artifact_refs", "artifact_refs"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                return value
    return None


def _normalize_clip_shorthand(value: Any, video_id: str, video_duration_s: float, field_hint: str | None) -> Any:
    if field_hint not in {"clip", "clips"} or not isinstance(value, str):
        return value
    normalized = "_".join(str(value or "").strip().lower().replace("-", " ").split())
    if normalized not in {"full_video", "whole_video", "entire_video"}:
        return value
    return {
        "video_id": video_id,
        "start_s": 0.0,
        "end_s": max(0.0, float(video_duration_s or 0.0)),
    }


def _task_video_duration_s(task) -> float:
    metadata = getattr(task, "metadata", {}) or {}
    for key in ("video_duration", "video_duration_s", "duration_s", "duration"):
        raw = metadata.get(key)
        if raw in (None, ""):
            continue
        try:
            return max(0.0, float(raw))
        except Exception:
            continue
    video_path = str(getattr(task, "video_path", "") or "").strip()
    if not video_path:
        return 0.0
    try:
        from ..tools.media import get_video_duration

        return max(0.0, float(get_video_duration(video_path) or 0.0))
    except Exception:
        return 0.0


def _payload_uses_clip_shorthand(value: Any, field_hint: str | None = None) -> bool:
    if isinstance(value, list):
        return any(_payload_uses_clip_shorthand(item, field_hint) for item in value)
    if isinstance(value, str) and field_hint in {"clip", "clips"}:
        normalized = "_".join(str(value or "").strip().lower().replace("-", " ").split())
        return normalized in {"full_video", "whole_video", "entire_video"}
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        child_hint = key if key in {"clip", "clips"} else field_hint
        if _payload_uses_clip_shorthand(item, child_hint):
            return True
    return False


def _hydrate_media_refs(value: Any, video_id: str, video_duration_s: float = 0.0, field_hint: str | None = None) -> Any:
    if isinstance(value, list):
        return [_hydrate_media_refs(item, video_id, video_duration_s, field_hint) for item in value]
    value = _normalize_clip_shorthand(value, video_id, video_duration_s, field_hint)
    if not isinstance(value, dict):
        return value
    hydrated = {}
    for key, item in value.items():
        child_hint = key if key in {"clip", "clips"} else field_hint
        hydrated[key] = _hydrate_media_refs(item, video_id, video_duration_s, child_hint)
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
    video_duration_s = _task_video_duration_s(task) if _payload_uses_clip_shorthand(arguments) else 0.0
    hydrated = _hydrate_media_refs(dict(arguments or {}), video_id, video_duration_s)
    hydrated.setdefault("video_id", video_id)
    return hydrated


def request_model_field_names(model_cls) -> List[str]:
    if hasattr(model_cls, "model_fields"):
        return list(getattr(model_cls, "model_fields").keys())
    if hasattr(model_cls, "__fields__"):
        return list(getattr(model_cls, "__fields__").keys())
    return []


_LIST_ARGUMENT_FIELDS = frozenset(
    {
        "clips",
        "frames",
        "regions",
        "transcripts",
        "text_contexts",
        "evidence_ids",
        "time_hints",
    }
)


def _coerce_dependency_list_value(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _merge_dependency_values(existing: Any, new_value: Any, target_field: str) -> Any:
    target_leaf = str(target_field or "").split(".")[-1].strip().lower()
    if target_leaf in _LIST_ARGUMENT_FIELDS:
        existing_items = _coerce_dependency_list_value(existing)
        new_items = _coerce_dependency_list_value(new_value)
        return existing_items + new_items
    if existing is None:
        return new_value
    if isinstance(existing, list):
        if isinstance(new_value, list):
            return list(existing) + list(new_value)
        return list(existing) + [new_value]
    if isinstance(new_value, list):
        return [existing] + list(new_value)
    return new_value


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
                value = fallback_dependency_value(source_obj, binding.source.field_path)
            if value is None:
                raise KeyError(
                    "Could not resolve %s from step %s" % (binding.source.field_path, binding.source.step_id)
                )
            existing_value = traverse_path(resolved, binding.target_field)
            value = _merge_dependency_values(existing_value, value, binding.target_field)
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
        if (
            str(getattr(step, "tool_name", "") or "").strip() == "asr"
            and "clip" in model_fields
            and payload.get("clip") is None
            and not payload.get("clips")
        ):
            payload["clip"] = "full_video"
        return payload

    def _load_cached_tool_result(self, cached: Dict[str, Any] | None, step_id: int, tool_name: str):
        if not cached:
            return None
        try:
            tool_result = ToolResult.parse_obj(cached.get("result") or {})
        except Exception:
            return None
        if not tool_result.ok:
            return None
        try:
            observations = [AtomicObservation.parse_obj(item) for item in cached.get("observations") or []]
        except Exception:
            return None
        tool_result.cache_hit = True
        summary_markdown = cached.get("summary_markdown") or render_tool_summary_markdown(
            step_id, tool_name, tool_result, observations
        )
        return tool_result, observations, summary_markdown

    def execute_plan(
        self,
        plan,
        execution_context: ToolExecutionContext,
        evidence_ledger: EvidenceLedger,
        video_fingerprint: str,
        progress_reporter=None,
        round_index: int | None = None,
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
            if progress_reporter is not None and hasattr(progress_reporter, "on_tool_start"):
                progress_reporter.on_tool_start(
                    round_index=round_index or 0,
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    purpose=step.purpose,
                    request_payload=sanitize_for_persistence(request.dict()),
                )
            request_hash = hash_payload(
                {
                    "tool": step.tool_name,
                    "request": request.dict(),
                    "video_fingerprint": video_fingerprint,
                    "implementation": tool_implementation(step.tool_name),
                    "model": self.models_config.tools[step.tool_name].model,
                    "prompt_version": self.models_config.tools[step.tool_name].prompt_version,
                    "extraction_version": getattr(self.extractor, "VERSION", "v1"),
                }
            )
            cached = self.evidence_cache.load(step.tool_name, request_hash)
            step_dir = execution_context.run.tool_step_dir(step.step_id, step.tool_name)
            write_json(step_dir / "request.json", sanitize_for_persistence(request.dict()))
            write_json(step_dir / "request_full.json", request.dict())
            if hasattr(adapter, "_runtime_payload"):
                try:
                    write_json(step_dir / "runtime.json", adapter._runtime_payload(execution_context))
                except Exception:
                    pass
            timing_metadata = {
                "started_at_utc": None,
                "finished_at_utc": None,
                "duration_s": 0.0,
                "cache_wait_s": 0.0,
                "execution_mode": "cache_hit",
            }
            cached_payload = self._load_cached_tool_result(cached, step.step_id, step.tool_name)
            if cached_payload is not None:
                tool_result, observations, summary_markdown = cached_payload
            else:
                cache_lock = self.evidence_cache.lock(step.tool_name, request_hash)
                cache_wait_started_at = time.perf_counter()
                with cache_lock:
                    timing_metadata["cache_wait_s"] = round(time.perf_counter() - cache_wait_started_at, 4)
                    cached = self.evidence_cache.load(step.tool_name, request_hash)
                    cached_payload = self._load_cached_tool_result(cached, step.step_id, step.tool_name)
                    if cached_payload is not None:
                        timing_metadata["execution_mode"] = "cache_hit_after_lock"
                        tool_result, observations, summary_markdown = cached_payload
                    else:
                        tool_started_at = datetime.now(timezone.utc)
                        execution_started_at = time.perf_counter()
                        tool_result = adapter.execute(request, execution_context)
                        tool_finished_at = datetime.now(timezone.utc)
                        tool_result.request_hash = request_hash
                        tool_result.cache_hit = False
                        timing_metadata = {
                            "started_at_utc": tool_started_at.isoformat(),
                            "finished_at_utc": tool_finished_at.isoformat(),
                            "duration_s": round(time.perf_counter() - execution_started_at, 4),
                            "cache_wait_s": timing_metadata["cache_wait_s"],
                            "execution_mode": "executed",
                        }
                        tool_result.metadata = {**dict(tool_result.metadata or {}), "timing": timing_metadata}
                        observations = self.extractor.extract(tool_result)
                        summary_markdown = render_tool_summary_markdown(step.step_id, step.tool_name, tool_result, observations)
                        self.evidence_cache.store_unlocked(
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
            tool_result.metadata = {**dict(tool_result.metadata or {}), "timing": timing_metadata}

            evidence_entry = EvidenceEntry(
                evidence_id="ev_%02d_%s" % (step.step_id, request_hash[:8]),
                tool_name=step.tool_name,
                evidence_text=tool_result.summary or tool_result.raw_output_text or "",
                confidence=tool_result.metadata.get("confidence") if isinstance(tool_result.metadata, dict) else None,
                **evidence_temporal_bounds(observations),
                artifact_refs=tool_result.artifact_refs,
                observation_ids=[item.observation_id for item in observations],
                metadata={"request_hash": request_hash, "cache_hit": tool_result.cache_hit},
            )
            evidence_ledger.append(evidence_entry, observations)
            write_json(step_dir / "result.json", sanitize_for_persistence(tool_result.dict()))
            write_json(step_dir / "timing.json", sanitize_for_persistence(timing_metadata))
            write_json(step_dir / "artifact_refs.json", [item.dict() for item in tool_result.artifact_refs])
            write_text(step_dir / "summary.md", summary_markdown)

            step_output_payload = dict(tool_result.data or {})
            serialized_artifacts = [item.dict() for item in tool_result.artifact_refs]
            if serialized_artifacts:
                step_output_payload.setdefault("artifact_refs", serialized_artifacts)
                step_output_payload.setdefault("source_artifact_refs", serialized_artifacts)
            if tool_result.summary:
                step_output_payload.setdefault("summary", tool_result.summary)
            if tool_result.raw_output_text:
                step_output_payload.setdefault("raw_output_text", tool_result.raw_output_text)
            step_outputs[step.step_id] = augment_dependency_output(step_output_payload)
            if progress_reporter is not None and hasattr(progress_reporter, "on_tool_end"):
                progress_reporter.on_tool_end(
                    round_index=round_index or 0,
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    result_payload=sanitize_for_persistence(tool_result.dict()),
                    observations=[sanitize_for_persistence(item.dict()) for item in observations],
                    step_dir=str(step_dir),
                )
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
