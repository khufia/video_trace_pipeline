from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import ValidationError

from ..common import assign_path, hash_payload, sanitize_for_persistence, traverse_path, write_json
from ..schemas import EvidenceEntry, ToolResult
from ..storage import EvidenceLedger
from ..tools import ObservationExtractor
from ..tools.extractors import build_evidence_text_digest, evidence_temporal_bounds
from ..tools.base import ToolExecutionContext
from ..tools.specs import tool_implementation


def augment_dependency_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    return dict(payload)


def request_model_field_names(model_cls) -> List[str]:
    if hasattr(model_cls, "model_fields"):
        return list(getattr(model_cls, "model_fields").keys())
    if hasattr(model_cls, "__fields__"):
        return list(getattr(model_cls, "__fields__").keys())
    return []


_LIST_ARGUMENT_FIELDS = frozenset({"clips", "frames", "regions", "transcripts", "text_contexts", "evidence_ids", "time_hints"})
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


def _iter_step_input_refs(step):
    for target_field, refs in dict(getattr(step, "input_refs", {}) or {}).items():
        for ref in list(refs or []):
            yield str(target_field or "").strip(), ref


class PlanExecutor(object):
    def __init__(self, tool_registry, extractor: ObservationExtractor, models_config):
        self.tool_registry = tool_registry
        self.extractor = extractor
        self.models_config = models_config

    def _resolve_arguments(self, step, step_outputs: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        resolved = dict(step.inputs or {})
        for target_field, ref in _iter_step_input_refs(step):
            source_obj = step_outputs.get(ref.step_id)
            if source_obj is None:
                raise KeyError("Missing step output for step %s" % ref.step_id)
            value = traverse_path(source_obj, ref.field_path)
            if value is None:
                raise KeyError(
                    "Could not resolve %s from step %s" % (ref.field_path, ref.step_id)
                )
            existing_value = traverse_path(resolved, target_field)
            value = _merge_dependency_values(existing_value, value, target_field)
            assign_path(resolved, target_field, value)
        return resolved

    def _record_tool_step_failure(
        self,
        *,
        step,
        exc: Exception,
        execution_context: ToolExecutionContext,
        evidence_ledger: EvidenceLedger,
        request_payload: Dict[str, Any],
        error_type: str,
        execution_mode: str,
        summary_prefix: str,
        progress_reporter=None,
        round_index: int | None = None,
    ) -> Dict[str, Any]:
        error_text = str(exc)
        summary = "%s: %s" % (summary_prefix, error_text)
        request_payload = sanitize_for_persistence(request_payload)
        request_hash = hash_payload(
            {
                "tool": step.tool_name,
                "step_id": step.step_id,
                "purpose": step.purpose,
                "error_type": error_type,
                "error": error_text,
                "request": request_payload,
            }
        )
        timing_metadata = {
            "started_at_utc": None,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": 0.0,
            "cache_wait_s": 0.0,
            "execution_mode": execution_mode,
        }
        tool_result = ToolResult(
            tool_name=step.tool_name,
            ok=False,
            data={"error": error_text, "error_type": error_type},
            raw_output_text=summary,
            request_hash=request_hash,
            cache_hit=False,
            summary=summary,
            metadata={"error": error_text, "error_type": error_type, "timing": timing_metadata},
        )
        evidence_entry = EvidenceEntry(
            evidence_id="ev_%02d_%s" % (step.step_id, request_hash[:8]),
            tool_name=step.tool_name,
            evidence_text=summary,
            status="unknown",
            artifact_refs=[],
            observation_ids=[],
            metadata={"request_hash": request_hash, "cache_hit": False, "error_type": error_type},
        )
        evidence_ledger.append(evidence_entry, [])

        step_dir = execution_context.run.tool_step_dir(step.step_id, step.tool_name, round_index=round_index)
        write_json(step_dir / "request.json", request_payload)
        write_json(step_dir / "result.json", sanitize_for_persistence(tool_result.dict()))
        write_json(step_dir / "timing.json", sanitize_for_persistence(timing_metadata))
        write_json(step_dir / "artifact_refs.json", [])
        write_json(step_dir / "observations.json", [])

        if progress_reporter is not None and hasattr(progress_reporter, "on_tool_end"):
            progress_reporter.on_tool_end(
                round_index=round_index or 0,
                step_id=step.step_id,
                tool_name=step.tool_name,
                result_payload=sanitize_for_persistence(tool_result.dict()),
                observations=[],
                step_dir=str(step_dir),
            )

        return {
            "step_id": step.step_id,
            "tool_name": step.tool_name,
            "purpose": step.purpose,
            "request": request_payload,
            "result": sanitize_for_persistence(tool_result.dict()),
            "evidence_entry": evidence_entry.dict(),
            "observations": [],
        }

    def _record_unresolved_dependency_failure(
        self,
        *,
        step,
        exc: Exception,
        execution_context: ToolExecutionContext,
        evidence_ledger: EvidenceLedger,
        progress_reporter=None,
        round_index: int | None = None,
    ) -> Dict[str, Any]:
        request_payload = {
            "inputs": sanitize_for_persistence(dict(step.inputs or {})),
            "input_refs": sanitize_for_persistence(
                {
                    target_field: [
                        item.model_dump() if hasattr(item, "model_dump") else item.dict() if hasattr(item, "dict") else item
                        for item in list(refs or [])
                    ]
                    for target_field, refs in dict(getattr(step, "input_refs", {}) or {}).items()
                }
            ),
        }
        return self._record_tool_step_failure(
            step=step,
            exc=exc,
            execution_context=execution_context,
            evidence_ledger=evidence_ledger,
            request_payload=request_payload,
            error_type="unresolved_dependency",
            execution_mode="unresolved_dependency",
            summary_prefix="Tool step skipped because a declared input reference could not be resolved",
            progress_reporter=progress_reporter,
            round_index=round_index,
        )

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
            try:
                resolved_arguments = self._resolve_arguments(step, step_outputs)
            except KeyError as exc:
                failure_record = self._record_unresolved_dependency_failure(
                    step=step,
                    exc=exc,
                    execution_context=execution_context,
                    evidence_ledger=evidence_ledger,
                    progress_reporter=progress_reporter,
                    round_index=round_index,
                )
                step_outputs[step.step_id] = {
                    "ok": False,
                    "summary": failure_record["result"].get("summary"),
                    "raw_output_text": failure_record["result"].get("raw_output_text"),
                    "error": failure_record["result"].get("data", {}).get("error"),
                    "error_type": failure_record["result"].get("data", {}).get("error_type"),
                }
                results.append(failure_record)
                continue
            adapter = self.tool_registry.get_adapter(step.tool_name)
            try:
                request = adapter.parse_request(resolved_arguments)
            except (ValidationError, ValueError) as exc:
                failure_record = self._record_tool_step_failure(
                    step=step,
                    exc=exc,
                    execution_context=execution_context,
                    evidence_ledger=evidence_ledger,
                    request_payload={
                        "resolved_arguments": sanitize_for_persistence(resolved_arguments),
                        "argument_keys": sorted(resolved_arguments.keys()),
                        "input_refs": sanitize_for_persistence(
                            {
                                target_field: [
                                    item.model_dump() if hasattr(item, "model_dump") else item.dict() if hasattr(item, "dict") else item
                                    for item in list(refs or [])
                                ]
                                for target_field, refs in dict(getattr(step, "input_refs", {}) or {}).items()
                            }
                        ),
                    },
                    error_type="invalid_request",
                    execution_mode="invalid_request",
                    summary_prefix="Tool step skipped because the resolved request did not satisfy the tool schema",
                    progress_reporter=progress_reporter,
                    round_index=round_index,
                )
                step_outputs[step.step_id] = {
                    "ok": False,
                    "summary": failure_record["result"].get("summary"),
                    "raw_output_text": failure_record["result"].get("raw_output_text"),
                    "error": failure_record["result"].get("data", {}).get("error"),
                    "error_type": failure_record["result"].get("data", {}).get("error_type"),
                }
                results.append(failure_record)
                continue
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
            step_dir = execution_context.run.tool_step_dir(step.step_id, step.tool_name, round_index=round_index)
            write_json(step_dir / "request.json", request.dict())
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
                "execution_mode": "executed",
            }
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
                "cache_wait_s": 0.0,
                "execution_mode": "executed",
            }
            tool_result.metadata = {**dict(tool_result.metadata or {}), "timing": timing_metadata}
            observations = self.extractor.extract(tool_result)
            tool_result.metadata = {**dict(tool_result.metadata or {}), "timing": timing_metadata}
            evidence_text = build_evidence_text_digest(tool_result, observations)
            tool_result.summary = evidence_text
            tool_result.metadata = {
                **dict(tool_result.metadata or {}),
                "digest_source_observation_ids": [item.observation_id for item in list(observations or [])],
                "digest_version": "observations_v1",
            }

            evidence_entry = EvidenceEntry(
                evidence_id="ev_%02d_%s" % (step.step_id, request_hash[:8]),
                tool_name=step.tool_name,
                evidence_text=evidence_text,
                confidence=tool_result.metadata.get("confidence") if isinstance(tool_result.metadata, dict) else None,
                status="candidate",
                **evidence_temporal_bounds(observations),
                artifact_refs=tool_result.artifact_refs,
                observation_ids=[item.observation_id for item in observations],
                metadata={
                    "request_hash": request_hash,
                    "cache_hit": tool_result.cache_hit,
                    "digest_source_observation_ids": [item.observation_id for item in list(observations or [])],
                    "digest_version": "observations_v1",
                },
            )
            evidence_ledger.append(evidence_entry, observations)
            write_json(step_dir / "result.json", sanitize_for_persistence(tool_result.dict()))
            write_json(step_dir / "timing.json", sanitize_for_persistence(timing_metadata))
            write_json(step_dir / "artifact_refs.json", [item.dict() for item in tool_result.artifact_refs])
            write_json(step_dir / "observations.json", [sanitize_for_persistence(item.dict()) for item in observations])

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
        return results
