from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import ValidationError

from ..common import extract_json_object, has_meaningful_text, hash_payload, is_low_signal_text, sanitize_for_persistence
from ..model_cache import describe_model_resolution
from ..prompts.shared import render_frame_sequence_context
from ..runtime_devices import describe_device_mapping, resolve_device_label
from ..tools.media import get_video_duration
from ..tool_wrappers.local_multimodal import TimeChatCaptionerRunner
from ..tool_wrappers.timechat_dense_caption_runner import execute_payload as execute_dense_caption_payload
from ..tool_wrappers.qwen35vl_runner import execute_payload as execute_generic_purpose_payload
from ..tool_wrappers.spatial_grounder_runner import execute_payload as execute_spatial_grounder_payload
from ..tool_wrappers.verifier_runner import execute_payload as execute_verifier_payload
from ..tool_wrappers.shared import resolve_generation_controls, resolve_model_path, resolved_device_label
from ..tool_wrappers.timelens_runner import execute_payload as execute_visual_temporal_grounder_payload
from ..schemas import (
    ArtifactRef,
    AudioTemporalGrounderOutput,
    AudioTemporalGrounderRequest,
    ClipRef,
    DenseCaptionOutput,
    DenseCaptionRequest,
    FrameRef,
    FrameRetrieverOutput,
    FrameRetrieverRequest,
    GenericPurposeOutput,
    GenericPurposeRequest,
    OCRRequest,
    OCROutput,
    RegionRef,
    SpatialGrounderOutput,
    SpatialGrounderRequest,
    ToolResult,
    VerifierOutput,
    VerifierRequest,
    VisualTemporalGrounderOutput,
    VisualTemporalGrounderRequest,
)
from .base import ToolAdapter


def _safe_confidences(values: List[Any]) -> List[float]:
    cleaned: List[float] = []
    for value in list(values or []):
        try:
            if value is None:
                continue
            cleaned.append(float(value))
        except Exception:
            continue
    return cleaned


def _confidence_metadata(values: List[Any], *, kind: str) -> Dict[str, Any]:
    cleaned = _safe_confidences(values)
    if not cleaned:
        return {}
    max_confidence = max(cleaned)
    mean_confidence = sum(cleaned) / float(len(cleaned))
    return {
        "confidence": round(float(max_confidence), 4),
        "confidence_avg": round(float(mean_confidence), 4),
        "confidence_count": len(cleaned),
        "confidence_kind": str(kind or "confidence"),
    }


def _model_to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _time_hints_inside_clip(time_hints: List[str], clip: Any) -> List[str]:
    hints = [str(item).strip() for item in list(time_hints or []) if str(item).strip()]
    if not hints or clip is None:
        return hints

    clip_data = _model_to_dict(clip)
    try:
        start_s = float(clip_data.get("start_s") or 0.0)
        end_s = float(clip_data.get("end_s") or start_s)
    except Exception:
        return hints

    try:
        from ..tool_wrappers.frame_retriever_runner import _time_hint_anchor_seconds
    except Exception:
        return hints

    kept: List[str] = []
    for hint in hints:
        try:
            anchor_s = _time_hint_anchor_seconds(hint, start_s, end_s)
        except Exception:
            anchor_s = None
        if anchor_s is not None:
            kept.append(hint)
    return kept


def _frame_sequence_context_for_request(request: Any) -> str:
    frames: List[Dict[str, Any]] = []
    for frame in list(getattr(request, "frames", []) or []):
        payload = _model_to_dict(frame)
        if payload:
            frames.append(payload)
    for region in list(getattr(request, "regions", []) or []):
        region_payload = _model_to_dict(region)
        frame_payload = region_payload.get("frame") if isinstance(region_payload, dict) else None
        if isinstance(frame_payload, dict):
            frames.append(dict(frame_payload))
    return render_frame_sequence_context(frames)


def _append_context_text(value: Optional[str], context_text: str) -> str:
    base = str(value or "").strip()
    context_text = str(context_text or "").strip()
    if not context_text:
        return base
    if context_text in base:
        return base
    if not base:
        return context_text
    return "%s\n\nFrame sequence context: %s" % (base, context_text)


def _request_with_query_sequence_context(model_cls, request: Any):
    context_text = _frame_sequence_context_for_request(request)
    if not context_text:
        return request
    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    payload["query"] = _append_context_text(payload.get("query"), context_text)
    return model_cls.parse_obj(payload)


def _generic_request_with_sequence_context(model_cls, request: Any):
    context_text = _frame_sequence_context_for_request(request)
    if not context_text:
        return request
    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    text_contexts = [str(item).strip() for item in list(payload.get("text_contexts") or []) if str(item).strip()]
    if context_text not in text_contexts:
        text_contexts.append(context_text)
    payload["text_contexts"] = text_contexts
    return model_cls.parse_obj(payload)


RAW_VLM_REASONING_RE = re.compile(
    r"\b(the user wants|let'?s|we need|i need|thinking process|step[- ]?by[- ]?step|first,|wait,|therefore,|"
    r"i will|i should|my answer)\b",
    re.I,
)


def generic_output_quality_flags(parsed: GenericPurposeOutput, raw: str = "") -> List[str]:
    flags: List[str] = []
    answer = str(getattr(parsed, "answer", "") or "").strip()
    analysis = str(getattr(parsed, "analysis", "") or "").strip()
    supporting_points = [
        str(item).strip()
        for item in list(getattr(parsed, "supporting_points", []) or [])
        if str(item).strip()
    ]
    combined = "\n".join(item for item in (answer, analysis, raw or "") if str(item).strip())
    if answer and len(answer.split()) > 48:
        flags.append("answer_too_long_for_strict_value")
    if combined and RAW_VLM_REASONING_RE.search(combined):
        flags.append("raw_reasoning_or_preamble_detected")
    if answer and not supporting_points:
        flags.append("missing_supporting_points")
    if getattr(parsed, "confidence", None) is None:
        flags.append("missing_numeric_confidence")
    if raw and raw.lstrip().startswith(("```", "Here", "Sure", "The ")):
        flags.append("non_json_surface_text")
    return flags


def generic_output_strict_json_usable(parsed: GenericPurposeOutput, raw: str = "") -> bool:
    flags = set(generic_output_quality_flags(parsed, raw))
    blocking = {
        "answer_too_long_for_strict_value",
        "raw_reasoning_or_preamble_detected",
        "non_json_surface_text",
    }
    return not bool(flags & blocking)


class JsonProcessMixin(object):
    def __init__(self, model_name: str, extra: Optional[Dict[str, Any]] = None):
        self.model_name = str(model_name or "").strip()
        self.extra = dict(extra or {})

    def _command(self) -> List[str]:
        command = self.extra.get("command") or self.extra.get("cmd")
        if not command:
            raise RuntimeError("No process command configured for tool %s" % getattr(self, "name", "<unknown>"))
        if isinstance(command, str):
            command = shlex.split(command)
        else:
            command = [str(item) for item in list(command)]
        if not command:
            raise RuntimeError("Empty process command configured for tool %s" % getattr(self, "name", "<unknown>"))
        executable = str(command[0] or "").strip()
        executable_name = Path(executable).name.lower()
        if executable and "/" not in executable and "\\" not in executable and executable_name.startswith("python"):
            command[0] = sys.executable
        return command

    def _scratch_dir(self, context) -> Path:
        tool_name = str(getattr(self, "name", "") or "tool")
        legacy_tools_dir = getattr(context.run, "tools_dir", None)
        if legacy_tools_dir is not None:
            return (Path(legacy_tools_dir).expanduser().resolve() / "_scratch" / tool_name).resolve()
        run_dir = getattr(context.run, "run_dir", None)
        if run_dir is not None:
            return (Path(run_dir).expanduser().resolve() / "_scratch" / tool_name).resolve()
        return (Path(context.workspace.workspace_root).expanduser().resolve() / "_scratch" / tool_name).resolve()

    def _runtime_payload(self, context) -> Dict[str, Any]:
        requested_device = self.extra.get("device") or context.workspace.profile.gpu_assignments.get(self.name)
        device = resolve_device_label(requested_device)
        device_mapping = describe_device_mapping(device)
        model_resolution = describe_model_resolution(
            self.model_name,
            hf_cache=context.workspace.profile.hf_cache,
        )
        runtime = {
            "backend": self.extra.get("backend_name") or getattr(self, "name", ""),
            "model_name": self.model_name,
            "device": device,
            "device_mapping": device_mapping,
            "hf_cache": context.workspace.profile.hf_cache,
            "resolved_model_path": model_resolution.get("resolved_path"),
            "model_resolution_status": model_resolution.get("status"),
            "workspace_root": str(context.workspace.workspace_root),
            "scratch_dir": str(self._scratch_dir(context)),
            "extra": {key: value for key, value in self.extra.items() if key not in {"command", "cmd", "env", "cwd"}},
        }
        return runtime

    def _task_payload(self, context) -> Dict[str, Any]:
        return {
            "benchmark": context.task.benchmark,
            "sample_key": context.task.sample_key,
            "question": context.task.question,
            "options": list(context.task.options or []),
            "video_path": context.task.video_path,
            "video_id": context.task.video_id,
            "question_id": context.task.question_id,
        }

    def _request_envelope(self, context, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        evidence_records = []
        evidence_ids = request_payload.get("evidence_ids")
        if context.evidence_lookup is not None and isinstance(evidence_ids, list) and evidence_ids:
            try:
                evidence_records = [
                    sanitize_for_persistence(dict(item))
                    for item in context.evidence_lookup(evidence_ids)
                    if isinstance(item, dict)
                ]
            except Exception:
                evidence_records = []
        return {
            "tool_name": getattr(self, "name", ""),
            "request": request_payload,
            "task": self._task_payload(context),
            "runtime": self._runtime_payload(context),
            "evidence_records": evidence_records,
        }

    def _run_command_json(self, context, request_payload: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        command = self._command()
        env = os.environ.copy()
        for key, value in dict(self.extra.get("env") or {}).items():
            env[str(key)] = str(value)
        cwd = self.extra.get("cwd")
        timeout = float(self.extra.get("timeout_s") or 0.0) or None
        payload = self._request_envelope(context, request_payload)
        completed = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            cwd=str(Path(cwd).expanduser().resolve()) if cwd else None,
            env=env,
            timeout=timeout,
            check=False,
        )
        raw_output = completed.stdout or ""
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(
                "Process adapter failed for %s (exit=%s): %s"
                % (getattr(self, "name", ""), completed.returncode, stderr or raw_output[:1000])
            )
        payload_json = extract_json_object(raw_output)
        if payload_json is None:
            raise ValueError(
                "Process adapter for %s did not return JSON. Output was: %s"
                % (getattr(self, "name", ""), raw_output[:1000])
            )
        return payload_json, raw_output


class BaseProcessToolAdapter(ToolAdapter, JsonProcessMixin):
    request_model = None
    output_model: Type = dict

    def __init__(
        self,
        name: str,
        model_name: str,
        extra: Optional[Dict[str, Any]] = None,
        model_pool=None,
    ):
        self.name = name
        self.model_pool = model_pool
        JsonProcessMixin.__init__(self, model_name=model_name, extra=extra)

    def _parse_output(self, payload: Dict[str, Any]):
        model_cls = getattr(self, "output_model", None)
        if model_cls is None:
            return payload
        if hasattr(model_cls, "model_validate"):
            return model_cls.model_validate(payload)
        return model_cls.parse_obj(payload)

    def _run_persisted_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Persistent execution is not implemented for %s" % self.name)

    def _run_json(self, context, request_payload: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        if self.model_pool is not None and self.model_pool.should_persist(self.name):
            payload = self._run_persisted_json(self._request_envelope(context, request_payload))
            return payload, json.dumps(payload, ensure_ascii=False)
        return self._run_command_json(context, request_payload)

    def _runtime_payload_for_profile(self, profile) -> Dict[str, Any]:
        requested_device = self.extra.get("device") or profile.gpu_assignments.get(self.name)
        device = resolve_device_label(requested_device)
        device_mapping = describe_device_mapping(device)
        model_resolution = describe_model_resolution(
            self.model_name,
            hf_cache=profile.hf_cache,
        )
        return {
            "backend": self.extra.get("backend_name") or getattr(self, "name", ""),
            "model_name": self.model_name,
            "device": device,
            "device_mapping": device_mapping,
            "hf_cache": profile.hf_cache,
            "resolved_model_path": model_resolution.get("resolved_path"),
            "model_resolution_status": model_resolution.get("status"),
            "workspace_root": str(profile.workspace_root),
            "extra": {key: value for key, value in self.extra.items() if key not in {"command", "cmd", "env", "cwd"}},
        }

    def _qwen_persistent_preload_spec(
        self,
        profile,
        *,
        processor_use_fast: Optional[bool] = None,
        processor_model_path: Optional[str] = None,
        device_map: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.model_pool is None or not self.model_pool.should_persist(self.name):
            return None
        if device_map is None:
            device_map = str(self.extra.get("device_map") or "").strip() or None
        runtime = self._runtime_payload_for_profile(profile)
        generation = resolve_generation_controls(runtime)
        model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
        device_label = resolved_device_label(runtime)
        attn_implementation = str((runtime.get("extra") or {}).get("attn_implementation") or "").strip() or None
        enable_thinking_value = (runtime.get("extra") or {}).get("enable_thinking")
        enable_thinking = None if enable_thinking_value is None else bool(enable_thinking_value)
        return {
            "tool_name": self.name,
            "runner_type": "qwen_style",
            "load_key": self.model_pool.qwen_style_key(
                tool_name=self.name,
                model_path=model_path,
                device_label=device_label,
                processor_use_fast=processor_use_fast,
                processor_model_path=processor_model_path,
                generate_do_sample=bool(generation.get("do_sample")),
                generate_temperature=generation.get("temperature"),
                attn_implementation=attn_implementation,
                device_map=device_map,
                enable_thinking=enable_thinking,
            ),
            "model_name": self.model_name,
            "resolved_model_path": model_path,
            "device_label": device_label,
            "processor_use_fast": processor_use_fast,
            "processor_model_path": processor_model_path,
            "generate_do_sample": bool(generation.get("do_sample")),
            "generate_temperature": generation.get("temperature"),
            "attn_implementation": attn_implementation,
            "device_map": device_map,
            "enable_thinking": enable_thinking,
        }

    def _penguin_persistent_preload_spec(self, profile) -> Optional[Dict[str, Any]]:
        if self.model_pool is None or not self.model_pool.should_persist(self.name):
            return None
        runtime = self._runtime_payload_for_profile(profile)
        generation = resolve_generation_controls(runtime)
        model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
        device_label = resolved_device_label(runtime)
        return {
            "tool_name": self.name,
            "runner_type": "penguin",
            "load_key": self.model_pool.penguin_key(
                tool_name=self.name,
                model_path=model_path,
                device_label=device_label,
                generate_do_sample=bool(generation.get("do_sample")),
                generate_temperature=generation.get("temperature"),
            ),
            "model_name": self.model_name,
            "resolved_model_path": model_path,
            "device_label": device_label,
            "generate_do_sample": bool(generation.get("do_sample")),
            "generate_temperature": generation.get("temperature"),
        }

    def _timechat_persistent_preload_spec(self, profile) -> Optional[Dict[str, Any]]:
        if self.model_pool is None or not self.model_pool.should_persist(self.name):
            return None
        runtime = self._runtime_payload_for_profile(profile)
        generation = resolve_generation_controls(runtime)
        model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
        device_label = resolved_device_label(runtime)
        use_audio_in_video = bool((runtime.get("extra") or {}).get("use_audio_in_video", True))
        attn_implementation = str((runtime.get("extra") or {}).get("attn_implementation") or "").strip() or None
        return {
            "tool_name": self.name,
            "runner_type": "timechat",
            "load_key": self.model_pool.timechat_key(
                tool_name=self.name,
                model_path=model_path,
                device_label=device_label,
                generate_do_sample=bool(generation.get("do_sample")),
                generate_temperature=generation.get("temperature"),
                use_audio_in_video=use_audio_in_video,
                attn_implementation=attn_implementation,
            ),
            "model_name": self.model_name,
            "resolved_model_path": model_path,
            "device_label": device_label,
            "generate_do_sample": bool(generation.get("do_sample")),
            "generate_temperature": generation.get("temperature"),
            "use_audio_in_video": use_audio_in_video,
            "attn_implementation": attn_implementation,
        }

    def persistent_preload_spec(self, profile) -> Optional[Dict[str, Any]]:
        return None


def _dedupe_artifact_refs(items):
    deduped = []
    seen = set()
    for item in list(items or []):
        key = (
            getattr(item, "artifact_id", None),
            getattr(item, "relpath", None),
            getattr(item, "kind", None),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _media_artifact_ref(payload: Dict[str, Any], *, kind: str, source_tool: str) -> Optional[ArtifactRef]:
    artifact_id = str(payload.get("artifact_id") or "").strip()
    relpath = str(payload.get("relpath") or "").strip() or None
    if not artifact_id and not relpath:
        return None
    if not artifact_id and relpath:
        artifact_id = Path(relpath).stem
    metadata = dict(payload.get("metadata") or {})
    for key in ("video_id", "start_s", "end_s", "timestamp_s"):
        if payload.get(key) is not None and key not in metadata:
            metadata[key] = payload.get(key)
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        relpath=relpath,
        media_type={"clip": "video", "frame": "image", "region": "image"}.get(kind),
        source_tool=source_tool,
        metadata=metadata,
    )


def _request_artifact_refs(request, *, source_tool: str) -> List[ArtifactRef]:
    artifacts: List[ArtifactRef] = []
    for clip in list(getattr(request, "clips", []) or []):
        payload = _model_to_dict(clip)
        artifact = _media_artifact_ref(payload, kind="clip", source_tool=source_tool)
        if artifact is not None:
            artifacts.append(artifact)
    for frame in list(getattr(request, "frames", []) or []):
        payload = _model_to_dict(frame)
        artifact = _media_artifact_ref(payload, kind="frame", source_tool=source_tool)
        if artifact is not None:
            artifacts.append(artifact)
        clip_payload = _model_to_dict(payload.get("clip"))
        clip_artifact = _media_artifact_ref(clip_payload, kind="clip", source_tool=source_tool)
        if clip_artifact is not None:
            artifacts.append(clip_artifact)
    for transcript in list(getattr(request, "transcripts", []) or []):
        payload = _model_to_dict(transcript)
        clip_payload = _model_to_dict(payload.get("clip"))
        clip_artifact = _media_artifact_ref(clip_payload, kind="clip", source_tool=source_tool)
        if clip_artifact is not None:
            artifacts.append(clip_artifact)
    return _dedupe_artifact_refs(artifacts)


def _unknown_verifier_output_for_validation_error(request, exc: Exception, payload: Dict[str, Any]) -> Dict[str, Any]:
    error_text = str(exc).replace("\n", " ")[:1000]
    try:
        payload_prefix = json.dumps(payload, ensure_ascii=False)[:1000]
    except Exception:
        payload_prefix = str(payload)[:1000]
    claim_results = []
    for claim in list(getattr(request, "claims", []) or []):
        claim_payload = _model_to_dict(claim)
        claim_id = str(claim_payload.get("claim_id") or "").strip()
        if not claim_id:
            continue
        claim_results.append(
            {
                "claim_id": claim_id,
                "verdict": "unknown",
                "confidence": 0.0,
                "answer_value": None,
                "claimed_value": None,
                "observed_value": None,
                "match_status": "unknown",
                "target_presence": "unknown",
                "supporting_observation_ids": [],
                "supporting_evidence_ids": [],
                "refuting_observation_ids": [],
                "refuting_evidence_ids": [],
                "time_intervals": [],
                "artifact_refs": [],
                "rationale": "Verifier output did not match the schema, so this claim was not validated.",
                "coverage": {
                    "checked_inputs": [],
                    "missing_inputs": ["valid verifier JSON schema"],
                    "sampling_summary": "Verifier output was discarded after schema validation failed.",
                },
            }
        )
    return {
        "claim_results": claim_results,
        "new_observations": [],
        "evidence_updates": [],
        "checklist_updates": [],
        "counter_updates": [],
        "referent_updates": [],
        "ocr_occurrence_updates": [],
        "unresolved_gaps": [
            "verifier_schema_validation_error: %s payload_prefix=%s" % (error_text, payload_prefix)
        ],
    }


def _join_text_blocks(items: List[str]) -> str:
    cleaned = [str(item or "").strip() for item in list(items or []) if str(item or "").strip()]
    return "\n\n".join(cleaned)


def _first_list_item(values: List[Any]) -> Any:
    items = list(values or [])
    return items[0] if items else None


def _spatial_frame_ref_for_result(request, parsed) -> Optional[FrameRef]:
    frame_ref = _first_list_item(getattr(request, "frames", []) or [])
    frame_path = getattr(parsed, "source_frame_path", None)
    if frame_ref is not None:
        if frame_path and not dict(frame_ref.metadata or {}).get("source_path"):
            payload = _model_to_dict(frame_ref)
            metadata = dict(payload.get("metadata") or {})
            metadata["source_path"] = frame_path
            payload["metadata"] = metadata
            return FrameRef.parse_obj(payload)
        return frame_ref

    clip_ref = _first_list_item(getattr(request, "clips", []) or [])
    if clip_ref is None:
        return None
    timestamp_s = getattr(parsed, "timestamp_s", None)
    if timestamp_s is None:
        timestamp_s = (float(clip_ref.start_s) + float(clip_ref.end_s)) / 2.0
    return FrameRef(
        video_id=clip_ref.video_id,
        timestamp_s=float(timestamp_s),
        clip=clip_ref,
        metadata={},
    )


def _frame_rank_sort_key(frame: Dict[str, Any]) -> tuple:
    metadata = dict(frame.get("metadata") or {})
    temporal_score = metadata.get("temporal_score")
    relevance_score = metadata.get("relevance_score")
    score = temporal_score if temporal_score is not None else relevance_score
    try:
        numeric_score = float(score)
    except Exception:
        numeric_score = 0.0
    anchor_distance = metadata.get("anchor_distance_s")
    try:
        numeric_anchor_distance = float(anchor_distance)
    except Exception:
        numeric_anchor_distance = float("inf")
    try:
        timestamp_s = float(frame.get("timestamp_s") or 0.0)
    except Exception:
        timestamp_s = 0.0
    return (-numeric_score, numeric_anchor_distance, timestamp_s)


def _frame_is_chronological_sequence(frame: Dict[str, Any]) -> bool:
    metadata = dict(frame.get("metadata") or {})
    return (
        str(metadata.get("sequence_mode") or "").strip().lower() in {"anchor_window", "chronological"}
        and str(metadata.get("sequence_sort_order") or "").strip().lower() == "chronological"
    )


def _frame_is_full_chronological_clip_sequence(frame: Dict[str, Any]) -> bool:
    metadata = dict(frame.get("metadata") or {})
    return (
        str(metadata.get("sequence_mode") or "").strip().lower() == "chronological"
        and str(metadata.get("sequence_role") or "").strip().lower() == "interval_frame"
        and str(metadata.get("selection_reason") or "").strip().lower() == "chronological_clip_sequence"
    )


def _frame_sequence_sort_key(frame: Dict[str, Any]) -> tuple:
    metadata = dict(frame.get("metadata") or {})

    def _float_value(value: Any, default: float = float("inf")) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _int_value(value: Any, default: int = 10**9) -> int:
        try:
            return int(value)
        except Exception:
            return default

    timestamp_s = _float_value(frame.get("timestamp_s"), 0.0)
    clip_start_s = _float_value(metadata.get("clip_start_s"), timestamp_s)
    requested_timestamp_s = _float_value(metadata.get("requested_timestamp_s"), timestamp_s)
    sequence_index = _int_value(metadata.get("sequence_index"))
    return (clip_start_s, requested_timestamp_s, sequence_index, timestamp_s)


def _frame_key(frame: Dict[str, Any]) -> tuple:
    return (
        str(frame.get("artifact_id") or "").strip(),
        str(frame.get("relpath") or "").strip(),
        float(frame.get("timestamp_s") or 0.0),
    )


def _select_multi_clip_frames(
    frame_groups: List[Dict[str, Any]],
    frames_per_input: int,
    *,
    sort_order: str = "ranked",
    sequence_mode: str = "ranked",
) -> List[Dict[str, Any]]:
    if frames_per_input <= 0:
        return []
    selected: List[Dict[str, Any]] = []
    seen = set()
    per_input_limit = max(1, int(frames_per_input or 1))
    groups = list(frame_groups or [])
    merged_frames = [frame for group in groups for frame in list(group.get("frames") or [])]
    chronological = (
        str(sort_order or "").strip().lower() == "chronological"
        or str(sequence_mode or "").strip().lower() in {"anchor_window", "chronological"}
        or any(_frame_is_chronological_sequence(frame) for frame in merged_frames)
    )

    group_sort_key = _frame_sequence_sort_key if chronological else _frame_rank_sort_key
    final_sort_key = _frame_sequence_sort_key if chronological else _frame_rank_sort_key

    for group in groups:
        group_selected = 0
        group_frames = sorted(list(group.get("frames") or []), key=group_sort_key)
        group_limit = len(group_frames) if any(_frame_is_full_chronological_clip_sequence(frame) for frame in group_frames) else per_input_limit
        for frame in group_frames:
            key = _frame_key(frame)
            if key in seen:
                continue
            seen.add(key)
            selected.append(frame)
            group_selected += 1
            if group_selected >= group_limit:
                break
    return sorted(selected, key=final_sort_key)


class VisualTemporalGrounderProcessAdapter(BaseProcessToolAdapter):
    request_model = VisualTemporalGrounderRequest
    output_model = VisualTemporalGrounderOutput

    def _run_persisted_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return execute_visual_temporal_grounder_payload(payload, runner_pool=self.model_pool)

    def persistent_preload_spec(self, profile) -> Optional[Dict[str, Any]]:
        return self._qwen_persistent_preload_spec(profile)

    def execute(self, request, context):
        payload, raw = self._run_json(context, request.dict())
        parsed = self._parse_output(payload)
        clips = [item.model_dump() for item in parsed.clips]
        confidence_metadata = _confidence_metadata(
            [item.confidence for item in parsed.clips],
            kind="temporal_grounding",
        )
        data = {
            "query": parsed.query,
            "clips": clips,
            "video_duration": parsed.video_duration,
            "retrieval_backend": parsed.retrieval_backend or self.model_name,
            "query_absent": parsed.query_absent,
            "summary": parsed.summary,
            "prefilter": dict(parsed.prefilter or {}),
        }
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
            summary=parsed.summary or "Found %d candidate clip(s)." % len(clips),
            metadata={
                "backend": parsed.retrieval_backend or self.model_name,
                "prefilter": dict(parsed.prefilter or {}),
                **confidence_metadata,
            },
        )


class AudioTemporalGrounderProcessAdapter(BaseProcessToolAdapter):
    request_model = AudioTemporalGrounderRequest
    output_model = AudioTemporalGrounderOutput

    def _execute_single(self, request, context):
        payload, raw = self._run_json(context, request.dict())
        parsed = self._parse_output(payload)
        clips = [item.model_dump() for item in parsed.clips]
        events = [
            {
                "event_label": item.event_label,
                "start": item.start_s,
                "end": item.end_s,
                "confidence": item.confidence,
                "metadata": item.metadata,
            }
            for item in parsed.events
        ]
        confidence_metadata = _confidence_metadata(
            [item.confidence for item in parsed.events] + [item.confidence for item in parsed.clips],
            kind="audio_grounding",
        )
        data = {
            "query": parsed.query,
            "clips": clips,
            "events": events,
            "retrieval_backend": parsed.retrieval_backend or self.model_name,
            "query_absent": parsed.query_absent,
            "summary": parsed.summary,
        }
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
            summary=parsed.summary or "Found %d candidate audio interval(s)." % len(clips),
            metadata={
                "backend": parsed.retrieval_backend or self.model_name,
                **confidence_metadata,
            },
        )

    def execute(self, request, context):
        clips = list(getattr(request, "clips", []) or [])
        if len(clips) > 1:
            subrequests = [
                self.request_model.parse_obj(
                    {
                        "tool_name": self.name,
                        "query": request.query,
                        "clips": [item.dict() if hasattr(item, "dict") else item],
                    }
                )
                for item in clips
            ]
            subresults = [self._execute_single(item, context) for item in subrequests]
            merged_clips = []
            merged_events = []
            raw_blocks = []
            summaries = []
            for subresult in subresults:
                merged_clips.extend(list(subresult.data.get("clips") or []))
                merged_events.extend(list(subresult.data.get("events") or []))
                if subresult.raw_output_text:
                    raw_blocks.append(subresult.raw_output_text)
                if subresult.summary:
                    summaries.append(subresult.summary)
            merged_data = {
                "query": request.query,
                "clips": merged_clips,
                "events": merged_events,
                "retrieval_backend": self.model_name,
                "query_absent": not bool(merged_clips or merged_events),
                "summary": _join_text_blocks(summaries),
            }
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data=merged_data,
                raw_output_text=_join_text_blocks(raw_blocks),
                request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
                summary="Found %d candidate audio interval(s) across %d input clip(s)." % (len(merged_clips), len(subrequests)),
                metadata={
                    "backend": self.model_name,
                    "group_count": len(subrequests),
                    **_confidence_metadata(
                        [item.get("confidence") for item in merged_events] + [item.get("confidence") for item in merged_clips],
                        kind="audio_grounding",
                    ),
                },
            )
        return self._execute_single(request, context)


class FrameRetrieverProcessAdapter(BaseProcessToolAdapter):
    request_model = FrameRetrieverRequest
    output_model = FrameRetrieverOutput

    def __init__(
        self,
        name: str,
        model_name: str,
        extra: Optional[Dict[str, Any]] = None,
        model_pool=None,
    ):
        super().__init__(name=name, model_name=model_name, extra=extra, model_pool=model_pool)
        self._persisted_frame_harnesses: Dict[tuple, Any] = {}

    def _persisted_frame_harness_key(self, payload: Dict[str, Any]) -> tuple:
        request = dict(payload.get("request") or {})
        task = dict(payload.get("task") or {})
        runtime = dict(payload.get("runtime") or {})
        clip = dict(_first_list_item(request.get("clips") or []) or {})
        clip_start_s = float(clip.get("start_s") or 0.0)
        clip_end_s = float(clip.get("end_s") or clip_start_s)
        clip_duration_s = max(1.0, clip_end_s - clip_start_s)
        extra = dict(runtime.get("extra") or {})
        return (
            str(task.get("video_id") or task.get("sample_key") or ""),
            str(task.get("video_path") or ""),
            str(runtime.get("device") or ""),
            str(runtime.get("model_name") or ""),
            str(runtime.get("resolved_model_path") or ""),
            str(extra.get("reranker_model") or ""),
            str(extra.get("attn_implementation") or ""),
            str(extra.get("device_map") or ""),
            round(float(clip_duration_s), 3),
        )

    def _persisted_frame_harness(self, payload: Dict[str, Any]):
        from ..tool_wrappers.reference_adapter import ReferenceHarness

        key = self._persisted_frame_harness_key(payload)
        harness = self._persisted_frame_harnesses.get(key)
        if harness is not None:
            return harness

        request = dict(payload.get("request") or {})
        task = dict(payload.get("task") or {})
        runtime = dict(payload.get("runtime") or {})
        clip = dict(_first_list_item(request.get("clips") or []) or {})
        clip_start_s = float(clip.get("start_s") or 0.0)
        clip_end_s = float(clip.get("end_s") or clip_start_s)
        harness = ReferenceHarness(
            task=task,
            runtime=runtime,
            clip_duration_s=max(1.0, clip_end_s - clip_start_s),
            embedder_model=str(runtime.get("model_name") or ""),
            reranker_model=str((runtime.get("extra") or {}).get("reranker_model") or ""),
        )
        self._persisted_frame_harnesses[key] = harness
        return harness

    def _run_persisted_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from ..tool_wrappers.frame_retriever_runner import execute_payload

        harness = self._persisted_frame_harness(payload)
        return execute_payload(payload, harness=harness, release_embedder=False)

    def _execute_single(self, request, context):
        payload, raw = self._run_json(context, request.dict())
        parsed = self._parse_output(payload)
        artifact_refs = []
        frames = []
        clip_ref = _first_list_item(getattr(request, "clips", []) or [])
        for item in parsed.frames:
            artifact = context.workspace.store_file_artifact(
                item.frame_path,
                kind="frame",
                source_tool=self.name,
                video_id=context.task.video_id or context.task.sample_key,
            )
            artifact_refs.append(artifact)
            frames.append(
                FrameRef(
                    video_id=context.task.video_id or context.task.sample_key,
                    timestamp_s=float(item.timestamp_s),
                    artifact_id=artifact.artifact_id,
                    relpath=artifact.relpath,
                    clip=clip_ref,
                    metadata={
                        "source_path": item.frame_path,
                        "relevance_score": item.relevance_score,
                        **dict(item.metadata or {}),
                    },
                ).dict()
            )
        confidence_metadata = _confidence_metadata(
            [item.relevance_score for item in parsed.frames],
            kind="frame_relevance",
        )
        cache_metadata = dict(parsed.cache_metadata or {})
        data = {
            "query": parsed.query,
            "frames": frames,
            "mode": parsed.mode,
            "cache_metadata": cache_metadata,
            "rationale": parsed.rationale,
        }
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            artifact_refs=artifact_refs,
            request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
            summary=parsed.rationale or "Retrieved %d frame(s)." % len(frames),
            metadata={"backend": self.model_name, **confidence_metadata, **cache_metadata},
        )

    def execute(self, request, context):
        clips = list(getattr(request, "clips", []) or [])
        time_hints = [str(item).strip() for item in list(getattr(request, "time_hints", []) or []) if str(item).strip()]
        if len(clips) > 1 or len(time_hints) > 1:
            subrequests = []
            skipped_time_hint_clip_count = 0
            if clips:
                for clip in clips:
                    clip_time_hints = _time_hints_inside_clip(time_hints, clip)
                    if time_hints and not clip_time_hints:
                        skipped_time_hint_clip_count += 1
                        continue
                    subrequests.append(
                        self.request_model.parse_obj(
                            {
                                "tool_name": self.name,
                                "clips": [clip.dict() if hasattr(clip, "dict") else clip],
                                "query": request.query,
                                "num_frames": request.num_frames,
                                "time_hints": list(clip_time_hints),
                                "sequence_mode": request.sequence_mode,
                                "neighbor_radius_s": request.neighbor_radius_s,
                                "include_anchor_neighbors": request.include_anchor_neighbors,
                                "sort_order": request.sort_order,
                            }
                        )
                    )
            else:
                for time_hint in time_hints:
                    subrequests.append(
                        self.request_model.parse_obj(
                            {
                                "tool_name": self.name,
                                "time_hints": [time_hint],
                                "query": request.query,
                                "num_frames": request.num_frames,
                                "sequence_mode": request.sequence_mode,
                                "neighbor_radius_s": request.neighbor_radius_s,
                                "include_anchor_neighbors": request.include_anchor_neighbors,
                                "sort_order": request.sort_order,
                            }
                        )
                    )
            if not subrequests:
                raise RuntimeError(
                    "frame_retriever could not match any time_hints to the provided clips; "
                    "provide a clip containing the anchor timestamp or use chronological mode without time_hints."
                )
            subresults = [self._execute_single(item, context) for item in subrequests]
            merged_frames = []
            frame_groups = []
            artifact_refs = []
            raw_blocks = []
            summaries = []
            cache_groups = []
            dense_frame_cache_hits = []
            dense_frame_counts = []
            bounded_frame_counts = []
            embedding_cache_ready = []
            for subrequest, subresult in zip(subrequests, subresults):
                group_frames = list(subresult.data.get("frames") or [])
                group_cache_metadata = dict(subresult.data.get("cache_metadata") or {})
                merged_frames.extend(group_frames)
                group_clip = _first_list_item(getattr(subrequest, "clips", []) or [])
                group_time_hints = [
                    str(item) for item in list(getattr(subrequest, "time_hints", []) or []) if str(item).strip()
                ]
                frame_groups.append(
                    {
                        "clips": [group_clip.dict() if hasattr(group_clip, "dict") else group_clip] if group_clip is not None else [],
                        "time_hints": group_time_hints,
                        "frames": group_frames,
                        "cache_metadata": group_cache_metadata,
                        "rationale": subresult.data.get("rationale") or subresult.summary,
                    }
                )
                if group_cache_metadata:
                    cache_groups.append(
                        {
                            "clips": [group_clip.dict() if hasattr(group_clip, "dict") else group_clip] if group_clip is not None else [],
                            "time_hints": group_time_hints,
                            "cache_metadata": group_cache_metadata,
                        }
                    )
                if group_cache_metadata.get("dense_frame_cache_hit") is not None:
                    dense_frame_cache_hits.append(bool(group_cache_metadata.get("dense_frame_cache_hit")))
                if group_cache_metadata.get("dense_frame_count") is not None:
                    dense_frame_counts.append(int(group_cache_metadata.get("dense_frame_count")))
                if group_cache_metadata.get("bounded_frame_count") is not None:
                    bounded_frame_counts.append(int(group_cache_metadata.get("bounded_frame_count")))
                if group_cache_metadata.get("embedding_cache_ready") is not None:
                    embedding_cache_ready.append(bool(group_cache_metadata.get("embedding_cache_ready")))
                artifact_refs.extend(list(subresult.artifact_refs or []))
                if subresult.raw_output_text:
                    raw_blocks.append(subresult.raw_output_text)
                if subresult.summary:
                    summaries.append(subresult.summary)
            cache_metadata = {}
            if dense_frame_cache_hits:
                cache_metadata["dense_frame_cache_hit"] = all(dense_frame_cache_hits)
            if dense_frame_counts:
                cache_metadata["dense_frame_count"] = max(dense_frame_counts)
            if bounded_frame_counts:
                cache_metadata["bounded_frame_count"] = sum(bounded_frame_counts)
            if embedding_cache_ready:
                cache_metadata["embedding_cache_ready"] = all(embedding_cache_ready)
            if skipped_time_hint_clip_count:
                cache_metadata["skipped_out_of_window_time_hint_clip_count"] = skipped_time_hint_clip_count
            frames_per_input = max(1, int(request.num_frames or 1))
            selected_frames = _select_multi_clip_frames(
                frame_groups,
                frames_per_input,
                sort_order=request.sort_order,
                sequence_mode=request.sequence_mode,
            )
            cache_metadata.update(
                {
                    "frame_count_policy": "per_clip",
                    "frames_per_input": frames_per_input,
                    "returned_frame_count": len(selected_frames),
                }
            )
            merged_data = {
                "query": request.query,
                "clips": [
                    first_clip.dict()
                    for first_clip in (_first_list_item(getattr(item, "clips", []) or []) for item in subrequests)
                    if first_clip is not None
                ],
                "frames": selected_frames,
                "frame_groups": frame_groups,
                "cache_groups": cache_groups,
                "cache_metadata": cache_metadata,
                "mode": "multi_clip_bounded",
                "rationale": _join_text_blocks(summaries),
            }
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data=merged_data,
                raw_output_text=_join_text_blocks(raw_blocks),
                artifact_refs=_dedupe_artifact_refs(artifact_refs),
                request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
                summary="Retrieved %d frame(s) across %d input clip(s)." % (len(selected_frames), len(frame_groups)),
                metadata={
                    "backend": self.model_name,
                    "group_count": len(frame_groups),
                    **_confidence_metadata(
                        [frame.get("metadata", {}).get("relevance_score") for frame in selected_frames],
                        kind="frame_relevance",
                    ),
                    **cache_metadata,
                },
            )
        return self._execute_single(request, context)


class DenseCaptionProcessAdapter(BaseProcessToolAdapter):
    request_model = DenseCaptionRequest
    output_model = DenseCaptionOutput

    def _reuse_preprocess_single(self, request, context):
        if str(getattr(request, "focus_query", "") or "").strip():
            return None
        clip = _first_list_item(getattr(request, "clips", []) or [])
        if clip is None:
            return None
        clip_payload = clip.model_dump() if hasattr(clip, "model_dump") else clip.dict() if hasattr(clip, "dict") else dict(clip)
        start_s = float(clip_payload.get("start_s") or 0.0)
        end_s = float(clip_payload.get("end_s") or start_s)
        bundle = getattr(context, "preprocess_bundle", None) if context is not None else None
        for segment in list(dict(bundle or {}).get("dense_caption_segments") or []):
            if not isinstance(segment, dict):
                continue
            if abs(float(segment.get("start_s") or 0.0) - start_s) > 1e-3:
                continue
            if abs(float(segment.get("end_s") or 0.0) - end_s) > 1e-3:
                continue
            dense_caption = dict(segment.get("dense_caption") or {})
            data = {
                key: value
                for key, value in dense_caption.items()
                if key != "backend" and value not in (None, "", [], {})
            }
            data.setdefault("clips", [clip_payload])
            summary = str(data.get("overall_summary") or "Reused dense captions from preprocessing.").strip()
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data=data,
                raw_output_text=json.dumps(data, ensure_ascii=False),
                artifact_refs=[],
                request_hash=hash_payload({"tool": self.name, "request": request.dict(), "source": "preprocess_reuse"}),
                summary=summary[:2000],
                metadata={"source": "preprocess_reuse"},
            )
        return None

    def _run_persisted_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return execute_dense_caption_payload(payload, runner_pool=self.model_pool)

    def persistent_preload_spec(self, profile) -> Optional[Dict[str, Any]]:
        return self._timechat_persistent_preload_spec(profile)

    def _build_tool_result(self, request, parsed, raw: str, context) -> ToolResult:
        artifact_refs = []
        sampled_frames = []
        for item in parsed.sampled_frames or []:
            frame_path = item.get("frame_path")
            if not frame_path:
                sampled_frames.append(dict(item))
                continue
            artifact = context.workspace.store_file_artifact(
                frame_path,
                kind="frame",
                source_tool=self.name,
                video_id=context.task.video_id or context.task.sample_key,
            )
            artifact_refs.append(artifact)
            enriched = dict(item)
            enriched["artifact_id"] = artifact.artifact_id
            enriched["relpath"] = artifact.relpath
            sampled_frames.append(enriched)
        data = {
            "clips": [item.dict() for item in parsed.clips],
            "captions": [item.dict() for item in parsed.captions],
            "overall_summary": parsed.overall_summary,
            "captioned_range": parsed.captioned_range.dict(),
        }
        if sampled_frames:
            data["sampled_frames"] = sampled_frames
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            artifact_refs=artifact_refs,
            request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
            summary=parsed.overall_summary[:2000],
            metadata={"backend": parsed.backend or self.model_name},
        )

    def _execute_single(self, request, context):
        reused = self._reuse_preprocess_single(request, context)
        if reused is not None:
            return reused
        payload, raw = self._run_json(context, request.dict())
        parsed = self._parse_output(payload)
        return self._build_tool_result(request, parsed, raw, context)

    def _preprocess_payload(self, request, context, preprocess_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = self._request_envelope(context, request.dict())
        runtime = dict(payload.get("runtime") or {})
        extra = dict(runtime.get("extra") or {})
        for key in ("sample_frames", "fps", "max_frames", "use_audio_in_video", "collect_sampled_frames", "max_new_tokens"):
            if preprocess_settings is not None and key in preprocess_settings:
                extra[key] = preprocess_settings[key]
        runtime["extra"] = extra
        payload["runtime"] = runtime
        return payload

    def execute(self, request, context):
        clips = list(getattr(request, "clips", []) or [])
        if len(clips) > 1:
            subrequests = [
                self.request_model.parse_obj(
                    {
                        "tool_name": self.name,
                        "clips": [item.dict() if hasattr(item, "dict") else item],
                        "granularity": request.granularity,
                        "focus_query": request.focus_query,
                    }
                )
                for item in clips
            ]
            subresults = [self._execute_single(item, context) for item in subrequests]
            artifact_refs = []
            raw_blocks = []
            captions = []
            sampled_frames = []
            summaries = []
            caption_groups = []
            for subrequest, subresult in zip(subrequests, subresults):
                artifact_refs.extend(list(subresult.artifact_refs or []))
                if subresult.raw_output_text:
                    raw_blocks.append(subresult.raw_output_text)
                captions.extend(list(subresult.data.get("captions") or []))
                sampled_frames.extend(list(subresult.data.get("sampled_frames") or []))
                group_summary = subresult.data.get("overall_summary") or ""
                if group_summary:
                    summaries.append(group_summary)
                group_clip = _first_list_item(getattr(subrequest, "clips", []) or [])
                caption_groups.append(
                    {
                        "clips": [group_clip.dict() if hasattr(group_clip, "dict") else group_clip] if group_clip is not None else [],
                        "captions": list(subresult.data.get("captions") or []),
                        "overall_summary": group_summary,
                    }
                )
            merged_data = {
                "clips": [
                    first_clip.dict()
                    for first_clip in (_first_list_item(getattr(item, "clips", []) or []) for item in subrequests)
                    if first_clip is not None
                ],
                "captions": captions,
                "overall_summary": _join_text_blocks(summaries),
                "caption_groups": caption_groups,
            }
            if sampled_frames:
                merged_data["sampled_frames"] = sampled_frames
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data=merged_data,
                raw_output_text=_join_text_blocks(raw_blocks),
                artifact_refs=_dedupe_artifact_refs(artifact_refs),
                request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
                summary="Dense captioning completed for %d clip(s)." % len(caption_groups),
                metadata={"backend": self.model_name, "group_count": len(caption_groups)},
            )
        return self._execute_single(request, context)

    def build_segment_cache(self, task, clip_duration_s: float, context, preprocess_settings: Optional[Dict[str, Any]] = None):
        settings = dict(preprocess_settings or {})
        effective_clip_duration_s = max(1.0, float(settings.get("clip_duration_s") or clip_duration_s or 1.0))
        runtime = self._runtime_payload(context)
        generation = resolve_generation_controls(runtime)
        model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
        device_label = resolved_device_label(runtime)
        attn_implementation = str((runtime.get("extra") or {}).get("attn_implementation") or "").strip() or None
        use_audio_in_video = bool(settings.get("use_audio_in_video", (runtime.get("extra") or {}).get("use_audio_in_video", True)))
        runner = None
        owns_runner = False
        if self.model_pool is not None:
            runner = self.model_pool.acquire_timechat_runner(
                tool_name=self.name,
                model_path=model_path,
                device_label=device_label,
                generate_do_sample=bool(generation.get("do_sample")),
                generate_temperature=generation.get("temperature"),
                use_audio_in_video=use_audio_in_video,
                attn_implementation=attn_implementation,
            )
        if runner is None:
            runner = TimeChatCaptionerRunner(
                model_path=model_path,
                device_label=device_label,
                generate_do_sample=bool(generation.get("do_sample")),
                generate_temperature=generation.get("temperature"),
                use_audio_in_video=use_audio_in_video,
                attn_implementation=attn_implementation,
            )
            owns_runner = True
        duration = get_video_duration(task.video_path) or 0.0
        segments = []
        start = 0.0
        try:
            while start < duration:
                end = min(duration, start + effective_clip_duration_s)
                request = self.request_model.parse_obj(
                    {
                        "tool_name": self.name,
                        "clips": [
                            {
                                "video_id": task.video_id or task.sample_key,
                                "start_s": start,
                                "end_s": end,
                            }
                        ],
                        "granularity": "segment",
                        "focus_query": "",
                    }
                )
                payload = self._preprocess_payload(request, context, preprocess_settings=settings)
                result_payload = execute_dense_caption_payload(payload, runner=runner)
                result = self._build_tool_result(
                    request,
                    self._parse_output(result_payload),
                    json.dumps(result_payload, ensure_ascii=False),
                    context,
                )
                segments.append(
                    {
                        "start": float(start),
                        "end": float(end),
                        "dense_caption": result.data,
                    }
                )
                if end <= start:
                    break
                start = end
        finally:
            if owns_runner and runner is not None:
                runner.close()
        return {"segments": segments, "summary": ""}

class OCRProcessAdapter(BaseProcessToolAdapter):
    request_model = OCRRequest
    output_model = OCROutput

    def parse_request(self, arguments: Dict[str, Any]):
        request = super().parse_request(arguments)
        return _request_with_query_sequence_context(self.request_model, request)

    def _execute_single(self, request, context):
        payload, raw = self._run_json(context, request.dict())
        parsed = self._parse_output(payload)
        artifact_refs = []
        if parsed.source_frame_path and context is not None:
            artifact_refs.append(
                context.workspace.store_file_artifact(
                    parsed.source_frame_path,
                    kind="frame",
                    source_tool=self.name,
                    video_id=context.task.video_id or context.task.sample_key,
                )
            )
        data = {
            "text": parsed.text,
            "lines": [item.dict() for item in parsed.lines],
            "query": parsed.query,
            "timestamp_s": parsed.timestamp_s,
            "source_frame_path": parsed.source_frame_path,
            "backend": parsed.backend or self.model_name,
        }
        if isinstance(payload, dict):
            for source_key in ("region", "frame", "clip"):
                if isinstance(payload.get(source_key), dict):
                    data[source_key] = dict(payload.get(source_key) or {})
        confidence_metadata = _confidence_metadata(
            [item.confidence for item in parsed.lines],
            kind="ocr_line",
        )
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            artifact_refs=artifact_refs,
            request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
            summary=(parsed.text or "No text detected.")[:2000],
            metadata={"backend": parsed.backend or self.model_name, **confidence_metadata},
        )

    def execute(self, request, context):
        request = _request_with_query_sequence_context(self.request_model, request)
        regions = list(getattr(request, "regions", []) or [])
        frames = list(getattr(request, "frames", []) or [])
        clips = list(getattr(request, "clips", []) or [])
        if len(regions) > 1 or len(frames) > 1 or clips:
            if regions:
                units = [("region", item) for item in regions]
            elif frames:
                units = [("frame", item) for item in frames]
            else:
                units = [("clip", item) for item in clips]
            payload, raw = self._run_json(context, request.dict())
            has_batch_results = isinstance(payload, dict) and "results" in payload
            batch_results = list(payload.get("results") or []) if has_batch_results else []
            if not has_batch_results:
                if len(units) == 1:
                    parsed = self._parse_output(payload)
                    artifact_refs = []
                    if parsed.source_frame_path and context is not None:
                        artifact_refs.append(
                            context.workspace.store_file_artifact(
                                parsed.source_frame_path,
                                kind="frame",
                                source_tool=self.name,
                                video_id=context.task.video_id or context.task.sample_key,
                            )
                        )
                    lines = [item.dict() for item in parsed.lines]
                    text = str(parsed.text or "").strip()
                    data = {
                        "text": text,
                        "lines": lines,
                        "query": parsed.query,
                        "timestamp_s": parsed.timestamp_s,
                        "source_frame_path": parsed.source_frame_path,
                        "backend": parsed.backend or self.model_name,
                    }
                    if isinstance(payload, dict):
                        for source_key in ("region", "frame", "clip"):
                            if isinstance(payload.get(source_key), dict):
                                data[source_key] = dict(payload.get(source_key) or {})
                    return ToolResult(
                        tool_name=self.name,
                        ok=True,
                        data=data,
                        raw_output_text=raw,
                        artifact_refs=artifact_refs,
                        request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
                        summary=(text or "No text detected.")[:2000],
                        metadata={
                            "backend": data["backend"],
                            **_confidence_metadata(
                                [line.get("confidence") for line in lines],
                                kind="ocr_line",
                            ),
                        },
                    )
                subrequests = []
                for field_name, item in units:
                    single_payload = {"tool_name": self.name, "query": request.query}
                    single_payload["%ss" % field_name] = [item.dict() if hasattr(item, "dict") else item]
                    subrequests.append(self.request_model.parse_obj(single_payload))
                subresults = [self._execute_single(item, context) for item in subrequests]
                artifact_refs = []
                raw_blocks = []
                texts = []
                merged_lines = []
                reads = []
                for field_name, item, subresult in zip(
                    [name for name, _ in units],
                    [value for _, value in units],
                    subresults,
                ):
                    artifact_refs.extend(list(subresult.artifact_refs or []))
                    if subresult.raw_output_text:
                        raw_blocks.append(subresult.raw_output_text)
                    text = str(subresult.data.get("text") or "").strip()
                    if text:
                        texts.append(text)
                    lines = list(subresult.data.get("lines") or [])
                    merged_lines.extend(lines)
                    reads.append(
                        {
                            field_name: item.dict() if hasattr(item, "dict") else item,
                            "text": text,
                            "lines": lines,
                            "timestamp_s": subresult.data.get("timestamp_s"),
                            "source_frame_path": subresult.data.get("source_frame_path"),
                            "backend": subresult.data.get("backend"),
                        }
                    )
                merged_data = {
                    "query": request.query,
                    "text": _join_text_blocks(texts),
                    "lines": merged_lines,
                    "reads": reads,
                    "backend": reads[0].get("backend") if reads else self.model_name,
                }
                return ToolResult(
                    tool_name=self.name,
                    ok=True,
                    data=merged_data,
                    raw_output_text=_join_text_blocks(raw_blocks),
                    artifact_refs=_dedupe_artifact_refs(artifact_refs),
                    request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
                    summary="OCR completed for %d input item(s)." % len(reads),
                    metadata={
                        "backend": merged_data["backend"],
                        "group_count": len(reads),
                        **_confidence_metadata(
                            [line.get("confidence") for line in merged_lines],
                            kind="ocr_line",
                        ),
                    },
                )
            has_embedded_sources = any(
                isinstance(item, dict) and any(isinstance(item.get(source_key), dict) for source_key in ("region", "frame", "clip"))
                for item in batch_results
            )
            if len(batch_results) != len(units) and not has_embedded_sources:
                raise ValueError(
                    "OCR batch runner returned %d result(s) for %d input item(s)." % (len(batch_results), len(units))
                )
            artifact_refs = []
            texts = []
            merged_lines = []
            reads = []
            batch_backend = str(payload.get("backend") or "").strip() if isinstance(payload, dict) else ""
            for index, result_payload in enumerate(batch_results):
                if not isinstance(result_payload, dict):
                    result_payload = {}
                field_name, item = units[index] if index < len(units) else (None, None)
                parsed = self._parse_output(result_payload)
                if parsed.source_frame_path and context is not None:
                    artifact_refs.append(
                        context.workspace.store_file_artifact(
                            parsed.source_frame_path,
                            kind="frame",
                            source_tool=self.name,
                            video_id=context.task.video_id or context.task.sample_key,
                        )
                    )
                lines = [line.dict() for line in parsed.lines]
                text = str(parsed.text or "").strip()
                if text:
                    texts.append(text)
                merged_lines.extend(lines)
                backend = parsed.backend or batch_backend or self.model_name
                read = {
                    "text": text,
                    "lines": lines,
                    "timestamp_s": parsed.timestamp_s,
                    "source_frame_path": parsed.source_frame_path,
                    "backend": backend,
                }
                for source_key in ("region", "frame", "clip"):
                    if isinstance(result_payload.get(source_key), dict):
                        read[source_key] = dict(result_payload.get(source_key) or {})
                if field_name and field_name not in read and item is not None:
                    read[field_name] = item.dict() if hasattr(item, "dict") else item
                reads.append(read)
            merged_data = {
                "query": request.query,
                "text": _join_text_blocks(texts),
                "lines": merged_lines,
                "reads": reads,
                "backend": reads[0].get("backend") if reads else self.model_name,
            }
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data=merged_data,
                raw_output_text=raw,
                artifact_refs=_dedupe_artifact_refs(artifact_refs),
                request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
                summary="OCR completed for %d input item(s)." % len(reads),
                metadata={
                    "backend": merged_data["backend"],
                    "group_count": len(reads),
                    "batch_execution": "single_subprocess",
                    **_confidence_metadata(
                        [line.get("confidence") for line in merged_lines],
                        kind="ocr_line",
                    ),
                },
            )
        return self._execute_single(request, context)


class SpatialGrounderProcessAdapter(BaseProcessToolAdapter):
    request_model = SpatialGrounderRequest
    output_model = SpatialGrounderOutput

    def parse_request(self, arguments: Dict[str, Any]):
        request = super().parse_request(arguments)
        return _request_with_query_sequence_context(self.request_model, request)

    def _run_persisted_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return execute_spatial_grounder_payload(payload, runner_pool=self.model_pool)

    def persistent_preload_spec(self, profile) -> Optional[Dict[str, Any]]:
        return self._qwen_persistent_preload_spec(profile)

    def _execute_single(self, request, context):
        payload, raw = self._run_json(context, request.dict())
        parsed = self._parse_output(payload)
        frame_ref = _spatial_frame_ref_for_result(request, parsed)
        frame_path = parsed.source_frame_path or ((frame_ref.metadata or {}).get("source_path") if frame_ref is not None else None)
        artifact_refs = []
        if frame_path and context is not None:
            artifact_refs.append(
                context.workspace.store_file_artifact(
                    frame_path,
                    kind="frame",
                    source_tool=self.name,
                    video_id=context.task.video_id or context.task.sample_key,
                )
            )
        detections = [item.dict() for item in parsed.detections]
        regions = []
        for item in parsed.detections:
            if item.bbox is None or frame_ref is None:
                continue
            regions.append(
                RegionRef(
                    frame=frame_ref,
                    bbox=item.bbox,
                    label=item.label,
                    metadata={"confidence": item.confidence, **dict(item.metadata or {})},
                ).dict()
            )
        data = {
            "query": parsed.query,
            "frames": [frame_ref.dict()] if frame_ref is not None else [],
            "detections": detections,
            "regions": regions,
            "spatial_description": parsed.spatial_description,
            "backend": parsed.backend or self.model_name,
        }
        confidence_metadata = _confidence_metadata(
            [item.confidence for item in parsed.detections],
            kind="spatial_grounding",
        )
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            artifact_refs=artifact_refs,
            request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
            summary=(parsed.spatial_description or "No grounded detection summary.")[:2000],
            metadata={"backend": parsed.backend or self.model_name, **confidence_metadata},
        )

    def execute(self, request, context):
        request = _request_with_query_sequence_context(self.request_model, request)
        frames = list(getattr(request, "frames", []) or [])
        clips = list(getattr(request, "clips", []) or [])
        media_field = "frames" if frames else "clips"
        media_items = frames if frames else clips
        if len(media_items) > 1:
            subrequests = [
                self.request_model.parse_obj(
                    {
                        "tool_name": self.name,
                        media_field: [item.dict() if hasattr(item, "dict") else item],
                        "query": request.query,
                    }
                )
                for item in media_items
            ]
            subresults = [self._execute_single(item, context) for item in subrequests]
            artifact_refs = []
            raw_blocks = []
            detections = []
            regions = []
            groundings = []
            summaries = []
            for subrequest, subresult in zip(subrequests, subresults):
                artifact_refs.extend(list(subresult.artifact_refs or []))
                if subresult.raw_output_text:
                    raw_blocks.append(subresult.raw_output_text)
                if subresult.summary:
                    summaries.append(subresult.summary)
                detections.extend(list(subresult.data.get("detections") or []))
                regions.extend(list(subresult.data.get("regions") or []))
                grounding_frame = _first_list_item(subresult.data.get("frames") or [])
                groundings.append(
                    {
                        "frames": [grounding_frame] if grounding_frame is not None else [],
                        "detections": list(subresult.data.get("detections") or []),
                        "regions": list(subresult.data.get("regions") or []),
                        "spatial_description": subresult.data.get("spatial_description") or "",
                    }
                )
            merged_data = {
                "query": request.query,
                "frames": [
                    first_frame
                    for first_frame in (_first_list_item(item.data.get("frames") or []) for item in subresults)
                    if first_frame is not None
                ],
                "detections": detections,
                "regions": regions,
                "groundings": groundings,
                "spatial_description": _join_text_blocks([item.get("spatial_description") for item in groundings]),
                "backend": self.model_name,
            }
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data=merged_data,
                raw_output_text=_join_text_blocks(raw_blocks),
                artifact_refs=_dedupe_artifact_refs(artifact_refs),
                request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
                summary="Spatial grounding completed for %d input item(s)." % len(groundings),
                metadata={
                    "backend": self.model_name,
                    "group_count": len(groundings),
                    **_confidence_metadata(
                        [item.get("confidence") for item in detections],
                        kind="spatial_grounding",
                    ),
                },
            )
        return self._execute_single(request, context)


class GenericPurposeProcessAdapter(BaseProcessToolAdapter):
    request_model = GenericPurposeRequest
    output_model = GenericPurposeOutput

    def parse_request(self, arguments: Dict[str, Any]):
        request = super().parse_request(arguments)
        return _generic_request_with_sequence_context(self.request_model, request)

    def _run_persisted_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return execute_generic_purpose_payload(payload, runner_pool=self.model_pool)

    def persistent_preload_spec(self, profile) -> Optional[Dict[str, Any]]:
        return self._qwen_persistent_preload_spec(profile)

    def execute(self, request, context):
        request = _generic_request_with_sequence_context(self.request_model, request)
        payload, raw = self._run_json(context, request.dict())
        parsed = self._parse_output(payload)
        artifact_refs = _request_artifact_refs(request, source_tool=self.name)
        low_signal_output = False
        candidate_chunks = [str(parsed.answer or "").strip(), str(parsed.analysis or "").strip()]
        candidate_chunks.extend(str(item).strip() for item in list(parsed.supporting_points or []) if str(item).strip())
        if candidate_chunks and all(is_low_signal_text(item) for item in candidate_chunks):
            low_signal_output = True
            parsed = GenericPurposeOutput(answer="", analysis="", supporting_points=[], confidence=parsed.confidence)
        quality_flags = generic_output_quality_flags(parsed, raw)
        strict_json_usable = generic_output_strict_json_usable(parsed, raw)
        raw_untrusted = bool(not strict_json_usable and not low_signal_output)
        raw_untrusted_text = ""
        if raw_untrusted:
            raw_untrusted_text = (
                str(raw or "").strip()
                or str(parsed.answer or parsed.analysis or "").strip()
            )
        data = {
            "response": "" if raw_untrusted else parsed.answer,
            "answer": "" if raw_untrusted else parsed.answer,
            "analysis": "" if raw_untrusted else (parsed.analysis or parsed.answer),
            "supporting_points": [] if raw_untrusted else list(parsed.supporting_points or []),
            "confidence": parsed.confidence,
        }
        if raw_untrusted:
            data["raw_untrusted_vlm_observation"] = raw_untrusted_text[:4000]
            data["raw_candidate_answer"] = str(parsed.answer or "").strip()[:1000]
            data["raw_candidate_analysis"] = str(parsed.analysis or "").strip()[:1000]
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            artifact_refs=artifact_refs,
            request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
            summary=(
                "generic_purpose produced a low-signal response and it was omitted from evidence."
                if low_signal_output
                else (
                    "generic_purpose returned raw/untrusted VLM text instead of a usable strict answer; "
                    "treat as candidate context only: %s" % raw_untrusted_text[:1800]
                )
                if raw_untrusted
                else (parsed.answer or parsed.analysis or "")[:2000]
            ),
            metadata={
                "backend": self.model_name,
                "low_signal_output": low_signal_output,
                "strict_json_usable": strict_json_usable,
                "raw_untrusted_vlm_observation": raw_untrusted,
                "quality_flags": quality_flags,
                **_confidence_metadata([parsed.confidence], kind="answer_confidence"),
            },
        )


class VerifierProcessAdapter(BaseProcessToolAdapter):
    request_model = VerifierRequest
    output_model = VerifierOutput

    def parse_request(self, arguments: Dict[str, Any]):
        request = super().parse_request(arguments)
        return _generic_request_with_sequence_context(self.request_model, request)

    def _run_persisted_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return execute_verifier_payload(payload, runner_pool=self.model_pool)

    def persistent_preload_spec(self, profile) -> Optional[Dict[str, Any]]:
        return self._qwen_persistent_preload_spec(profile)

    def execute(self, request, context):
        request = _generic_request_with_sequence_context(self.request_model, request)
        request_payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        task_options = list(getattr(getattr(context, "task", None), "options", []) or [])
        if len(task_options) >= 2 and request_payload.get("verification_mode") == "strict":
            request_payload["verification_mode"] = "mcq_comparative"
        payload, raw = self._run_json(context, request_payload)
        schema_validation_failed = False
        try:
            parsed = self._parse_output(payload)
        except ValidationError as exc:
            schema_validation_failed = True
            payload = _unknown_verifier_output_for_validation_error(request, exc, payload)
            parsed = self._parse_output(payload)
        artifact_refs = _request_artifact_refs(request, source_tool=self.name)
        supported = sum(1 for item in parsed.claim_results if item.verdict == "supported")
        refuted = sum(1 for item in parsed.claim_results if item.verdict == "refuted")
        unknown = sum(1 for item in parsed.claim_results if item.verdict == "unknown")
        partial = sum(1 for item in parsed.claim_results if item.verdict == "partially_supported")
        data = {
            "claim_results": [item.dict() for item in parsed.claim_results],
            "new_observations": list(parsed.new_observations or []),
            "evidence_updates": list(parsed.evidence_updates or []),
            "checklist_updates": list(parsed.checklist_updates or []),
            "counter_updates": list(parsed.counter_updates or []),
            "referent_updates": list(parsed.referent_updates or []),
            "ocr_occurrence_updates": list(parsed.ocr_occurrence_updates or []),
            "unresolved_gaps": list(parsed.unresolved_gaps or []),
        }
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            artifact_refs=artifact_refs,
            request_hash=hash_payload({"tool": self.name, "request": request_payload, "model": self.model_name}),
            summary="Verifier results: supported=%d, refuted=%d, partial=%d, unknown=%d." % (supported, refuted, partial, unknown),
            metadata={
                "backend": self.model_name,
                "supported_count": supported,
                "refuted_count": refuted,
                "partial_count": partial,
                "unknown_count": unknown,
                "schema_validation_failed": schema_validation_failed,
                **_confidence_metadata([item.confidence for item in parsed.claim_results], kind="verifier_claim"),
            },
        )
