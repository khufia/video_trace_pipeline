from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..common import extract_json_object, has_meaningful_text, hash_payload, is_low_signal_text, sanitize_for_persistence
from ..model_cache import describe_model_resolution
from ..runtime_devices import describe_device_mapping, resolve_device_label
from ..tool_wrappers.local_multimodal import TimeChatCaptionerRunner
from ..tool_wrappers.timechat_dense_caption_runner import execute_payload as execute_dense_caption_payload
from ..tool_wrappers.qwen35vl_runner import execute_payload as execute_generic_purpose_payload
from ..tool_wrappers.spatial_grounder_runner import execute_payload as execute_spatial_grounder_payload
from ..tool_wrappers.shared import resolve_generation_controls, resolve_model_path, resolved_device_label
from ..tool_wrappers.timelens_runner import execute_payload as execute_visual_temporal_grounder_payload
from ..schemas import (
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
    VisualTemporalGrounderOutput,
    VisualTemporalGrounderRequest,
)
from .base import ToolAdapter
from .media import get_video_duration


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

    def _runtime_payload(self, context) -> Dict[str, Any]:
        device = resolve_device_label(context.workspace.profile.gpu_assignments.get(self.name))
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
            "scratch_dir": str((context.run.tools_dir / "_scratch" / getattr(self, "name", "")).resolve()),
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
        device = resolve_device_label(profile.gpu_assignments.get(self.name))
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
    ) -> Optional[Dict[str, Any]]:
        if self.model_pool is None or not self.model_pool.should_persist(self.name):
            return None
        runtime = self._runtime_payload_for_profile(profile)
        generation = resolve_generation_controls(runtime)
        model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
        device_label = resolved_device_label(runtime)
        attn_implementation = str((runtime.get("extra") or {}).get("attn_implementation") or "").strip() or None
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
            ),
            "model_name": self.model_name,
            "resolved_model_path": model_path,
            "device_label": device_label,
            "processor_use_fast": processor_use_fast,
            "processor_model_path": processor_model_path,
            "generate_do_sample": bool(generation.get("do_sample")),
            "generate_temperature": generation.get("temperature"),
            "attn_implementation": attn_implementation,
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


def _join_text_blocks(items: List[str]) -> str:
    cleaned = [str(item or "").strip() for item in list(items or []) if str(item or "").strip()]
    return "\n\n".join(cleaned)


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


def _select_multi_clip_frames(frame_groups: List[Dict[str, Any]], total_limit: int) -> List[Dict[str, Any]]:
    if total_limit <= 0:
        return []
    selected: List[Dict[str, Any]] = []
    seen = set()

    def _frame_key(frame: Dict[str, Any]) -> tuple:
        return (
            str(frame.get("artifact_id") or "").strip(),
            str(frame.get("relpath") or "").strip(),
            float(frame.get("timestamp_s") or 0.0),
        )

    ordered_groups = sorted(
        list(frame_groups or []),
        key=lambda item: _frame_rank_sort_key((item.get("frames") or [{}])[0]) if list(item.get("frames") or []) else (0.0, float("inf"), 0.0),
    )
    for group in ordered_groups:
        frames = sorted(list(group.get("frames") or []), key=_frame_rank_sort_key)
        if not frames:
            continue
        key = _frame_key(frames[0])
        if key in seen:
            continue
        seen.add(key)
        selected.append(frames[0])
        if len(selected) >= total_limit:
            return selected

    merged_frames = []
    for group in list(frame_groups or []):
        merged_frames.extend(list(group.get("frames") or []))
    for frame in sorted(merged_frames, key=_frame_rank_sort_key):
        key = _frame_key(frame)
        if key in seen:
            continue
        seen.add(key)
        selected.append(frame)
        if len(selected) >= total_limit:
            break
    return selected


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
                        "clip": item.dict() if hasattr(item, "dict") else item,
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
        clip = dict(request.get("clip") or {})
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
        clip = dict(request.get("clip") or {})
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
        for item in parsed.frames:
            artifact = context.workspace.store_file_artifact(item.frame_path, kind="frame", source_tool=self.name)
            artifact_refs.append(artifact)
            frames.append(
                FrameRef(
                    video_id=context.task.video_id or context.task.sample_key,
                    timestamp_s=float(item.timestamp_s),
                    artifact_id=artifact.artifact_id,
                    relpath=artifact.relpath,
                    clip=request.clip,
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
            if clips:
                for clip in clips:
                    subrequests.append(
                        self.request_model.parse_obj(
                            {
                                "tool_name": self.name,
                                "clip": clip.dict() if hasattr(clip, "dict") else clip,
                                "query": request.query,
                                "num_frames": request.num_frames,
                                "time_hints": time_hints,
                            }
                        )
                    )
            else:
                for time_hint in time_hints:
                    subrequests.append(
                        self.request_model.parse_obj(
                            {
                                "tool_name": self.name,
                                "time_hint": time_hint,
                                "query": request.query,
                                "num_frames": request.num_frames,
                            }
                        )
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
                frame_groups.append(
                    {
                        "clip": subrequest.clip.dict() if getattr(subrequest, "clip", None) is not None else None,
                        "time_hint": getattr(subrequest, "time_hint", None),
                        "frames": group_frames,
                        "cache_metadata": group_cache_metadata,
                        "rationale": subresult.data.get("rationale") or subresult.summary,
                    }
                )
                if group_cache_metadata:
                    cache_groups.append(
                        {
                            "clip": subrequest.clip.dict() if getattr(subrequest, "clip", None) is not None else None,
                            "time_hint": getattr(subrequest, "time_hint", None),
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
            selected_frames = _select_multi_clip_frames(frame_groups, max(1, int(request.num_frames or 1)))
            merged_data = {
                "query": request.query,
                "clips": [item.clip.dict() for item in subrequests if getattr(item, "clip", None) is not None],
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
            artifact = context.workspace.store_file_artifact(frame_path, kind="frame", source_tool=self.name)
            artifact_refs.append(artifact)
            enriched = dict(item)
            enriched["artifact_id"] = artifact.artifact_id
            enriched["relpath"] = artifact.relpath
            sampled_frames.append(enriched)
        data = {
            "clip": parsed.clip.dict(),
            "captions": [item.dict() for item in parsed.captions],
            "overall_summary": parsed.overall_summary,
            "captioned_range": parsed.captioned_range.dict(),
            "sampled_frames": sampled_frames,
            "backend": parsed.backend or self.model_name,
        }
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
                        "clip": item.dict() if hasattr(item, "dict") else item,
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
                caption_groups.append(
                    {
                        "clip": subrequest.clip.dict() if getattr(subrequest, "clip", None) is not None else None,
                        "captions": list(subresult.data.get("captions") or []),
                        "overall_summary": group_summary,
                    }
                )
            merged_data = {
                "clips": [item.clip.dict() for item in subrequests if getattr(item, "clip", None) is not None],
                "captions": captions,
                "overall_summary": _join_text_blocks(summaries),
                "sampled_frames": sampled_frames,
                "caption_groups": caption_groups,
                "backend": self.model_name,
            }
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
        duration = get_video_duration(task.video_path)
        segments = []
        start = 0.0
        try:
            while start < duration:
                end = min(duration, start + effective_clip_duration_s)
                request = self.request_model.parse_obj(
                    {
                        "tool_name": self.name,
                        "clip": {
                            "video_id": task.video_id or task.sample_key,
                            "start_s": start,
                            "end_s": end,
                        },
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

    def _execute_single(self, request, context):
        payload, raw = self._run_json(context, request.dict())
        parsed = self._parse_output(payload)
        artifact_refs = []
        if parsed.source_frame_path:
            artifact_refs.append(
                context.workspace.store_file_artifact(parsed.source_frame_path, kind="frame", source_tool=self.name)
            )
        data = {
            "text": parsed.text,
            "lines": [item.dict() for item in parsed.lines],
            "query": parsed.query,
            "timestamp_s": parsed.timestamp_s,
            "source_frame_path": parsed.source_frame_path,
            "backend": parsed.backend or self.model_name,
        }
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
        regions = list(getattr(request, "regions", []) or [])
        frames = list(getattr(request, "frames", []) or [])
        clips = list(getattr(request, "clips", []) or [])
        if len(regions) > 1 or len(frames) > 1 or len(clips) > 1:
            if regions:
                units = [("region", item) for item in regions]
            elif frames:
                units = [("frame", item) for item in frames]
            else:
                units = [("clip", item) for item in clips]
            payload, raw = self._run_json(context, request.dict())
            batch_results = list(payload.get("results") or []) if isinstance(payload, dict) else []
            if not batch_results:
                subrequests = []
                for field_name, item in units:
                    single_payload = {"tool_name": self.name, "query": request.query}
                    single_payload[field_name] = item.dict() if hasattr(item, "dict") else item
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
            if len(batch_results) != len(units):
                raise ValueError(
                    "OCR batch runner returned %d result(s) for %d input item(s)." % (len(batch_results), len(units))
                )
            artifact_refs = []
            texts = []
            merged_lines = []
            reads = []
            batch_backend = str(payload.get("backend") or "").strip() if isinstance(payload, dict) else ""
            for field_name, item, result_payload in zip(
                [name for name, _ in units],
                [value for _, value in units],
                batch_results,
            ):
                parsed = self._parse_output(result_payload)
                if parsed.source_frame_path:
                    artifact_refs.append(
                        context.workspace.store_file_artifact(parsed.source_frame_path, kind="frame", source_tool=self.name)
                    )
                lines = [line.dict() for line in parsed.lines]
                text = str(parsed.text or "").strip()
                if text:
                    texts.append(text)
                merged_lines.extend(lines)
                backend = parsed.backend or batch_backend or self.model_name
                reads.append(
                    {
                        field_name: item.dict() if hasattr(item, "dict") else item,
                        "text": text,
                        "lines": lines,
                        "timestamp_s": parsed.timestamp_s,
                        "source_frame_path": parsed.source_frame_path,
                        "backend": backend,
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

    def _run_persisted_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return execute_spatial_grounder_payload(payload, runner_pool=self.model_pool)

    def persistent_preload_spec(self, profile) -> Optional[Dict[str, Any]]:
        return self._qwen_persistent_preload_spec(profile)

    def _execute_single(self, request, context):
        payload, raw = self._run_json(context, request.dict())
        parsed = self._parse_output(payload)
        frame_path = parsed.source_frame_path or request.frame.metadata.get("source_path")
        artifact_refs = []
        if frame_path:
            artifact_refs.append(context.workspace.store_file_artifact(frame_path, kind="frame", source_tool=self.name))
        detections = [item.dict() for item in parsed.detections]
        regions = []
        for item in parsed.detections:
            if item.bbox is None:
                continue
            regions.append(
                RegionRef(
                    frame=request.frame,
                    bbox=item.bbox,
                    label=item.label,
                    metadata={"confidence": item.confidence, **dict(item.metadata or {})},
                ).dict()
            )
        data = {
            "query": parsed.query,
            "frame": request.frame.dict(),
            "best_frame": request.frame.dict(),
            "timestamp": parsed.timestamp_s or request.frame.timestamp_s,
            "detections": detections,
            "regions": regions,
            "region": regions[0] if regions else None,
            "puzzle_bbox": regions[0] if regions else None,
            "spatial_description": parsed.spatial_description,
            "analysis": parsed.spatial_description,
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
        frames = list(getattr(request, "frames", []) or [])
        if len(frames) > 1:
            subrequests = [
                self.request_model.parse_obj(
                    {
                        "tool_name": self.name,
                        "frame": item.dict() if hasattr(item, "dict") else item,
                        "query": request.query,
                    }
                )
                for item in frames
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
                groundings.append(
                    {
                        "frame": subrequest.frame.dict() if getattr(subrequest, "frame", None) is not None else None,
                        "timestamp": subresult.data.get("timestamp"),
                        "detections": list(subresult.data.get("detections") or []),
                        "regions": list(subresult.data.get("regions") or []),
                        "spatial_description": subresult.data.get("spatial_description") or "",
                    }
                )
            merged_data = {
                "query": request.query,
                "frames": [item.frame.dict() for item in subrequests if getattr(item, "frame", None) is not None],
                "detections": detections,
                "regions": regions,
                "groundings": groundings,
                "region": regions[0] if regions else None,
                "puzzle_bbox": regions[0] if regions else None,
                "spatial_description": _join_text_blocks([item.get("spatial_description") for item in groundings]),
                "analysis": _join_text_blocks([item.get("spatial_description") for item in groundings]),
                "backend": self.model_name,
            }
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data=merged_data,
                raw_output_text=_join_text_blocks(raw_blocks),
                artifact_refs=_dedupe_artifact_refs(artifact_refs),
                request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
                summary="Spatial grounding completed for %d frame(s)." % len(groundings),
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

    def _run_persisted_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return execute_generic_purpose_payload(payload, runner_pool=self.model_pool)

    def persistent_preload_spec(self, profile) -> Optional[Dict[str, Any]]:
        return self._qwen_persistent_preload_spec(profile)

    def execute(self, request, context):
        payload, raw = self._run_json(context, request.dict())
        parsed = self._parse_output(payload)
        low_signal_output = False
        candidate_chunks = [str(parsed.answer or "").strip(), str(parsed.analysis or "").strip()]
        candidate_chunks.extend(str(item).strip() for item in list(parsed.supporting_points or []) if str(item).strip())
        if candidate_chunks and all(is_low_signal_text(item) for item in candidate_chunks):
            low_signal_output = True
            parsed = GenericPurposeOutput(answer="", analysis="", supporting_points=[], confidence=parsed.confidence)
        data = {
            "response": parsed.answer,
            "answer": parsed.answer,
            "analysis": parsed.analysis or parsed.answer,
            "supporting_points": list(parsed.supporting_points or []),
            "confidence": parsed.confidence,
        }
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            request_hash=hash_payload({"tool": self.name, "request": request.dict(), "model": self.model_name}),
            summary=(
                "generic_purpose produced a low-signal response and it was omitted from evidence."
                if low_signal_output
                else (parsed.answer or parsed.analysis or "")[:2000]
            ),
            metadata={
                "backend": self.model_name,
                "low_signal_output": low_signal_output,
                **_confidence_metadata([parsed.confidence], kind="answer_confidence"),
            },
        )
