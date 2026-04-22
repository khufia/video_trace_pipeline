from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..common import extract_json_object, hash_payload, sanitize_for_persistence
from ..model_cache import describe_model_resolution
from ..runtime_devices import resolve_device_label
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
            return shlex.split(command)
        return [str(item) for item in list(command)]

    def _runtime_payload(self, context) -> Dict[str, Any]:
        device = resolve_device_label(context.workspace.profile.gpu_assignments.get(self.name))
        model_resolution = describe_model_resolution(
            self.model_name,
            hf_cache=context.workspace.profile.hf_cache,
        )
        runtime = {
            "backend": self.extra.get("backend_name") or getattr(self, "name", ""),
            "model_name": self.model_name,
            "device": device,
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

    def _run_command_json(self, context, request_payload: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        command = self._command()
        env = os.environ.copy()
        for key, value in dict(self.extra.get("env") or {}).items():
            env[str(key)] = str(value)
        cwd = self.extra.get("cwd")
        timeout = float(self.extra.get("timeout_s") or 0.0) or None
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
        payload = {
            "tool_name": getattr(self, "name", ""),
            "request": request_payload,
            "task": self._task_payload(context),
            "runtime": self._runtime_payload(context),
            "evidence_records": evidence_records,
        }
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

    def __init__(self, name: str, model_name: str, extra: Optional[Dict[str, Any]] = None):
        self.name = name
        JsonProcessMixin.__init__(self, model_name=model_name, extra=extra)

    def _parse_output(self, payload: Dict[str, Any]):
        model_cls = getattr(self, "output_model", None)
        if model_cls is None:
            return payload
        if hasattr(model_cls, "model_validate"):
            return model_cls.model_validate(payload)
        return model_cls.parse_obj(payload)


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


class VisualTemporalGrounderProcessAdapter(BaseProcessToolAdapter):
    request_model = VisualTemporalGrounderRequest
    output_model = VisualTemporalGrounderOutput

    def execute(self, request, context):
        payload, raw = self._run_command_json(context, request.dict())
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
        payload, raw = self._run_command_json(context, request.dict())
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

    def _execute_single(self, request, context):
        payload, raw = self._run_command_json(context, request.dict())
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
        data = {
            "query": parsed.query,
            "frames": frames,
            "mode": parsed.mode,
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
            metadata={"backend": self.model_name, **confidence_metadata},
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
            for subrequest, subresult in zip(subrequests, subresults):
                group_frames = list(subresult.data.get("frames") or [])
                merged_frames.extend(group_frames)
                frame_groups.append(
                    {
                        "clip": subrequest.clip.dict() if getattr(subrequest, "clip", None) is not None else None,
                        "time_hint": getattr(subrequest, "time_hint", None),
                        "frames": group_frames,
                        "rationale": subresult.data.get("rationale") or subresult.summary,
                    }
                )
                artifact_refs.extend(list(subresult.artifact_refs or []))
                if subresult.raw_output_text:
                    raw_blocks.append(subresult.raw_output_text)
                if subresult.summary:
                    summaries.append(subresult.summary)
            merged_data = {
                "query": request.query,
                "clips": [item.clip.dict() for item in subrequests if getattr(item, "clip", None) is not None],
                "frames": merged_frames,
                "frame_groups": frame_groups,
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
                summary="Retrieved %d frame(s) across %d input clip(s)." % (len(merged_frames), len(frame_groups)),
                metadata={
                    "backend": self.model_name,
                    "group_count": len(frame_groups),
                    **_confidence_metadata(
                        [frame.get("metadata", {}).get("relevance_score") for frame in merged_frames],
                        kind="frame_relevance",
                    ),
                },
            )
        return self._execute_single(request, context)


class DenseCaptionProcessAdapter(BaseProcessToolAdapter):
    request_model = DenseCaptionRequest
    output_model = DenseCaptionOutput

    def _execute_single(self, request, context):
        payload, raw = self._run_command_json(context, request.dict())
        parsed = self._parse_output(payload)
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

    def build_segment_cache(self, task, clip_duration_s: float, context):
        duration = get_video_duration(task.video_path)
        segments = []
        start = 0.0
        while start < duration:
            end = min(duration, start + float(clip_duration_s))
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
            result = self.execute(request, context)
            segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "dense_caption": result.data,
                    "caption_summary": result.data.get("overall_summary", ""),
                }
            )
            if end <= start:
                break
            start = end
        summary = " ".join(item.get("caption_summary", "") for item in segments if item.get("caption_summary")).strip()
        if context.llm_client is not None and segments:
            prompt = "\n".join(
                [
                    "Summarize the full video from the segment summaries below.",
                    "Mention the main setting, entities, and event progression.",
                    "Return plain text only.",
                    "",
                    "\n".join(
                        "%0.1fs-%0.1fs: %s" % (item["start"], item["end"], item.get("caption_summary", ""))
                        for item in segments
                    ),
                ]
            )
            try:
                summary = context.llm_client.complete_text(
                    endpoint_name="default",
                    model_name=self.model_name,
                    system_prompt="You summarize videos from segment summaries.",
                    user_prompt=prompt,
                    temperature=float(self.extra.get("temperature", 0.0) or 0.0),
                    max_tokens=1200,
                )
            except Exception:
                pass
        return {"segments": segments, "summary": summary}


class OCRProcessAdapter(BaseProcessToolAdapter):
    request_model = OCRRequest
    output_model = OCROutput

    def _execute_single(self, request, context):
        payload, raw = self._run_command_json(context, request.dict())
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
            subrequests = []
            for field_name, item in units:
                payload = {"tool_name": self.name, "query": request.query}
                payload[field_name] = item.dict() if hasattr(item, "dict") else item
                subrequests.append(self.request_model.parse_obj(payload))
            subresults = [self._execute_single(item, context) for item in subrequests]
            artifact_refs = []
            raw_blocks = []
            texts = []
            merged_lines = []
            reads = []
            for field_name, item, subresult in zip([name for name, _ in units], [value for _, value in units], subresults):
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
        return self._execute_single(request, context)


class SpatialGrounderProcessAdapter(BaseProcessToolAdapter):
    request_model = SpatialGrounderRequest
    output_model = SpatialGrounderOutput

    def _execute_single(self, request, context):
        payload, raw = self._run_command_json(context, request.dict())
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

    def execute(self, request, context):
        payload, raw = self._run_command_json(context, request.dict())
        parsed = self._parse_output(payload)
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
            summary=(parsed.answer or parsed.analysis or "")[:2000],
            metadata={
                "backend": self.model_name,
                **_confidence_metadata([parsed.confidence], kind="answer_confidence"),
            },
        )
