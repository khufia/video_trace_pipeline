from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..common import extract_json_object, hash_payload
from ..schemas import (
    ASRRequest,
    AudioTemporalGrounderRequest,
    ClipRef,
    DenseCaptionRequest,
    FrameRef,
    FrameRetrieverRequest,
    GenericPurposeRequest,
    OCRRequest,
    RegionRef,
    SpatialGrounderRequest,
    ToolResult,
    VisualTemporalGrounderRequest,
)
from .base import ToolAdapter
from .media import cleanup_temp_path, extract_audio_clip, get_video_duration, midpoint_frame, normalize_clip_bounds, sample_frames


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[A-Za-z0-9']+", str(text or "").lower()) if len(token) >= 2]


def _score_overlap(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    text_tokens = set(_tokenize(text))
    if not query_tokens:
        return 0.0
    if not text_tokens:
        return 0.0
    overlap = query_tokens.intersection(text_tokens)
    return float(len(overlap)) / float(len(query_tokens))


def _clip_from_time_hint(video_id: str, video_path: str, time_hint: Optional[str]) -> Optional[ClipRef]:
    hint = str(time_hint or "").strip().lower()
    if not hint:
        return None
    duration = max(0.0, float(get_video_duration(video_path) or 0.0))
    if duration <= 0.0:
        return ClipRef(video_id=video_id, start_s=0.0, end_s=0.0, metadata={"time_hint": time_hint})

    anchor = None
    if any(token in hint for token in ("last", "final", "ending", "end")):
        anchor = "end"
    elif any(token in hint for token in ("first", "opening", "start", "beginning")):
        anchor = "start"
    elif "middle" in hint or "mid" in hint:
        anchor = "middle"
    if anchor is None:
        anchor = "end"

    window = None
    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", hint)
    if percent_match:
        fraction = max(0.01, min(1.0, float(percent_match.group(1)) / 100.0))
        window = duration * fraction
    else:
        seconds_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", hint)
        if seconds_match:
            window = float(seconds_match.group(1))

    if window is None:
        window = duration * 0.2
    window = max(1.0, min(duration, window))

    if anchor == "start":
        start_s = 0.0
        end_s = min(duration, window)
    elif anchor == "middle":
        center = duration / 2.0
        start_s = max(0.0, center - (window / 2.0))
        end_s = min(duration, start_s + window)
    else:
        end_s = duration
        start_s = max(0.0, end_s - window)
    return ClipRef(
        video_id=video_id,
        start_s=round(float(start_s), 3),
        end_s=round(float(max(start_s, end_s)), 3),
        metadata={"time_hint": time_hint},
    )


class OpenAIVisionMixin(object):
    def __init__(self, endpoint_name: str, model_name: str, extra: Optional[Dict[str, Any]] = None):
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        self.extra = dict(extra or {})

    def _vision_json(
        self,
        context,
        system_prompt: str,
        user_prompt: str,
        image_paths: List[str],
        fallback: Dict[str, Any],
        max_tokens: int = 1800,
    ) -> Tuple[Dict[str, Any], str]:
        if context.llm_client is None:
            return dict(fallback), ""
        text = context.llm_client.complete_text(
            endpoint_name=self.endpoint_name,
            model_name=self.model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=max_tokens,
            image_paths=image_paths or None,
        )
        payload = extract_json_object(text)
        if payload is None or not isinstance(payload, dict):
            return dict(fallback), text
        merged = dict(fallback)
        merged.update(payload)
        return merged, text


class DenseCaptionToolAdapter(ToolAdapter, OpenAIVisionMixin):
    request_model = DenseCaptionRequest

    def __init__(self, name: str, endpoint_name: str, model_name: str, extra: Optional[Dict[str, Any]] = None):
        self.name = name
        OpenAIVisionMixin.__init__(self, endpoint_name=endpoint_name, model_name=model_name, extra=extra)

    def describe_clip(self, request, context, persist_dir: Path):
        start_s, end_s = normalize_clip_bounds(context.task.video_path, request.clip.start_s, request.clip.end_s)
        sample_count = int(self.extra.get("sample_frames", 8) or 8)
        if str(request.granularity or "segment").lower() == "frame":
            sample_count = max(sample_count, 10)
        sampled = sample_frames(
            context.task.video_path,
            start_s,
            end_s,
            sample_count,
            str(persist_dir),
            prefix="dense_caption",
        )
        image_paths = [item["frame_path"] for item in sampled]
        artifact_refs = [
            context.workspace.store_file_artifact(item["frame_path"], kind="frame", source_tool=self.name)
            for item in sampled
        ]
        fallback = {
            "captioned_range": {"start": start_s, "end": end_s},
            "captions": [
                {
                    "start": start_s,
                    "end": end_s,
                    "visual": "Unable to produce a detailed caption from sampled frames.",
                    "audio": "unknown",
                    "on_screen_text": "none",
                    "actions": [],
                    "objects": [],
                }
            ],
            "overall_summary": "No detailed summary available.",
        }
        system_prompt = (
            "You are a dense captioning tool for sampled video frames. "
            "Return JSON only. Be factual. Never mention tools or hidden reasoning."
        )
        user_prompt = "\n".join(
            [
                "Describe the video clip represented by these frames.",
                "The frames are in chronological order.",
                "Clip range: %.3fs to %.3fs." % (start_s, end_s),
                "Granularity: %s." % request.granularity,
                "Focus query: %s" % (request.focus_query or "none"),
                'Return JSON with keys: captioned_range, captions, overall_summary.',
                'Each caption must have: start, end, visual, audio, on_screen_text, actions, objects.',
            ]
        )
        payload, raw = self._vision_json(
            context=context,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=image_paths,
            fallback=fallback,
            max_tokens=2200,
        )
        data = {
            "clip": request.clip.dict(),
            "captions": list(payload.get("captions") or fallback["captions"]),
            "overall_summary": str(payload.get("overall_summary") or "").strip() or fallback["overall_summary"],
            "captioned_range": payload.get("captioned_range") or {"start": start_s, "end": end_s},
            "sampled_frames": [
                {
                    "timestamp_s": item["timestamp_s"],
                    "artifact_id": artifact.artifact_id,
                    "relpath": artifact.relpath,
                }
                for item, artifact in zip(sampled, artifact_refs)
            ],
        }
        summary = data["overall_summary"]
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            artifact_refs=artifact_refs,
            request_hash=hash_payload({"tool": self.name, "clip": request.clip.dict(), "focus_query": request.focus_query}),
            summary=summary[:2000],
            metadata={"sample_frame_count": len(sampled)},
        )

    def build_segment_cache(self, task, clip_duration_s: float, context):
        duration = get_video_duration(task.video_path)
        segments = []
        start = 0.0
        segment_index = 0
        while start < duration:
            end = min(duration, start + float(clip_duration_s))
            clip = ClipRef(video_id=task.video_id or task.sample_key, start_s=start, end_s=end)
            request = self.request_model.parse_obj(
                {
                    "tool_name": self.name,
                    "clip": clip.dict(),
                    "granularity": "segment",
                    "focus_query": "",
                }
            )
            persist_dir = context.workspace.preprocess_dir(
                video_fingerprint_value=context.workspace.video_fingerprint(task.video_path),
                model_id=self.model_name or "dense_captioner",
                clip_duration_s=clip_duration_s,
                prompt_version=context.models_config.tools[self.name].prompt_version,
            ) / "frames" / ("%03d" % segment_index)
            result = self.describe_clip(request=request, context=context, persist_dir=persist_dir)
            segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "dense_caption": result.data,
                    "caption_summary": result.data.get("overall_summary", ""),
                }
            )
            start = end
            segment_index += 1
            if end <= start:
                break
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
                    endpoint_name=self.endpoint_name,
                    model_name=self.model_name,
                    system_prompt="You summarize videos from segment summaries.",
                    user_prompt=prompt,
                    temperature=0.1,
                    max_tokens=1200,
                )
            except Exception:
                pass
        return {"segments": segments, "summary": summary}

    def execute(self, request, context):
        persist_dir = context.run.tools_dir / "_scratch" / self.name / request.clip.video_id / (
            "%0.3f_%0.3f" % (request.clip.start_s, request.clip.end_s)
        )
        persist_dir.mkdir(parents=True, exist_ok=True)
        return self.describe_clip(request=request, context=context, persist_dir=persist_dir)


class TemporalGrounderToolAdapter(ToolAdapter):
    request_model = VisualTemporalGrounderRequest

    def __init__(self, name: str, top_k: int = 5):
        self.name = name
        self.top_k = int(top_k or 5)

    def execute(self, request, context):
        preprocess_bundle = context.preprocess_bundle or {}
        segments = list(preprocess_bundle.get("segments") or [])
        ranked = []
        for segment in segments:
            dense = segment.get("dense_caption") or {}
            segment_text = " ".join(
                [
                    str(segment.get("caption_summary") or ""),
                    str(dense.get("overall_summary") or ""),
                    json.dumps(dense.get("captions") or [], ensure_ascii=False),
                ]
            )
            score = _score_overlap(request.query, segment_text)
            if score > 0.0:
                ranked.append(
                    {
                        "start": float(segment.get("start", 0.0) or 0.0),
                        "end": float(segment.get("end", 0.0) or 0.0),
                        "confidence": round(score, 4),
                    }
                )
        if not ranked and segments:
            for segment in segments[: self.top_k]:
                ranked.append(
                    {
                        "start": float(segment.get("start", 0.0) or 0.0),
                        "end": float(segment.get("end", 0.0) or 0.0),
                        "confidence": 0.0,
                    }
                )
        ranked = sorted(ranked, key=lambda item: (-float(item["confidence"]), float(item["start"])))[: request.top_k]
        data = {
            "query": request.query,
            "clips": [
                ClipRef(
                    video_id=context.task.video_id or context.task.sample_key,
                    start_s=item["start"],
                    end_s=item["end"],
                    metadata={"confidence": item["confidence"]},
                ).dict()
                for item in ranked
            ],
            "video_duration": get_video_duration(context.task.video_path),
        }
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=json.dumps(data, ensure_ascii=False),
            request_hash=hash_payload({"tool": self.name, "query": request.query}),
            summary="Matched %d candidate clips for query: %s" % (len(data["clips"]), request.query),
            metadata={},
        )


class FrameRetrieverToolAdapter(ToolAdapter, OpenAIVisionMixin):
    request_model = FrameRetrieverRequest

    def __init__(self, name: str, endpoint_name: str, model_name: str, extra: Optional[Dict[str, Any]] = None):
        self.name = name
        OpenAIVisionMixin.__init__(self, endpoint_name=endpoint_name, model_name=model_name, extra=extra)

    def execute(self, request, context):
        clip = request.clip
        mode = "clip_bounded"
        if clip is None:
            clip = _clip_from_time_hint(
                video_id=context.task.video_id or context.task.sample_key,
                video_path=context.task.video_path,
                time_hint=request.time_hint,
            )
            mode = "time_hint_bounded"
        if clip is None:
            clip = ClipRef(
                video_id=context.task.video_id or context.task.sample_key,
                start_s=0.0,
                end_s=get_video_duration(context.task.video_path),
                metadata={"time_hint": request.time_hint or ""},
            )
            mode = "full_video_fallback"
        start_s, end_s = normalize_clip_bounds(context.task.video_path, clip.start_s, clip.end_s)
        candidate_count = max(int(self.extra.get("candidate_frames", 8) or 8), int(request.num_frames))
        persist_dir = context.run.tools_dir / "_scratch" / self.name / clip.video_id / (
            "%0.3f_%0.3f" % (clip.start_s, clip.end_s)
        )
        persist_dir.mkdir(parents=True, exist_ok=True)
        candidates = sample_frames(
            context.task.video_path,
            start_s,
            end_s,
            candidate_count,
            str(persist_dir),
            prefix="frame_retriever",
        )
        selected = list(candidates[: request.num_frames])
        raw = ""
        if request.query and candidates and context.llm_client is not None:
            fallback = {
                "selected_indices": list(range(1, min(len(candidates), int(request.num_frames)) + 1)),
                "reasoning": "Uniform fallback selection.",
            }
            payload, raw = self._vision_json(
                context=context,
                system_prompt=(
                    "You select the most relevant frames from a chronological set. "
                    "Return JSON only with selected_indices and reasoning."
                ),
                user_prompt="\n".join(
                    [
                        "Choose the frames that best answer this query:",
                        request.query,
                        "",
                        "Frames are in chronological order and are implicitly numbered starting at 1.",
                        "Return selected_indices as a list of 1-based indices ordered by usefulness.",
                    ]
                ),
                image_paths=[item["frame_path"] for item in candidates],
                fallback=fallback,
                max_tokens=900,
            )
            chosen = []
            for item in payload.get("selected_indices") or []:
                try:
                    index = int(item) - 1
                except Exception:
                    continue
                if 0 <= index < len(candidates):
                    chosen.append(candidates[index])
            dedup = []
            seen = set()
            for item in chosen:
                key = item["frame_path"]
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(item)
            if dedup:
                selected = dedup[: request.num_frames]

        artifact_refs = [
            context.workspace.store_file_artifact(item["frame_path"], kind="frame", source_tool=self.name)
            for item in selected
        ]
        frames = []
        for item, artifact in zip(selected, artifact_refs):
            frames.append(
                FrameRef(
                    video_id=context.task.video_id or context.task.sample_key,
                    timestamp_s=float(item["timestamp_s"]),
                    artifact_id=artifact.artifact_id,
                    relpath=artifact.relpath,
                    clip=clip,
                    metadata={"source_path": item["frame_path"]},
                ).dict()
            )
        data = {"query": request.query, "frames": frames, "mode": mode}
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            artifact_refs=artifact_refs,
            request_hash=hash_payload(
                {
                    "tool": self.name,
                    "clip": clip.dict(),
                    "query": request.query,
                    "time_hint": request.time_hint,
                }
            ),
            summary="Retrieved %d frame(s) from %.2fs-%.2fs." % (len(frames), start_s, end_s),
            metadata={},
        )


class OCRToolAdapter(ToolAdapter, OpenAIVisionMixin):
    request_model = OCRRequest

    def __init__(self, name: str, endpoint_name: str, model_name: str, extra: Optional[Dict[str, Any]] = None):
        self.name = name
        OpenAIVisionMixin.__init__(self, endpoint_name=endpoint_name, model_name=model_name, extra=extra)

    def execute(self, request, context):
        persist_dir = context.run.tools_dir / "_scratch" / self.name
        persist_dir.mkdir(parents=True, exist_ok=True)
        frame_path = None
        timestamp = None
        artifact_refs = []
        if request.region is not None:
            frame_path = request.region.frame.metadata.get("source_path")
            timestamp = request.region.frame.timestamp_s
        elif request.frame is not None:
            frame_path = request.frame.metadata.get("source_path")
            timestamp = request.frame.timestamp_s
        elif request.clip is not None:
            sampled = midpoint_frame(
                context.task.video_path,
                request.clip.start_s,
                request.clip.end_s,
                str(persist_dir),
                prefix="ocr",
            )
            if sampled:
                frame_path = sampled["frame_path"]
                timestamp = sampled["timestamp_s"]
        if frame_path:
            artifact_refs.append(context.workspace.store_file_artifact(frame_path, kind="frame", source_tool=self.name))
        fallback = {"text": "", "lines": []}
        payload, raw = self._vision_json(
            context=context,
            system_prompt="You are an OCR tool. Return JSON only with text and lines.",
            user_prompt="\n".join(
                [
                    "Read all visible text from the image.",
                    "Return JSON with keys: text, lines.",
                    "Each line should be an object with text and optional bbox [x1,y1,x2,y2].",
                    "Query focus: %s" % (request.query or "none"),
                ]
            ),
            image_paths=[frame_path] if frame_path else [],
            fallback=fallback,
            max_tokens=1000,
        )
        data = {
            "text": str(payload.get("text") or "").strip(),
            "lines": list(payload.get("lines") or []),
            "query": request.query,
            "timestamp_s": timestamp,
        }
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            artifact_refs=artifact_refs,
            request_hash=hash_payload({"tool": self.name, "query": request.query, "timestamp": timestamp}),
            summary=(data["text"] or "No text detected.")[:2000],
            metadata={},
        )


class SpatialGrounderToolAdapter(ToolAdapter, OpenAIVisionMixin):
    request_model = SpatialGrounderRequest

    def __init__(self, name: str, endpoint_name: str, model_name: str, extra: Optional[Dict[str, Any]] = None):
        self.name = name
        OpenAIVisionMixin.__init__(self, endpoint_name=endpoint_name, model_name=model_name, extra=extra)

    def execute(self, request, context):
        frame_path = request.frame.metadata.get("source_path")
        artifact_refs = []
        if frame_path:
            artifact_refs.append(context.workspace.store_file_artifact(frame_path, kind="frame", source_tool=self.name))
        fallback = {"detections": [], "spatial_description": ""}
        payload, raw = self._vision_json(
            context=context,
            system_prompt="You are a spatial grounding tool. Return JSON only.",
            user_prompt="\n".join(
                [
                    "Locate the queried target in the image.",
                    "Query: %s" % request.query,
                    "Return JSON with keys: detections, spatial_description.",
                    "Each detection should have: label, bbox, confidence.",
                    "Use approximate pixel coordinates when possible.",
                ]
            ),
            image_paths=[frame_path] if frame_path else [],
            fallback=fallback,
            max_tokens=1000,
        )
        detections = []
        regions = []
        for item in payload.get("detections") or []:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                bbox = [float(v) for v in bbox]
            else:
                bbox = None
            label = str(item.get("label") or request.query).strip() or request.query
            confidence = float(item.get("confidence", 0.5) or 0.5)
            detections.append({"label": label, "bbox": bbox, "confidence": confidence})
            if bbox is not None:
                regions.append(
                    RegionRef(
                        frame=request.frame,
                        bbox=bbox,
                        label=label,
                        metadata={"confidence": confidence},
                    ).dict()
                )
        data = {
            "query": request.query,
            "frame": request.frame.dict(),
            "best_frame": request.frame.dict(),
            "timestamp": request.frame.timestamp_s,
            "detections": detections,
            "regions": regions,
            "region": regions[0] if regions else None,
            "puzzle_bbox": regions[0] if regions else None,
            "spatial_description": str(payload.get("spatial_description") or "").strip(),
            "analysis": str(payload.get("spatial_description") or "").strip(),
        }
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            artifact_refs=artifact_refs,
            request_hash=hash_payload({"tool": self.name, "query": request.query, "timestamp": request.frame.timestamp_s}),
            summary=(data["spatial_description"] or "No grounded detection summary.")[:2000],
            metadata={},
        )


def _transcribe_with_whisperx(audio_path: str, model_name: str, device_label: str, language: Optional[str] = None):
    import whisperx  # pragma: no cover - heavy optional dependency

    device = str(device_label or "cpu")
    device_name = "cpu"
    device_index = None
    if device.startswith("cuda"):
        device_name = "cuda"
        if ":" in device:
            try:
                device_index = int(device.split(":", 1)[1])
            except Exception:
                device_index = 0
    model = whisperx.load_model(
        model_name,
        device_name,
        device_index=device_index,
        compute_type="float16" if device_name == "cuda" else "int8",
    )
    kwargs = {}
    if language:
        kwargs["language"] = language
    result = model.transcribe(audio_path, batch_size=8, **kwargs)
    return result


class ASRToolAdapter(ToolAdapter):
    request_model = ASRRequest

    def __init__(self, name: str, extra: Optional[Dict[str, Any]] = None):
        self.name = name
        self.extra = dict(extra or {})

    def execute(self, request, context):
        start_s, end_s = normalize_clip_bounds(context.task.video_path, request.clip.start_s, request.clip.end_s)
        audio_path = None
        try:
            audio_path = extract_audio_clip(
                context.task.video_path,
                context.workspace.profile.ffmpeg_bin,
                start_s,
                end_s,
            )
            model_name = str(self.extra.get("model_name") or "small")
            language = self.extra.get("language")
            device = context.workspace.profile.gpu_assignments.get("asr", "cpu")
            try:
                result = _transcribe_with_whisperx(audio_path, model_name=model_name, device_label=device, language=language)
            except Exception as exc:
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    data={"clip": request.clip.dict(), "text": "", "segments": [], "error": str(exc)},
                    raw_output_text="",
                    artifact_refs=[],
                    request_hash=hash_payload({"tool": self.name, "clip": request.clip.dict()}),
                    summary="ASR unavailable: %s" % exc,
                    metadata={},
                )
            segments = []
            for item in result.get("segments") or []:
                start = float(item.get("start", 0.0) or 0.0) + start_s
                end = float(item.get("end", start) or start) + start_s
                segments.append(
                    {
                        "start_s": start,
                        "end_s": end,
                        "text": str(item.get("text") or "").strip(),
                        "speaker_id": None,
                        "confidence": float(item.get("avg_logprob", 0.0) or 0.0) if item.get("avg_logprob") is not None else None,
                    }
                )
            text = " ".join(segment["text"] for segment in segments).strip()
            data = {"clip": request.clip.dict(), "text": text, "segments": segments, "backend": "whisperx_local"}
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data=data,
                raw_output_text=json.dumps(data, ensure_ascii=False),
                artifact_refs=[],
                request_hash=hash_payload({"tool": self.name, "clip": request.clip.dict()}),
                summary=(text or "No speech detected.")[:2000],
                metadata={},
            )
        finally:
            cleanup_temp_path(audio_path)


class AudioTemporalGrounderToolAdapter(ToolAdapter):
    request_model = AudioTemporalGrounderRequest

    def __init__(self, name: str, asr_adapter: ASRToolAdapter):
        self.name = name
        self.asr_adapter = asr_adapter

    def execute(self, request, context):
        clip = request.clip
        if clip is None:
            clip = ClipRef(
                video_id=context.task.video_id or context.task.sample_key,
                start_s=0.0,
                end_s=get_video_duration(context.task.video_path),
            )
        asr_request = ASRRequest(tool_name="asr", clip=clip, speaker_attribution=True)
        asr_result = self.asr_adapter.execute(asr_request, context)
        transcript = str(asr_result.data.get("text") or "").strip()
        score = _score_overlap(request.query, transcript)
        clips = []
        if transcript and score > 0.0:
            clips.append(
                ClipRef(
                    video_id=clip.video_id,
                    start_s=clip.start_s,
                    end_s=clip.end_s,
                    metadata={"confidence": round(score, 4)},
                ).dict()
            )
        data = {
            "query": request.query,
            "clips": clips,
            "events": [
                {
                    "start": clip.start_s,
                    "end": clip.end_s,
                    "confidence": round(score, 4),
                    "event_label": request.query,
                }
            ]
            if clips
            else [],
            "backend": "transcript_search",
        }
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=json.dumps(data, ensure_ascii=False),
            artifact_refs=[],
            request_hash=hash_payload({"tool": self.name, "query": request.query, "clip": clip.dict()}),
            summary="Found %d transcript-matched audio interval(s)." % len(clips),
            metadata={},
        )


class OpenAIMultimodalToolAdapter(ToolAdapter, OpenAIVisionMixin):
    request_model = GenericPurposeRequest

    def __init__(self, name: str, endpoint_name: str, model_name: str, extra: Optional[Dict[str, Any]] = None):
        self.name = name
        OpenAIVisionMixin.__init__(self, endpoint_name=endpoint_name, model_name=model_name, extra=extra)

    def execute(self, request, context):
        prompt_parts = [
            "Answer the query using the provided evidence and media references.",
            "Return JSON with keys: answer, supporting_points, confidence.",
            "Do not mention hidden tools or APIs.",
            "",
            "QUERY:",
            request.query,
        ]
        image_paths = []
        artifact_refs = []
        if request.transcript is not None:
            prompt_parts.extend(["", "TRANSCRIPT:", request.transcript.text])
        if context.evidence_lookup is not None and request.evidence_ids:
            prompt_parts.extend(["", "EVIDENCE:"])
            for item in context.evidence_lookup(request.evidence_ids):
                prompt_parts.append("- %s" % item.get("atomic_text", ""))
        if request.frame is not None:
            source_path = request.frame.metadata.get("source_path")
            if source_path:
                image_paths.append(source_path)
                artifact_refs.append(context.workspace.store_file_artifact(source_path, kind="frame", source_tool=self.name))
        elif request.clip is not None:
            persist_dir = context.run.tools_dir / "_scratch" / self.name
            persist_dir.mkdir(parents=True, exist_ok=True)
            sampled = sample_frames(
                context.task.video_path,
                request.clip.start_s,
                request.clip.end_s,
                4,
                str(persist_dir),
                prefix="generic",
            )
            for item in sampled:
                image_paths.append(item["frame_path"])
                artifact_refs.append(context.workspace.store_file_artifact(item["frame_path"], kind="frame", source_tool=self.name))
        payload, raw = self._vision_json(
            context=context,
            system_prompt="You are a careful multimodal extraction tool.",
            user_prompt="\n".join(prompt_parts),
            image_paths=image_paths,
            fallback={"answer": "", "supporting_points": [], "confidence": 0.0},
            max_tokens=1500,
        )
        answer = str(payload.get("answer") or "").strip()
        data = {
            "response": answer,
            "answer": answer,
            "analysis": answer,
            "supporting_points": list(payload.get("supporting_points") or []),
            "confidence": payload.get("confidence"),
        }
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=raw,
            artifact_refs=artifact_refs,
            request_hash=hash_payload({"tool": self.name, "query": request.query, "evidence_ids": request.evidence_ids}),
            summary=(answer or raw or "")[:2000],
            metadata={},
        )
