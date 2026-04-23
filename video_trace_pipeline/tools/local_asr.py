from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from ..common import hash_payload
from ..runtime_devices import resolve_device_label
from ..schemas import ASRRequest, ASROutput, ClipRef, ToolResult
from .base import ToolAdapter
from .media import cleanup_temp_path, extract_audio_clip, get_video_duration, normalize_clip_bounds


def _join_text_blocks(items: List[str]) -> str:
    cleaned = [str(item or "").strip() for item in list(items or []) if str(item or "").strip()]
    return "\n\n".join(cleaned)


def _dedupe_existing_dirs(paths: List[str]) -> List[Path]:
    deduped: List[Path] = []
    seen = set()
    for raw_path in paths:
        candidate = Path(str(raw_path or "").strip()).expanduser()
        if not str(candidate):
            continue
        try:
            normalized = str(candidate.resolve()) if candidate.exists() else str(candidate)
        except Exception:
            normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.is_dir():
            deduped.append(candidate)
    return deduped


def _whisperx_library_dirs() -> List[Path]:
    candidates: List[str] = []
    for raw_entry in str(os.environ.get("LD_LIBRARY_PATH") or "").split(":"):
        entry = raw_entry.strip()
        if entry:
            candidates.append(entry)

    cuda_home = str(os.environ.get("CUDA_HOME") or "").strip()
    if cuda_home:
        candidates.append(str(Path(cuda_home) / "lib64"))

    for module_name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            module = importlib.import_module(module_name)
            candidates.extend(list(getattr(module, "__path__", []) or []))
        except Exception:
            continue

    try:
        import ctranslate2

        candidates.append(str(Path(ctranslate2.__file__).resolve().parent.parent / "ctranslate2.libs"))
    except Exception:
        pass

    return _dedupe_existing_dirs(candidates)


def _find_shared_library(lib_name: str) -> Optional[Path]:
    for directory in _whisperx_library_dirs():
        candidate = directory / lib_name
        if candidate.exists():
            return candidate
    return None


def _whisperx_gpu_runtime_issue() -> Optional[str]:
    missing = []
    for lib_name in ("libcublas.so.12", "libcudnn_ops_infer.so.8", "libcudnn_cnn_infer.so.8", "libcudnn_adv_infer.so.8"):
        if _find_shared_library(lib_name) is None:
            missing.append(lib_name)
    if not missing:
        return None
    searched = [str(path) for path in _whisperx_library_dirs()]
    if not searched:
        return "missing %s; no candidate CUDA/cuDNN library directories were found" % ", ".join(missing)
    return "missing %s; searched %s" % (", ".join(missing), ", ".join(searched))


def _resolve_whisperx_runtime(device_label: str) -> Tuple[str, Optional[str]]:
    resolved = resolve_device_label(device_label, default="cpu")
    if not resolved.startswith("cuda"):
        return resolved, None
    runtime_issue = _whisperx_gpu_runtime_issue()
    if runtime_issue is None:
        return resolved, None
    warning = (
        "WhisperX GPU runtime unavailable on %s (%s). Falling back to CPU int8 ASR."
        % (resolved, runtime_issue)
    )
    return "cpu", warning


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


@contextlib.contextmanager
def _temporary_torch_load_weights_only_false():
    import torch

    original_load = torch.load
    if getattr(original_load, "_video_trace_pipeline_weights_only_compat", False):
        yield
        return

    def _compat_load(*args, **kwargs):
        # PyTorch 2.6 changed torch.load to default to weights_only=True.
        # WhisperX's pyannote VAD checkpoint path still expects full checkpoint
        # deserialization for trusted local model files. Lightning/pyannote may
        # pass weights_only=None explicitly, which still triggers the new
        # PyTorch default, so normalize both missing and None to False here.
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    _compat_load._video_trace_pipeline_weights_only_compat = True
    torch.load = _compat_load
    try:
        yield
    finally:
        torch.load = original_load


def _transcribe_with_whisperx(
    audio_path: str,
    model_name: str,
    device_label: str,
    language: Optional[str] = None,
) -> Tuple[dict, Optional[str]]:
    import whisperx  # pragma: no cover - heavy optional dependency

    device, runtime_warning = _resolve_whisperx_runtime(device_label)
    device_name = "cpu"
    load_kwargs = {}
    if device.startswith("cuda"):
        device_name = "cuda"
        if ":" in device:
            try:
                load_kwargs["device_index"] = int(device.split(":", 1)[1])
            except Exception:
                load_kwargs["device_index"] = 0
    with _temporary_torch_load_weights_only_false():
        model = whisperx.load_model(
            model_name,
            device_name,
            compute_type="float16" if device_name == "cuda" else "int8",
            **load_kwargs,
        )
        kwargs = {}
        if language:
            kwargs["language"] = language
        return model.transcribe(audio_path, batch_size=8, **kwargs), runtime_warning


class LocalASRAdapter(ToolAdapter):
    request_model = ASRRequest
    output_model = ASROutput

    def __init__(self, name: str, extra: Optional[dict] = None):
        self.name = name
        self.extra = dict(extra or {})

    def _execute_single(self, request, context):
        clip = request.clip or (request.clips[0] if getattr(request, "clips", None) else None)
        if clip is None:
            raise ValueError("ASR requires at least one clip")
        start_s, end_s = normalize_clip_bounds(context.task.video_path, clip.start_s, clip.end_s)
        audio_path = None
        try:
            audio_path = extract_audio_clip(
                context.task.video_path,
                context.workspace.profile.ffmpeg_bin,
                start_s,
                end_s,
            )
            model_name = str(self.extra.get("model_name") or "large-v3")
            language = self.extra.get("language")
            device = context.workspace.profile.gpu_assignments.get("asr", "cpu")
            runtime_warning = None
            try:
                result, runtime_warning = _transcribe_with_whisperx(
                    audio_path,
                    model_name=model_name,
                    device_label=device,
                    language=language,
                )
            except Exception as exc:
                error_text = str(exc)
                failed_data = ASROutput.model_validate(
                    {
                        "clip": clip.model_dump(),
                        "text": "",
                        "segments": [],
                        "backend": "whisperx_local",
                    }
                ).model_dump()
                failed_data["error"] = error_text
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    data=failed_data,
                    raw_output_text="",
                    artifact_refs=[],
                    request_hash=hash_payload({"tool": self.name, "clip": clip.model_dump()}),
                    summary="ASR unavailable.",
                    metadata={"error": error_text},
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
            data = ASROutput.model_validate(
                {
                    "clip": clip.model_dump(),
                    "text": text,
                    "segments": segments,
                    "backend": "whisperx_local",
                }
            ).model_dump()
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data=data,
                raw_output_text=json.dumps(data, ensure_ascii=False),
                artifact_refs=[],
                request_hash=hash_payload({"tool": self.name, "clip": clip.model_dump()}),
                summary=(text or "No speech detected.")[:2000],
                metadata={"runtime_warning": runtime_warning} if runtime_warning else {},
            )
        finally:
            cleanup_temp_path(audio_path)

    def execute(self, request, context):
        clips = list(getattr(request, "clips", []) or [])
        if len(clips) > 1:
            subrequests = [
                self.request_model.model_validate(
                    {
                        "tool_name": self.name,
                        "clip": item.model_dump() if hasattr(item, "model_dump") else item,
                        "speaker_attribution": request.speaker_attribution,
                    }
                )
                for item in clips
            ]
            subresults = [self._execute_single(item, context) for item in subrequests]
            raw_blocks = []
            texts = []
            segments = []
            transcripts = []
            for subrequest, subresult in zip(subrequests, subresults):
                if subresult.raw_output_text:
                    raw_blocks.append(subresult.raw_output_text)
                text = str(subresult.data.get("text") or "").strip()
                if text:
                    texts.append(text)
                sub_segments = list(subresult.data.get("segments") or [])
                segments.extend(sub_segments)
                transcripts.append(
                    {
                        "clip": subrequest.clip.model_dump() if getattr(subrequest, "clip", None) is not None else None,
                        "text": text,
                        "segments": sub_segments,
                        "backend": subresult.data.get("backend") or "whisperx_local",
                    }
                )
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data={
                    "clips": [item.clip.model_dump() for item in subrequests if getattr(item, "clip", None) is not None],
                    "text": _join_text_blocks(texts),
                    "segments": segments,
                    "transcripts": transcripts,
                    "backend": transcripts[0].get("backend") if transcripts else "whisperx_local",
                },
                raw_output_text=_join_text_blocks(raw_blocks),
                request_hash=hash_payload({"tool": self.name, "request": request.model_dump()}),
                summary="ASR completed for %d clip(s)." % len(transcripts),
                metadata={"group_count": len(transcripts)},
            )
        return self._execute_single(request, context)
