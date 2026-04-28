from __future__ import annotations

import contextlib
import difflib
import importlib
import json
import os
import re
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..common import hash_payload
from ..runtime_devices import resolve_device_label
from ..schemas import ASRRequest, ASROutput, ClipRef, ToolResult
from .base import ToolAdapter
from .media import cleanup_temp_path, extract_audio_clip, get_video_duration, normalize_clip_bounds


def _join_text_blocks(items: List[str]) -> str:
    cleaned = [str(item or "").strip() for item in list(items or []) if str(item or "").strip()]
    return "\n\n".join(cleaned)


def _normalize_phrase_text(text: str) -> str:
    cleaned = str(text or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in cleaned)
    return " ".join(cleaned.split())


def _extract_quoted_spans(text: str, quote_char: str) -> List[str]:
    spans: List[str] = []
    source = str(text or "")
    start_index: Optional[int] = None
    for index, char in enumerate(source):
        if char != quote_char:
            continue
        if start_index is None:
            start_index = index + 1
            continue
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if next_char.isalnum():
            continue
        candidate = source[start_index:index].strip()
        if len(candidate) >= 3:
            spans.append(candidate)
        start_index = None
    return spans


def _quoted_task_phrases(question: str) -> List[str]:
    phrases: List[str] = []
    for match in re.findall(r'"([^"\n]{3,})"', str(question or "")):
        text = str(match or "").strip()
        if text:
            phrases.append(text)
    phrases.extend(_extract_quoted_spans(question, "'"))
    ordered: List[str] = []
    seen = set()
    for phrase in phrases:
        normalized = _normalize_phrase_text(phrase)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(phrase)
    return ordered


def _phrase_matches(question: str, transcript_text: str, segments: List[Dict[str, object]]) -> List[Dict[str, object]]:
    phrases = _quoted_task_phrases(question)
    if not phrases:
        return []
    candidates = [str(transcript_text or "").strip()]
    candidates.extend(str(item.get("text") or "").strip() for item in list(segments or []))
    candidates = [item for item in candidates if item]
    if not candidates:
        return []

    matches: List[Dict[str, object]] = []
    for phrase in phrases:
        normalized_phrase = _normalize_phrase_text(phrase)
        best_candidate = ""
        best_score = 0.0
        for candidate in candidates:
            normalized_candidate = _normalize_phrase_text(candidate)
            if not normalized_candidate:
                continue
            score = difflib.SequenceMatcher(None, normalized_phrase, normalized_candidate).ratio()
            if score > best_score:
                best_score = score
                best_candidate = candidate
        if best_candidate:
            matches.append(
                {
                    "phrase": phrase,
                    "matched_text": best_candidate,
                    "similarity": round(float(best_score), 4),
                }
            )
    return matches


def _missing_audio_error(error_text: str) -> bool:
    lowered = str(error_text or "").strip().lower()
    if not lowered:
        return False
    markers = (
        "does not contain any stream",
        "matches no streams",
        "stream map",
        "no such stream",
        "output file #0 does not contain any stream",
    )
    return any(marker in lowered for marker in markers)


def _wav_frame_count(audio_path: str) -> Optional[int]:
    candidate = Path(str(audio_path or "").strip()).expanduser()
    if not candidate.is_file():
        return None
    try:
        with wave.open(str(candidate), "rb") as handle:
            return int(handle.getnframes())
    except Exception:
        return None


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
    frame_count = _wav_frame_count(audio_path)
    if frame_count is not None and frame_count <= 0:
        return {"segments": [], "language": str(language or "").strip() or None}, runtime_warning
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
        try:
            return model.transcribe(audio_path, batch_size=8, **kwargs), runtime_warning
        except IndexError:
            return {"segments": [], "language": str(language or "").strip() or None}, runtime_warning
        except Exception as exc:
            if "list index out of range" in str(exc).strip().lower():
                return {"segments": [], "language": str(language or "").strip() or None}, runtime_warning
            raise


class LocalASRAdapter(ToolAdapter):
    request_model = ASRRequest
    output_model = ASROutput

    def __init__(self, name: str, extra: Optional[dict] = None):
        self.name = name
        self.extra = dict(extra or {})

    def _build_transcript_payload(self, clip_payload, text, segments, backend, extra_metadata=None):
        metadata = {"backend": backend or "whisperx_local"}
        metadata.update(dict(extra_metadata or {}))
        return {
            "transcript_id": "tx_%s"
            % hash_payload(
                {
                    "clip": clip_payload,
                    "text": str(text or "").strip(),
                    "segments": list(segments or []),
                    "backend": backend or "whisperx_local",
                    "metadata": metadata,
                },
                12,
            ),
            "clip": clip_payload,
            "segments": list(segments or []),
            "metadata": metadata,
        }

    def _empty_success_result(self, clip_payload, summary, *, metadata=None):
        transcript_payload = self._build_transcript_payload(
            clip_payload,
            "",
            [],
            "whisperx_local",
            extra_metadata=metadata,
        )
        data = {
            "clips": [clip_payload],
            "transcripts": [transcript_payload],
        }
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=json.dumps(data, ensure_ascii=False),
            artifact_refs=[],
            request_hash=hash_payload({"tool": self.name, "clip": clip_payload}),
            summary=str(summary or "").strip() or "No speech detected.",
            metadata=dict(metadata or {}),
        )

    def _preprocess_segments_for_clip(self, context, clip_payload: Dict[str, object]) -> Optional[List[Dict[str, object]]]:
        bundle = getattr(context, "preprocess_bundle", None)
        transcripts = list(dict(bundle or {}).get("asr_transcripts") or [])
        request_start = float(clip_payload.get("start_s") or 0.0)
        request_end = float(clip_payload.get("end_s") or request_start)
        matched_segments: List[Dict[str, object]] = []
        covered = False
        for transcript in transcripts:
            if not isinstance(transcript, dict):
                continue
            transcript_clip = dict(transcript.get("clip") or {})
            clip_start = float(transcript_clip.get("start_s") or 0.0)
            clip_end = float(transcript_clip.get("end_s") or clip_start)
            if clip_start > request_start + 1e-3 or clip_end < request_end - 1e-3:
                continue
            covered = True
            for segment in list(transcript.get("segments") or []):
                if not isinstance(segment, dict):
                    continue
                start_s = float(segment.get("start_s", segment.get("start", request_start)) or request_start)
                end_s = float(segment.get("end_s", segment.get("end", start_s)) or start_s)
                if end_s < request_start or start_s > request_end:
                    continue
                item = dict(segment)
                item["start_s"] = max(request_start, start_s)
                item["end_s"] = min(request_end, end_s)
                matched_segments.append(item)
        if not covered:
            return None
        return matched_segments

    def _reuse_preprocess_single(self, request, context):
        clip = request.clips[0] if getattr(request, "clips", None) else None
        if clip is None:
            return None
        clip_payload = clip.model_dump()
        segments = self._preprocess_segments_for_clip(context, clip_payload)
        if segments is None:
            return None
        text = " ".join(str(item.get("text") or "").strip() for item in segments if str(item.get("text") or "").strip())
        transcript_payload = self._build_transcript_payload(
            clip_payload,
            text,
            segments,
            "preprocess_reuse",
            extra_metadata={"source": "preprocess_reuse"},
        )
        phrase_matches = _phrase_matches(getattr(context.task, "question", ""), text, segments)
        data = {
            "clips": [clip_payload],
            "transcripts": [transcript_payload],
        }
        if phrase_matches:
            data["phrase_matches"] = phrase_matches
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=data,
            raw_output_text=json.dumps(data, ensure_ascii=False),
            artifact_refs=[],
            request_hash=hash_payload({"tool": self.name, "clip": clip_payload, "source": "preprocess_reuse"}),
            summary=(text or "No speech detected in reused preprocess transcript.")[:2000],
            metadata={"source": "preprocess_reuse"},
        )

    def _execute_single(self, request, context):
        reused = self._reuse_preprocess_single(request, context)
        if reused is not None:
            return reused
        clip = request.clips[0] if getattr(request, "clips", None) else None
        if clip is None:
            raise ValueError("ASR requires at least one clip")
        clip_payload = clip.model_dump()
        start_s, end_s = normalize_clip_bounds(context.task.video_path, clip.start_s, clip.end_s)
        if end_s <= start_s:
            return self._empty_success_result(
                clip_payload,
                "ASR skipped because the grounded clip has no positive duration after bounds normalization.",
                metadata={
                    "warning": "clip_collapsed_after_bounds_normalization",
                    "normalized_start_s": float(start_s),
                    "normalized_end_s": float(end_s),
                },
            )
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
                failure_payload = {
                    "clips": [clip_payload],
                    "transcripts": [],
                    "error": error_text,
                }
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    data=failure_payload,
                    raw_output_text="",
                    artifact_refs=[],
                    request_hash=hash_payload({"tool": self.name, "clip": clip_payload}),
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
            phrase_matches = _phrase_matches(getattr(context.task, "question", ""), text, segments)
            validated_output = ASROutput.model_validate(
                {
                    "clips": [clip_payload],
                    "transcripts": [
                        self._build_transcript_payload(
                            clip_payload,
                            text,
                            segments,
                            "whisperx_local",
                        )
                    ],
                }
            )
            transcript_payload = validated_output.transcripts[0].model_dump()
            data = {
                "clips": [clip_payload],
                "transcripts": [transcript_payload],
            }
            if phrase_matches:
                data["phrase_matches"] = phrase_matches
                best_match = dict(phrase_matches[0])
                data["phrase_match_summary"] = (
                    'Closest ASR match for quoted phrase "%s" is "%s" (similarity=%0.2f).'
                    % (
                        str(best_match.get("phrase") or ""),
                        str(best_match.get("matched_text") or ""),
                        float(best_match.get("similarity") or 0.0),
                    )
                )
            summary_prefix = str(data.get("phrase_match_summary") or "").strip()
            summary_text = text or "No speech detected."
            if summary_prefix:
                summary_text = "%s %s" % (summary_prefix, summary_text)
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data=data,
                raw_output_text=json.dumps(data, ensure_ascii=False),
                artifact_refs=[],
                request_hash=hash_payload({"tool": self.name, "clip": clip_payload}),
                summary=summary_text[:2000],
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
                        "clips": [item.model_dump() if hasattr(item, "model_dump") else item],
                        "speaker_attribution": request.speaker_attribution,
                    }
                )
                for item in clips
            ]
            subresults = [self._execute_single(item, context) for item in subrequests]
            raw_blocks = []
            transcripts = []
            phrase_matches = []
            for subrequest, subresult in zip(subrequests, subresults):
                if subresult.raw_output_text:
                    raw_blocks.append(subresult.raw_output_text)
                transcripts.extend(list(subresult.data.get("transcripts") or []))
                phrase_matches.extend(list(subresult.data.get("phrase_matches") or []))
            data = {
                "clips": [item.clips[0].model_dump() for item in subrequests if list(getattr(item, "clips", []) or [])],
                "transcripts": transcripts,
            }
            if phrase_matches:
                data["phrase_matches"] = phrase_matches
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data=data,
                raw_output_text=_join_text_blocks(raw_blocks),
                request_hash=hash_payload({"tool": self.name, "request": request.model_dump()}),
                summary="ASR completed for %d clip(s)." % len(transcripts),
                metadata={"group_count": len(transcripts)},
            )
        return self._execute_single(request, context)

    def build_preprocess_transcript(self, task, context):
        duration = max(0.0, float(get_video_duration(task.video_path) or 0.0))
        clip = ClipRef(
            video_id=task.video_id or task.sample_key,
            start_s=0.0,
            end_s=duration,
        )
        request = self.request_model.model_validate(
            {
                "tool_name": self.name,
                "clips": [clip.model_dump()],
                "speaker_attribution": False,
            }
        )
        try:
            result = self._execute_single(request, context)
        except Exception as exc:
            if _missing_audio_error(str(exc)):
                return {
                    "clip": clip.model_dump(),
                    "transcripts": [
                        self._build_transcript_payload(clip.model_dump(), "", [], "whisperx_local")
                    ],
                    "warning": str(exc),
                }
            raise
        if getattr(result, "ok", True):
            clips = list(result.data.get("clips") or [])
            return {
                "clip": clips[0] if clips else clip.model_dump(),
                "transcripts": list(result.data.get("transcripts") or []),
            }
        error_text = str((result.metadata or {}).get("error") or "").strip()
        if _missing_audio_error(error_text):
            return {
                "clip": clip.model_dump(),
                "transcripts": [
                    self._build_transcript_payload(clip.model_dump(), "", [], "whisperx_local")
                ],
                "warning": error_text,
            }
        raise RuntimeError(error_text or "ASR preprocess failed")
