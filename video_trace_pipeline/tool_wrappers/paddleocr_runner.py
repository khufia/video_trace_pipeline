from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
import os
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .protocol import emit_json, fail_runtime, load_request
from .shared import (
    absolute_frame_path,
    crop_region,
    ensure_frame_for_request,
    resolved_device_label,
    scratch_dir,
    workspace_root_from_runtime,
)
from ..tools.media import sample_frames


def _normalize_bbox(raw_bbox: Any) -> List[float] | None:
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, Sequence) and len(raw_bbox) == 4 and all(isinstance(item, (int, float)) for item in raw_bbox):
        return [float(item) for item in raw_bbox]
    if not isinstance(raw_bbox, Sequence):
        return None
    xs: List[float] = []
    ys: List[float] = []
    for point in raw_bbox:
        if not isinstance(point, Sequence) or len(point) < 2:
            continue
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except Exception:
            continue
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _normalize_confidence(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_line(*, text: Any, bbox: Any, confidence: Any) -> Dict[str, Any] | None:
    cleaned = _normalize_text(text)
    if not cleaned:
        return None
    return {
        "text": cleaned,
        "bbox": _normalize_bbox(bbox),
        "confidence": _normalize_confidence(confidence),
    }


def _looks_like_legacy_line(item: Any) -> bool:
    return isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2


def _parse_legacy_lines(items: Iterable[Any]) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    for item in items:
        if not _looks_like_legacy_line(item):
            continue
        bbox = item[0]
        payload = item[1]
        text = ""
        confidence = None
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)) and payload:
            text = payload[0] if len(payload) > 0 else ""
            confidence = payload[1] if len(payload) > 1 else None
        else:
            text = payload
        normalized = _normalize_line(text=text, bbox=bbox, confidence=confidence)
        if normalized is not None:
            lines.append(normalized)
    return lines


def _mapping_value(mapping: Dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if value is not None:
        return value
    for candidate_key, candidate_value in mapping.items():
        if str(candidate_key) == key:
            return candidate_value
    return None


def _extract_lines(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        explicit_lines = _mapping_value(payload, "lines")
        if isinstance(explicit_lines, list):
            normalized_explicit = []
            for item in explicit_lines:
                if not isinstance(item, dict):
                    continue
                normalized = _normalize_line(
                    text=item.get("text"),
                    bbox=item.get("bbox"),
                    confidence=item.get("confidence"),
                )
                if normalized is not None:
                    normalized_explicit.append(normalized)
            if normalized_explicit:
                return normalized_explicit

        texts = _mapping_value(payload, "rec_texts")
        if not isinstance(texts, list):
            texts = _mapping_value(payload, "texts")
        if isinstance(texts, list):
            boxes = _mapping_value(payload, "dt_polys")
            if not isinstance(boxes, list):
                boxes = _mapping_value(payload, "boxes")
            if not isinstance(boxes, list):
                boxes = _mapping_value(payload, "polygons")
            scores = _mapping_value(payload, "rec_scores")
            if not isinstance(scores, list):
                scores = _mapping_value(payload, "scores")
            lines = []
            for index, text in enumerate(texts):
                bbox = boxes[index] if isinstance(boxes, list) and index < len(boxes) else None
                confidence = scores[index] if isinstance(scores, list) and index < len(scores) else None
                normalized = _normalize_line(text=text, bbox=bbox, confidence=confidence)
                if normalized is not None:
                    lines.append(normalized)
            if lines:
                return lines

        for nested_key in ("result", "results", "res", "data", "ocr_res"):
            nested = _mapping_value(payload, nested_key)
            nested_lines = _extract_lines(nested)
            if nested_lines:
                return nested_lines
        return []

    if isinstance(payload, list):
        if payload and all(_looks_like_legacy_line(item) for item in payload):
            return _parse_legacy_lines(payload)
        lines: List[Dict[str, Any]] = []
        for item in payload:
            lines.extend(_extract_lines(item))
        return lines

    extracted_attrs = {}
    for attr_name in ("lines", "rec_texts", "texts", "rec_scores", "scores", "dt_polys", "boxes", "polygons", "result", "results", "res", "data"):
        if hasattr(payload, attr_name):
            extracted_attrs[attr_name] = getattr(payload, attr_name)
    if extracted_attrs:
        return _extract_lines(extracted_attrs)
    return []


def _prepare_ocr_image(frame_path: str, out_dir: Path, *, max_longest_dim: int) -> str:
    source = Path(frame_path).resolve()
    with Image.open(source) as image:
        image = image.convert("RGB")
        longest_dim = max(image.size)
        if longest_dim <= max_longest_dim:
            return str(source)
        scale = float(max_longest_dim) / float(longest_dim)
        resized = image.resize(
            (
                max(1, int(round(image.size[0] * scale))),
                max(1, int(round(image.size[1] * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
        prepared_path = out_dir / ("ocr_prepared_%s.png" % source.stem)
        resized.save(prepared_path)
        return str(prepared_path.resolve())


def _extract_request_items(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    query = str(request.get("query") or "").strip() or None
    regions = list(request.get("regions") or [])
    if regions:
        return [{"tool_name": "ocr", "query": query, "regions": [item]} for item in regions]
    frames = list(request.get("frames") or [])
    if frames:
        return [{"tool_name": "ocr", "query": query, "frames": [item]} for item in frames]
    clips = list(request.get("clips") or [])
    if clips:
        return [{"tool_name": "ocr", "query": query, "clips": [item]} for item in clips]
    return [dict(request or {})]


def _ocr_sample_fps(runtime: Dict[str, Any]) -> float:
    extra = dict(runtime.get("extra") or {})
    raw = extra.get("fps")
    if raw in (None, ""):
        raw = extra.get("ocr_fps")
    try:
        fps = float(raw)
    except Exception:
        fps = 2.0
    return max(0.01, fps)


def _clip_sample_count(clip: Dict[str, Any], fps: float) -> int:
    start_s = float(clip.get("start_s") or 0.0)
    end_s = float(clip.get("end_s") if clip.get("end_s") is not None else start_s)
    return max(1, int(math.ceil(max(0.0, end_s - start_s) * float(fps))))


def _request_clip_frame_items(
    request: Dict[str, Any],
    task: Dict[str, Any],
    runtime: Dict[str, Any],
    frame_out_dir: Path,
) -> List[Dict[str, Any]]:
    clips = list(request.get("clips") or [])
    if not clips:
        return [request]
    video_path = str(task.get("video_path") or "").strip()
    if not video_path:
        raise RuntimeError("OCR clip input requires task.video_path for frame sampling.")

    query = str(request.get("query") or "").strip() or None
    fps = _ocr_sample_fps(runtime)
    expanded: List[Dict[str, Any]] = []
    for clip_index, raw_clip in enumerate(clips):
        clip = dict(raw_clip or {})
        clip.setdefault("metadata", {})
        start_s = float(clip.get("start_s") or 0.0)
        end_s = float(clip.get("end_s") if clip.get("end_s") is not None else start_s)
        frame_count = _clip_sample_count(clip, fps)
        sampled = sample_frames(
            video_path,
            start_s,
            end_s,
            frame_count,
            str(frame_out_dir),
            prefix="ocr_clip_%02d" % (clip_index + 1),
        )
        if not sampled:
            raise RuntimeError("Could not sample OCR frames from clip %.3f-%.3fs." % (start_s, end_s))
        for sample in sampled:
            timestamp_s = float(sample.get("timestamp_s") or start_s)
            frame = {
                "video_id": clip.get("video_id") or task.get("video_id") or task.get("sample_key") or "video",
                "timestamp_s": timestamp_s,
                "clip": clip,
                "metadata": {
                    "source_path": str(sample.get("frame_path") or ""),
                    "ocr_sample_fps": fps,
                    "ocr_source": "clip",
                },
            }
            expanded.append({"tool_name": "ocr", "query": query, "frames": [frame]})
    return expanded


def _prepare_single_request(request: Dict[str, Any], task: Dict[str, Any], runtime: Dict[str, Any], frame_out_dir: Path):
    region = dict((request.get("regions") or [{}])[0] or {})
    region_frame = dict(region.get("frame") or {}) if region else {}
    frame_payload = dict((request.get("frames") or [{}])[0] or {})
    frame_path = absolute_frame_path(region_frame or frame_payload, runtime)
    if frame_path:
        timestamp_payload = region_frame or frame_payload
        timestamp_s = float(timestamp_payload.get("timestamp_s") or timestamp_payload.get("timestamp") or 0.0)
    else:
        frame_path, timestamp_s = ensure_frame_for_request(
            request,
            task,
            runtime,
            out_dir=frame_out_dir,
            prefix="ocr_frame",
        )
    source_frame_path = str(Path(frame_path).resolve())
    if region:
        frame_path = crop_region(
            frame_path,
            region.get("bbox"),
            frame_out_dir / ("ocr_crop_%s.png" % Path(source_frame_path).stem),
        )
    max_longest_dim = int((runtime.get("extra") or {}).get("max_longest_dim") or 1600)
    prepared_frame_path = _prepare_ocr_image(frame_path, frame_out_dir, max_longest_dim=max(256, max_longest_dim))
    query = str(request.get("query") or "").strip()
    return prepared_frame_path, source_frame_path, float(timestamp_s), query


def _prepared_request_source(
    request: Dict[str, Any],
    *,
    source_frame_path: str,
    timestamp_s: float,
) -> Dict[str, Any]:
    region = dict((request.get("regions") or [{}])[0] or {})
    frame = dict((request.get("frames") or [{}])[0] or {})
    clip = dict((request.get("clips") or [{}])[0] or {})

    if region:
        region_frame = dict(region.get("frame") or {})
        if region_frame:
            frame = region_frame
            if not clip:
                clip = dict(frame.get("clip") or {})
        return {"region": region, "frame": frame or None, "clip": clip or None}

    if frame:
        metadata = dict(frame.get("metadata") or {})
        if source_frame_path and not metadata.get("source_path"):
            metadata["source_path"] = source_frame_path
            frame["metadata"] = metadata
        frame["timestamp_s"] = float(frame.get("timestamp_s") or timestamp_s)
        if not clip:
            clip = dict(frame.get("clip") or {})
        return {"frame": frame, "clip": clip or None}

    if clip:
        return {"clip": clip}
    return {}


def _prepare_ocr_item(
    request: Dict[str, Any],
    *,
    task: Dict[str, Any],
    runtime: Dict[str, Any],
    frame_out_dir: Path,
) -> Dict[str, Any]:
    prepared_frame_path, source_frame_path, timestamp_s, query = _prepare_single_request(
        request,
        task,
        runtime,
        frame_out_dir,
    )
    source = _prepared_request_source(
        request,
        source_frame_path=source_frame_path,
        timestamp_s=timestamp_s,
    )
    return {
        "prepared_frame_path": prepared_frame_path,
        "source_frame_path": source_frame_path,
        "timestamp_s": float(timestamp_s),
        "query": query,
        **source,
    }


def _configure_paddleocr_environment(runtime: Dict[str, Any]) -> Path:
    hf_cache = str(runtime.get("hf_cache") or os.environ.get("HF_HOME") or "").strip()
    if hf_cache:
        base = Path(hf_cache).expanduser().resolve()
    else:
        base = workspace_root_from_runtime(runtime) / "cache" / "paddleocr_runtime"
    paddlex_cache = base / "paddlex"
    matplotlib_cache = base / "matplotlib"
    fallback_home = base / "paddleocr_home"
    hub_cache = base / "hub"
    for path in (paddlex_cache, matplotlib_cache, fallback_home):
        path.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    paddlex_hub_link = paddlex_cache / "hub"
    if not paddlex_hub_link.exists() and not paddlex_hub_link.is_symlink():
        try:
            paddlex_hub_link.symlink_to(hub_cache, target_is_directory=True)
        except OSError:
            pass

    os.environ.setdefault("HF_HOME", str(base))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddlex_cache)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    current_home = Path(str(os.environ.get("HOME") or "")).expanduser()
    if not current_home.exists() or not os.access(current_home, os.W_OK):
        os.environ["HOME"] = str(fallback_home)
    return paddlex_cache


def _paddleocr_device_label(runtime: Dict[str, Any]) -> str:
    device_label = str((runtime.get("extra") or {}).get("device") or resolved_device_label(runtime) or "cpu").strip().lower()
    if device_label.startswith("cuda"):
        suffix = device_label[4:]
        return "gpu%s" % suffix
    if device_label.startswith("gpu"):
        return device_label
    return "cpu"


def _optional_runtime_value(extra: Dict[str, Any], key: str) -> str | None:
    value = str(extra.get(key) or "").strip()
    return value or None


def _optional_runtime_int(extra: Dict[str, Any], key: str) -> int | None:
    value = extra.get(key)
    if value in (None, ""):
        return None
    try:
        return max(1, int(value))
    except Exception:
        return None


def _probe_paddleocr_device(device_label: str) -> Dict[str, Any]:
    normalized = str(device_label or "").strip().lower()
    probe: Dict[str, Any] = {
        "requested_device": normalized or "cpu",
        "available": True,
    }
    if not normalized.startswith("gpu"):
        return probe

    try:
        import paddle
    except Exception as exc:  # pragma: no cover - runtime-env dependent
        probe["available"] = None
        probe["probe_error"] = str(exc)
        return probe

    try:
        compiled_with_cuda = bool(paddle.device.is_compiled_with_cuda())
        visible_gpu_count = int(paddle.device.cuda.device_count()) if compiled_with_cuda else 0
    except Exception as exc:  # pragma: no cover - runtime-env dependent
        probe["available"] = None
        probe["probe_error"] = str(exc)
        return probe

    probe["compiled_with_cuda"] = compiled_with_cuda
    probe["visible_gpu_count"] = visible_gpu_count
    if not compiled_with_cuda or visible_gpu_count <= 0:
        probe["available"] = False
        probe["reason"] = "gpu_runtime_unavailable"
        return probe

    if ":" not in normalized:
        return probe

    try:
        requested_index = int(normalized.split(":", 1)[1])
    except Exception:
        probe["available"] = None
        probe["reason"] = "unparseable_gpu_index"
        return probe

    probe["requested_gpu_index"] = requested_index
    if requested_index < 0 or requested_index >= visible_gpu_count:
        probe["available"] = False
        probe["reason"] = "gpu_index_out_of_range"
    return probe


def create_paddleocr_engine(runtime: Dict[str, Any]):
    paddlex_cache = _configure_paddleocr_environment(runtime)
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # pragma: no cover - import is runtime-env dependent
        fail_runtime(
            "PaddleOCR requires both `paddleocr` and a matching `paddlepaddle` runtime in the environment.",
            extra={"import_error": str(exc)},
        )

    extra = dict(runtime.get("extra") or {})
    require_gpu = bool(extra.get("require_gpu", False))
    init_kwargs = {
        "lang": str(extra.get("lang") or "en"),
        "ocr_version": str(extra.get("ocr_version") or "PP-OCRv5"),
        "use_textline_orientation": bool(extra.get("use_textline_orientation", False)),
        "text_detection_model_dir": _optional_runtime_value(extra, "text_detection_model_dir"),
        "text_recognition_model_dir": _optional_runtime_value(extra, "text_recognition_model_dir"),
        "textline_orientation_model_dir": _optional_runtime_value(extra, "textline_orientation_model_dir"),
        "text_recognition_batch_size": _optional_runtime_int(extra, "text_recognition_batch_size") or 16,
        "textline_orientation_batch_size": _optional_runtime_int(extra, "textline_orientation_batch_size"),
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    attempted_devices = []
    requested_device = _paddleocr_device_label(runtime)
    device_probe = _probe_paddleocr_device(requested_device)
    if require_gpu and requested_device == "cpu":
        fail_runtime(
            "OCR is configured to require GPU execution, but the resolved device is CPU.",
            extra={
                "requested_device": requested_device,
                "device_probe": device_probe,
                "paddlex_cache_home": str(paddlex_cache),
            },
        )
    if require_gpu and requested_device != "cpu" and device_probe.get("available") is not True:
        fail_runtime(
            "OCR is configured to require GPU execution, but the Paddle GPU runtime is unavailable.",
            extra={
                "requested_device": requested_device,
                "device_probe": device_probe,
                "paddlex_cache_home": str(paddlex_cache),
                "hint": "Install a GPU-enabled `paddlepaddle-gpu` runtime in this environment.",
            },
        )
    candidate_devices = [requested_device]
    if device_probe.get("available") is False and requested_device != "cpu":
        candidate_devices = ["cpu"]
    elif requested_device != "cpu":
        candidate_devices.append("cpu")
    last_error: Exception | None = None
    for device in candidate_devices:
        if device in attempted_devices:
            continue
        attempted_devices.append(device)
        try:
            attempt_kwargs = {**init_kwargs, "device": device}
            if device == "cpu":
                attempt_kwargs["enable_mkldnn"] = bool(extra.get("enable_mkldnn", False))
            return PaddleOCR(**attempt_kwargs)
        except Exception as exc:  # pragma: no cover - runtime-env dependent
            last_error = exc
            if require_gpu and device != "cpu":
                fail_runtime(
                    "OCR is configured to require GPU execution, but PaddleOCR failed to initialize on the requested GPU.",
                    extra={
                        "requested_device": requested_device,
                        "attempted_devices": attempted_devices,
                        "device_probe": device_probe,
                        "init_error": str(exc),
                        "paddlex_cache_home": str(paddlex_cache),
                    },
                )
    fail_runtime(
        "Failed to initialize PaddleOCR. Make sure the Paddle runtime is installed and OCR models are available.",
        extra={
            "init_error": str(last_error),
            "requested_device": requested_device,
            "attempted_devices": attempted_devices,
            "device_probe": device_probe,
            "paddlex_cache_home": str(paddlex_cache),
        },
    )


def run_paddleocr_image(engine: Any, image_path: str, *, use_textline_orientation: bool) -> List[Dict[str, Any]]:
    if hasattr(engine, "predict"):
        raw_result = engine.predict(
            input=image_path,
            use_textline_orientation=use_textline_orientation,
        )
    elif hasattr(engine, "ocr"):
        raw_result = engine.ocr(
            image_path,
            use_textline_orientation=use_textline_orientation,
        )
    else:  # pragma: no cover - defensive guard
        raise RuntimeError("Unsupported PaddleOCR engine: missing predict/ocr entrypoint.")
    return _extract_lines(raw_result)


def run_paddleocr_batch(engine: Any, image_paths: List[str], *, use_textline_orientation: bool) -> List[List[Dict[str, Any]]]:
    paths = [str(path) for path in list(image_paths or [])]
    if not paths:
        return []
    if len(paths) > 1 and hasattr(engine, "predict"):
        try:
            raw_result = engine.predict(
                input=paths,
                use_textline_orientation=use_textline_orientation,
            )
            if isinstance(raw_result, list):
                raw_items = raw_result
            else:
                raw_items = list(raw_result or [])
            if len(raw_items) == len(paths):
                return [_extract_lines(item) for item in raw_items]
        except Exception:
            pass
    return [
        run_paddleocr_image(
            engine,
            image_path,
            use_textline_orientation=use_textline_orientation,
        )
        for image_path in paths
    ]


def _normalize_single_output(
    lines: List[Dict[str, Any]],
    *,
    query: str,
    timestamp_s: float,
    source_frame_path: str,
    frame: Dict[str, Any] | None = None,
    clip: Dict[str, Any] | None = None,
    region: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cleaned_lines = []
    for item in list(lines or []):
        if not isinstance(item, dict):
            continue
        normalized = _normalize_line(
            text=item.get("text"),
            bbox=item.get("bbox"),
            confidence=item.get("confidence"),
        )
        if normalized is not None:
            cleaned_lines.append(normalized)
    text = "\n".join(item["text"] for item in cleaned_lines)
    payload = {
        "text": text,
        "lines": cleaned_lines,
        "query": query or None,
        "timestamp_s": float(timestamp_s),
        "source_frame_path": source_frame_path,
        "backend": "paddleocr",
    }
    if frame:
        payload["frame"] = frame
    if clip:
        payload["clip"] = clip
    if region:
        payload["region"] = region
    return payload


def _run_single_request(
    request: Dict[str, Any],
    *,
    task: Dict[str, Any],
    runtime: Dict[str, Any],
    frame_out_dir: Path,
    engine: Any,
) -> Dict[str, Any]:
    prepared = _prepare_ocr_item(
        request,
        task=task,
        runtime=runtime,
        frame_out_dir=frame_out_dir,
    )
    lines = run_paddleocr_image(
        engine,
        prepared["prepared_frame_path"],
        use_textline_orientation=bool((runtime.get("extra") or {}).get("use_textline_orientation", False)),
    )
    return _normalize_single_output(
        lines,
        query=prepared["query"],
        timestamp_s=prepared["timestamp_s"],
        source_frame_path=prepared["source_frame_path"],
        frame=prepared.get("frame"),
        clip=prepared.get("clip"),
        region=prepared.get("region"),
    )


def _run_prepared_requests(
    prepared_items: List[Dict[str, Any]],
    *,
    runtime: Dict[str, Any],
    engine: Any,
) -> List[Dict[str, Any]]:
    use_textline_orientation = bool((runtime.get("extra") or {}).get("use_textline_orientation", False))
    lines_by_item = run_paddleocr_batch(
        engine,
        [item["prepared_frame_path"] for item in prepared_items],
        use_textline_orientation=use_textline_orientation,
    )
    results = []
    for prepared, lines in zip(prepared_items, lines_by_item):
        results.append(
            _normalize_single_output(
                lines,
                query=prepared["query"],
                timestamp_s=prepared["timestamp_s"],
                source_frame_path=prepared["source_frame_path"],
                frame=prepared.get("frame"),
                clip=prepared.get("clip"),
                region=prepared.get("region"),
            )
        )
    return results


def main() -> None:
    payload = load_request()
    request = dict(payload.get("request") or {})
    task = dict(payload.get("task") or {})
    runtime = dict(payload.get("runtime") or {})
    frame_out_dir = scratch_dir(runtime, "ocr")
    request_items = _extract_request_items(request)
    expanded_items = []
    for item in request_items:
        expanded_items.extend(_request_clip_frame_items(item, task, runtime, frame_out_dir))
    prepared_items = [
        _prepare_ocr_item(
            item,
            task=task,
            runtime=runtime,
            frame_out_dir=frame_out_dir,
        )
        for item in expanded_items
    ]
    engine = create_paddleocr_engine(runtime)
    results = _run_prepared_requests(
        prepared_items,
        runtime=runtime,
        engine=engine,
    )
    if len(results) <= 1:
        emit_json(results[0] if results else _normalize_single_output([], query="", timestamp_s=0.0, source_frame_path=""))
        return
    emit_json({"results": results, "backend": "paddleocr"})


if __name__ == "__main__":
    main()
