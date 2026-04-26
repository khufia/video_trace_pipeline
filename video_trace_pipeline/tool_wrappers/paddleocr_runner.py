from __future__ import annotations

from collections.abc import Iterable, Sequence
import os
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .protocol import emit_json, fail_runtime, load_request
from .shared import crop_region, ensure_frame_for_request, resolved_device_label, scratch_dir, workspace_root_from_runtime


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


def _prepare_single_request(request: Dict[str, Any], task: Dict[str, Any], runtime: Dict[str, Any], frame_out_dir: Path):
    frame_path, timestamp_s = ensure_frame_for_request(
        request,
        task,
        runtime,
        out_dir=frame_out_dir,
        prefix="ocr_frame",
    )
    source_frame_path = str(Path(frame_path).resolve())
    region = dict((request.get("regions") or [{}])[0] or {})
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


def _configure_paddleocr_environment(runtime: Dict[str, Any]) -> Path:
    base = workspace_root_from_runtime(runtime) / "cache" / "paddleocr_runtime"
    paddlex_cache = base / "paddlex"
    matplotlib_cache = base / "matplotlib"
    fallback_home = base / "home"
    for path in (paddlex_cache, matplotlib_cache, fallback_home):
        path.mkdir(parents=True, exist_ok=True)

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


def _normalize_single_output(
    lines: List[Dict[str, Any]],
    *,
    query: str,
    timestamp_s: float,
    source_frame_path: str,
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
    return {
        "text": text,
        "lines": cleaned_lines,
        "query": query or None,
        "timestamp_s": float(timestamp_s),
        "source_frame_path": source_frame_path,
        "backend": "paddleocr",
    }


def _run_single_request(
    request: Dict[str, Any],
    *,
    task: Dict[str, Any],
    runtime: Dict[str, Any],
    frame_out_dir: Path,
    engine: Any,
) -> Dict[str, Any]:
    prepared_frame_path, source_frame_path, timestamp_s, query = _prepare_single_request(
        request,
        task,
        runtime,
        frame_out_dir,
    )
    lines = run_paddleocr_image(
        engine,
        prepared_frame_path,
        use_textline_orientation=bool((runtime.get("extra") or {}).get("use_textline_orientation", False)),
    )
    return _normalize_single_output(
        lines,
        query=query,
        timestamp_s=timestamp_s,
        source_frame_path=source_frame_path,
    )


def main() -> None:
    payload = load_request()
    request = dict(payload.get("request") or {})
    task = dict(payload.get("task") or {})
    runtime = dict(payload.get("runtime") or {})
    frame_out_dir = scratch_dir(runtime, "ocr")
    request_items = _extract_request_items(request)
    engine = create_paddleocr_engine(runtime)
    results = []
    for item in request_items:
        results.append(
            _run_single_request(
                item,
                task=task,
                runtime=runtime,
                frame_out_dir=frame_out_dir,
                engine=engine,
            )
        )
    if len(results) <= 1:
        emit_json(results[0] if results else _normalize_single_output([], query="", timestamp_s=0.0, source_frame_path=""))
        return
    emit_json({"results": results, "backend": "paddleocr"})


if __name__ == "__main__":
    main()
