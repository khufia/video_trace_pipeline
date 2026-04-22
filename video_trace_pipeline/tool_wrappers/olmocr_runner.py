from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..common import extract_json_object
from .local_multimodal import make_qwen_image_messages, run_qwen_style_messages
from .protocol import emit_json, fail_runtime, load_request
from .shared import crop_region, ensure_frame_for_request, resolve_model_path, resolved_device_label, scratch_dir


def _normalize_lines(parsed: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    raw_lines = parsed.get("lines")
    if isinstance(raw_lines, list):
        lines: List[Dict[str, Any]] = []
        for item in raw_lines:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                bbox = [float(value) for value in bbox[:4]]
            else:
                bbox = None
            confidence = item.get("confidence")
            lines.append(
                {
                    "text": str(item.get("text") or "").strip(),
                    "bbox": bbox,
                    "confidence": None if confidence is None else float(confidence),
                }
            )
        if lines:
            return lines
    cleaned = str(text or "").strip()
    return [{"text": cleaned, "bbox": None, "confidence": None}] if cleaned else []


def main() -> None:
    payload = load_request()
    request = dict(payload.get("request") or {})
    task = dict(payload.get("task") or {})
    runtime = dict(payload.get("runtime") or {})
    frame_out_dir = scratch_dir(runtime, "ocr")

    frame_path, timestamp_s = ensure_frame_for_request(
        request,
        task,
        runtime,
        out_dir=frame_out_dir,
        prefix="ocr_frame",
    )
    source_frame_path = str(Path(frame_path).resolve())
    region = dict(request.get("region") or {})
    if region:
        frame_path = crop_region(
            frame_path,
            region.get("bbox"),
            frame_out_dir / ("ocr_crop_%s.png" % Path(source_frame_path).stem),
        )

    query = str(request.get("query") or "").strip()
    model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
    prompt = (
        "Extract all visible text from this image.\n"
        "Return JSON only with keys: text, lines.\n"
        "Each lines item must contain text, bbox, confidence.\n"
        "If bbox values are unavailable, set bbox to null.\n"
        f"Focus query: {query or '<none>'}"
    )
    raw_text = run_qwen_style_messages(
        model_path=model_path,
        messages=make_qwen_image_messages(prompt, [frame_path]),
        device_label=resolved_device_label(runtime),
        max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 384),
    )
    parsed = extract_json_object(raw_text) or {}
    text = str(parsed.get("text") or "").strip()
    fallback_lines = _normalize_lines(parsed, text)

    if not text and not fallback_lines:
        fail_runtime("OCR produced no output", extra={"frame_path": frame_path})

    emit_json(
        {
            "text": text,
            "lines": fallback_lines,
            "query": query or None,
            "timestamp_s": float(timestamp_s),
            "source_frame_path": source_frame_path,
            "backend": "olmocr_transformers",
        }
    )


if __name__ == "__main__":
    main()
