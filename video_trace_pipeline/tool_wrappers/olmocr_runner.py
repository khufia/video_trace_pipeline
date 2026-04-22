from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from ..common import extract_json_object
from .local_multimodal import make_qwen_image_messages, run_qwen_style_messages
from .protocol import emit_json, load_request
from .shared import (
    crop_region,
    ensure_frame_for_request,
    resolve_generation_controls,
    resolve_model_path,
    resolved_device_label,
    scratch_dir,
)

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


def _prepare_olmocr_image(frame_path: str, out_dir: Path) -> str:
    source = Path(frame_path).resolve()
    with Image.open(source) as image:
        image = image.convert("RGB")
        longest_dim = max(image.size)
        if longest_dim <= 1288:
            return str(source)
        scale = 1288.0 / float(longest_dim)
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


def _extract_text_from_raw_output(raw_text: str) -> str:
    cleaned = str(raw_text or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
    if cleaned.startswith("---"):
        lines = cleaned.splitlines()
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                body = "\n".join(lines[index + 1 :]).strip()
                if body:
                    return body
                break
    return cleaned


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
    prepared_frame_path = _prepare_olmocr_image(frame_path, frame_out_dir)

    query = str(request.get("query") or "").strip()
    model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
    generation = resolve_generation_controls(runtime)
    prompt = (
        "Extract all visible text from this image.\n"
        "This may be a chart or slide, so include titles, legends, labels, numbers, percentages, and source text.\n"
        "Return JSON only with keys: text, lines.\n"
        "Each lines item must contain text, bbox, confidence.\n"
        "If bbox values are unavailable, set bbox to null.\n"
        f"Focus query: {query or '<none>'}"
    )
    raw_text = run_qwen_style_messages(
        model_path=model_path,
        messages=make_qwen_image_messages(prompt, [prepared_frame_path]),
        device_label=resolved_device_label(runtime),
        max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 384),
        processor_use_fast=False,
        generate_do_sample=bool(generation.get("do_sample")),
        generate_temperature=generation.get("temperature"),
    )
    parsed = extract_json_object(raw_text) or {}
    text = str(parsed.get("text") or "").strip()
    if not text:
        text = _extract_text_from_raw_output(raw_text)
    fallback_lines = _normalize_lines(parsed, text)
    if not text and fallback_lines:
        text = "\n".join(str(item.get("text") or "").strip() for item in fallback_lines if str(item.get("text") or "").strip())

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
