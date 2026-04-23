from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from ..common import extract_json_object
from .local_multimodal import QwenStyleRunner, make_qwen_image_messages, run_qwen_style_messages
from .protocol import emit_json, load_request
from .shared import (
    cleanup_torch,
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


def _ocr_prompt(query: str) -> str:
    return (
        "Extract all visible text from this image.\n"
        "This may be a chart or slide, so include titles, legends, labels, numbers, percentages, and source text.\n"
        "Return JSON only with keys: text, lines.\n"
        "Each lines item must contain text, bbox, confidence.\n"
        "If bbox values are unavailable, set bbox to null.\n"
        f"Focus query: {query or '<none>'}"
    )


def _extract_request_items(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    query = str(request.get("query") or "").strip() or None
    regions = list(request.get("regions") or [])
    if regions:
        return [{"tool_name": "ocr", "query": query, "region": item} for item in regions]
    frames = list(request.get("frames") or [])
    if frames:
        return [{"tool_name": "ocr", "query": query, "frame": item} for item in frames]
    clips = list(request.get("clips") or [])
    if clips:
        return [{"tool_name": "ocr", "query": query, "clip": item} for item in clips]
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
    region = dict(request.get("region") or {})
    if region:
        frame_path = crop_region(
            frame_path,
            region.get("bbox"),
            frame_out_dir / ("ocr_crop_%s.png" % Path(source_frame_path).stem),
        )
    prepared_frame_path = _prepare_olmocr_image(frame_path, frame_out_dir)
    query = str(request.get("query") or "").strip()
    return prepared_frame_path, source_frame_path, float(timestamp_s), query


def _normalize_single_output(raw_text: str, *, query: str, timestamp_s: float, source_frame_path: str) -> Dict[str, Any]:
    parsed = extract_json_object(raw_text) or {}
    text = str(parsed.get("text") or "").strip()
    if not text:
        text = _extract_text_from_raw_output(raw_text)
    fallback_lines = _normalize_lines(parsed, text)
    if not text and fallback_lines:
        text = "\n".join(str(item.get("text") or "").strip() for item in fallback_lines if str(item.get("text") or "").strip())
    return {
        "text": text,
        "lines": fallback_lines,
        "query": query or None,
        "timestamp_s": float(timestamp_s),
        "source_frame_path": source_frame_path,
        "backend": "olmocr_transformers",
    }


def _run_single_request(
    request: Dict[str, Any],
    *,
    task: Dict[str, Any],
    runtime: Dict[str, Any],
    frame_out_dir: Path,
    model_path: str,
    generation: Dict[str, Any],
    attn_implementation: str | None,
    runner: QwenStyleRunner | None = None,
) -> Dict[str, Any]:
    prepared_frame_path, source_frame_path, timestamp_s, query = _prepare_single_request(
        request,
        task,
        runtime,
        frame_out_dir,
    )
    prompt = _ocr_prompt(query)
    messages = make_qwen_image_messages(prompt, [prepared_frame_path])
    if runner is None:
        raw_text = run_qwen_style_messages(
            model_path=model_path,
            messages=messages,
            device_label=resolved_device_label(runtime),
            max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 384),
            processor_use_fast=False,
            generate_do_sample=bool(generation.get("do_sample")),
            generate_temperature=generation.get("temperature"),
            attn_implementation=attn_implementation,
        )
    else:
        raw_text = runner.generate(
            messages,
            max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 384),
        )
    return _normalize_single_output(
        raw_text,
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
    model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
    generation = resolve_generation_controls(runtime)
    attn_implementation = str((runtime.get("extra") or {}).get("attn_implementation") or "").strip() or None
    request_items = _extract_request_items(request)
    if len(request_items) <= 1:
        emit_json(
            _run_single_request(
                request_items[0] if request_items else request,
                task=task,
                runtime=runtime,
                frame_out_dir=frame_out_dir,
                model_path=model_path,
                generation=generation,
                attn_implementation=attn_implementation,
            )
        )
        return

    runner = QwenStyleRunner(
        model_path=model_path,
        device_label=resolved_device_label(runtime),
        processor_use_fast=False,
        generate_do_sample=bool(generation.get("do_sample")),
        generate_temperature=generation.get("temperature"),
        attn_implementation=attn_implementation,
    )
    try:
        results = []
        for item in request_items:
            results.append(
                _run_single_request(
                    item,
                    task=task,
                    runtime=runtime,
                    frame_out_dir=frame_out_dir,
                    model_path=model_path,
                    generation=generation,
                    attn_implementation=attn_implementation,
                    runner=runner,
                )
            )
            with contextlib.suppress(Exception):
                cleanup_torch()
    finally:
        runner.close()

    emit_json({"results": results, "backend": "olmocr_transformers"})


if __name__ == "__main__":
    main()
