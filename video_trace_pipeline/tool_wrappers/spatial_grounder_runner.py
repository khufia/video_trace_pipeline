from __future__ import annotations

from typing import Any, Dict, List

from ..common import extract_json_object
from .local_multimodal import make_qwen_image_messages, run_qwen_style_messages
from .protocol import emit_json, fail_runtime, load_request
from .shared import absolute_frame_path, resolve_model_path, resolved_device_label


def _build_prompt(request: Dict[str, Any]) -> str:
    query = str(request.get("query") or "").strip()
    parts = [
        "Locate the target in the image.",
        "Return JSON only with keys: detections, spatial_description.",
        "Each detection must be an object with: label, bbox, confidence.",
        "bbox must be [x1, y1, x2, y2] in image pixel coordinates.",
        "If the target is absent, return detections as an empty list.",
        "",
        "TARGET:",
        query,
    ]
    return "\n".join(parts)


def _normalize_detections(raw_items: Any, default_label: str) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for item in list(raw_items or []):
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = None
        else:
            try:
                bbox = [float(value) for value in bbox]
            except Exception:
                bbox = None
        label = str(item.get("label") or default_label).strip() or default_label
        confidence = item.get("confidence")
        try:
            confidence = None if confidence is None else float(confidence)
        except Exception:
            confidence = None
        detections.append(
            {
                "label": label,
                "bbox": bbox,
                "confidence": confidence,
                "metadata": {},
            }
        )
    return detections


def main() -> None:
    payload = load_request()
    request = dict(payload.get("request") or {})
    runtime = dict(payload.get("runtime") or {})

    query = str(request.get("query") or "").strip()
    if not query:
        fail_runtime("spatial_grounder requires a non-empty query")

    frame_payload = dict(request.get("frame") or {})
    frame_path = absolute_frame_path(frame_payload, runtime)
    if not frame_path:
        fail_runtime("spatial_grounder requires a resolved frame path")

    model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
    device_label = resolved_device_label(runtime)
    prompt = _build_prompt(request)

    raw_text = run_qwen_style_messages(
        model_path=model_path,
        messages=make_qwen_image_messages(prompt, [frame_path]),
        device_label=device_label,
        max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 384),
    )
    parsed = extract_json_object(raw_text) or {}
    detections = _normalize_detections(parsed.get("detections") or [], query)
    emit_json(
        {
            "query": query,
            "timestamp_s": frame_payload.get("timestamp_s") or frame_payload.get("timestamp"),
            "detections": detections,
            "spatial_description": str(parsed.get("spatial_description") or raw_text).strip(),
            "source_frame_path": frame_path,
            "backend": str(runtime.get("model_name") or "").strip(),
        }
    )


if __name__ == "__main__":
    main()
