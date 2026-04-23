from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from ..common import extract_json_object
from .local_multimodal import QwenStyleRunner, make_qwen_image_messages
from .protocol import emit_json, fail_runtime, load_request
from .shared import absolute_frame_path, resolve_generation_controls, resolve_model_path, resolved_device_label

if TYPE_CHECKING:
    from .persistent_pool import PersistentModelPool


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


def execute_payload(payload: Dict[str, Any], *, runner_pool: "PersistentModelPool | None" = None) -> Dict[str, Any]:
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
    generation = resolve_generation_controls(runtime)
    attn_implementation = str((runtime.get("extra") or {}).get("attn_implementation") or "").strip() or None
    runner = None
    owns_runner = False
    if runner_pool is not None:
        runner = runner_pool.acquire_qwen_style_runner(
            tool_name="spatial_grounder",
            model_path=model_path,
            device_label=device_label,
            generate_do_sample=bool(generation.get("do_sample")),
            generate_temperature=generation.get("temperature"),
            attn_implementation=attn_implementation,
        )
    if runner is None:
        runner = QwenStyleRunner(
            model_path=model_path,
            device_label=device_label,
            generate_do_sample=bool(generation.get("do_sample")),
            generate_temperature=generation.get("temperature"),
            attn_implementation=attn_implementation,
        )
        owns_runner = True
    try:
        raw_text = runner.generate(
            make_qwen_image_messages(prompt, [frame_path]),
            max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 384),
        )
    finally:
        if owns_runner:
            runner.close()
    parsed = extract_json_object(raw_text) or {}
    detections = _normalize_detections(parsed.get("detections") or [], query)
    return {
        "query": query,
        "timestamp_s": frame_payload.get("timestamp_s") or frame_payload.get("timestamp"),
        "detections": detections,
        "spatial_description": str(parsed.get("spatial_description") or raw_text).strip(),
        "source_frame_path": frame_path,
        "backend": str(runtime.get("model_name") or "").strip(),
    }


def main() -> None:
    emit_json(execute_payload(load_request()))


if __name__ == "__main__":
    main()
