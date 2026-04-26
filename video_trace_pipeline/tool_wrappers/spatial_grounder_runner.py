from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from PIL import Image

from ..common import extract_json_object
from .local_multimodal import QwenStyleRunner, make_qwen_image_messages
from .protocol import emit_json, fail_runtime, load_request
from .shared import (
    absolute_frame_path,
    fit_bbox_to_image,
    resolve_generation_controls,
    resolve_model_path,
    resolved_device_label,
)

if TYPE_CHECKING:
    from .persistent_pool import PersistentModelPool


def _build_prompt(request: Dict[str, Any], *, image_size: tuple[int, int] | None = None) -> str:
    query = str(request.get("query") or "").strip()
    parts = [
        "Locate the target in the image.",
        "Return JSON only with keys: detections, spatial_description.",
        "Each detection must be an object with: label, bbox, confidence.",
        "bbox must be [x1, y1, x2, y2] in image pixel coordinates.",
    ]
    if image_size and image_size[0] > 0 and image_size[1] > 0:
        parts.extend(
            [
                "The image size is %dx%d pixels." % (int(image_size[0]), int(image_size[1])),
                "Use that exact coordinate system for bbox values. Do not use guessed HD coordinates.",
            ]
        )
    parts.extend(
        [
        "If the target is absent, return detections as an empty list.",
        "",
        "TARGET:",
        query,
        ]
    )
    return "\n".join(parts)


def _normalize_detections(
    raw_items: Any,
    default_label: str,
    *,
    image_size: tuple[int, int] | None = None,
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for item in list(raw_items or []):
        if not isinstance(item, dict):
            continue
        metadata: Dict[str, Any] = {}
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = None
        else:
            try:
                bbox = [float(value) for value in bbox]
            except Exception:
                bbox = None
        if bbox is not None and image_size is not None:
            fitted_bbox = fit_bbox_to_image(
                bbox,
                image_size=image_size,
                allow_scaled_canvas=True,
            )
            if fitted_bbox is not None:
                if fitted_bbox != bbox:
                    metadata["bbox_normalized_to_image"] = True
                    metadata["raw_bbox"] = [float(value) for value in bbox]
                bbox = fitted_bbox
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
                "metadata": metadata,
            }
        )
    return detections


def execute_payload(payload: Dict[str, Any], *, runner_pool: "PersistentModelPool | None" = None) -> Dict[str, Any]:
    request = dict(payload.get("request") or {})
    runtime = dict(payload.get("runtime") or {})

    query = str(request.get("query") or "").strip()
    if not query:
        fail_runtime("spatial_grounder requires a non-empty query")

    frame_payload = dict((list(request.get("frames") or []) or [{}])[0] or {})
    frame_path = absolute_frame_path(frame_payload, runtime)
    if not frame_path:
        fail_runtime("spatial_grounder requires a resolved frames[0] path")

    image_size = None
    try:
        with Image.open(Path(frame_path)) as image:
            image_size = image.size
    except Exception:
        image_size = None

    model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
    device_label = resolved_device_label(runtime)
    prompt = _build_prompt(request, image_size=image_size)
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
    detections = _normalize_detections(
        parsed.get("detections") or [],
        query,
        image_size=image_size,
    )
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
