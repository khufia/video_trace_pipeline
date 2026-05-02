from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from PIL import Image

from ..common import extract_json_object
from .local_multimodal import QwenStyleRunner, make_qwen_image_messages, make_qwen_video_message
from .protocol import emit_json, fail_runtime, load_request
from .shared import (
    absolute_frame_path,
    ensure_frame_for_request,
    extracted_clip,
    fit_bbox_to_image,
    resolve_generation_controls,
    resolve_model_path,
    resolved_device_label,
    tool_cache_root,
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
        "Do not write prose outside the JSON object.",
        "Every bbox must fit inside the actual image coordinate system; if uncertain, set bbox to null instead of guessing off-image coordinates.",
        "For visible text targets, localize the visible text region that matches the requested relation; do not invent the word.",
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
            "If the target is absent, occluded, or not in the requested relation, return detections as an empty list.",
            "",
            "TARGET:",
            query,
        ]
    )
    return "\n".join(parts)


def _build_video_prompt(request: Dict[str, Any], clip: Dict[str, Any]) -> str:
    query = str(request.get("query") or "").strip()
    start_s = float(clip.get("start_s") or 0.0)
    end_s = float(clip.get("end_s") or start_s)
    return "\n".join(
        [
            "Locate the target in this video clip.",
            "Choose the single clearest frame where the target is visible.",
            "Return JSON only with keys: detections, spatial_description, timestamp_s.",
            "timestamp_s is optional; when present, use seconds relative to the original full video.",
            "Each detection must be an object with: label, bbox, confidence.",
            "bbox must be [x1, y1, x2, y2] in video-frame pixel coordinates at timestamp_s.",
            "Do not write prose outside the JSON object.",
            "Every bbox must fit inside the actual video-frame coordinate system; if uncertain, set bbox to null instead of guessing off-frame coordinates.",
            "For visible text targets, localize the visible text region that matches the requested relation; do not invent the word.",
            "If the target is absent, occluded, or not in the requested relation, return detections as an empty list.",
            "Original full-video interval: %.3f to %.3f seconds." % (start_s, end_s),
            "",
            "TARGET:",
            query,
        ]
    )


def _clip_timestamp(parsed: Dict[str, Any], clip: Dict[str, Any]) -> float:
    start_s = float(clip.get("start_s") or 0.0)
    end_s = float(clip.get("end_s") or start_s)
    raw_timestamp = parsed.get("timestamp_s")
    if raw_timestamp is None:
        raw_timestamp = parsed.get("timestamp")
    try:
        timestamp_s = float(raw_timestamp)
    except Exception:
        return (start_s + end_s) / 2.0
    if timestamp_s < start_s:
        timestamp_s += start_s
    return timestamp_s


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
    task = dict(payload.get("task") or {})
    runtime = dict(payload.get("runtime") or {})

    query = str(request.get("query") or "").strip()
    if not query:
        fail_runtime("spatial_grounder requires a non-empty query")

    frames = list(request.get("frames") or [])
    clips = list(request.get("clips") or [])
    use_frame_input = bool(frames)
    frame_payload = dict((frames or [{}])[0] or {})
    clip_payload = dict((clips or [{}])[0] or {})
    frame_path = None
    timestamp_s = None
    image_size = None
    if use_frame_input:
        frame_path = absolute_frame_path(frame_payload, runtime)
        timestamp_s = frame_payload.get("timestamp_s") or frame_payload.get("timestamp")
        if not frame_path:
            video_id = str(
                frame_payload.get("video_id")
                or clip_payload.get("video_id")
                or task.get("video_id")
                or task.get("sample_key")
                or "video"
            )
            out_dir = tool_cache_root(runtime, "spatial_grounder", video_id)
            try:
                frame_path, timestamp_s = ensure_frame_for_request(
                    request,
                    task,
                    runtime,
                    out_dir=out_dir,
                    prefix="spatial",
                )
            except Exception as exc:
                fail_runtime("spatial_grounder requires a resolved frame: %s" % exc)
        try:
            with Image.open(Path(frame_path)) as image:
                image_size = image.size
        except Exception:
            image_size = None
        prompt = _build_prompt(request, image_size=image_size)
    else:
        if not clip_payload:
            fail_runtime("spatial_grounder requires a resolved frame or clip")
        video_path = str(task.get("video_path") or "").strip()
        if not video_path:
            fail_runtime("spatial_grounder requires task.video_path for clip inputs")
        prompt = _build_video_prompt(request, clip_payload)

    model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
    device_label = resolved_device_label(runtime)
    generation = resolve_generation_controls(runtime)
    attn_implementation = str((runtime.get("extra") or {}).get("attn_implementation") or "").strip() or None
    sample_fps = float((runtime.get("extra") or {}).get("fps") or 2.0)
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
        if use_frame_input:
            raw_text = runner.generate(
                make_qwen_image_messages(prompt, [frame_path]),
                max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 384),
            )
        else:
            start_s = float(clip_payload.get("start_s") or 0.0)
            end_s = float(clip_payload.get("end_s") or start_s)
            with extracted_clip(video_path, start_s, end_s, include_audio=False) as video_clip_path:
                raw_text = runner.generate(
                    make_qwen_video_message(prompt, video_clip_path, fps=sample_fps),
                    max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 384),
                )
    finally:
        if owns_runner:
            runner.close()
    parsed = extract_json_object(raw_text) or {}
    if not use_frame_input:
        timestamp_s = _clip_timestamp(parsed, clip_payload)
    detections = _normalize_detections(
        parsed.get("detections") or [],
        query,
        image_size=image_size,
    )
    return {
        "query": query,
        "timestamp_s": timestamp_s,
        "detections": detections,
        "spatial_description": str(parsed.get("spatial_description") or raw_text).strip(),
        "source_frame_path": frame_path if use_frame_input else None,
        "backend": str(runtime.get("model_name") or "").strip(),
    }


def main() -> None:
    emit_json(execute_payload(load_request()))


if __name__ == "__main__":
    main()
