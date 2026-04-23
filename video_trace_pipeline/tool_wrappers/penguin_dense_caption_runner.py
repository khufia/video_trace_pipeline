from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from ..common import extract_json_object, is_low_signal_text
from .local_multimodal import PenguinRunner, make_penguin_conversation
from .protocol import emit_json, fail_runtime, load_request
from .shared import resolve_generation_controls, resolve_model_path, resolved_device_label, sample_request_frames, scratch_dir

if TYPE_CHECKING:
    from .persistent_pool import PersistentModelPool


def _normalize_span(raw: Dict[str, Any], *, start_s: float, end_s: float) -> Dict[str, Any]:
    def _text(value: Any) -> str:
        text = str(value or "").strip()
        return "" if is_low_signal_text(text) else text

    def _list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in (_text(item) for item in value) if item]
        text = _text(value)
        return [text] if text else []

    return {
        "start": float(raw.get("start") if raw.get("start") is not None else start_s),
        "end": float(raw.get("end") if raw.get("end") is not None else end_s),
        "visual": _text(raw.get("visual")),
        "audio": _text(raw.get("audio")),
        "on_screen_text": " | ".join(_list(raw.get("on_screen_text"))),
        "actions": _list(raw.get("actions")),
        "objects": _list(raw.get("objects")),
        "attributes": _list(raw.get("attributes")),
    }


def execute_payload(payload: Dict[str, Any], *, runner_pool: "PersistentModelPool | None" = None) -> Dict[str, Any]:
    request = dict(payload.get("request") or {})
    task = dict(payload.get("task") or {})
    runtime = dict(payload.get("runtime") or {})

    clip = dict(request.get("clip") or {})
    if not clip:
        fail_runtime("dense_captioner requires request.clip")

    start_s = float(clip.get("start_s") or 0.0)
    end_s = float(clip.get("end_s") or start_s)
    model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
    device_label = resolved_device_label(runtime)
    out_dir = scratch_dir(runtime, "dense_captioner")
    sampled = sample_request_frames(
        request,
        task,
        out_dir=out_dir,
        prefix="dense_caption",
        num_frames=int((runtime.get("extra") or {}).get("sample_frames") or 10),
    )
    if not sampled:
        fail_runtime("dense_captioner could not sample any frames from the requested clip")

    focus_query = str(request.get("focus_query") or "").strip()
    granularity = str(request.get("granularity") or "segment").strip()
    generation = resolve_generation_controls(runtime)
    prompt = (
        "You are a dense captioning model for a chronological sample of frames from one video clip.\n"
        "Return JSON only with keys: captions, overall_summary.\n"
        "Each captions item must contain: start, end, visual, audio, on_screen_text, actions, objects, attributes.\n"
        "Use one atomic item per action/object/attribute list entry.\n"
        "If audio cannot be inferred from the frames, leave audio as an empty string.\n"
        "Use absolute seconds within the clip window shown below.\n\n"
        f"Clip start: {start_s:.3f}\n"
        f"Clip end: {end_s:.3f}\n"
        f"Granularity: {granularity}\n"
        f"Focus query: {focus_query or '<none>'}"
    )
    runner = None
    owns_runner = False
    if runner_pool is not None:
        runner = runner_pool.acquire_penguin_runner(
            tool_name="dense_captioner",
            model_path=model_path,
            device_label=device_label,
            generate_do_sample=bool(generation.get("do_sample")),
            generate_temperature=generation.get("temperature"),
        )
    if runner is None:
        runner = PenguinRunner(
            model_path=model_path,
            device_label=device_label,
            generate_do_sample=bool(generation.get("do_sample")),
            generate_temperature=generation.get("temperature"),
        )
        owns_runner = True
    try:
        raw_text = runner.generate(
            make_penguin_conversation(prompt, [item["frame_path"] for item in sampled]),
            max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 700),
        )
    finally:
        if owns_runner:
            runner.close()
    parsed = extract_json_object(raw_text) or {}
    fallback_text = str(raw_text or "").strip()
    if is_low_signal_text(fallback_text):
        fallback_text = ""
    captions = parsed.get("captions") if isinstance(parsed.get("captions"), list) else []
    if not captions:
        captions = [
            {
                "start": start_s,
                "end": end_s,
                "visual": str(parsed.get("overall_summary") or fallback_text or "").strip(),
                "audio": "",
                "on_screen_text": "",
                "actions": [],
                "objects": [],
                "attributes": [],
            }
        ]

    return {
        "clip": clip,
        "captioned_range": {"start_s": start_s, "end_s": end_s},
        "captions": [_normalize_span(item, start_s=start_s, end_s=end_s) for item in captions],
        "overall_summary": str(parsed.get("overall_summary") or fallback_text or "").strip(),
        "sampled_frames": sampled,
        "backend": "penguin_vl_transformers",
    }


def main() -> None:
    emit_json(execute_payload(load_request()))


if __name__ == "__main__":
    main()
