from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List

from ..common import extract_json_object
from .local_multimodal import QwenStyleRunner, make_qwen_image_messages
from .protocol import emit_json, fail_runtime, load_request
from .shared import (
    absolute_frame_path,
    resolve_generation_controls,
    resolve_model_path,
    resolved_device_label,
    sample_request_frames,
    scratch_dir,
)

if TYPE_CHECKING:
    from .persistent_pool import PersistentModelPool


def _build_prompt(
    request: Dict[str, Any],
    task: Dict[str, Any],
    transcript_text: str,
    evidence_lines: List[str],
    text_contexts: List[str],
) -> str:
    parts = [
        "Answer the query using the supplied evidence and any sampled media.",
        "Return JSON only with keys: answer, supporting_points, confidence, analysis.",
        "Set confidence to a numeric score from 0.0 to 1.0, not a label like High, Medium, or Low.",
        "Do not mention hidden tools or APIs.",
    ]
    task_question = str(task.get("question") or "").strip()
    if task_question:
        parts.extend(["", "TASK QUESTION:", task_question])
    options = [str(item).strip() for item in list(task.get("options") or []) if str(item).strip()]
    if options:
        parts.extend(["", "ANSWER OPTIONS:"])
        parts.extend("- %s" % option for option in options)
    parts.extend(["", "QUERY:", str(request.get("query") or "").strip()])
    if transcript_text:
        parts.extend(["", "TRANSCRIPT:", transcript_text])
    if evidence_lines:
        parts.extend(["", "STRUCTURED EVIDENCE:"])
        parts.extend("- %s" % line for line in evidence_lines)
    if text_contexts:
        parts.extend(["", "TEXT CONTEXT:"])
        parts.extend("- %s" % line for line in text_contexts)
    extra_fields: List[str] = []
    for key, value in sorted(request.items()):
        if key in {"tool_name", "query", "clip", "clips", "frame", "frames", "transcript", "transcripts", "text_contexts", "evidence_ids"}:
            continue
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value).strip()
        if rendered:
            extra_fields.append("%s: %s" % (key, rendered))
    if extra_fields:
        parts.extend(["", "REQUEST CONTEXT:"])
        parts.extend("- %s" % line for line in extra_fields)
    return "\n".join(parts)


def execute_payload(payload: Dict[str, Any], *, runner_pool: "PersistentModelPool | None" = None) -> Dict[str, Any]:
    request = dict(payload.get("request") or {})
    task = dict(payload.get("task") or {})
    runtime = dict(payload.get("runtime") or {})

    query = str(request.get("query") or "").strip()
    if not query:
        fail_runtime("generic_purpose requires a non-empty query")

    model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
    device_label = resolved_device_label(runtime)
    prompt_dir = scratch_dir(runtime, "generic_purpose")

    image_paths: List[str] = []
    frame_payloads = []
    if isinstance(request.get("frames"), list) and request.get("frames"):
        frame_payloads.extend([dict(item or {}) for item in request.get("frames") or [] if isinstance(item, dict)])
    elif request.get("frame"):
        frame_payloads.append(dict(request.get("frame") or {}))
    for frame_payload in frame_payloads:
        frame_path = absolute_frame_path(frame_payload, runtime)
        if frame_path and frame_path not in image_paths:
            image_paths.append(frame_path)

    clip_payloads = []
    if isinstance(request.get("clips"), list) and request.get("clips"):
        clip_payloads.extend([dict(item or {}) for item in request.get("clips") or [] if isinstance(item, dict)])
    elif request.get("clip"):
        clip_payloads.append(dict(request.get("clip") or {}))
    if not image_paths and clip_payloads:
        for clip_index, clip_payload in enumerate(clip_payloads, start=1):
            sampled = sample_request_frames(
                {"clip": clip_payload},
                task,
                out_dir=prompt_dir,
                prefix="generic_%02d" % clip_index,
                num_frames=4,
            )
            for item in sampled:
                frame_path = str(item["frame_path"])
                if frame_path not in image_paths:
                    image_paths.append(frame_path)

    transcript_payloads = []
    if isinstance(request.get("transcripts"), list) and request.get("transcripts"):
        transcript_payloads.extend([dict(item or {}) for item in request.get("transcripts") or [] if isinstance(item, dict)])
    elif request.get("transcript"):
        transcript_payloads.append(dict(request.get("transcript") or {}))
    transcript_blocks = []
    for index, transcript in enumerate(transcript_payloads, start=1):
        transcript_text = str(transcript.get("text") or "").strip()
        if transcript_text:
            transcript_blocks.append("Transcript %d:\n%s" % (index, transcript_text))
    transcript_text = "\n\n".join(transcript_blocks).strip()
    evidence_records = payload.get("evidence_records") or []
    evidence_lines = [
        str(item.get("atomic_text") or item.get("text") or "").strip()
        for item in evidence_records
        if isinstance(item, dict) and str(item.get("atomic_text") or item.get("text") or "").strip()
    ]
    text_contexts = [str(item).strip() for item in list(request.get("text_contexts") or []) if str(item).strip()]
    prompt = _build_prompt(request, task, transcript_text, evidence_lines, text_contexts)
    generation = resolve_generation_controls(runtime)
    attn_implementation = str((runtime.get("extra") or {}).get("attn_implementation") or "").strip() or None
    runner = None
    owns_runner = False
    if runner_pool is not None:
        runner = runner_pool.acquire_qwen_style_runner(
            tool_name="generic_purpose",
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
            make_qwen_image_messages(prompt, image_paths),
            max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 512),
        )
    finally:
        if owns_runner:
            runner.close()
    parsed = extract_json_object(raw_text) or {}
    answer = str(parsed.get("answer") or "").strip()
    supporting_points = parsed.get("supporting_points")
    if not isinstance(supporting_points, list):
        supporting_points = []
    return {
        "answer": answer or raw_text.strip(),
        "supporting_points": [str(item).strip() for item in supporting_points if str(item).strip()],
        "confidence": parsed.get("confidence"),
        "analysis": str(parsed.get("analysis") or answer or raw_text).strip(),
    }


def main() -> None:
    emit_json(execute_payload(load_request()))


if __name__ == "__main__":
    main()
