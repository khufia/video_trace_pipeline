from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List

from ..common import extract_json_object
from ..temporal import render_temporal_anchor
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


def _format_seconds(value: Any) -> str:
    text = "%.3f" % float(value)
    text = text.rstrip("0").rstrip(".")
    return "%ss" % text


def _format_time_range(start_s: Any, end_s: Any) -> str:
    if start_s is None and end_s is None:
        return ""
    if start_s is None:
        start_s = end_s
    if end_s is None:
        end_s = start_s
    if float(start_s) == float(end_s):
        return "[%s]" % _format_seconds(start_s)
    return "[%s-%s]" % (_format_seconds(start_s), _format_seconds(end_s))


def _media_line(index: int, payload: Dict[str, Any]) -> str:
    parts = ["Image %d" % index]
    artifact_id = str(payload.get("artifact_id") or "").strip()
    if artifact_id:
        parts.append("artifact_id=%s" % artifact_id)
    timestamp_s = payload.get("timestamp_s", payload.get("timestamp"))
    if timestamp_s not in (None, ""):
        parts.append("timestamp=%s" % _format_seconds(timestamp_s))
    relpath = str(payload.get("relpath") or payload.get("frame_path") or payload.get("source_frame_path") or "").strip()
    if relpath:
        parts.append("relpath=%s" % relpath)
    video_id = str(payload.get("video_id") or "").strip()
    if video_id:
        parts.append("video_id=%s" % video_id)
    clip = payload.get("clip") if isinstance(payload.get("clip"), dict) else {}
    if clip:
        clip_anchor = _format_time_range(clip.get("start_s"), clip.get("end_s"))
        if clip_anchor:
            parts.append("source_clip=%s" % clip_anchor)
    return " | ".join(parts)


def _evidence_line(record: Dict[str, Any]) -> str:
    text = str(
        record.get("atomic_text")
        or record.get("evidence_text")
        or record.get("text")
        or ""
    ).strip()
    if not text:
        return ""
    parts = []
    evidence_id = str(record.get("evidence_id") or "").strip()
    if evidence_id:
        parts.append(evidence_id)
    temporal_anchor = render_temporal_anchor(record)
    if temporal_anchor:
        parts.append(temporal_anchor)
    if not parts:
        return text
    return "[%s] %s" % (" | ".join(parts), text)


def _is_image_artifact(record: Dict[str, Any]) -> bool:
    kind = str(record.get("kind") or "").strip().lower()
    media_type = str(record.get("media_type") or "").strip().lower()
    relpath = str(record.get("relpath") or "").strip().lower()
    if kind in {"frame", "image", "screenshot"}:
        return True
    if media_type.startswith("image/"):
        return True
    return relpath.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))


def _frame_payload_from_artifact(artifact: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(artifact.get("metadata") or {})
    frame_payload: Dict[str, Any] = {
        "artifact_id": artifact.get("artifact_id"),
        "relpath": artifact.get("relpath"),
        "metadata": metadata,
    }
    timestamp_s = metadata.get("timestamp_s")
    if timestamp_s is None:
        timestamp_s = record.get("frame_ts_s")
    if timestamp_s is not None:
        frame_payload["timestamp_s"] = timestamp_s
    video_id = str(metadata.get("video_id") or record.get("video_id") or "").strip()
    if video_id:
        frame_payload["video_id"] = video_id
    clip_start_s = metadata.get("clip_start_s")
    if clip_start_s is None:
        clip_start_s = record.get("time_start_s")
    clip_end_s = metadata.get("clip_end_s")
    if clip_end_s is None:
        clip_end_s = record.get("time_end_s")
    if video_id and (clip_start_s is not None or clip_end_s is not None):
        start_s = clip_start_s if clip_start_s is not None else clip_end_s
        end_s = clip_end_s if clip_end_s is not None else clip_start_s
        frame_payload["clip"] = {
            "video_id": video_id,
            "start_s": start_s,
            "end_s": end_s,
        }
    return frame_payload


def _evidence_frame_payloads(evidence_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    seen = set()
    for record in evidence_records:
        artifact_refs: List[Dict[str, Any]] = []
        for key in ("artifact_refs", "source_artifact_refs"):
            value = record.get(key)
            if isinstance(value, list):
                artifact_refs.extend([dict(item or {}) for item in value if isinstance(item, dict)])
        for artifact in artifact_refs:
            if not _is_image_artifact(artifact):
                continue
            frame_payload = _frame_payload_from_artifact(artifact, record)
            signature = (
                str(frame_payload.get("relpath") or "").strip(),
                str(frame_payload.get("timestamp_s") or "").strip(),
            )
            if signature in seen:
                continue
            seen.add(signature)
            payloads.append(frame_payload)
    return payloads


def _render_transcript_payloads(transcript_payloads: List[Dict[str, Any]]) -> str:
    blocks = []
    for index, transcript in enumerate(list(transcript_payloads or []), start=1):
        if not isinstance(transcript, dict):
            continue
        header_parts = ["Transcript %d" % index]
        clip = dict(transcript.get("clip") or {})
        if clip:
            clip_anchor = _format_time_range(clip.get("start_s"), clip.get("end_s"))
            if clip_anchor:
                header_parts.append(clip_anchor)
            video_id = str(clip.get("video_id") or "").strip()
            if video_id:
                header_parts.append(video_id)
        header = " ".join(part for part in header_parts if part).strip()
        lines = []
        for segment in list(transcript.get("segments") or []):
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            time_anchor = _format_time_range(segment.get("start_s", segment.get("start")), segment.get("end_s", segment.get("end")))
            speaker = str(segment.get("speaker_id") or segment.get("speaker") or "").strip()
            prefix_parts = [part for part in (time_anchor, ("%s:" % speaker) if speaker and speaker.lower() != "unknown_speaker" else "") if part]
            if prefix_parts:
                lines.append("%s %s" % (" ".join(prefix_parts), text))
            else:
                lines.append(text)
        if not lines:
            transcript_text = str(transcript.get("text") or "").strip()
            if transcript_text:
                lines.append(transcript_text)
        if lines:
            block = "%s:\n%s" % (header, "\n".join(lines)) if header else "\n".join(lines)
            blocks.append(block.strip())
    return "\n\n".join(blocks).strip()


def _build_prompt(
    request: Dict[str, Any],
    task: Dict[str, Any],
    transcript_text: str,
    evidence_lines: List[str],
    text_contexts: List[str],
    media_lines: List[str] | None = None,
) -> str:
    parts = [
        "Answer the query using only the supplied evidence and any sampled media.",
        "Do not rely on scene priors, world knowledge, or what usually happens in similar videos.",
        "Treat the request/query wording as instructions, not as evidence that the queried description is true.",
        "If a queried state such as empty/full/open/closed is not directly visible or stated, answer indeterminate instead of guessing.",
        "If a question counts objects in a queried state, verify the state of each counted object; visible object presence alone does not prove that state.",
        "For sound-count questions, collapse near-synonymous labels or repeated phases of the same action into one counted sound unless the evidence explicitly distinguishes separate answer-critical sounds.",
        "For sound-count questions, do not count background chatter, music texture, silverware, or other ambient layers as separate answer-level sounds unless the question explicitly asks for them.",
        "For cause-or-inference multiple-choice questions, prefer the option that best matches the directly grounded phenomenon; do not swap in a more remote presumed cause unless the evidence explicitly supports that causal step.",
        "For earliest/first questions with multiple candidate moments, identify the earliest validated candidate first and analyze only that candidate's downstream attribute; do not mix later-candidate details into the earliest event.",
        "For repeated-text, quoted-span, or mentioned-in-text tasks, compare full surface forms and exact span boundaries before considering substrings or paraphrases.",
        "For repeated place/name/entity questions, count whole named entities or repeated surface phrases. Prefer the longest repeated matching name/phrase over a shorter substring embedded inside that phrase unless the task explicitly asks about words or tokens.",
        "If a repeated longer phrase and a shorter substring start at the same occurrence, use the longer repeated phrase as the boundary; do not use the shorter substring unless the task asks for words or tokens.",
        "Treat prior structured evidence and answer options as provisional context, not an answer key. If the query asks you to verify a missing attribute, use the supplied media/transcripts to verify that attribute directly.",
        "For chart, table, scoreboard, or graph images, read label-value pairs from explicit visual alignment. If multiple images show the same progressive display, prefer the latest stable complete image and do not treat missing early bars or labels as zero.",
        "Use the INPUT MEDIA image numbers, artifact ids, and timestamps when comparing images; do not confuse Image N with timestamps or prior evidence ids.",
        "If the grounded evidence still leaves multiple answer options compatible, answer indeterminate instead of forcing a best guess.",
        "Return JSON only with keys: answer, supporting_points, confidence, analysis.",
        "Return the JSON object directly; do not write a preamble, markdown, or private step-by-step reasoning.",
        "Do not put phrases like 'Thinking Process' or hidden chain-of-thought in any JSON field.",
        "Keep answer short: the final value/choice, or indeterminate. Put concise evidence bullets in supporting_points and one short sentence in analysis.",
        "Set confidence to a numeric score from 0.0 to 1.0, not a label like High, Medium, or Low.",
        "Do not mention hidden tools or APIs.",
    ]
    parts.extend(
        [
            "",
            "OUTPUT EXAMPLES:",
            '{"answer":"B. Example Store, 20 percentage points","supporting_points":["Image 2 is the complete chart for Metric X.","Example Store: |70-50| = 20 points."],"confidence":0.82,"analysis":"The selected option has the largest grounded difference."}',
            '{"answer":"indeterminate","supporting_points":["Image 1 shows the object, but its required state is occluded."],"confidence":0.35,"analysis":"The answer-critical state is not directly visible."}',
            '{"answer":"C. Example phrase","supporting_points":["The repeated full name is Example Station, not the substring Example.","The interval between the two full mentions contains options A and B but not C."],"confidence":0.78,"analysis":"The answer uses full repeated-entity span boundaries."}',
        ]
    )
    task_question = str(task.get("question") or "").strip()
    if task_question:
        parts.extend(["", "TASK QUESTION:", task_question])
    options = [str(item).strip() for item in list(task.get("options") or []) if str(item).strip()]
    if options:
        parts.extend(["", "ANSWER OPTIONS:"])
        parts.extend("- %s" % option for option in options)
    parts.extend(["", "QUERY:", str(request.get("query") or "").strip()])
    if media_lines:
        parts.extend(["", "INPUT MEDIA:"])
        parts.extend("- %s" % line for line in media_lines if str(line).strip())
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
    evidence_records = [dict(item or {}) for item in list(payload.get("evidence_records") or []) if isinstance(item, dict)]

    image_paths: List[str] = []
    media_lines: List[str] = []
    frame_payloads = []
    if isinstance(request.get("frames"), list) and request.get("frames"):
        frame_payloads.extend([dict(item or {}) for item in request.get("frames") or [] if isinstance(item, dict)])
    if not frame_payloads and evidence_records:
        frame_payloads.extend(_evidence_frame_payloads(evidence_records))
    for frame_payload in frame_payloads:
        frame_path = absolute_frame_path(frame_payload, runtime)
        if frame_path and frame_path not in image_paths:
            image_paths.append(frame_path)
            media_lines.append(_media_line(len(image_paths), frame_payload))

    clip_payloads = []
    if isinstance(request.get("clips"), list) and request.get("clips"):
        clip_payloads.extend([dict(item or {}) for item in request.get("clips") or [] if isinstance(item, dict)])
    if not image_paths and clip_payloads:
        for clip_index, clip_payload in enumerate(clip_payloads, start=1):
            sampled = sample_request_frames(
                {"clips": [clip_payload]},
                task,
                out_dir=prompt_dir,
                prefix="generic_%02d" % clip_index,
                num_frames=4,
            )
            for item in sampled:
                frame_path = str(item["frame_path"])
                if frame_path not in image_paths:
                    image_paths.append(frame_path)
                    media_payload = {
                        "frame_path": frame_path,
                        "timestamp_s": item.get("timestamp_s"),
                        "clip": clip_payload,
                        "video_id": clip_payload.get("video_id") or task.get("video_id") or task.get("sample_key"),
                    }
                    media_lines.append(_media_line(len(image_paths), media_payload))

    transcript_payloads = []
    if isinstance(request.get("transcripts"), list) and request.get("transcripts"):
        transcript_payloads.extend([dict(item or {}) for item in request.get("transcripts") or [] if isinstance(item, dict)])
    transcript_text = _render_transcript_payloads(transcript_payloads)
    evidence_lines = [
        _evidence_line(item)
        for item in evidence_records
        if isinstance(item, dict)
        and str(item.get("atomic_text") or item.get("evidence_text") or item.get("text") or "").strip()
    ]
    text_contexts = [str(item).strip() for item in list(request.get("text_contexts") or []) if str(item).strip()]
    prompt = _build_prompt(request, task, transcript_text, evidence_lines, text_contexts, media_lines=media_lines)
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
            max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 768),
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
