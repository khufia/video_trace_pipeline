from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List

from ..common import extract_json_object
from .local_multimodal import QwenStyleRunner, make_qwen_image_messages
from .protocol import emit_json, fail_runtime, load_request
from .qwen35vl_runner import (
    _evidence_frame_payloads,
    _evidence_line,
    _media_line,
    _render_transcript_payloads,
)
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


def _final_json(raw_text: str) -> Dict[str, Any]:
    raw_text = str(raw_text or "")
    marker_start = raw_text.rfind("<final>")
    marker_end = raw_text.rfind("</final>")
    if marker_start >= 0 and marker_end > marker_start:
        candidate = raw_text[marker_start + len("<final>") : marker_end]
        parsed = extract_json_object(candidate)
        if isinstance(parsed, dict):
            return parsed
    parsed = extract_json_object(raw_text)
    if isinstance(parsed, dict):
        return parsed
    return {}


def _claim_lines(claims: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for claim in list(claims or []):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id") or "").strip()
        text = str(claim.get("text") or "").strip()
        claim_type = str(claim.get("claim_type") or "").strip()
        option = str(claim.get("expected_answer_option") or "").strip()
        parts = [part for part in (claim_id, claim_type, option) if part]
        prefix = "[%s] " % " | ".join(parts) if parts else ""
        if text:
            lines.append("%s%s" % (prefix, text))
    return lines


def _region_lines(regions: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for index, region in enumerate(list(regions or []), start=1):
        if not isinstance(region, dict):
            continue
        frame = dict(region.get("frame") or {})
        parts = ["Region %d" % index]
        if region.get("label"):
            parts.append("label=%s" % region.get("label"))
        if region.get("bbox"):
            parts.append("bbox=%s" % region.get("bbox"))
        if frame.get("artifact_id"):
            parts.append("frame_artifact_id=%s" % frame.get("artifact_id"))
        if frame.get("timestamp_s") is not None:
            parts.append("timestamp=%.3fs" % float(frame.get("timestamp_s")))
        lines.append(" | ".join(parts))
    return lines


def _build_prompt(
    request: Dict[str, Any],
    task: Dict[str, Any],
    transcript_text: str,
    evidence_lines: List[str],
    media_lines: List[str],
) -> str:
    parts = [
        "You are a strict verifier in a video QA evidence pipeline.",
        "Verify each claim only against the supplied media, transcripts, OCR results, observations, and evidence records.",
        "Return supported only for direct support. Return refuted only for direct contradiction. Return unknown when coverage is missing or ambiguous.",
        "Do not answer the benchmark question directly except through claim_results.",
        "Do not use prior beliefs or world knowledge. Do not include hidden reasoning.",
        "Return JSON only with keys: claim_results, new_observations, evidence_updates, checklist_updates, counter_updates, referent_updates, ocr_occurrence_updates, unresolved_gaps.",
        "",
        "TASK QUESTION:",
        str(task.get("question") or "").strip(),
    ]
    options = [str(item).strip() for item in list(task.get("options") or []) if str(item).strip()]
    if options:
        parts.extend(["", "ANSWER OPTIONS:"])
        parts.extend("- %s" % option for option in options)
    parts.extend(["", "VERIFICATION QUERY:", str(request.get("query") or "").strip()])
    claim_lines = _claim_lines([dict(item or {}) for item in list(request.get("claims") or []) if isinstance(item, dict)])
    if claim_lines:
        parts.extend(["", "CLAIMS TO VERIFY:"])
        parts.extend("- %s" % line for line in claim_lines)
    if media_lines:
        parts.extend(["", "INPUT MEDIA:"])
        parts.extend("- %s" % line for line in media_lines)
    region_lines = _region_lines([dict(item or {}) for item in list(request.get("regions") or []) if isinstance(item, dict)])
    if region_lines:
        parts.extend(["", "INPUT REGIONS:"])
        parts.extend("- %s" % line for line in region_lines)
    if transcript_text:
        parts.extend(["", "TRANSCRIPTS:", transcript_text])
    text_contexts = [str(item).strip() for item in list(request.get("text_contexts") or []) if str(item).strip()]
    if text_contexts:
        parts.extend(["", "TEXT CONTEXTS:"])
        parts.extend("- %s" % item for item in text_contexts[:80])
    if request.get("ocr_results"):
        parts.extend(["", "OCR RESULTS:", json.dumps(request.get("ocr_results"), ensure_ascii=False)])
    if request.get("dense_captions"):
        parts.extend(["", "DENSE CAPTIONS:", json.dumps(request.get("dense_captions"), ensure_ascii=False)])
    observations = [item for item in list(request.get("observations") or []) if isinstance(item, dict)]
    if observations:
        parts.extend(["", "OBSERVATIONS:"])
        for item in observations[:80]:
            text = str(item.get("atomic_text") or item.get("text") or "").strip()
            if text:
                identifier = str(item.get("observation_id") or "").strip()
                parts.append("- [%s] %s" % (identifier, text) if identifier else "- %s" % text)
    if evidence_lines:
        parts.extend(["", "EVIDENCE RECORDS:"])
        parts.extend("- %s" % line for line in evidence_lines)
    if request.get("retrieved_context"):
        parts.extend(["", "RETRIEVED CONTEXT:", json.dumps(request.get("retrieved_context"), ensure_ascii=False)])
    if request.get("verification_policy"):
        parts.extend(["", "VERIFICATION POLICY:", json.dumps(request.get("verification_policy"), ensure_ascii=False)])
    parts.extend(
        [
            "",
            "Required JSON shape:",
            '{"claim_results":[{"claim_id":"claim_1","verdict":"supported|refuted|unknown|partially_supported","confidence":0.0,"answer_value":null,"supporting_observation_ids":[],"supporting_evidence_ids":[],"refuting_observation_ids":[],"refuting_evidence_ids":[],"time_intervals":[],"artifact_refs":[],"rationale":"short reason","coverage":{"checked_inputs":[],"missing_inputs":[],"sampling_summary":"what was checked"}}],"new_observations":[],"evidence_updates":[],"checklist_updates":[],"counter_updates":[],"referent_updates":[],"ocr_occurrence_updates":[],"unresolved_gaps":[]}',
        ]
    )
    return "\n".join(parts).strip()


def execute_payload(payload: Dict[str, Any], *, runner_pool: "PersistentModelPool | None" = None) -> Dict[str, Any]:
    request = dict(payload.get("request") or {})
    task = dict(payload.get("task") or {})
    runtime = dict(payload.get("runtime") or {})
    if not str(request.get("query") or "").strip():
        fail_runtime("verifier requires a non-empty query")
    if not list(request.get("claims") or []):
        fail_runtime("verifier requires at least one claim")

    model_path = resolve_model_path(str(runtime.get("model_name") or ""), runtime)
    device_label = resolved_device_label(runtime)
    prompt_dir = scratch_dir(runtime, "verifier")
    evidence_records = [dict(item or {}) for item in list(payload.get("evidence_records") or []) if isinstance(item, dict)]

    image_paths: List[str] = []
    media_lines: List[str] = []
    frame_payloads = []
    for key in ("frames",):
        if isinstance(request.get(key), list):
            frame_payloads.extend([dict(item or {}) for item in request.get(key) or [] if isinstance(item, dict)])
    for region in [dict(item or {}) for item in list(request.get("regions") or []) if isinstance(item, dict)]:
        frame = dict(region.get("frame") or {})
        if frame:
            frame_payloads.append(frame)
    if not frame_payloads and evidence_records:
        frame_payloads.extend(_evidence_frame_payloads(evidence_records))
    for frame_payload in frame_payloads:
        frame_path = absolute_frame_path(frame_payload, runtime)
        if frame_path and frame_path not in image_paths:
            image_paths.append(frame_path)
            media_lines.append(_media_line(len(image_paths), frame_payload))

    clip_payloads = [dict(item or {}) for item in list(request.get("clips") or []) if isinstance(item, dict)]
    if not image_paths and clip_payloads:
        for clip_index, clip_payload in enumerate(clip_payloads, start=1):
            sampled = sample_request_frames(
                {"clips": [clip_payload]},
                task,
                out_dir=prompt_dir,
                prefix="verifier_%02d" % clip_index,
                num_frames=6,
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

    transcript_payloads = [dict(item or {}) for item in list(request.get("transcripts") or []) if isinstance(item, dict)]
    transcript_text = _render_transcript_payloads(transcript_payloads)
    evidence_lines = [
        _evidence_line(item)
        for item in evidence_records
        if isinstance(item, dict)
        and str(item.get("atomic_text") or item.get("evidence_text") or item.get("text") or "").strip()
    ]
    prompt = _build_prompt(request, task, transcript_text, evidence_lines, media_lines)
    generation = resolve_generation_controls(runtime)
    attn_implementation = str((runtime.get("extra") or {}).get("attn_implementation") or "").strip() or None
    runner = None
    owns_runner = False
    if runner_pool is not None:
        runner = runner_pool.acquire_qwen_style_runner(
            tool_name="verifier",
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
            max_new_tokens=int((runtime.get("extra") or {}).get("max_new_tokens") or 1024),
        )
    finally:
        if owns_runner:
            runner.close()
    parsed = _final_json(raw_text)
    if not parsed:
        fail_runtime("verifier did not return parseable JSON", extra={"raw_prefix": raw_text[:1000]})
    parsed.setdefault("new_observations", [])
    parsed.setdefault("evidence_updates", [])
    parsed.setdefault("checklist_updates", [])
    parsed.setdefault("counter_updates", [])
    parsed.setdefault("referent_updates", [])
    parsed.setdefault("ocr_occurrence_updates", [])
    parsed.setdefault("unresolved_gaps", [])
    return parsed


def main() -> None:
    emit_json(execute_payload(load_request()))


if __name__ == "__main__":
    main()
