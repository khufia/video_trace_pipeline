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


def _unknown_claim_result(claim_id: str, rationale: str, missing_input: str) -> Dict[str, Any]:
    return {
        "claim_id": claim_id,
        "verdict": "unknown",
        "confidence": 0.0,
        "answer_value": None,
        "claimed_value": None,
        "observed_value": None,
        "match_status": "unknown",
        "target_presence": "unknown",
        "supporting_observation_ids": [],
        "supporting_evidence_ids": [],
        "refuting_observation_ids": [],
        "refuting_evidence_ids": [],
        "time_intervals": [],
        "artifact_refs": [],
        "rationale": rationale,
        "coverage": {
            "checked_inputs": [],
            "missing_inputs": [missing_input],
            "sampling_summary": rationale,
        },
    }


def _unknown_results_for_unparseable_output(request: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    raw_prefix = str(raw_text or "")[:1000]
    claim_results = []
    for claim in [dict(item or {}) for item in list(request.get("claims") or []) if isinstance(item, dict)]:
        claim_id = str(claim.get("claim_id") or "").strip()
        if not claim_id:
            continue
        claim_results.append(
            _unknown_claim_result(
                claim_id,
                "Verifier model output was not parseable JSON, so this claim was not validated.",
                "parseable verifier JSON",
            )
        )
    return {
        "claim_results": claim_results,
        "new_observations": [],
        "evidence_updates": [],
        "checklist_updates": [],
        "counter_updates": [],
        "referent_updates": [],
        "ocr_occurrence_updates": [],
        "unresolved_gaps": [
            "verifier_unparseable_output: The verifier model output could not be parsed as JSON. raw_prefix=%s"
            % raw_prefix
        ],
    }


def _repair_claim_results(parsed: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
    expected_claim_ids = [
        str(item.get("claim_id") or "").strip()
        for item in [dict(claim or {}) for claim in list(request.get("claims") or []) if isinstance(claim, dict)]
        if str(item.get("claim_id") or "").strip()
    ]
    if not expected_claim_ids:
        return parsed
    raw_results = parsed.get("claim_results")
    if isinstance(raw_results, dict):
        raw_results = [
            dict(value or {}, claim_id=str(key))
            if isinstance(value, dict)
            else {"claim_id": str(key), "verdict": "unknown", "rationale": str(value or "").strip()}
            for key, value in raw_results.items()
        ]
    if not isinstance(raw_results, list):
        raw_results = []
    normalized_results = [dict(item or {}) for item in raw_results if isinstance(item, dict)]

    result_ids = [str(item.get("claim_id") or "").strip() for item in normalized_results]
    expected_set = set(expected_claim_ids)
    if normalized_results and (not set(result_ids).intersection(expected_set)) and len(normalized_results) == len(expected_claim_ids):
        for item, claim_id in zip(normalized_results, expected_claim_ids):
            item["claim_id"] = claim_id

    by_id: Dict[str, Dict[str, Any]] = {}
    extras: List[Dict[str, Any]] = []
    for item in normalized_results:
        claim_id = str(item.get("claim_id") or "").strip()
        if claim_id in expected_set and claim_id not in by_id:
            by_id[claim_id] = item
        else:
            extras.append(item)
    for claim_id, extra in zip([claim_id for claim_id in expected_claim_ids if claim_id not in by_id], extras):
        fixed = dict(extra)
        fixed["claim_id"] = claim_id
        by_id[claim_id] = fixed
    parsed["claim_results"] = [
        by_id.get(claim_id)
        or _unknown_claim_result(
            claim_id,
            "Verifier did not return a result for this exact claim_id.",
            "claim_result with matching claim_id",
        )
        for claim_id in expected_claim_ids
    ]
    return parsed


def _normalize_unresolved_gaps(value: Any) -> List[str]:
    gaps: List[str] = []
    for item in list(value or []):
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            gap = str(item.get("gap") or item.get("id") or item.get("name") or "verifier_gap").strip()
            details = str(item.get("details") or item.get("reason") or item.get("description") or "").strip()
            raw_prefix = str(item.get("raw_prefix") or "").strip()
            parts = [gap]
            if details:
                parts.append(details)
            if raw_prefix:
                parts.append("raw_prefix=%s" % raw_prefix)
            text = ": ".join(parts[:2])
            if len(parts) > 2:
                text = "%s %s" % (text, parts[2])
        else:
            text = str(item or "").strip()
        if text:
            gaps.append(text)
    return gaps


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


def _clip_lines(clips: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for index, clip in enumerate(list(clips or []), start=1):
        if not isinstance(clip, dict):
            continue
        parts = ["Clip %d" % index]
        if clip.get("video_id"):
            parts.append("video_id=%s" % clip.get("video_id"))
        if clip.get("start_s") is not None or clip.get("end_s") is not None:
            parts.append("time=%.3f-%.3fs" % (float(clip.get("start_s") or 0.0), float(clip.get("end_s") or 0.0)))
        if clip.get("artifact_id"):
            parts.append("artifact_id=%s" % clip.get("artifact_id"))
        metadata = clip.get("metadata") if isinstance(clip.get("metadata"), dict) else {}
        if metadata.get("tool_backend"):
            parts.append("tool_backend=%s" % metadata.get("tool_backend"))
        if metadata.get("event_label"):
            parts.append("event_label=%s" % metadata.get("event_label"))
        if metadata.get("source"):
            parts.append("source=%s" % metadata.get("source"))
        lines.append(" | ".join(parts))
    return lines


def _build_prompt(
    request: Dict[str, Any],
    task: Dict[str, Any],
    transcript_text: str,
    evidence_lines: List[str],
    media_lines: List[str],
) -> str:
    verification_mode = str(request.get("verification_mode") or "strict").strip() or "strict"
    parts = [
        "You are a strict verifier in a video QA evidence pipeline.",
        "Verify each claim only against the supplied media, transcripts, OCR results, observations, and evidence records.",
        "Return supported only for direct support. Return refuted only for direct contradiction. Return unknown when coverage is missing or ambiguous.",
        "You must return exactly one claim_result for every input claim and preserve each input claim_id exactly.",
        "Do not answer the benchmark question directly except through claim_results.",
        "Do not use prior beliefs or world knowledge. Do not include hidden reasoning.",
        "Return JSON only with keys: claim_results, new_observations, evidence_updates, checklist_updates, counter_updates, referent_updates, ocr_occurrence_updates, unresolved_gaps.",
        "For visual/action/relation/count claims, first decide whether the target was actually present in the supplied inputs. If target presence is missing or partial, verdict must be unknown unless there is direct contradiction.",
        "For count/value/comparison claims, fill claimed_value, observed_value, and match_status (match, mismatch, partial, unknown).",
        "For count tasks, emit counter_updates with counter_id `task_count` unless the request policy names a different counter_id. Include candidates with canonical_label, raw_mentions, status, reason, dedupe_rationale, accepted_observation_ids, rejected_observation_ids, count, and status.",
        "For same/different/person/object relation tasks, emit referent_updates for each grounded slot when the evidence supports it.",
        "Confidence calibration: high confidence only when the supplied evidence covers the needed target, interval, and discriminator; use low confidence for weak or indirect evidence.",
        "",
        "TASK QUESTION:",
        str(task.get("question") or "").strip(),
    ]
    if verification_mode == "mcq_comparative":
        parts.extend(
            [
                "",
                "VERIFICATION MODE: mcq_comparative",
                "This is a multiple-choice benchmark reduction. Compare the supplied claims/options against each other.",
                "Support the option whose core discriminator is best grounded by the supplied evidence, even if the option wording is not a verbatim transcript label.",
                "Do not mark every option unknown merely because evaluative labels such as tone, emotion, relation, or intent require interpretation from supplied audio/video/transcript context.",
                "Use refuted for options contradicted by the supported discriminator. Use unknown only when the supplied evidence does not discriminate that option.",
                "For over-specific wording, judge whether the answer-critical discriminator is supported; explain caveats in rationale instead of refusing all options.",
                "For tone/emotion claims, use audio/video clips, frame sequence context, facial/body cues, and transcript semantics together. Transcript text alone is not sufficient unless the tone is explicitly stated.",
                "For count claims, deduplicate candidate mentions by canonical sound/event/object label before setting observed_value.",
            ]
        )
    else:
        parts.extend(
            [
                "",
                "VERIFICATION MODE: strict",
                "Strict mode is claim-by-claim. Do not choose a best option unless that specific claim has direct support.",
            ]
        )
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
    clip_lines = _clip_lines([dict(item or {}) for item in list(request.get("clips") or []) if isinstance(item, dict)])
    if clip_lines:
        parts.extend(
            [
                "",
                "INPUT CLIPS / AUDIO EVENT CANDIDATES:",
                "These clips are evidence inputs. If a clip comes from audio_temporal_grounder, treat its interval and metadata as an audio-event candidate from that tool; do not say no audio evidence exists merely because the waveform is not shown as text.",
            ]
        )
        parts.extend("- %s" % line for line in clip_lines)
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
            '{"claim_results":[{"claim_id":"<copy_input_claim_id_exactly>","verdict":"supported|refuted|unknown|partially_supported","confidence":0.0,"answer_value":null,"claimed_value":null,"observed_value":null,"match_status":"match|mismatch|partial|unknown","target_presence":"present|absent|partial|unknown","supporting_observation_ids":[],"supporting_evidence_ids":[],"refuting_observation_ids":[],"refuting_evidence_ids":[],"time_intervals":[],"artifact_refs":[],"rationale":"short reason","coverage":{"checked_inputs":[],"missing_inputs":[],"sampling_summary":"what was checked"}}],"new_observations":[],"evidence_updates":[],"checklist_updates":[],"counter_updates":[{"counter_id":"task_count","target":"","inclusion_rule":"","exclusion_rule":"","candidates":[{"canonical_label":"","raw_mentions":[],"status":"accepted|rejected|unknown","reason":"","dedupe_rationale":""}],"accepted_observation_ids":[],"rejected_observation_ids":[],"count":null,"status":"open|candidate|validated|unknown"}],"referent_updates":[],"ocr_occurrence_updates":[],"unresolved_gaps":[]}',
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
        parsed = _unknown_results_for_unparseable_output(request, raw_text)
    parsed = _repair_claim_results(parsed, request)
    parsed.setdefault("new_observations", [])
    parsed.setdefault("evidence_updates", [])
    parsed.setdefault("checklist_updates", [])
    parsed.setdefault("counter_updates", [])
    parsed.setdefault("referent_updates", [])
    parsed.setdefault("ocr_occurrence_updates", [])
    parsed["unresolved_gaps"] = _normalize_unresolved_gaps(parsed.get("unresolved_gaps"))
    return parsed


def main() -> None:
    emit_json(execute_payload(load_request()))


if __name__ == "__main__":
    main()
