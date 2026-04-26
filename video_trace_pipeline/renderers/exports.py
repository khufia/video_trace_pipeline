from __future__ import annotations

from typing import Dict, List

from ..temporal import render_temporal_anchor


def _format_seconds(value) -> str:
    text = "%.3f" % float(value)
    text = text.rstrip("0").rstrip(".")
    return "%ss" % text


def export_trace_for_benchmark(benchmark: str, task, trace_package: dict) -> Dict[str, object]:
    benchmark_key = str(benchmark or "").strip().lower()
    inference_steps = []
    for item in trace_package.get("inference_steps") or []:
        text = str(item.get("text", "") or "").strip()
        temporal_anchor = render_temporal_anchor(item)
        if temporal_anchor:
            inference_steps.append("[%s] %s" % (temporal_anchor, text))
        else:
            inference_steps.append(text)
    evidence_entries = trace_package.get("evidence_entries") or []
    final_answer = trace_package.get("final_answer", "")
    if benchmark_key == "videomathqa":
        return {
            "question_id": task.question_id,
            "video_id": task.video_id,
            "steps": ["%d. %s" % (idx + 1, step) for idx, step in enumerate(inference_steps)],
            "answer": final_answer,
        }
    if benchmark_key == "minerva":
        return {
            "key": task.question_id,
            "video_id": task.video_id,
            "trace": "\n".join(inference_steps),
            "answer": final_answer,
        }
    if benchmark_key == "omnivideobench":
        triples = []
        max_len = max(len(evidence_entries), len(inference_steps))
        for idx in range(max_len):
            evidence = evidence_entries[idx]["evidence_text"] if idx < len(evidence_entries) else ""
            inference = inference_steps[idx] if idx < len(inference_steps) else ""
            triples.append({"modality": "mixed", "evidence": evidence, "inference": inference})
        return {"sample_key": task.sample_key, "trace": triples, "answer": final_answer}
    return {
        "sample_key": task.sample_key,
        "trace": {
            "inference_steps": inference_steps,
            "evidence": [
                {
                    "tool_name": item.get("tool_name", ""),
                    "evidence_text": item.get("evidence_text", ""),
                    "observation_ids": list(item.get("observation_ids") or []),
                    "time_start_s": item.get("time_start_s"),
                    "time_end_s": item.get("time_end_s"),
                    "frame_ts_s": item.get("frame_ts_s"),
                    "time_intervals": list(item.get("time_intervals") or []),
                }
                for item in evidence_entries
            ],
        },
        "answer": final_answer,
    }


def render_trace_markdown(trace_package: dict) -> str:
    lines = ["# Trace", ""]
    lines.append("## Final Answer")
    lines.append("")
    lines.append(trace_package.get("final_answer", ""))
    lines.append("")
    lines.append("## Inference")
    lines.append("")
    for step in trace_package.get("inference_steps") or []:
        line = "%s. %s" % (step.get("step_id"), step.get("text", ""))
        temporal_anchor = render_temporal_anchor(step)
        if temporal_anchor:
            line = "%s [%s]" % (line, temporal_anchor)
        lines.append(line)
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    for item in trace_package.get("evidence_entries") or []:
        lines.append("### %s" % item.get("evidence_id", "evidence"))
        lines.append("")
        if item.get("status"):
            lines.append("- Status: %s" % item.get("status"))
        temporal_anchor = render_temporal_anchor(item)
        if temporal_anchor:
            lines.append("- Time: %s" % temporal_anchor)
        lines.append("- Text: %s" % item.get("evidence_text", ""))
        if item.get("observation_ids"):
            lines.append("- Observations: %s" % ", ".join(item.get("observation_ids") or []))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
