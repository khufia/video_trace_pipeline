from __future__ import annotations

from typing import Any, Dict, List

from .protocol import emit_json, fail_runtime, load_request


def _reference_harness_cls():
    from .reference_adapter import ReferenceHarness

    return ReferenceHarness


def main() -> None:
    payload = load_request()
    request = dict(payload.get("request") or {})
    task = dict(payload.get("task") or {})
    runtime = dict(payload.get("runtime") or {})

    query = str(request.get("query") or "").strip()
    if not query:
        fail_runtime("audio_temporal_grounder requires a non-empty query")

    clip = dict(request.get("clip") or {})
    harness = _reference_harness_cls()(
        task=task,
        runtime=runtime,
        clip_duration_s=max(1.0, float((clip.get("end_s") or 0.0)) - float((clip.get("start_s") or 0.0)) or 30.0),
    )
    result = harness._audio_grounder_clap(
        {
            "query": query,
            "start_time": clip.get("start_s") if clip else None,
            "end_time": clip.get("end_s") if clip else None,
        }
    )

    events = []
    clips = []
    for item in result.get("events") or []:
        if not isinstance(item, dict):
            continue
        start_s = float(item.get("start") or item.get("start_s") or 0.0)
        end_s = float(item.get("end") or item.get("end_s") or start_s)
        confidence = item.get("confidence")
        event = {
            "event_label": str(item.get("event_label") or query),
            "start_s": start_s,
            "end_s": end_s,
            "confidence": None if confidence is None else float(confidence),
            "metadata": {"tool_backend": result.get("backend") or "laion_clap"},
        }
        events.append(event)
        clips.append(
            {
                "video_id": str(task.get("video_id") or task.get("sample_key") or ""),
                "start_s": start_s,
                "end_s": end_s,
                "confidence": None if confidence is None else float(confidence),
                "metadata": {"event_label": event["event_label"]},
            }
        )

    emit_json(
        {
            "query": query,
            "clips": clips,
            "events": events,
            "retrieval_backend": str(result.get("backend") or "local_clap"),
            "query_absent": not bool(events),
            "summary": str(result.get("audio_summary") or ("No matching audio event found." if not events else "")),
        }
    )


if __name__ == "__main__":
    main()
