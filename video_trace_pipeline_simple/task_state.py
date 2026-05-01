from __future__ import annotations

from typing import Any


def initial_state(task: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for option in list(task.get("options") or []):
        text = str(option or "").strip()
        if text:
            candidates.append({"text": text, "status": "unknown"})
    return {
        "answer_candidates": candidates,
        "known_facts": [],
        "open_questions": [str(task.get("question") or "").strip()],
        "tool_failures": [],
    }


def update_state(
    task: dict[str, Any],
    state: dict[str, Any],
    previous_steps: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    trace: dict[str, Any] | None,
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    updated = dict(state or initial_state(task))
    failures = list(updated.get("tool_failures") or [])
    for record in list(previous_steps or []):
        result = dict(record.get("result") or {})
        if result.get("ok") is False:
            step = dict(record.get("step") or {})
            item = {"step_id": step.get("id"), "tool": step.get("tool"), "error": result.get("error")}
            if item not in failures:
                failures.append(item)
    updated["tool_failures"] = failures[-20:]
    known = list(updated.get("known_facts") or [])
    for observation in list(observations or [])[-20:]:
        text = str(observation.get("text") or observation.get("atomic_text") or "").strip()
        if text and text not in known:
            known.append(text)
    updated["known_facts"] = known[-40:]
    audit = dict(audit or {})
    open_questions = [str(item).strip() for item in list(audit.get("missing_information") or []) if str(item).strip()]
    if not open_questions and audit.get("verdict") != "PASS":
        feedback = str(audit.get("feedback") or "").strip()
        if feedback:
            open_questions = [feedback]
    updated["open_questions"] = open_questions
    answer = str((trace or {}).get("answer") or (trace or {}).get("final_answer") or "").strip()
    if answer:
        for candidate in list(updated.get("answer_candidates") or []):
            if answer in str(candidate.get("text") or "") or str(candidate.get("text") or "") in answer:
                candidate["status"] = "selected_candidate"
    return updated
