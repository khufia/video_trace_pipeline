from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from ..common import extract_json_object
from ..prompts.trace_synthesizer_prompt import SYNTHESIZER_SYSTEM_PROMPT, build_synthesizer_prompt
from ..schemas import TracePackage


def _nonempty_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_evidence_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"validated", "provisional", "superseded"}:
        return normalized
    if normalized in {
        "grounded",
        "supported",
        "supporting",
        "confirmed",
        "corroborated",
        "consistent",
        "relevant",
        "usable",
        "valid",
    }:
        return "provisional"
    if normalized in {
        "discarded",
        "obsolete",
        "overridden",
        "replaced",
        "stale",
        "contradicted",
        "invalidated",
    }:
        return "superseded"
    return "provisional"


def _temporal_anchor_from_observation_ids(
    observation_ids: Iterable[Any],
    observation_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    time_starts = []
    time_ends = []
    frame_timestamps = []
    for raw_observation_id in list(observation_ids or []):
        observation_id = _nonempty_text(raw_observation_id)
        if observation_id is None:
            continue
        observation = observation_by_id.get(observation_id)
        if not isinstance(observation, dict):
            continue
        start_s = _coerce_float(observation.get("time_start_s"))
        end_s = _coerce_float(observation.get("time_end_s"))
        frame_ts_s = _coerce_float(observation.get("frame_ts_s"))
        if start_s is not None:
            time_starts.append(start_s)
        if end_s is not None:
            time_ends.append(end_s)
        elif start_s is not None:
            time_ends.append(start_s)
        if frame_ts_s is not None:
            frame_timestamps.append(frame_ts_s)

    payload: Dict[str, Any] = {}
    if time_starts or time_ends:
        payload["time_start_s"] = round(min(time_starts or time_ends), 3)
        payload["time_end_s"] = round(max(time_ends or time_starts), 3)
    unique_frames = sorted({round(value, 3) for value in frame_timestamps})
    if len(unique_frames) == 1:
        payload["frame_ts_s"] = unique_frames[0]
    return payload


def _repair_trace_package_payload(
    payload: Dict[str, Any],
    *,
    evidence_entries: Iterable[Dict[str, Any]] | None,
    observations: Iterable[Dict[str, Any]] | None,
    current_trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    repaired = dict(payload or {})

    evidence_by_id: Dict[str, Dict[str, Any]] = {}
    for source in list(evidence_entries or []):
        if not isinstance(source, dict):
            continue
        evidence_id = _nonempty_text(source.get("evidence_id"))
        if evidence_id and evidence_id not in evidence_by_id:
            evidence_by_id[evidence_id] = dict(source)
    for source in list((current_trace or {}).get("evidence_entries") or []):
        if not isinstance(source, dict):
            continue
        evidence_id = _nonempty_text(source.get("evidence_id"))
        if evidence_id and evidence_id not in evidence_by_id:
            evidence_by_id[evidence_id] = dict(source)

    tool_by_observation_id: Dict[str, str] = {}
    observation_by_id: Dict[str, Dict[str, Any]] = {}
    for item in list(observations or []):
        if not isinstance(item, dict):
            continue
        observation_id = _nonempty_text(item.get("observation_id"))
        source_tool = _nonempty_text(item.get("source_tool"))
        if observation_id and source_tool:
            tool_by_observation_id[observation_id] = source_tool
        if observation_id:
            observation_by_id[observation_id] = dict(item)

    repaired_entries = []
    for item in list(repaired.get("evidence_entries") or []):
        if not isinstance(item, dict):
            repaired_entries.append(item)
            continue
        fixed = dict(item)
        fixed["status"] = _normalize_evidence_status(fixed.get("status"))
        tool_name = _nonempty_text(fixed.get("tool_name"))
        if tool_name is None:
            evidence_id = _nonempty_text(fixed.get("evidence_id"))
            cached_entry = evidence_by_id.get(evidence_id or "")
            tool_name = _nonempty_text((cached_entry or {}).get("tool_name"))
        if tool_name is None:
            observation_ids = [
                _nonempty_text(observation_id)
                for observation_id in list(fixed.get("observation_ids") or [])
            ]
            source_tools = {
                tool_by_observation_id[observation_id]
                for observation_id in observation_ids
                if observation_id and observation_id in tool_by_observation_id
            }
            if len(source_tools) == 1:
                tool_name = next(iter(source_tools))
        if tool_name is not None:
            fixed["tool_name"] = tool_name
        repaired_entries.append(fixed)
    repaired["evidence_entries"] = repaired_entries

    repaired_steps = []
    for item in list(repaired.get("inference_steps") or []):
        if not isinstance(item, dict):
            repaired_steps.append(item)
            continue
        fixed = dict(item)
        temporal_anchor = _temporal_anchor_from_observation_ids(
            fixed.get("supporting_observation_ids") or [],
            observation_by_id,
        )
        for key, value in temporal_anchor.items():
            if fixed.get(key) is None and value is not None:
                fixed[key] = value
        repaired_steps.append(fixed)
    repaired["inference_steps"] = repaired_steps
    return repaired


class TraceSynthesizerAgent(object):
    def __init__(self, llm_client, agent_config):
        self.llm_client = llm_client
        self.agent_config = agent_config

    def synthesize(self, task, mode, evidence_entries, observations, current_trace, refinement_instructions):
        prompt = build_synthesizer_prompt(
            task=task,
            mode=mode,
            evidence_entries=evidence_entries,
            observations=observations,
            current_trace=current_trace,
            refinement_instructions=refinement_instructions,
        )
        raw = self.llm_client.complete_text(
            endpoint_name=self.agent_config.endpoint or "default",
            model_name=self.agent_config.model,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_format={"type": "json_object"},
            temperature=self.agent_config.temperature,
            max_tokens=self.agent_config.max_tokens,
        )
        payload = extract_json_object(raw)
        if payload is None:
            raise ValueError("Trace synthesizer did not return a JSON object: %s" % raw[:1000])
        payload = _repair_trace_package_payload(
            payload,
            evidence_entries=evidence_entries,
            observations=observations,
            current_trace=current_trace,
        )
        if hasattr(TracePackage, "model_validate"):
            parsed = TracePackage.model_validate(payload)
        else:
            parsed = TracePackage.parse_obj(payload)
        return raw, parsed
