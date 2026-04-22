from __future__ import annotations

from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from ..schemas import ModelsConfig, TaskSpec
from ..storage import RunContext, WorkspaceManager


def _request_model_field_names(model_cls) -> set:
    if hasattr(model_cls, "model_fields"):
        return set(getattr(model_cls, "model_fields").keys())
    if hasattr(model_cls, "__fields__"):
        return set(getattr(model_cls, "__fields__").keys())
    return set()


def _normalize_request_payload(payload: Dict[str, Any], request_model: Type[BaseModel]) -> Dict[str, Any]:
    normalized = dict(payload or {})
    model_fields = _request_model_field_names(request_model)
    if "clips" in model_fields:
        if isinstance(normalized.get("clip"), list) and normalized["clip"]:
            normalized["clips"] = list(normalized["clip"])
            normalized["clip"] = None
        elif normalized.get("clip") is not None and not normalized.get("clips"):
            normalized["clips"] = [normalized["clip"]]
        elif isinstance(normalized.get("clips"), dict):
            normalized["clips"] = [normalized["clips"]]
    elif "clip" in model_fields and isinstance(normalized.get("clip"), list) and normalized["clip"]:
        normalized["clip"] = normalized["clip"][0]
    if "frames" in model_fields:
        if isinstance(normalized.get("frame"), list) and normalized["frame"]:
            normalized["frames"] = list(normalized["frame"])
            normalized["frame"] = None
        elif normalized.get("frame") is not None and not normalized.get("frames"):
            normalized["frames"] = [normalized["frame"]]
        elif isinstance(normalized.get("frames"), dict):
            normalized["frames"] = [normalized["frames"]]
    elif "frame" in model_fields and isinstance(normalized.get("frame"), list) and normalized["frame"]:
        normalized["frame"] = normalized["frame"][0]
    if "regions" in model_fields:
        if isinstance(normalized.get("region"), list) and normalized["region"]:
            normalized["regions"] = list(normalized["region"])
            normalized["region"] = None
        elif normalized.get("region") is not None and not normalized.get("regions"):
            normalized["regions"] = [normalized["region"]]
        elif isinstance(normalized.get("regions"), dict):
            normalized["regions"] = [normalized["regions"]]
    elif "region" in model_fields and isinstance(normalized.get("region"), list) and normalized["region"]:
        normalized["region"] = normalized["region"][0]
    if "transcripts" in model_fields:
        if isinstance(normalized.get("transcript"), list) and normalized["transcript"]:
            normalized["transcripts"] = list(normalized["transcript"])
            normalized["transcript"] = None
        elif normalized.get("transcript") is not None and not normalized.get("transcripts"):
            normalized["transcripts"] = [normalized["transcript"]]
        elif isinstance(normalized.get("transcripts"), dict):
            normalized["transcripts"] = [normalized["transcripts"]]
    elif "transcript" in model_fields and isinstance(normalized.get("transcript"), list) and normalized["transcript"]:
        normalized["transcript"] = normalized["transcript"][0]
    if "time_hints" in model_fields:
        if isinstance(normalized.get("time_hint"), list) and normalized["time_hint"]:
            normalized["time_hints"] = [str(item).strip() for item in normalized["time_hint"] if str(item).strip()]
            normalized["time_hint"] = None
        elif normalized.get("time_hint") is not None and not normalized.get("time_hints"):
            rendered = str(normalized.get("time_hint") or "").strip()
            normalized["time_hints"] = [rendered] if rendered else []
        elif isinstance(normalized.get("time_hints"), str):
            rendered = str(normalized["time_hints"]).strip()
            normalized["time_hints"] = [rendered] if rendered else []
    if "query" in model_fields and "query" not in normalized:
        for alias in ("task", "question", "instruction", "prompt"):
            if normalized.get(alias) is not None:
                normalized["query"] = normalized[alias]
                break
    if "num_frames" in model_fields and "num_frames" not in normalized and "count" in normalized:
        normalized["num_frames"] = normalized["count"]
    if "num_frames" in model_fields and "num_frames" not in normalized and "max_frames" in normalized:
        normalized["num_frames"] = normalized["max_frames"]
    if "num_frames" in model_fields and "num_frames" not in normalized and "top_k" in normalized:
        normalized["num_frames"] = normalized["top_k"]
    if "top_k" in model_fields and "top_k" not in normalized and "max_segments" in normalized:
        normalized["top_k"] = normalized["max_segments"]
    if "frame" in model_fields and normalized.get("frame") is None:
        if (
            isinstance(normalized.get("frames"), list)
            and len(normalized["frames"]) == 1
        ):
            normalized["frame"] = normalized["frames"][0]
        elif isinstance(normalized.get("region"), dict) and normalized["region"].get("frame") is not None:
            normalized["frame"] = normalized["region"]["frame"]
    if "clip" in model_fields and normalized.get("clip") is None:
        if normalized.get("segment") is not None:
            normalized["clip"] = normalized["segment"]
        elif isinstance(normalized.get("frame"), dict) and normalized["frame"].get("clip") is not None:
            normalized["clip"] = normalized["frame"]["clip"]
        elif isinstance(normalized.get("segments"), list) and len(normalized["segments"]) == 1:
            normalized["clip"] = normalized["segments"][0]
        elif isinstance(normalized.get("clips"), list) and len(normalized["clips"]) == 1:
            normalized["clip"] = normalized["clips"][0]
        elif normalized.get("clip_start_s") is not None or normalized.get("clip_end_s") is not None:
            start_s = normalized.get("clip_start_s")
            if start_s is None:
                start_s = normalized.get("start_s")
            end_s = normalized.get("clip_end_s")
            if end_s is None:
                end_s = normalized.get("end_s")
            if start_s is None:
                start_s = 0.0
            if end_s is None:
                end_s = start_s
            video_id = (
                normalized.get("video_id")
                or normalized.get("clip_video_id")
                or normalized.get("frame_video_id")
                or normalized.get("sample_key")
            )
            if video_id:
                normalized["clip"] = {
                    "video_id": video_id,
                    "start_s": start_s,
                    "end_s": end_s,
                }
    return normalized


class ToolExecutionContext(object):
    def __init__(
        self,
        workspace: WorkspaceManager,
        run: RunContext,
        task: TaskSpec,
        models_config: ModelsConfig,
        llm_client=None,
        evidence_lookup=None,
        preprocess_bundle=None,
    ):
        self.workspace = workspace
        self.run = run
        self.task = task
        self.models_config = models_config
        self.llm_client = llm_client
        self.evidence_lookup = evidence_lookup
        self.preprocess_bundle = preprocess_bundle


class ToolAdapter(object):
    name = ""
    request_model = BaseModel

    def parse_request(self, arguments: Dict[str, Any]):
        payload = _normalize_request_payload(arguments, self.request_model)
        payload.setdefault("tool_name", self.name)
        return self.request_model.parse_obj(payload)

    def execute(self, request, context: ToolExecutionContext):
        raise NotImplementedError
