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
    if "query" in model_fields and "query" not in normalized:
        for alias in ("task", "question", "instruction", "prompt"):
            if normalized.get(alias) is not None:
                normalized["query"] = normalized[alias]
                break
    if "time_hint" in model_fields and "time_hint" not in normalized:
        for alias in ("clip_hint", "time_window", "temporal_hint", "window_hint"):
            if normalized.get(alias) is not None:
                normalized["time_hint"] = normalized[alias]
                break
    if "num_frames" in model_fields and "num_frames" not in normalized and "count" in normalized:
        normalized["num_frames"] = normalized["count"]
    if "num_frames" in model_fields and "num_frames" not in normalized and "max_frames" in normalized:
        normalized["num_frames"] = normalized["max_frames"]
    if "top_k" in model_fields and "top_k" not in normalized and "max_segments" in normalized:
        normalized["top_k"] = normalized["max_segments"]
    if "region" in model_fields and normalized.get("region") is None and normalized.get("image_region") is not None:
        normalized["region"] = normalized["image_region"]
    if "region" in model_fields and normalized.get("region") is None:
        for key, value in normalized.items():
            if key.endswith("_region") and value is not None:
                normalized["region"] = value
                break
    if "frame" in model_fields and normalized.get("frame") is None:
        if normalized.get("image") is not None:
            normalized["frame"] = normalized["image"]
        elif isinstance(normalized.get("frames"), list) and normalized["frames"]:
            normalized["frame"] = normalized["frames"][0]
        elif isinstance(normalized.get("image_region"), dict) and normalized["image_region"].get("frame") is not None:
            normalized["frame"] = normalized["image_region"]["frame"]
        elif isinstance(normalized.get("region"), dict) and normalized["region"].get("frame") is not None:
            normalized["frame"] = normalized["region"]["frame"]
        else:
            for key, value in normalized.items():
                if key.endswith("_region") and isinstance(value, dict) and value.get("frame") is not None:
                    normalized["frame"] = value["frame"]
                    break
    if "clip" in model_fields and normalized.get("clip") is None:
        if normalized.get("segment") is not None:
            normalized["clip"] = normalized["segment"]
        elif isinstance(normalized.get("frame"), dict) and normalized["frame"].get("clip") is not None:
            normalized["clip"] = normalized["frame"]["clip"]
        elif isinstance(normalized.get("image"), dict) and normalized["image"].get("clip") is not None:
            normalized["clip"] = normalized["image"]["clip"]
        elif isinstance(normalized.get("segments"), list) and normalized["segments"]:
            normalized["clip"] = normalized["segments"][0]
        elif isinstance(normalized.get("clips"), list) and normalized["clips"]:
            normalized["clip"] = normalized["clips"][0]
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
