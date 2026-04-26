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


def _validate_request_payload(payload: Dict[str, Any], request_model: Type[BaseModel]) -> Dict[str, Any]:
    normalized = dict(payload or {})
    model_fields = _request_model_field_names(request_model)
    unexpected_fields = sorted(
        field_name
        for field_name in normalized
        if field_name != "tool_name" and field_name not in model_fields
    )
    if unexpected_fields:
        raise ValueError(
            "Unexpected request field(s) for %s: %s. Use canonical schema fields only."
            % (
                getattr(request_model, "__name__", "request"),
                ", ".join(unexpected_fields),
            )
        )
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
        payload = _validate_request_payload(arguments, self.request_model)
        payload.setdefault("tool_name", self.name)
        return self.request_model.parse_obj(payload)

    def execute(self, request, context: ToolExecutionContext):
        raise NotImplementedError
