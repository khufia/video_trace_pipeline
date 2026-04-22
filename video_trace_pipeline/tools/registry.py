from __future__ import annotations

from typing import Dict

from ..schemas import MachineProfile, ModelsConfig
from ..storage import WorkspaceManager
from .base import ToolAdapter
from .local_asr import LocalASRAdapter
from .process_adapters import (
    AudioTemporalGrounderProcessAdapter,
    DenseCaptionProcessAdapter,
    FrameRetrieverProcessAdapter,
    GenericPurposeProcessAdapter,
    OCRProcessAdapter,
    SpatialGrounderProcessAdapter,
    VisualTemporalGrounderProcessAdapter,
)
from .specs import tool_implementation


def _request_field_names(model_cls) -> list[str]:
    if hasattr(model_cls, "model_fields"):
        return list(getattr(model_cls, "model_fields").keys())
    if hasattr(model_cls, "__fields__"):
        return list(getattr(model_cls, "__fields__").keys())
    return []


class ToolRegistry(object):
    def __init__(self, workspace: WorkspaceManager, profile: MachineProfile, models_config: ModelsConfig, llm_client=None):
        self.workspace = workspace
        self.profile = profile
        self.models_config = models_config
        self.llm_client = llm_client
        self.adapters = self._build_adapters()

    def _build_adapters(self) -> Dict[str, ToolAdapter]:
        adapters = {}
        for tool_name, config in sorted(self.models_config.tools.items()):
            if not config.enabled:
                continue
            if tool_name == "dense_captioner":
                adapters[tool_name] = DenseCaptionProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                )
            elif tool_name == "visual_temporal_grounder":
                adapters[tool_name] = VisualTemporalGrounderProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                )
            elif tool_name == "frame_retriever":
                adapters[tool_name] = FrameRetrieverProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                )
            elif tool_name == "ocr":
                adapters[tool_name] = OCRProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                )
            elif tool_name == "spatial_grounder":
                adapters[tool_name] = SpatialGrounderProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                )
            elif tool_name == "asr":
                adapters[tool_name] = LocalASRAdapter(name=tool_name, extra=config.extra)
            elif tool_name == "audio_temporal_grounder":
                adapters[tool_name] = AudioTemporalGrounderProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                )
            elif tool_name == "generic_purpose":
                adapters[tool_name] = GenericPurposeProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                )
            else:
                raise ValueError("Unsupported tool %s" % tool_name)
        return adapters

    def get_adapter(self, tool_name: str) -> ToolAdapter:
        if tool_name not in self.adapters:
            raise KeyError("Unknown tool adapter: %s" % tool_name)
        return self.adapters[tool_name]

    def execute(self, tool_name: str, arguments: Dict[str, object], context):
        adapter = self.get_adapter(tool_name)
        request = adapter.parse_request(arguments)
        return adapter.execute(request, context)

    def build_dense_caption_cache(self, task, clip_duration_s: float, context):
        adapter = self.get_adapter("dense_captioner")
        if not hasattr(adapter, "build_segment_cache"):
            raise RuntimeError("dense_captioner adapter does not support preprocessing")
        return adapter.build_segment_cache(task=task, clip_duration_s=clip_duration_s, context=context)

    def implementation_name(self, tool_name: str) -> str:
        return tool_implementation(tool_name)

    def close(self) -> None:
        return None

    def tool_catalog(self) -> Dict[str, Dict[str, object]]:
        catalog = {}
        for tool_name, config in sorted(self.models_config.tools.items()):
            if not config.enabled:
                continue
            adapter = self.adapters.get(tool_name)
            request_fields = [
                field_name
                for field_name in _request_field_names(getattr(adapter, "request_model", None))
                if field_name != "tool_name"
            ]
            catalog[tool_name] = {
                "implementation": self.implementation_name(tool_name),
                "model": config.model,
                "description": config.description or "",
                "extra": dict(config.extra or {}),
                "request_fields": request_fields,
            }
        return catalog
