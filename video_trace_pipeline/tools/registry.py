from __future__ import annotations

from typing import Dict

from ..agents import OpenAIChatClient
from ..schemas import MachineProfile, ModelsConfig
from ..storage import WorkspaceManager
from .base import ToolAdapter
from .internal_backends import (
    ASRToolAdapter,
    AudioTemporalGrounderToolAdapter,
    DenseCaptionToolAdapter,
    FrameRetrieverToolAdapter,
    OCRToolAdapter,
    OpenAIMultimodalToolAdapter,
    SpatialGrounderToolAdapter,
    TemporalGrounderToolAdapter,
)


class ToolRegistry(object):
    def __init__(self, workspace: WorkspaceManager, profile: MachineProfile, models_config: ModelsConfig, llm_client=None):
        self.workspace = workspace
        self.profile = profile
        self.models_config = models_config
        self.llm_client = llm_client
        self.adapters = self._build_adapters()

    def _build_adapters(self) -> Dict[str, ToolAdapter]:
        adapters = {}
        asr_adapter = None
        for tool_name, config in sorted(self.models_config.tools.items()):
            if not config.enabled:
                continue
            backend = str(config.backend or "").strip().lower()
            if backend == "internal_dense_captioner":
                adapters[tool_name] = DenseCaptionToolAdapter(
                    name=tool_name,
                    endpoint_name=config.endpoint or "default",
                    model_name=config.model or self.models_config.agents["planner"].model,
                    extra=config.extra,
                )
            elif backend == "internal_temporal_grounder":
                adapters[tool_name] = TemporalGrounderToolAdapter(
                    name=tool_name,
                    top_k=config.top_k or 5,
                )
            elif backend == "internal_frame_retriever":
                adapters[tool_name] = FrameRetrieverToolAdapter(
                    name=tool_name,
                    endpoint_name=config.endpoint or "default",
                    model_name=config.model or self.models_config.agents["planner"].model,
                    extra=config.extra,
                )
            elif backend == "internal_ocr":
                adapters[tool_name] = OCRToolAdapter(
                    name=tool_name,
                    endpoint_name=config.endpoint or "default",
                    model_name=config.model or self.models_config.agents["planner"].model,
                    extra=config.extra,
                )
            elif backend == "internal_spatial_grounder":
                adapters[tool_name] = SpatialGrounderToolAdapter(
                    name=tool_name,
                    endpoint_name=config.endpoint or "default",
                    model_name=config.model or self.models_config.agents["planner"].model,
                    extra=config.extra,
                )
            elif backend == "internal_asr":
                asr_adapter = ASRToolAdapter(name=tool_name, extra=config.extra)
                adapters[tool_name] = asr_adapter
            elif backend == "internal_audio_temporal_grounder":
                if asr_adapter is None:
                    asr_adapter = ASRToolAdapter(name="asr", extra=self.models_config.tools.get("asr").extra if self.models_config.tools.get("asr") else {})
                adapters[tool_name] = AudioTemporalGrounderToolAdapter(name=tool_name, asr_adapter=asr_adapter)
            elif backend == "openai_multimodal":
                adapters[tool_name] = OpenAIMultimodalToolAdapter(
                    name=tool_name,
                    endpoint_name=config.endpoint or "default",
                    model_name=config.model or self.models_config.agents["planner"].model,
                    extra=config.extra,
                )
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
        if not isinstance(adapter, DenseCaptionToolAdapter):
            raise RuntimeError("dense_captioner adapter does not support preprocessing")
        return adapter.build_segment_cache(task=task, clip_duration_s=clip_duration_s, context=context)

    def close(self) -> None:
        return None
