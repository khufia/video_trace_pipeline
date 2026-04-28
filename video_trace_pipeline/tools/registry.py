from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, get_args, get_origin

from pydantic import BaseModel

from ..schemas import MachineProfile, ModelsConfig
from ..storage import WorkspaceManager
from ..tool_wrappers.persistent_pool import PersistentModelPool
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


def _output_field_names(model_cls) -> list[str]:
    if hasattr(model_cls, "model_fields"):
        return list(getattr(model_cls, "model_fields").keys())
    if hasattr(model_cls, "__fields__"):
        return list(getattr(model_cls, "__fields__").keys())
    return []


def _model_fields(model_cls) -> Dict[str, object]:
    if hasattr(model_cls, "model_fields"):
        return dict(getattr(model_cls, "model_fields") or {})
    if hasattr(model_cls, "__fields__"):
        return dict(getattr(model_cls, "__fields__") or {})
    return {}


def _field_annotation(field_info) -> Any:
    if hasattr(field_info, "annotation"):
        return getattr(field_info, "annotation")
    if hasattr(field_info, "outer_type_"):
        return getattr(field_info, "outer_type_")
    if hasattr(field_info, "type_"):
        return getattr(field_info, "type_")
    return Any


def _is_basemodel_subclass(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)


def _format_annotation(annotation: Any) -> str:
    if annotation is None:
        return "Any"
    if annotation is Any:
        return "Any"
    origin = get_origin(annotation)
    if origin is None:
        if annotation is type(None):
            return "None"
        if _is_basemodel_subclass(annotation):
            return annotation.__name__
        if isinstance(annotation, type):
            return annotation.__name__
        rendered = str(annotation).replace("typing.", "")
        return rendered or "Any"
    args = list(get_args(annotation) or [])
    if origin is list:
        inner = _format_annotation(args[0]) if args else "Any"
        return "List[%s]" % inner
    if origin is dict:
        key_type = _format_annotation(args[0]) if len(args) > 0 else "Any"
        value_type = _format_annotation(args[1]) if len(args) > 1 else "Any"
        return "Dict[%s, %s]" % (key_type, value_type)
    if origin is tuple:
        inner = ", ".join(_format_annotation(arg) for arg in args) if args else "Any"
        return "Tuple[%s]" % inner
    if str(origin).endswith("Union"):
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) + 1 == len(args):
            if len(non_none) == 1:
                return "Optional[%s]" % _format_annotation(non_none[0])
            return "Optional[Union[%s]]" % ", ".join(_format_annotation(arg) for arg in non_none)
        return "Union[%s]" % ", ".join(_format_annotation(arg) for arg in args)
    origin_name = getattr(origin, "__name__", str(origin).replace("typing.", ""))
    if args:
        return "%s[%s]" % (origin_name, ", ".join(_format_annotation(arg) for arg in args))
    return str(origin_name or "Any")


def _nested_model_specs(annotation: Any) -> list[tuple[str, type[BaseModel]]]:
    origin = get_origin(annotation)
    if origin is None:
        if _is_basemodel_subclass(annotation):
            return [("", annotation)]
        return []
    nested = []
    for arg in list(get_args(annotation) or []):
        if arg is type(None):
            continue
        child_specs = _nested_model_specs(arg)
        if child_specs:
            if origin is list:
                return [("[]", model_cls) for _, model_cls in child_specs]
            return child_specs
    return nested


def _model_signature_lines(model_cls, *, exclude_fields: set[str] | None = None) -> list[str]:
    lines = []
    exclude_fields = set(exclude_fields or set())
    for field_name, field_info in _model_fields(model_cls).items():
        if field_name in exclude_fields:
            continue
        annotation = _field_annotation(field_info)
        lines.append("%s: %s" % (field_name, _format_annotation(annotation)))
    return lines


def _model_nested_lines(model_cls, *, exclude_fields: set[str] | None = None) -> list[str]:
    lines = []
    exclude_fields = set(exclude_fields or set())
    for field_name, field_info in _model_fields(model_cls).items():
        if field_name in exclude_fields:
            continue
        annotation = _field_annotation(field_info)
        for suffix, nested_model in _nested_model_specs(annotation):
            nested_fields = _model_signature_lines(nested_model)
            if not nested_fields:
                continue
            lines.append("%s%s -> %s" % (field_name, suffix, ", ".join(nested_fields)))
    return lines

_CANONICAL_OUTPUT_OVERRIDES = {
    "asr": {
        "output_fields": ["clips", "transcripts", "phrase_matches"],
        "output_schema": [
            "clips: List[ClipRef]",
            "transcripts: List[TranscriptRef]",
            "phrase_matches: List[Dict[str, Any]]",
        ],
        "output_nested": [
            "clips[] -> video_id: str, start_s: float, end_s: float, metadata: Dict[str, Any]",
            "transcripts[] -> transcript_id: str, clip: Optional[ClipRef], relpath: Optional[str], text: str, segments: List[TranscriptSegment], metadata: Dict[str, Any]",
        ],
    },
    "dense_captioner": {
        "output_fields": ["clips", "captions", "overall_summary", "captioned_range", "sampled_frames"],
        "output_schema": [
            "clips: List[ClipRef]",
            "captions: List[DenseCaptionSpan]",
            "overall_summary: str",
            "captioned_range: TimeRange",
            "sampled_frames: List[Dict[str, Any]]",
        ],
        "output_nested": [
            "clips[] -> video_id: str, start_s: float, end_s: float, metadata: Dict[str, Any]",
            "captions[] -> start: float, end: float, visual: str, audio: str, on_screen_text: str, actions: List[str], objects: List[str], attributes: List[str]",
        ],
    },
    "spatial_grounder": {
        "output_fields": ["frames", "detections", "regions", "spatial_description", "backend"],
        "output_schema": [
            "frames: List[FrameRef]",
            "detections: List[SpatialDetectionOutput]",
            "regions: List[RegionRef]",
            "spatial_description: str",
            "backend: Optional[str]",
        ],
        "output_nested": [
            "frames[] -> video_id: str, timestamp_s: float, artifact_id: Optional[str], relpath: Optional[str], clip: Optional[ClipRef], metadata: Dict[str, Any]",
            "detections[] -> label: str, bbox: Optional[List[float]], confidence: Optional[float], metadata: Dict[str, Any]",
            "regions[] -> frame: FrameRef, bbox: List[float], label: Optional[str], artifact_id: Optional[str], relpath: Optional[str], metadata: Dict[str, Any]",
        ],
    },
}


class ToolRegistry(object):
    def __init__(
        self,
        workspace: WorkspaceManager,
        profile: MachineProfile,
        models_config: ModelsConfig,
        llm_client=None,
        persist_tool_models=None,
    ):
        self.workspace = workspace
        self.profile = profile
        self.models_config = models_config
        self.llm_client = llm_client
        self.model_pool = PersistentModelPool(persist_tool_models)
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
                    model_pool=self.model_pool,
                )
            elif tool_name == "visual_temporal_grounder":
                adapters[tool_name] = VisualTemporalGrounderProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                    model_pool=self.model_pool,
                )
            elif tool_name == "frame_retriever":
                adapters[tool_name] = FrameRetrieverProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                    model_pool=self.model_pool,
                )
            elif tool_name == "ocr":
                adapters[tool_name] = OCRProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                    model_pool=self.model_pool,
                )
            elif tool_name == "spatial_grounder":
                adapters[tool_name] = SpatialGrounderProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                    model_pool=self.model_pool,
                )
            elif tool_name == "asr":
                adapters[tool_name] = LocalASRAdapter(name=tool_name, extra=config.extra)
            elif tool_name == "audio_temporal_grounder":
                adapters[tool_name] = AudioTemporalGrounderProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                    model_pool=self.model_pool,
                )
            elif tool_name == "generic_purpose":
                adapters[tool_name] = GenericPurposeProcessAdapter(
                    name=tool_name,
                    model_name=config.model or tool_name,
                    extra=config.extra,
                    model_pool=self.model_pool,
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

    def build_dense_caption_cache(self, task, clip_duration_s: float, context, preprocess_settings=None):
        adapter = self.get_adapter("dense_captioner")
        if not hasattr(adapter, "build_segment_cache"):
            raise RuntimeError("dense_captioner adapter does not support preprocessing")
        return adapter.build_segment_cache(
            task=task,
            clip_duration_s=clip_duration_s,
            context=context,
            preprocess_settings=preprocess_settings,
        )

    def build_asr_preprocess_transcript(self, task, context):
        adapter = self.get_adapter("asr")
        if not hasattr(adapter, "build_preprocess_transcript"):
            raise RuntimeError("asr adapter does not support preprocessing")
        return adapter.build_preprocess_transcript(task=task, context=context)

    def implementation_name(self, tool_name: str) -> str:
        return tool_implementation(tool_name)

    def persistent_tool_names(self) -> list[str]:
        names = []
        for tool_name in sorted(self.model_pool.enabled_tools):
            if tool_name in self.adapters:
                names.append(tool_name)
        return names

    def _load_persistent_spec(self, spec: Dict[str, object]) -> Dict[str, object]:
        runner_type = str(spec.get("runner_type") or "").strip()
        if runner_type == "qwen_style":
            self.model_pool.acquire_qwen_style_runner(
                tool_name=str(spec.get("tool_name") or ""),
                model_path=str(spec.get("resolved_model_path") or ""),
                device_label=str(spec.get("device_label") or ""),
                processor_use_fast=spec.get("processor_use_fast"),
                processor_model_path=str(spec.get("processor_model_path") or "") or None,
                generate_do_sample=bool(spec.get("generate_do_sample")),
                generate_temperature=spec.get("generate_temperature"),
                attn_implementation=str(spec.get("attn_implementation") or "") or None,
            )
        elif runner_type == "penguin":
            self.model_pool.acquire_penguin_runner(
                tool_name=str(spec.get("tool_name") or ""),
                model_path=str(spec.get("resolved_model_path") or ""),
                device_label=str(spec.get("device_label") or ""),
                generate_do_sample=bool(spec.get("generate_do_sample")),
                generate_temperature=spec.get("generate_temperature"),
            )
        elif runner_type == "timechat":
            self.model_pool.acquire_timechat_runner(
                tool_name=str(spec.get("tool_name") or ""),
                model_path=str(spec.get("resolved_model_path") or ""),
                device_label=str(spec.get("device_label") or ""),
                generate_do_sample=bool(spec.get("generate_do_sample")),
                generate_temperature=spec.get("generate_temperature"),
                use_audio_in_video=bool(spec.get("use_audio_in_video", True)),
                attn_implementation=str(spec.get("attn_implementation") or "") or None,
            )
        else:
            raise RuntimeError("Unsupported persistent preload runner type: %s" % runner_type)
        return {
            "tool_name": str(spec.get("tool_name") or ""),
            "runner_type": runner_type,
            "model_name": str(spec.get("model_name") or ""),
            "resolved_model_path": str(spec.get("resolved_model_path") or ""),
            "device_label": str(spec.get("device_label") or ""),
        }

    def preload_persistent_models(self) -> Dict[str, object]:
        requested_tools = self.persistent_tool_names()
        if not requested_tools:
            return {
                "enabled": False,
                "requested_tools": [],
                "loaded_models": [],
                "parallel_workers": 0,
                "shared_tools": [],
            }

        preload_specs = []
        seen_keys = {}
        shared_tools = []
        for tool_name in requested_tools:
            adapter = self.adapters.get(tool_name)
            describe_spec = getattr(adapter, "persistent_preload_spec", None)
            if describe_spec is None:
                continue
            spec = describe_spec(self.profile)
            if not spec:
                continue
            load_key = tuple(spec.get("load_key") or ())
            if load_key and load_key in seen_keys:
                shared_tools.append(
                    {
                        "tool_name": tool_name,
                        "shared_with": seen_keys[load_key],
                    }
                )
                continue
            if load_key:
                seen_keys[load_key] = tool_name
            preload_specs.append(spec)

        if not preload_specs:
            return {
                "enabled": True,
                "requested_tools": requested_tools,
                "loaded_models": [],
                "parallel_workers": 0,
                "shared_tools": shared_tools,
            }

        specs_by_device = {}
        for spec in preload_specs:
            specs_by_device.setdefault(str(spec.get("device_label") or "cpu"), []).append(spec)

        ordered_devices = sorted(specs_by_device)
        parallel_workers = max(1, len(ordered_devices))

        def _load_device_specs(device_label: str) -> list[Dict[str, object]]:
            return [self._load_persistent_spec(spec) for spec in specs_by_device.get(device_label, [])]

        loaded_models = []
        if parallel_workers == 1:
            loaded_models = _load_device_specs(ordered_devices[0])
        else:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                future_map = {
                    executor.submit(_load_device_specs, device_label): device_label
                    for device_label in ordered_devices
                }
                for future in as_completed(future_map):
                    loaded_models.extend(future.result())

        order_map = {name: index for index, name in enumerate(requested_tools)}
        loaded_models.sort(key=lambda item: order_map.get(str(item.get("tool_name") or ""), 10**6))
        return {
            "enabled": True,
            "requested_tools": requested_tools,
            "loaded_models": loaded_models,
            "parallel_workers": parallel_workers,
            "shared_tools": shared_tools,
        }

    def close(self) -> None:
        self.model_pool.close()

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
            output_fields = _output_field_names(getattr(adapter, "output_model", None))
            request_schema = _model_signature_lines(getattr(adapter, "request_model", None), exclude_fields={"tool_name"})
            output_schema = _model_signature_lines(getattr(adapter, "output_model", None))
            request_nested = _model_nested_lines(getattr(adapter, "request_model", None), exclude_fields={"tool_name"})
            output_nested = _model_nested_lines(getattr(adapter, "output_model", None))
            output_override = _CANONICAL_OUTPUT_OVERRIDES.get(tool_name) or {}
            if output_override.get("output_fields"):
                output_fields = list(output_override.get("output_fields") or [])
            if output_override.get("output_schema"):
                output_schema = list(output_override.get("output_schema") or [])
            if output_override.get("output_nested"):
                output_nested = list(output_override.get("output_nested") or [])
            catalog[tool_name] = {
                "implementation": self.implementation_name(tool_name),
                "model": config.model,
                "description": config.description or "",
                "extra": dict(config.extra or {}),
                "request_fields": request_fields,
                "output_fields": output_fields,
                "request_schema": request_schema,
                "output_schema": output_schema,
                "request_nested": request_nested,
                "output_nested": output_nested,
            }
        return catalog
