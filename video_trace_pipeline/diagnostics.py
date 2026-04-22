from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from packaging.requirements import Requirement

from .config import resolve_api_key
from .model_cache import describe_model_resolution
from .tools.specs import tool_implementation, uses_process_wrapper


_PLANNED_AGENT_SPECS = {
    "atomicizer": {"backend": "openai", "model": "gpt-5.4"},
    "planner": {"backend": "openai", "model": "gpt-5.4"},
    "trace_auditor": {"backend": "openai", "model": "gpt-5.4"},
    "trace_synthesizer": {"backend": "openai", "model": "gpt-5.4"},
}

_PLANNED_TOOL_IMPLEMENTATIONS = {
    "asr": "local_whisperx",
    "audio_temporal_grounder": "local_process",
    "dense_captioner": "local_process",
    "frame_retriever": "local_process",
    "generic_purpose": "local_process",
    "ocr": "local_process",
    "spatial_grounder": "local_process",
    "visual_temporal_grounder": "local_process",
}

_PLANNED_TOOL_MODELS = {
    "asr": "Systran/faster-whisper-large-v3",
    "audio_temporal_grounder": "Loie/SpotSound",
    "dense_captioner": "tencent/Penguin-VL-8B",
    "frame_retriever": "Qwen/Qwen3-VL-Embedding-8B",
    "generic_purpose": "Qwen/Qwen3-VL-8B-Instruct",
    "ocr": "allenai/olmOCR-2-7B-1025-FP8",
    "spatial_grounder": "Qwen/Qwen3-VL-8B-Instruct",
    "visual_temporal_grounder": "TencentARC/TimeLens-8B",
}


def _requirement_lines(path: Path) -> Iterable[str]:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            yield line


def package_report(requirement_files: Iterable[Path], optional_packages: Optional[Set[str]] = None) -> List[Dict[str, object]]:
    report: List[Dict[str, object]] = []
    optional = set(optional_packages or set())
    for path in requirement_files:
        if not path.exists():
            report.append(
                {
                    "requirement_file": str(path),
                    "requirement": None,
                    "name": None,
                    "installed_version": None,
                    "status": "missing_requirement_file",
                }
            )
            continue
        for line in _requirement_lines(path):
            requirement = Requirement(line)
            is_optional = requirement.name in optional
            try:
                installed_version = importlib.metadata.version(requirement.name)
                status = "ok" if installed_version in requirement.specifier else (
                    "optional_version_mismatch" if is_optional else "version_mismatch"
                )
            except importlib.metadata.PackageNotFoundError:
                installed_version = None
                status = "optional_missing" if is_optional else "missing"
            report.append(
                {
                    "requirement_file": str(path),
                    "requirement": line,
                    "name": requirement.name,
                    "installed_version": installed_version,
                    "status": status,
                    "optional": is_optional,
                }
            )
    return report


def _wrapper_status(extra: Dict[str, object]) -> Dict[str, Optional[str]]:
    command = extra.get("command") or extra.get("cmd")
    if not command:
        return {"status": "no_command", "module": None}
    if isinstance(command, list):
        items = [str(item) for item in command]
    else:
        items = str(command).split()
    if "-m" in items:
        index = items.index("-m")
        if index + 1 < len(items):
            module_name = items[index + 1]
            module_path = None
            if module_name.startswith("video_trace_pipeline."):
                relative = module_name.split(".", 1)[1].replace(".", "/") + ".py"
                module_path = Path(__file__).resolve().parent / relative
            if module_path is not None and not module_path.exists():
                return {"status": "missing_wrapper_module", "module": module_name}
            if module_path is not None:
                text = module_path.read_text(encoding="utf-8")
                if "fail_stub(" in text or '"status": "not_implemented"' in text:
                    return {"status": "stub_wrapper", "module": module_name}
            return {"status": "configured", "module": module_name}
    return {"status": "configured", "module": None}


def _auxiliary_model_resolutions(extra: Dict[str, object], hf_cache: Optional[str]) -> List[Dict[str, Optional[str]]]:
    items: List[Dict[str, Optional[str]]] = []
    for field_name in ("reranker_model", "base_model", "pretrain_model"):
        value = extra.get(field_name)
        if not isinstance(value, str) or not value.strip():
            continue
        resolution = describe_model_resolution(value, hf_cache=hf_cache)
        items.append({"field": field_name, **resolution})
    return items


def _normalize_model_name(model_name: Optional[str]) -> str:
    return str(model_name or "").strip()


def _agent_plan_report(agent_name: str, backend: str, model_name: Optional[str]) -> Dict[str, object]:
    expected = _PLANNED_AGENT_SPECS.get(agent_name)
    if expected is None:
        return {"plan_status": "not_checked"}
    expected_backend = expected["backend"]
    expected_model = expected["model"]
    if str(backend or "").strip() != expected_backend:
        return {
            "plan_status": "backend_mismatch",
            "expected_backend": expected_backend,
            "expected_model": expected_model,
        }
    if _normalize_model_name(model_name) != expected_model:
        return {
            "plan_status": "model_mismatch",
            "expected_backend": expected_backend,
            "expected_model": expected_model,
        }
    return {
        "plan_status": "planned",
        "expected_backend": expected_backend,
        "expected_model": expected_model,
    }


def _tool_plan_report(tool_name: str, implementation: str, model_name: Optional[str]) -> Dict[str, object]:
    expected_implementation = _PLANNED_TOOL_IMPLEMENTATIONS.get(tool_name)
    if expected_implementation is not None:
        expected_model = _PLANNED_TOOL_MODELS.get(tool_name)
        if implementation == expected_implementation:
            status = "planned"
            if expected_model is not None and _normalize_model_name(model_name) and _normalize_model_name(model_name) != expected_model:
                status = "model_mismatch"
            return {
                "plan_status": status,
                "expected_implementation": expected_implementation,
                "expected_model": expected_model,
            }
        return {
            "plan_status": "implementation_mismatch",
            "expected_implementation": expected_implementation,
            "expected_model": expected_model,
        }
    return {"plan_status": "custom_extension"}


def dataset_report(profile, benchmark: Optional[str] = None) -> List[Dict[str, object]]:
    names = [benchmark] if benchmark else sorted(profile.datasets.keys())
    report = []
    for name in names:
        config = profile.datasets.get(name)
        if config is None:
            report.append({"benchmark": name, "status": "missing_from_profile"})
            continue
        root = Path(config.root).expanduser()
        annotations = Path(config.annotations).expanduser() if config.annotations else None
        videos_dir = root / config.videos_subdir
        status = "ok"
        if not root.exists():
            status = "missing_root"
        elif annotations is not None and not annotations.exists():
            status = "missing_annotations"
        elif not videos_dir.exists():
            status = "missing_videos_dir"
        report.append(
            {
                "benchmark": name,
                "root": str(root),
                "annotations": str(annotations) if annotations is not None else None,
                "videos_dir": str(videos_dir),
                "status": status,
            }
        )
    return report


def model_report(profile, models_config) -> List[Dict[str, object]]:
    report: List[Dict[str, object]] = []
    for agent_name, config in sorted(models_config.agents.items()):
        backend = str(config.backend or "").strip().lower()
        endpoint_name = config.endpoint or "default"
        if backend == "openai":
            api_key = resolve_api_key(profile, endpoint_name)
            entry = {
                "kind": "agent",
                "name": agent_name,
                "backend": config.backend,
                "model": config.model,
                "endpoint": endpoint_name,
                "status": "ok" if api_key else "missing_api_key",
            }
            entry.update(_agent_plan_report(agent_name, config.backend, config.model))
            report.append(entry)
        else:
            resolution = describe_model_resolution(config.model, hf_cache=profile.hf_cache)
            entry = {
                "kind": "agent",
                "name": agent_name,
                "backend": config.backend,
                "model": config.model,
                "endpoint": endpoint_name,
                **resolution,
            }
            entry.update(_agent_plan_report(agent_name, config.backend, config.model))
            report.append(entry)

    for tool_name, config in sorted(models_config.tools.items()):
        implementation = tool_implementation(tool_name)
        if not config.enabled:
            report.append(
                {
                    "kind": "tool",
                    "name": tool_name,
                    "implementation": implementation,
                    "model": config.model,
                    "status": "disabled",
                }
            )
            continue
        entry = {
            "kind": "tool",
            "name": tool_name,
            "implementation": implementation,
            "model": config.model,
        }
        wrapper_info = None
        if uses_process_wrapper(tool_name):
            wrapper_info = _wrapper_status(config.extra)
            entry.update(wrapper_info)
        resolution = None
        if implementation == "local_whisperx":
            resolution = describe_model_resolution(
                (config.extra or {}).get("model_name") or config.model or "large-v3",
                hf_cache=profile.hf_cache,
            )
        elif config.model:
            resolution = describe_model_resolution(config.model, hf_cache=profile.hf_cache)
        else:
            entry["status"] = entry.get("status") or "ok"

        if resolution is not None:
            model_status = resolution.get("status")
            entry.update({key: value for key, value in resolution.items() if key != "status"})
            entry["model_resolution_status"] = model_status
            if "status" not in entry:
                entry["status"] = model_status
        auxiliary_models = _auxiliary_model_resolutions(config.extra or {}, hf_cache=profile.hf_cache)
        if auxiliary_models:
            entry["auxiliary_models"] = auxiliary_models
            if entry.get("status") in {None, "ok", "configured"} and any(
                item.get("status") != "ok" for item in auxiliary_models
            ):
                entry["status"] = "missing_auxiliary_model"
        if wrapper_info is not None:
            wrapper_status = wrapper_info.get("status")
            entry["wrapper_status"] = wrapper_status
            if wrapper_status not in {"configured"}:
                entry["status"] = wrapper_status
        entry.update(_tool_plan_report(tool_name, implementation, config.model))
        report.append(entry)
    return report


def summarize_status(
    items: Iterable[Dict[str, object]],
    ok_statuses: Optional[Iterable[str]] = None,
    status_field: str = "status",
) -> str:
    allowed = set(ok_statuses or {"ok", "configured", "disabled"})
    return "ok" if all(str(item.get(status_field)) in allowed for item in items) else "needs_attention"
