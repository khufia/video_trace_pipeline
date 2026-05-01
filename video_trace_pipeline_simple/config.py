from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .common import apply_env_map

CONTROL_TOOLS = frozenset({"planner", "synthesizer", "auditor"})
HIDDEN_PLAN_TOOLS = frozenset({"verifier", "generic_purpose"})


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path).expanduser().resolve()
    with path_obj.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Expected a YAML object in %s" % path_obj)
    return payload


def load_profile(path: str) -> dict[str, Any]:
    profile = _load_yaml(path)
    profile.setdefault("workspace_root", str((Path.cwd() / "workspace").resolve()))
    profile.setdefault("hf_cache", None)
    profile.setdefault("ffmpeg_bin", "ffmpeg")
    profile.setdefault("datasets", {})
    profile.setdefault("agent_endpoints", {})
    profile.setdefault("gpu_assignments", {})
    profile.setdefault("env_overrides", {})
    apply_env_map(profile.get("env_overrides") or {})
    if profile.get("hf_cache"):
        os.environ.setdefault("HF_HOME", str(profile["hf_cache"]))
    return profile


def load_models(path: str) -> dict[str, Any]:
    models = _load_yaml(path)
    tools = models.get("tools")
    if not isinstance(tools, dict):
        raise ValueError("Models config must contain a top-level `tools` mapping.")
    return {"tools": tools}


def _endpoint_runtime(profile: dict[str, Any], endpoint_name: str | None) -> dict[str, Any]:
    name = str(endpoint_name or "default")
    endpoint = dict((profile.get("agent_endpoints") or {}).get(name) or {})
    api_key = endpoint.get("api_key")
    api_key_env = endpoint.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.environ.get(str(api_key_env))
    return {
        "endpoint": name,
        "base_url": endpoint.get("base_url"),
        "api_key_env": api_key_env,
        "api_key": api_key,
    }


def tool_runtime(profile: dict[str, Any], models: dict[str, Any], tool_name: str, run_dir: str | Path) -> dict[str, Any]:
    tools = dict(models.get("tools") or {})
    if tool_name not in tools:
        raise KeyError("Unknown tool in models config: %s" % tool_name)
    config = dict(tools[tool_name] or {})
    endpoint_info = _endpoint_runtime(profile, config.get("endpoint"))
    extra = dict(config.get("extra") or {})
    gpu_assignments = dict(profile.get("gpu_assignments") or {})
    device_aliases = {"multimodal_reasoner": "generic_purpose"}
    device = extra.get("device") or gpu_assignments.get(tool_name) or gpu_assignments.get(device_aliases.get(tool_name, ""))
    runtime = {
        "tool": tool_name,
        "run_dir": str(Path(run_dir).expanduser().resolve()),
        "workspace_root": str(Path(str(profile.get("workspace_root") or ".")).expanduser().resolve()),
        "hf_cache": profile.get("hf_cache"),
        "ffmpeg_bin": profile.get("ffmpeg_bin") or "ffmpeg",
        "device": device,
        "backend": config.get("backend"),
        "model": config.get("model"),
        "model_name": config.get("model"),
        "temperature": config.get("temperature", 0.0),
        "max_tokens": config.get("max_tokens", 4096),
        "extra": extra,
    }
    runtime.update(endpoint_info)
    return runtime


def enabled_plan_tools(models: dict[str, Any]) -> dict[str, dict[str, Any]]:
    enabled: dict[str, dict[str, Any]] = {}
    for name, config in sorted((models.get("tools") or {}).items()):
        if name in CONTROL_TOOLS or name in HIDDEN_PLAN_TOOLS:
            continue
        if dict(config or {}).get("enabled", True):
            enabled[name] = dict(config or {})
    return enabled


def redacted_runtime(profile: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    endpoints = {}
    for name, endpoint in sorted((profile.get("agent_endpoints") or {}).items()):
        endpoint = dict(endpoint or {})
        endpoints[name] = {
            "base_url": endpoint.get("base_url"),
            "api_key_env": endpoint.get("api_key_env"),
            "api_key": "<redacted>" if endpoint.get("api_key") else None,
        }
    return {
        "machine": {
            "workspace_root": profile.get("workspace_root"),
            "hf_cache": "<redacted>" if profile.get("hf_cache") else None,
            "ffmpeg_bin": profile.get("ffmpeg_bin"),
            "datasets": sorted((profile.get("datasets") or {}).keys()),
            "agent_endpoints": endpoints,
            "gpu_assignments": dict(profile.get("gpu_assignments") or {}),
            "env_overrides": dict(profile.get("env_overrides") or {}),
        },
        "tools": {
            name: {
                key: value
                for key, value in dict(config or {}).items()
                if key not in {"api_key"}
            }
            for name, config in sorted((models.get("tools") or {}).items())
        },
    }
