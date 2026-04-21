from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .common import apply_env_map, read_json, write_json
from .schemas import MachineProfile, ModelsConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Expected a YAML object in %s" % path)
    return payload


def load_models_config(path: str) -> ModelsConfig:
    payload = _load_yaml(Path(path).expanduser().resolve())
    return ModelsConfig.parse_obj(payload)


def load_machine_profile(path: str, workspace_root: Optional[str] = None) -> MachineProfile:
    payload = _load_yaml(Path(path).expanduser().resolve())
    if workspace_root:
        payload["workspace_root"] = str(Path(workspace_root).expanduser().resolve())
    profile = MachineProfile.parse_obj(payload)
    apply_env_map(profile.env_overrides)
    if profile.hf_cache:
        os.environ.setdefault("HF_HOME", profile.hf_cache)
    return profile


def resolve_api_key(profile: MachineProfile, endpoint_name: str) -> Optional[str]:
    endpoint = profile.agent_endpoints.get(endpoint_name)
    if endpoint is None:
        return None
    if endpoint.api_key:
        return endpoint.api_key
    if endpoint.api_key_env:
        return os.environ.get(endpoint.api_key_env)
    return None


def save_runtime_snapshot(path: str, payload: Dict[str, Any]) -> None:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".json":
        write_json(path_obj, payload)
        return
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)
