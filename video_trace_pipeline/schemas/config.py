from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class DatasetConfig(BaseModel):
    root: str
    annotations: Optional[str] = None
    videos_subdir: str = "videos"


class ApiEndpointConfig(BaseModel):
    base_url: str
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None

    @validator("base_url")
    def _validate_base_url(cls, value):  # noqa: N805
        value = str(value or "").strip()
        if not value:
            raise ValueError("base_url must be non-empty")
        return value


class AgentConfig(BaseModel):
    backend: str
    model: str
    endpoint: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 4096
    prompt_version: str = "v1"


class ToolConfig(BaseModel):
    enabled: bool = True
    backend: str
    prompt_version: str = "v1"
    top_k: Optional[int] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    agents: Dict[str, AgentConfig]
    tools: Dict[str, ToolConfig]


class MachineProfile(BaseModel):
    workspace_root: str
    hf_cache: Optional[str] = None
    ffmpeg_bin: str = "ffmpeg"
    datasets: Dict[str, DatasetConfig] = Field(default_factory=dict)
    agent_endpoints: Dict[str, ApiEndpointConfig] = Field(default_factory=dict)
    gpu_assignments: Dict[str, str] = Field(default_factory=dict)
    env_overrides: Dict[str, str] = Field(default_factory=dict)

    def redacted_snapshot(self) -> Dict[str, Any]:
        return {
            "workspace_root": "<redacted>",
            "hf_cache": "<redacted>" if self.hf_cache else None,
            "ffmpeg_bin": self.ffmpeg_bin,
            "datasets": sorted(self.datasets.keys()),
            "agent_endpoints": {
                name: {
                    "base_url": endpoint.base_url,
                    "api_key_env": endpoint.api_key_env,
                    "api_key": "<redacted>" if endpoint.api_key else None,
                }
                for name, endpoint in sorted(self.agent_endpoints.items())
            },
            "gpu_assignments": dict(sorted(self.gpu_assignments.items())),
            "env_overrides": dict(sorted(self.env_overrides.items())),
        }


class RuntimeSnapshot(BaseModel):
    machine: Dict[str, Any]
    agent_models: Dict[str, Dict[str, Any]]
    tool_backends: Dict[str, Dict[str, Any]]
