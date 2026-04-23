from pathlib import Path

from video_trace_pipeline.diagnostics import dataset_report, model_report, package_report
from video_trace_pipeline.schemas import AgentConfig, ApiEndpointConfig, DatasetConfig, MachineProfile, ModelsConfig, ToolConfig


def _models_config():
    return ModelsConfig(
        agents={
            "planner": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_synthesizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_auditor": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "atomicizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
        },
        tools={
            "dense_captioner": ToolConfig(
                enabled=True,
                model="yaolily/TimeChat-Captioner-GRPO-7B",
                extra={"command": ["python3", "-m", "video_trace_pipeline.tool_wrappers.timechat_dense_caption_runner"]},
            ),
            "visual_temporal_grounder": ToolConfig(enabled=True, model="TencentARC/TimeLens-8B", extra={"command": ["python3", "-m", "video_trace_pipeline.tool_wrappers.timelens_runner"]}),
            "asr": ToolConfig(enabled=True, extra={"model_name": "large-v3"}),
        },
    )


def test_dataset_report_detects_existing_paths(tmp_path):
    root = tmp_path / "dataset"
    videos = root / "videos"
    videos.mkdir(parents=True)
    annotations = root / "annotations.json"
    annotations.write_text("[]", encoding="utf-8")
    profile = MachineProfile(
        workspace_root=str(tmp_path / "workspace"),
        datasets={"demo": DatasetConfig(root=str(root), annotations=str(annotations), videos_subdir="videos")},
    )
    report = dataset_report(profile, benchmark="demo")
    assert report[0]["status"] == "ok"


def test_package_report_flags_missing_package(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("this-package-should-not-exist==0.0.1\n", encoding="utf-8")
    report = package_report([requirements])
    assert report[0]["status"] == "missing"


def test_package_report_handles_bare_wheel_url(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "https://example.com/packages/flash_attn-2.8.3%2Bcu124torch2.6-cp310-cp310-linux_x86_64.whl\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("importlib.metadata.version", lambda _: "2.8.3+cu124torch2.6")
    report = package_report([requirements])
    assert report[0]["name"] == "flash-attn"
    assert report[0]["status"] == "ok"


def test_package_report_surfaces_invalid_requirements(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("not a valid requirement line ???\n", encoding="utf-8")
    report = package_report([requirements])
    assert report[0]["status"] == "invalid_requirement"


def test_model_report_tracks_endpoint_and_wrapper_status(tmp_path):
    profile = MachineProfile(
        workspace_root=str(tmp_path / "workspace"),
        hf_cache=str(tmp_path / "hf"),
        agent_endpoints={"default": ApiEndpointConfig(base_url="https://api.openai.com/v1", api_key="sk-test")},
    )
    report = model_report(profile, _models_config())
    by_name = {(item["kind"], item["name"]): item for item in report}
    assert by_name[("agent", "planner")]["status"] == "ok"
    assert by_name[("agent", "planner")]["plan_status"] == "planned"
    assert by_name[("tool", "dense_captioner")]["plan_status"] == "planned"
    assert by_name[("tool", "visual_temporal_grounder")]["module"] == "video_trace_pipeline.tool_wrappers.timelens_runner"
    assert by_name[("tool", "visual_temporal_grounder")]["wrapper_status"] == "configured"
    assert by_name[("tool", "visual_temporal_grounder")]["model_resolution_status"] == "missing"
    assert by_name[("tool", "visual_temporal_grounder")]["plan_status"] == "planned"
    assert by_name[("tool", "asr")]["status"] in {"ok", "missing"}
    assert by_name[("tool", "asr")].get("model_resolution_status") in {None, "ok", "missing"}
    assert by_name[("tool", "asr")]["plan_status"] in {"planned", "implementation_mismatch"}


def test_model_report_tracks_spotsound_wrapper_as_configured(tmp_path):
    profile = MachineProfile(
        workspace_root=str(tmp_path / "workspace"),
        hf_cache=str(tmp_path / "hf"),
        agent_endpoints={"default": ApiEndpointConfig(base_url="https://api.openai.com/v1", api_key="sk-test")},
    )
    models = ModelsConfig(
        agents={"planner": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default")},
        tools={
            "audio_temporal_grounder": ToolConfig(
                enabled=True,
                model="Loie/SpotSound",
                extra={"command": ["python3", "-m", "video_trace_pipeline.tool_wrappers.spotsound_runner"]},
            )
        },
    )

    report = model_report(profile, models)
    entry = report[1]
    assert entry["name"] == "audio_temporal_grounder"
    assert entry["module"] == "video_trace_pipeline.tool_wrappers.spotsound_runner"
    assert entry["wrapper_status"] == "configured"
    assert entry["status"] == "configured"
    assert entry["model_resolution_status"] == "missing"
    assert entry["expected_model"] == "Loie/SpotSound"
    assert entry["plan_status"] == "planned"


def test_model_report_tracks_auxiliary_models(tmp_path):
    primary_model = tmp_path / "primary_model"
    primary_model.mkdir()
    reranker_model = tmp_path / "reranker_model"
    reranker_model.mkdir()
    prefilter_model = tmp_path / "prefilter_model"
    prefilter_model.mkdir()
    profile = MachineProfile(
        workspace_root=str(tmp_path / "workspace"),
        hf_cache=str(tmp_path / "hf"),
        agent_endpoints={"default": ApiEndpointConfig(base_url="https://api.openai.com/v1", api_key="sk-test")},
    )
    models = ModelsConfig(
        agents={"planner": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default")},
        tools={
            "frame_retriever": ToolConfig(
                enabled=True,
                model=str(primary_model),
                extra={
                    "command": ["python3", "-m", "video_trace_pipeline.tool_wrappers.frame_retriever_runner"],
                    "reranker_model": str(reranker_model),
                    "prefilter_embedder_model": str(prefilter_model),
                },
            )
        },
    )

    report = model_report(profile, models)
    entry = report[1]
    assert entry["name"] == "frame_retriever"
    assert entry["model_resolution_status"] == "ok"
    assert entry["auxiliary_models"][0]["field"] == "reranker_model"
    assert entry["auxiliary_models"][0]["status"] == "ok"
    assert entry["auxiliary_models"][1]["field"] == "prefilter_embedder_model"
    assert entry["auxiliary_models"][1]["status"] == "ok"


def test_model_report_uses_current_planned_qwen_vl_model(tmp_path):
    vl_model = tmp_path / "vl_model"
    vl_model.mkdir()
    profile = MachineProfile(
        workspace_root=str(tmp_path / "workspace"),
        hf_cache=str(tmp_path / "hf"),
        agent_endpoints={"default": ApiEndpointConfig(base_url="https://api.openai.com/v1", api_key="sk-test")},
    )
    models = ModelsConfig(
        agents={"planner": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default")},
        tools={
            "generic_purpose": ToolConfig(
                enabled=True,
                model="Qwen/Qwen3.5-9B",
                extra={"command": ["python3", "-m", "video_trace_pipeline.tool_wrappers.qwen35vl_runner"]},
            ),
            "spatial_grounder": ToolConfig(
                enabled=True,
                model="Qwen/Qwen3.5-9B",
                extra={"command": ["python3", "-m", "video_trace_pipeline.tool_wrappers.spatial_grounder_runner"]},
            ),
        },
    )

    report = model_report(profile, models)
    by_name = {(item["kind"], item["name"]): item for item in report}
    assert by_name[("tool", "generic_purpose")]["expected_model"] == "Qwen/Qwen3.5-9B"
    assert by_name[("tool", "generic_purpose")]["plan_status"] == "planned"
    assert by_name[("tool", "spatial_grounder")]["expected_model"] == "Qwen/Qwen3.5-9B"
    assert by_name[("tool", "spatial_grounder")]["plan_status"] == "planned"
