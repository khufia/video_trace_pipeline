import sys

from video_trace_pipeline.schemas import MachineProfile, TaskSpec
from video_trace_pipeline.storage import WorkspaceManager
from video_trace_pipeline.tools.process_adapters import JsonProcessMixin


class _TestProcessMixin(JsonProcessMixin):
    name = "demo_tool"


def test_command_uses_current_interpreter_for_bare_python3():
    mixin = _TestProcessMixin("demo-model", extra={"command": ["python3", "-m", "demo.module"]})

    command = mixin._command()

    assert command == [sys.executable, "-m", "demo.module"]


def test_command_uses_current_interpreter_for_bare_python_string_command():
    mixin = _TestProcessMixin("demo-model", extra={"command": "python -m demo.module"})

    command = mixin._command()

    assert command == [sys.executable, "-m", "demo.module"]


def test_command_preserves_explicit_non_python_executable():
    mixin = _TestProcessMixin("demo-model", extra={"command": ["/usr/bin/env", "python3", "-m", "demo.module"]})

    command = mixin._command()

    assert command == ["/usr/bin/env", "python3", "-m", "demo.module"]


def test_runtime_payload_uses_run_dir_when_tools_dir_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "video_trace_pipeline.tools.process_adapters.describe_model_resolution",
        lambda model_name, hf_cache=None: {"resolved_path": "/models/%s" % model_name, "status": "resolved"},
    )
    monkeypatch.setattr(
        "video_trace_pipeline.tools.process_adapters.describe_device_mapping",
        lambda device: {"resolved_device": device},
    )
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    task = TaskSpec(
        benchmark="adhoc",
        sample_key="sample1",
        question="What happens?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
        video_id="video1",
    )
    (tmp_path / "video.mp4").write_bytes(b"video")
    run = workspace.create_run(task)
    context = type(
        "_Context",
        (),
        {
            "workspace": workspace,
            "run": run,
            "task": task,
        },
    )()

    mixin = _TestProcessMixin("demo-model", extra={})

    runtime = mixin._runtime_payload(context)

    assert runtime["scratch_dir"] == str((run.run_dir / "_scratch" / "demo_tool").resolve())
