import sys

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
