import importlib
import sys


def test_tools_package_exports_tool_registry_lazily():
    for module_name in (
        "video_trace_pipeline.tools",
        "video_trace_pipeline.tools.registry",
    ):
        sys.modules.pop(module_name, None)

    tools = importlib.import_module("video_trace_pipeline.tools")

    assert "video_trace_pipeline.tools.registry" not in sys.modules

    registry_cls = tools.ToolRegistry

    assert registry_cls.__name__ == "ToolRegistry"
    assert "video_trace_pipeline.tools.registry" in sys.modules


def test_local_multimodal_import_does_not_trigger_tool_registry_cycle():
    for module_name in (
        "video_trace_pipeline.tool_wrappers.local_multimodal",
        "video_trace_pipeline.tool_wrappers.shared",
        "video_trace_pipeline.tools.media",
        "video_trace_pipeline.tools.registry",
        "video_trace_pipeline.tool_wrappers.persistent_pool",
        "video_trace_pipeline.tools",
    ):
        sys.modules.pop(module_name, None)

    module = importlib.import_module("video_trace_pipeline.tool_wrappers.local_multimodal")

    assert module.__name__ == "video_trace_pipeline.tool_wrappers.local_multimodal"
