from video_trace_pipeline.orchestration.preprocess import DenseCaptionPreprocessor
from video_trace_pipeline.schemas import AgentConfig, MachineProfile, ModelsConfig, TaskSpec, ToolConfig
from video_trace_pipeline.storage import WorkspaceManager
from video_trace_pipeline.tools.specs import tool_implementation
from video_trace_pipeline.common import write_json, write_text


class FakeDenseCaptionBackend(object):
    def __init__(self):
        self.calls = 0

    def build_dense_caption_cache(self, task, clip_duration_s):
        self.calls += 1
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": float(clip_duration_s),
                    "caption_summary": "A short summary.",
                    "dense_caption": {"captions": []},
                }
            ],
            "summary": "Whole video summary.",
        }


class FakeLowSignalDenseCaptionBackend(object):
    def __init__(self):
        self.calls = 0

    def build_dense_caption_cache(self, task, clip_duration_s):
        self.calls += 1
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": float(clip_duration_s),
                    "caption_summary": "!!!!!!!!!!!!!!!!!!!!!!!!",
                    "dense_caption": {
                        "overall_summary": "!!!!!!!!!!!!!!!!!!!!!!!!",
                        "captions": [{"visual": "!!!!!!!!!!!!!!!!!!!!!!!!", "audio": "", "on_screen_text": ""}],
                    },
                }
            ],
            "summary": "!!!!!!!!!!!!!!!!!!!!!!!!",
        }


class FakeToolRegistry(object):
    def __init__(self, backend):
        self.backend = backend
        self.llm_client = None

    def build_dense_caption_cache(self, task, clip_duration_s, context):
        del context
        return self.backend.build_dense_caption_cache(task, clip_duration_s)


def _models_config():
    return ModelsConfig(
        agents={
            "planner": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_synthesizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_auditor": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "atomicizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
        },
        tools={"dense_captioner": ToolConfig(enabled=True, model="yaolily/TimeChat-Captioner-GRPO-7B")},
    )


def test_preprocess_cache_reuses_same_signature(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-bytes")
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    runtime = FakeDenseCaptionBackend()
    preprocessor = DenseCaptionPreprocessor(workspace, FakeToolRegistry(runtime), _models_config())
    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What happens?",
        options=[],
        video_path=str(video),
    )
    first = preprocessor.get_or_build(task, clip_duration_s=30.0)
    second = preprocessor.get_or_build(task, clip_duration_s=30.0)
    third = preprocessor.get_or_build(task, clip_duration_s=60.0)
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert third["cache_hit"] is False
    assert runtime.calls == 2


def test_preprocess_cache_rebuilds_blank_summary_bundle(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-bytes")
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    runtime = FakeDenseCaptionBackend()
    preprocessor = DenseCaptionPreprocessor(workspace, FakeToolRegistry(runtime), _models_config())
    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What happens?",
        options=[],
        video_path=str(video),
    )
    model_name = "yaolily/TimeChat-Captioner-GRPO-7B"
    model_id = "%s__%s" % (tool_implementation("dense_captioner"), model_name)
    cache_dir = workspace.preprocess_dir(
        video_fingerprint_value=workspace.video_fingerprint(task.video_path),
        model_id=model_id,
        clip_duration_s=30.0,
        prompt_version="v1",
    )
    write_json(
        cache_dir / "manifest.json",
        {
            "video_fingerprint": workspace.video_fingerprint(task.video_path),
            "clip_duration_s": 30.0,
            "model_id": model_id,
            "prompt_version": "v1",
            "segment_count": 1,
        },
    )
    write_json(
        cache_dir / "segments.json",
        [
            {
                "start": 0.0,
                "end": 30.0,
                "caption_summary": "",
                "dense_caption": {"overall_summary": "", "captions": [{"visual": "", "audio": "", "on_screen_text": ""}]},
            }
        ],
    )
    write_text(cache_dir / "summary.txt", "")

    first = preprocessor.get_or_build(task, clip_duration_s=30.0)
    second = preprocessor.get_or_build(task, clip_duration_s=30.0)

    assert first["cache_hit"] is False
    assert first["summary"] == "Whole video summary."
    assert second["cache_hit"] is True
    assert second["summary"] == "Whole video summary."
    assert runtime.calls == 1


def test_preprocess_cache_keeps_unavailable_low_signal_bundle_without_rebuild(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-bytes")
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    runtime = FakeLowSignalDenseCaptionBackend()
    preprocessor = DenseCaptionPreprocessor(workspace, FakeToolRegistry(runtime), _models_config())
    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What happens?",
        options=[],
        video_path=str(video),
    )

    first = preprocessor.get_or_build(task, clip_duration_s=30.0)
    second = preprocessor.get_or_build(task, clip_duration_s=30.0)

    assert first["cache_hit"] is False
    assert first["summary"] == ""
    assert first["manifest"]["summary_status"] == "unavailable_low_signal"
    assert second["cache_hit"] is True
    assert second["summary"] == ""
    assert second["manifest"]["summary_status"] == "unavailable_low_signal"
    assert runtime.calls == 1
