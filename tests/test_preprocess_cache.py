from video_trace_pipeline.orchestration.preprocess import DenseCaptionPreprocessor
from video_trace_pipeline.schemas import AgentConfig, MachineProfile, ModelsConfig, TaskSpec, ToolConfig
from video_trace_pipeline.storage import WorkspaceManager


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
        tools={"dense_captioner": ToolConfig(enabled=True, backend="internal_dense_captioner", model="gpt-5.4", endpoint="default")},
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
