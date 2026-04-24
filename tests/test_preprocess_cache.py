from video_trace_pipeline.orchestration.preprocess import DenseCaptionPreprocessor
from video_trace_pipeline.schemas import AgentConfig, MachineProfile, ModelsConfig, TaskSpec, ToolConfig
from video_trace_pipeline.storage import WorkspaceManager
from video_trace_pipeline.tools.specs import tool_implementation
from video_trace_pipeline.common import hash_payload, write_json, write_text


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
                    "dense_caption": {
                        "overall_summary": "A short summary.",
                        "captions": [
                            {
                                "start": 0.0,
                                "end": float(clip_duration_s),
                                "visual": "A shopper pushes a cart past a low-price display.",
                                "audio": "",
                                "on_screen_text": "LOW PRICES",
                                "actions": ["pushes a cart"],
                                "objects": ["cart", "price display"],
                                "attributes": ["camera_state: static", "bright aisle"],
                            }
                        ],
                        "sampled_frames": [],
                    },
                }
            ],
            "summary": "",
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
                    "dense_caption": {
                        "overall_summary": "",
                        "captions": [],
                        "sampled_frames": [],
                    },
                }
            ],
            "summary": "",
        }


class FakeToolRegistry(object):
    def __init__(self, backend, transcript_segments=None):
        self.backend = backend
        self.llm_client = None
        self.transcript_segments = list(transcript_segments or [])
        self.asr_calls = 0

    def build_dense_caption_cache(self, task, clip_duration_s, context, preprocess_settings=None):
        del context
        return self.backend.build_dense_caption_cache(task, clip_duration_s)

    def build_asr_preprocess_transcript(self, task, context):
        del task, context
        self.asr_calls += 1
        return {
            "clip": {"video_id": "sample1", "start_s": 0.0, "end_s": 60.0},
            "text": "A narrator explains the store prices.",
            "segments": list(self.transcript_segments),
            "backend": "whisperx_local",
        }


def _models_config(preprocess=None):
    extra = {}
    if preprocess is not None:
        extra["preprocess"] = dict(preprocess)
    return ModelsConfig(
        agents={
            "planner": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_synthesizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_auditor": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "atomicizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
        },
        tools={
            "dense_captioner": ToolConfig(enabled=True, model="yaolily/TimeChat-Captioner-GRPO-7B", extra=extra),
            "asr": ToolConfig(enabled=True, extra={}),
        },
    )


def test_preprocess_cache_reuses_same_signature(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-bytes")
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    runtime = FakeDenseCaptionBackend()
    preprocessor = DenseCaptionPreprocessor(
        workspace,
        FakeToolRegistry(runtime, transcript_segments=[{"start_s": 2.0, "end_s": 5.0, "text": "The narrator mentions a price drop."}]),
        _models_config(),
    )
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
    assert "Visual: A shopper pushes a cart past a low-price display." in first["summary"]
    assert "Speech: The narrator mentions a price drop." in first["summary"]


def test_preprocess_cache_signature_changes_when_preprocess_settings_change(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-bytes")
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    runtime = FakeDenseCaptionBackend()
    transcript_segments = [{"start_s": 1.0, "end_s": 2.0, "text": "Milk is $2.18."}]
    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What happens?",
        options=[],
        video_path=str(video),
    )
    first_preprocessor = DenseCaptionPreprocessor(
        workspace,
        FakeToolRegistry(runtime, transcript_segments=transcript_segments),
        _models_config(preprocess={"clip_duration_s": 60, "sample_frames": 6}),
    )
    second_preprocessor = DenseCaptionPreprocessor(
        workspace,
        FakeToolRegistry(runtime, transcript_segments=transcript_segments),
        _models_config(preprocess={"clip_duration_s": 60, "sample_frames": 8}),
    )

    first = first_preprocessor.get_or_build(task, clip_duration_s=None)
    second = second_preprocessor.get_or_build(task, clip_duration_s=None)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is False
    assert first["manifest"]["preprocess_signature"] != second["manifest"]["preprocess_signature"]
    assert runtime.calls == 2


def test_preprocess_cache_merges_asr_once_and_records_dense_summary_metadata(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-bytes")
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    runtime = FakeDenseCaptionBackend()
    registry = FakeToolRegistry(
        runtime,
        transcript_segments=[{"start_s": 4.0, "end_s": 8.0, "text": "The narrator points out the low prices."}],
    )
    preprocessor = DenseCaptionPreprocessor(workspace, registry, _models_config())
    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What happens?",
        options=[],
        video_path=str(video),
    )

    result = preprocessor.get_or_build(task, clip_duration_s=60.0)

    assert registry.asr_calls == 1
    assert result["segments"][0]["transcript_segments"] == [
        {"start_s": 4.0, "end_s": 8.0, "text": "The narrator points out the low prices."}
    ]
    assert result["manifest"]["summary_format"] == "dense_interleaved"
    assert result["manifest"]["include_asr"] is True
    assert result["manifest"]["transcript_segment_count"] == 1
    assert "[00:04-00:08] Speech: The narrator points out the low prices." in result["summary"]


def test_preprocess_cache_rebuilds_blank_summary_bundle(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-bytes")
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    runtime = FakeDenseCaptionBackend()
    registry = FakeToolRegistry(runtime, transcript_segments=[{"start_s": 2.0, "end_s": 5.0, "text": "The narrator mentions a price drop."}])
    preprocessor = DenseCaptionPreprocessor(workspace, registry, _models_config())
    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What happens?",
        options=[],
        video_path=str(video),
    )
    model_name = "yaolily/TimeChat-Captioner-GRPO-7B"
    model_id = "%s__%s" % (tool_implementation("dense_captioner"), model_name)
    preprocess_settings = preprocessor.resolve_preprocess_settings(30.0)
    preprocess_signature = hash_payload(preprocess_settings, 12)
    cache_dir = workspace.preprocess_dir(
        video_fingerprint_value=workspace.video_fingerprint(task.video_path),
        model_id=model_id,
        clip_duration_s=preprocess_settings["clip_duration_s"],
        prompt_version="v1",
        settings_signature=preprocess_signature,
    )
    write_json(
        cache_dir / "manifest.json",
        {
            "video_fingerprint": workspace.video_fingerprint(task.video_path),
            "clip_duration_s": preprocess_settings["clip_duration_s"],
            "model_id": model_id,
            "prompt_version": "v1",
            "preprocess_settings": preprocess_settings,
            "preprocess_signature": preprocess_signature,
            "segment_count": 1,
        },
    )
    write_json(
        cache_dir / "segments.json",
        [
            {
                "start": 0.0,
                "end": 30.0,
                "dense_caption": {"overall_summary": "", "captions": [], "sampled_frames": []},
            }
        ],
    )
    write_text(cache_dir / "summary.txt", "")

    first = preprocessor.get_or_build(task, clip_duration_s=30.0)
    second = preprocessor.get_or_build(task, clip_duration_s=30.0)

    assert first["cache_hit"] is False
    assert "Visual: A shopper pushes a cart past a low-price display." in first["summary"]
    assert second["cache_hit"] is True
    assert "Speech: The narrator mentions a price drop." in second["summary"]
    assert second["manifest"]["preprocess_signature"] == preprocess_signature
    assert runtime.calls == 1
    assert registry.asr_calls == 1


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
