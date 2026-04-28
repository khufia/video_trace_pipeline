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
                    "dense_caption": {
                        "clips": [{"video_id": task.video_id, "start_s": 0.0, "end_s": float(clip_duration_s)}],
                        "captioned_range": {"start_s": 0.0, "end_s": float(clip_duration_s)},
                        "overall_summary": "A shopper pushes a cart past a low-price display.",
                        "captions": [
                            {
                                "start": 0.0,
                                "end": float(clip_duration_s),
                                "visual": "A shopper pushes a cart past a low-price display.",
                                "audio": "A metallic bang echoes near the doorway.",
                                "on_screen_text": "LOW PRICES",
                                "actions": ["pushes a cart"],
                                "objects": ["cart", "price display"],
                                "attributes": [
                                    "camera_state: static",
                                    "video_background: bright aisle",
                                    "storyline: a shopper moves through the store",
                                    "shooting_style: handheld",
                                ],
                                "metadata": {},
                            }
                        ],
                        "sampled_frames": [],
                        "backend": "legacy_backend",
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
        del context, preprocess_settings
        return self.backend.build_dense_caption_cache(task, clip_duration_s)

    def build_asr_preprocess_transcript(self, task, context):
        del context
        self.asr_calls += 1
        return {
            "clip": {"video_id": task.video_id, "start_s": 0.0, "end_s": 60.0},
            "segments": list(self.transcript_segments),
            "transcripts": [
                {
                    "transcript_id": "tx_pre",
                    "clip": {"video_id": task.video_id, "start_s": 0.0, "end_s": 60.0},
                    "text": "The narrator mentions a price drop.",
                    "segments": list(self.transcript_segments),
                }
            ],
        }


def _models_config(preprocess=None):
    extra = {}
    if preprocess is not None:
        extra["preprocess"] = dict(preprocess)
    return ModelsConfig(
        agents={"planner": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default")},
        tools={
            "dense_captioner": ToolConfig(enabled=True, model="yaolily/TimeChat-Captioner-GRPO-7B", extra=extra),
            "asr": ToolConfig(enabled=True, extra={}),
        },
    )


def _task(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-bytes")
    return TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        video_id="video_1",
        question="What happens?",
        options=[],
        video_path=str(video),
    )


def test_preprocess_writes_video_id_layout_and_rich_planner_segments(tmp_path):
    workspace = WorkspaceManager(MachineProfile(workspace_root=str(tmp_path / "workspace")))
    backend = FakeDenseCaptionBackend()
    registry = FakeToolRegistry(
        backend,
        transcript_segments=[{"start_s": 2.0, "end_s": 5.0, "text": "The narrator mentions a price drop."}],
    )
    preprocessor = DenseCaptionPreprocessor(workspace, registry, _models_config())

    first = preprocessor.get_or_build(_task(tmp_path), clip_duration_s=30.0)
    second = preprocessor.get_or_build(_task(tmp_path), clip_duration_s=30.0)

    cache_dir = tmp_path / "workspace" / "preprocess" / "video_1"
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert backend.calls == 1
    assert registry.asr_calls == 1
    assert (cache_dir / "manifest.json").exists()
    assert (cache_dir / "raw_segments.json").exists()
    assert (cache_dir / "planner_segments.json").exists()
    assert (cache_dir / "dense_caption" / "segments.json").exists()
    assert (cache_dir / "asr" / "transcripts.json").exists()
    assert not (cache_dir / "planner_context.json").exists()

    segment = first["planner_segments"][0]
    dense = segment["dense_caption"]
    assert dense["overall_summary"] == "A shopper pushes a cart past a low-price display."
    assert dense["clips"][0]["artifact_id"].startswith("clip_")
    assert "artifacts/video_1/clips/" in dense["clips"][0]["relpath"]
    assert dense["captions"][0]["attributes"] == [
        "camera_state: static",
        "video_background: bright aisle",
        "storyline: a shopper moves through the store",
        "shooting_style: handheld",
    ]
    assert segment["asr"]["transcript_spans"][0]["text"] == "The narrator mentions a price drop."
    assert "backend" not in str(segment)
    assert "sampled_frames" not in str(segment)


def test_preprocess_manifest_records_counts_without_planner_context(tmp_path):
    workspace = WorkspaceManager(MachineProfile(workspace_root=str(tmp_path / "workspace")))
    registry = FakeToolRegistry(
        FakeDenseCaptionBackend(),
        transcript_segments=[{"start_s": 1.0, "end_s": 2.0, "text": "Milk is cheap."}],
    )
    preprocessor = DenseCaptionPreprocessor(workspace, registry, _models_config())

    result = preprocessor.get_or_build(_task(tmp_path), clip_duration_s=30.0)

    assert result["manifest"]["segment_count"] == 1
    assert result["manifest"]["planner_segment_count"] == 1
    assert result["manifest"]["dense_caption_span_count"] == 1
    assert result["manifest"]["transcript_segment_count"] == 1
    assert "identity_memory_count" not in result["manifest"]
    assert "audio_event_memory_count" not in result["manifest"]
