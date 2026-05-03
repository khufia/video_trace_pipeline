from video_trace_pipeline.orchestration.preprocess import DenseCaptionPreprocessor
from video_trace_pipeline.schemas import MachineProfile, ModelsConfig, TaskSpec, ToolConfig
from video_trace_pipeline.storage import WorkspaceManager


class FailingToolRegistry(object):
    llm_client = None

    def build_dense_caption_cache(self, *args, **kwargs):
        raise AssertionError("legacy cache should be used without rebuilding")


def test_preprocessor_reads_legacy_preprocess_cache(tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    preprocess_cache_root = tmp_path / "preprocess_cache"
    cache_dir = preprocess_cache_root / "video_1"
    cache_dir.mkdir(parents=True)
    (cache_dir / "manifest.json").write_text(
        """{
  "schema_version": 6,
  "video_id": "video_1",
  "video_duration_s": 12.0
}""",
        encoding="utf-8",
    )
    (cache_dir / "preprocess.json").write_text(
        """{
  "ok": true,
  "segments": [
    {
      "id": "seg_001",
      "start_s": 0.0,
      "end_s": 12.0,
      "dense_caption_summary": "A red sign appears on screen.",
      "dense_captions": [
        {
          "start": 1.0,
          "end": 2.0,
          "visual": "A red sign is visible.",
          "audio": "No speech.",
          "on_screen_text": "SALE",
          "attributes": ["static camera"]
        }
      ],
      "transcript": [
        {
          "start_s": 3.0,
          "end_s": 4.0,
          "text": "hello there"
        }
      ]
    }
  ],
  "asr_transcripts": [
    {
      "clip": {"video_id": "video_1", "start_s": 0.0, "end_s": 12.0},
      "segments": [{"start_s": 3.0, "end_s": 4.0, "text": "hello there"}]
    }
  ]
}""",
        encoding="utf-8",
    )
    workspace = WorkspaceManager(
        MachineProfile(
            workspace_root=str(tmp_path / "workspace"),
            preprocess_cache_root=str(preprocess_cache_root),
        )
    )
    preprocessor = DenseCaptionPreprocessor(
        workspace,
        FailingToolRegistry(),
        ModelsConfig(
            agents={},
            tools={
                "dense_captioner": ToolConfig(
                    enabled=True,
                    model="mock_dense_captioner",
                    extra={"preprocess": {"enabled": True, "include_asr": True, "clip_duration_s": 30.0}},
                ),
                "asr": ToolConfig(enabled=True),
            },
        ),
    )

    bundle = preprocessor.get_or_build(
        TaskSpec(
            benchmark="omnivideobench",
            sample_key="sample1",
            video_id="video_1",
            question="What is shown?",
            options=["A", "B"],
            video_path=str(video_path),
        ),
        clip_duration_s=30.0,
    )

    assert bundle["cache_hit"] is True
    assert bundle["cache_dir"].endswith("preprocess_cache/video_1")
    assert bundle["planner_segments_source"] == "preprocess.json"
    assert bundle["planner_segments"][0]["dense_caption"]["captions"][0]["visual"] == "A red sign is visible."
    assert bundle["planner_segments"][0]["asr"]["transcript_spans"][0]["text"] == "hello there"
