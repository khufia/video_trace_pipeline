import video_trace_pipeline.tools.process_adapters as process_adapters

from video_trace_pipeline.schemas import AgentConfig, MachineProfile, ModelsConfig, TaskSpec
from video_trace_pipeline.storage import WorkspaceManager
from video_trace_pipeline.tools.process_adapters import DenseCaptionProcessAdapter


class _Run(object):
    def __init__(self, tools_dir):
        self.tools_dir = tools_dir


class _LLMClient(object):
    def __init__(self):
        self.calls = []

    def complete_text(self, **kwargs):
        self.calls.append(kwargs)
        return "Global summary."


class _Context(object):
    def __init__(self, workspace, task, tools_dir):
        self.workspace = workspace
        self.task = task
        self.run = _Run(tools_dir)
        self.llm_client = _LLMClient()
        self.evidence_lookup = None
        self.models_config = ModelsConfig(
            agents={"planner": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default")},
            tools={},
        )


def test_dense_caption_preprocess_reuses_single_runner_and_preserves_bundle_shape(tmp_path, monkeypatch):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What happens?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
        video_id="video1",
    )
    (tmp_path / "video.mp4").write_bytes(b"video")
    context = _Context(workspace, task, tmp_path / "run_tools")

    runner_events = {"init_count": 0, "close_count": 0, "runner_ids": []}

    class _FakeRunner(object):
        def __init__(self, **kwargs):
            del kwargs
            runner_events["init_count"] += 1

        def close(self):
            runner_events["close_count"] += 1

    def _fake_execute_payload(payload, *, runner_pool=None, runner=None):
        del runner_pool
        clip = dict(payload.get("request") or {}).get("clip") or {}
        start_s = float(clip.get("start_s") or 0.0)
        end_s = float(clip.get("end_s") or start_s)
        runner_events["runner_ids"].append(id(runner))
        return {
            "clip": clip,
            "captioned_range": {"start_s": start_s, "end_s": end_s},
            "captions": [
                {
                    "start": 0.0,
                    "end": max(0.0, end_s - start_s),
                    "visual": "segment %.0f-%.0f" % (start_s, end_s),
                    "audio": "",
                    "on_screen_text": "",
                    "actions": [],
                    "objects": [],
                    "attributes": [],
                }
            ],
            "overall_summary": "segment %.0f-%.0f" % (start_s, end_s),
            "sampled_frames": [],
            "backend": "fake-timechat",
        }

    monkeypatch.setattr(process_adapters, "get_video_duration", lambda _: 130.0)
    monkeypatch.setattr(process_adapters, "resolve_model_path", lambda *args, **kwargs: "/models/fake")
    monkeypatch.setattr(process_adapters, "TimeChatCaptionerRunner", _FakeRunner)
    monkeypatch.setattr(process_adapters, "execute_dense_caption_payload", _fake_execute_payload)

    adapter = DenseCaptionProcessAdapter(
        name="dense_captioner",
        model_name="fake-model",
        extra={"use_audio_in_video": True},
    )
    result = adapter.build_segment_cache(
        task=task,
        clip_duration_s=60.0,
        context=context,
        preprocess_settings={
            "clip_duration_s": 60.0,
            "sample_frames": 6,
            "fps": 1.0,
            "max_frames": 96,
            "use_audio_in_video": False,
            "collect_sampled_frames": False,
            "max_new_tokens": 700,
        },
    )

    assert len(result["segments"]) == 3
    assert runner_events["init_count"] == 1
    assert runner_events["close_count"] == 1
    assert len(set(runner_events["runner_ids"])) == 1
    first_dense_caption = result["segments"][0]["dense_caption"]
    assert first_dense_caption["clip"]["start_s"] == 0.0
    assert "captions" in first_dense_caption
    assert "overall_summary" in first_dense_caption
    assert "sampled_frames" in first_dense_caption
    assert result["summary"] == ""
    assert context.llm_client.calls == []


def test_dense_caption_preprocess_passes_visual_first_runtime_settings(tmp_path, monkeypatch):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What happens?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
        video_id="video1",
    )
    (tmp_path / "video.mp4").write_bytes(b"video")
    context = _Context(workspace, task, tmp_path / "run_tools")
    captured_payloads = []
    runner_kwargs = []

    class _FakeRunner(object):
        def __init__(self, **kwargs):
            runner_kwargs.append(dict(kwargs))

        def close(self):
            return None

    def _fake_execute_payload(payload, *, runner_pool=None, runner=None):
        del runner_pool, runner
        captured_payloads.append(dict(payload))
        clip = dict(payload.get("request") or {}).get("clip") or {}
        start_s = float(clip.get("start_s") or 0.0)
        end_s = float(clip.get("end_s") or start_s)
        return {
            "clip": clip,
            "captioned_range": {"start_s": start_s, "end_s": end_s},
            "captions": [],
            "overall_summary": "segment %.0f-%.0f" % (start_s, end_s),
            "sampled_frames": [],
            "backend": "fake-timechat",
        }

    monkeypatch.setattr(process_adapters, "get_video_duration", lambda _: 61.0)
    monkeypatch.setattr(process_adapters, "resolve_model_path", lambda *args, **kwargs: "/models/fake")
    monkeypatch.setattr(process_adapters, "TimeChatCaptionerRunner", _FakeRunner)
    monkeypatch.setattr(process_adapters, "execute_dense_caption_payload", _fake_execute_payload)

    adapter = DenseCaptionProcessAdapter(
        name="dense_captioner",
        model_name="fake-model",
        extra={"use_audio_in_video": True},
    )
    result = adapter.build_segment_cache(
        task=task,
        clip_duration_s=60.0,
        context=context,
        preprocess_settings={
            "clip_duration_s": 60.0,
            "sample_frames": 6,
            "fps": 1.0,
            "max_frames": 96,
            "use_audio_in_video": False,
            "collect_sampled_frames": False,
            "max_new_tokens": 700,
        },
    )

    assert result["summary"] == ""
    assert context.llm_client.calls == []
    assert len(captured_payloads) == 2
    assert runner_kwargs[0]["use_audio_in_video"] is False
    for payload in captured_payloads:
        extra = dict(dict(payload.get("runtime") or {}).get("extra") or {})
        assert extra["use_audio_in_video"] is False
        assert extra["collect_sampled_frames"] is False
        assert extra["max_new_tokens"] == 700
