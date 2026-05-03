from video_trace_pipeline.tool_wrappers import persistent_pool
from video_trace_pipeline.tools.registry import ToolRegistry


def test_generic_purpose_persistence_shares_qwen_runner_with_spatial_grounder(monkeypatch):
    created = []
    closed = []

    class FakeQwenRunner(object):
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            created.append(self.kwargs)

        def close(self):
            closed.append(self.kwargs["model_path"])

    monkeypatch.setattr(persistent_pool, "QwenStyleRunner", FakeQwenRunner)

    pool = persistent_pool.PersistentModelPool(["generic_purpose"])
    generic_runner = pool.acquire_qwen_style_runner(
        tool_name="generic_purpose",
        model_path="/models/qwen35",
        device_label="cuda:3",
        generate_do_sample=False,
        generate_temperature=None,
        attn_implementation="flash_attention_2",
    )
    spatial_runner = pool.acquire_qwen_style_runner(
        tool_name="spatial_grounder",
        model_path="/models/qwen35",
        device_label="cuda:3",
        generate_do_sample=False,
        generate_temperature=None,
        attn_implementation="flash_attention_2",
    )

    assert pool.should_persist("generic_purpose") is True
    assert pool.should_persist("spatial_grounder") is True
    assert generic_runner is spatial_runner
    assert len(created) == 1
    assert created[0]["attn_implementation"] == "flash_attention_2"

    pool.close()
    assert closed == ["/models/qwen35"]


def test_qwen_runner_device_map_is_part_of_persistence_key(monkeypatch):
    created = []

    class FakeQwenRunner(object):
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            created.append(self.kwargs)

        def close(self):
            pass

    monkeypatch.setattr(persistent_pool, "QwenStyleRunner", FakeQwenRunner)

    pool = persistent_pool.PersistentModelPool(["visual_temporal_grounder"])
    single_gpu = pool.acquire_qwen_style_runner(
        tool_name="visual_temporal_grounder",
        model_path="/models/timelens",
        device_label="cuda:0",
    )
    two_gpu = pool.acquire_qwen_style_runner(
        tool_name="visual_temporal_grounder",
        model_path="/models/timelens",
        device_label="cuda:0",
        device_map="balanced_cuda:0,1",
    )
    two_gpu_again = pool.acquire_qwen_style_runner(
        tool_name="visual_temporal_grounder",
        model_path="/models/timelens",
        device_label="cuda:0",
        device_map="balanced_cuda:0,1",
    )

    assert single_gpu is not two_gpu
    assert two_gpu is two_gpu_again
    assert len(created) == 2
    assert created[0]["device_map"] is None
    assert created[1]["device_map"] == "balanced_cuda:0,1"


def test_dense_captioner_persistence_reuses_timechat_runner(monkeypatch):
    created = []
    closed = []

    class FakeTimeChatRunner(object):
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            created.append(self.kwargs)

        def close(self):
            closed.append(self.kwargs["model_path"])

    monkeypatch.setattr(persistent_pool, "TimeChatCaptionerRunner", FakeTimeChatRunner)

    pool = persistent_pool.PersistentModelPool(["dense_captioner"])
    runner_a = pool.acquire_timechat_runner(
        tool_name="dense_captioner",
        model_path="/models/timechat",
        device_label="cuda:1",
        generate_do_sample=False,
        generate_temperature=None,
        use_audio_in_video=True,
        attn_implementation="eager",
    )
    runner_b = pool.acquire_timechat_runner(
        tool_name="dense_captioner",
        model_path="/models/timechat",
        device_label="cuda:1",
        generate_do_sample=False,
        generate_temperature=None,
        use_audio_in_video=True,
        attn_implementation="eager",
    )

    assert runner_a is runner_b
    assert len(created) == 1
    assert created[0]["attn_implementation"] == "eager"

    pool.close()
    assert closed == ["/models/timechat"]


def test_tool_registry_preload_dedupes_shared_qwen_runner():
    registry = ToolRegistry.__new__(ToolRegistry)
    registry.model_pool = persistent_pool.PersistentModelPool(["generic_purpose", "spatial_grounder", "dense_captioner"])
    registry.profile = object()

    class FakeAdapter(object):
        def __init__(self, spec):
            self.spec = dict(spec)

        def persistent_preload_spec(self, profile):
            assert profile is registry.profile
            return dict(self.spec)

    generic_key = registry.model_pool.qwen_style_key(
        tool_name="generic_purpose",
        model_path="/models/qwen35",
        device_label="cuda:3",
        generate_do_sample=False,
        generate_temperature=None,
        attn_implementation="flash_attention_2",
    )
    spatial_key = registry.model_pool.qwen_style_key(
        tool_name="spatial_grounder",
        model_path="/models/qwen35",
        device_label="cuda:3",
        generate_do_sample=False,
        generate_temperature=None,
        attn_implementation="flash_attention_2",
    )
    dense_key = registry.model_pool.timechat_key(
        tool_name="dense_captioner",
        model_path="/models/timechat",
        device_label="cuda:1",
        generate_do_sample=False,
        generate_temperature=None,
        use_audio_in_video=True,
        attn_implementation="eager",
    )
    assert generic_key == spatial_key

    registry.adapters = {
        "generic_purpose": FakeAdapter(
            {
                "tool_name": "generic_purpose",
                "runner_type": "qwen_style",
                "load_key": generic_key,
                "model_name": "Qwen/Qwen3.5-9B",
                "resolved_model_path": "/models/qwen35",
                "device_label": "cuda:3",
                "generate_do_sample": False,
                "generate_temperature": None,
                "attn_implementation": "flash_attention_2",
            }
        ),
        "spatial_grounder": FakeAdapter(
            {
                "tool_name": "spatial_grounder",
                "runner_type": "qwen_style",
                "load_key": spatial_key,
                "model_name": "Qwen/Qwen3.5-9B",
                "resolved_model_path": "/models/qwen35",
                "device_label": "cuda:3",
                "generate_do_sample": False,
                "generate_temperature": None,
                "attn_implementation": "flash_attention_2",
            }
        ),
        "dense_captioner": FakeAdapter(
            {
                "tool_name": "dense_captioner",
                "runner_type": "timechat",
                "load_key": dense_key,
                "model_name": "yaolily/TimeChat-Captioner-GRPO-7B",
                "resolved_model_path": "/models/timechat",
                "device_label": "cuda:1",
                "generate_do_sample": False,
                "generate_temperature": None,
                "use_audio_in_video": True,
                "attn_implementation": "eager",
            }
        ),
    }

    loaded = []

    def fake_load(spec):
        loaded.append(str(spec["tool_name"]))
        return {
            "tool_name": str(spec["tool_name"]),
            "runner_type": str(spec["runner_type"]),
            "model_name": str(spec["model_name"]),
            "resolved_model_path": str(spec["resolved_model_path"]),
            "device_label": str(spec["device_label"]),
        }

    registry._load_persistent_spec = fake_load

    payload = registry.preload_persistent_models()

    assert set(loaded) == {"generic_purpose", "dense_captioner"}
    assert payload["parallel_workers"] == 2
    assert payload["shared_tools"] == [{"tool_name": "spatial_grounder", "shared_with": "generic_purpose"}]
