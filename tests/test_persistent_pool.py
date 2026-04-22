from video_trace_pipeline.tool_wrappers import persistent_pool


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
    )
    spatial_runner = pool.acquire_qwen_style_runner(
        tool_name="spatial_grounder",
        model_path="/models/qwen35",
        device_label="cuda:3",
        generate_do_sample=False,
        generate_temperature=None,
    )

    assert pool.should_persist("generic_purpose") is True
    assert pool.should_persist("spatial_grounder") is True
    assert generic_runner is spatial_runner
    assert len(created) == 1

    pool.close()
    assert closed == ["/models/qwen35"]


def test_dense_captioner_persistence_reuses_penguin_runner(monkeypatch):
    created = []
    closed = []

    class FakePenguinRunner(object):
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            created.append(self.kwargs)

        def close(self):
            closed.append(self.kwargs["model_path"])

    monkeypatch.setattr(persistent_pool, "PenguinRunner", FakePenguinRunner)

    pool = persistent_pool.PersistentModelPool(["dense_captioner"])
    runner_a = pool.acquire_penguin_runner(
        tool_name="dense_captioner",
        model_path="/models/penguin",
        device_label="cuda:1",
        generate_do_sample=False,
        generate_temperature=None,
    )
    runner_b = pool.acquire_penguin_runner(
        tool_name="dense_captioner",
        model_path="/models/penguin",
        device_label="cuda:1",
        generate_do_sample=False,
        generate_temperature=None,
    )

    assert runner_a is runner_b
    assert len(created) == 1

    pool.close()
    assert closed == ["/models/penguin"]
