import sys
import types

from video_trace_pipeline.tool_wrappers import local_multimodal


def test_qwen_style_runner_forwards_processor_use_fast(monkeypatch):
    calls = []

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls.append({"model_path": model_path, **kwargs})
            return object()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeAutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(local_multimodal, "_qwen_style_model", lambda *args, **kwargs: object())

    runner = local_multimodal.QwenStyleRunner(
        model_path="/tmp/demo-model",
        device_label="cpu",
        processor_use_fast=False,
    )

    assert calls
    assert calls[0]["model_path"] == "/tmp/demo-model"
    assert calls[0]["padding_side"] == "left"
    assert calls[0]["use_fast"] is False
    runner.model = None


def test_apply_generation_controls_disables_checkpoint_sampling_defaults():
    class _Config:
        do_sample = True
        temperature = 0.7
        top_k = 20
        top_p = 0.8
        min_p = None
        typical_p = None
        epsilon_cutoff = None
        eta_cutoff = None

    class _Model:
        generation_config = _Config()

    model = _Model()
    kwargs = local_multimodal._apply_generation_controls(model, do_sample=False, temperature=0.0)

    assert kwargs == {"do_sample": False}
    assert model.generation_config.do_sample is False
    assert model.generation_config.temperature is None
    assert model.generation_config.top_k is None
    assert model.generation_config.top_p is None
