import sys
import types
import json

import pytest

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


def test_qwen35_checkpoints_require_newer_transformers(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"transformers>=5\.2\.0"):
        local_multimodal._require_supported_transformers_for_checkpoint(
            str(tmp_path),
            transformers_version="4.57.1",
        )


def test_qwen35_checkpoints_allow_supported_transformers(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")

    local_multimodal._require_supported_transformers_for_checkpoint(
        str(tmp_path),
        transformers_version="5.2.0",
    )


def test_penguin_processor_signature_compat_accepts_processor_dict(tmp_path, monkeypatch):
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"auto_map": {"AutoProcessor": "processing_penguinvl.PenguinVLQwen3Processor"}}),
        encoding="utf-8",
    )

    calls = []

    class FakeProcessor:
        @classmethod
        def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            calls.append((pretrained_model_name_or_path, dict(kwargs)))
            return ["ok"]

    fake_dynamic = types.ModuleType("transformers.dynamic_module_utils")
    fake_dynamic.get_class_from_dynamic_module = lambda *args, **kwargs: FakeProcessor
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.dynamic_module_utils = fake_dynamic
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", fake_dynamic)

    local_multimodal._install_penguin_processor_signature_compat(str(tmp_path))

    result = FakeProcessor._get_arguments_from_pretrained("demo-model", {"processor_class": "legacy"}, use_fast=False)

    assert result == ["ok"]
    assert calls == [("demo-model", {"use_fast": False})]


def test_penguin_runner_disables_low_cpu_mem_usage(monkeypatch):
    model_calls = []

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            return object()

    class FakeModel:
        def to(self, device_label):
            return self

        def eval(self):
            return self

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            model_calls.append({"model_path": model_path, **kwargs})
            return FakeModel()

    class FakePreTrainedModel:
        @staticmethod
        def _load_pretrained_model(*args, **kwargs):
            return None

    fake_import_utils = types.ModuleType("transformers.utils.import_utils")
    fake_import_utils.is_flash_attn_2_available = lambda: True
    fake_import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: True
    fake_utils = types.ModuleType("transformers.utils")
    fake_utils.import_utils = fake_import_utils
    fake_utils.is_flash_attn_2_available = lambda: True
    fake_utils.is_flash_attn_greater_or_equal_2_10 = lambda: True
    fake_modeling_utils = types.ModuleType("transformers.modeling_utils")
    fake_modeling_utils.check_and_set_device_map = lambda device_map: device_map
    fake_rope_utils = types.ModuleType("transformers.modeling_rope_utils")
    fake_rope_utils.ROPE_INIT_FUNCTIONS = {"default": lambda *args, **kwargs: ("inv_freq", 1.0)}
    fake_accelerate = types.ModuleType("transformers.integrations.accelerate")
    fake_accelerate.check_and_set_device_map = lambda device_map: device_map
    fake_integrations = types.ModuleType("transformers.integrations")
    fake_integrations.accelerate = fake_accelerate
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_transformers.PreTrainedModel = FakePreTrainedModel
    fake_transformers.utils = fake_utils
    fake_transformers.modeling_utils = fake_modeling_utils
    fake_transformers.modeling_rope_utils = fake_rope_utils
    fake_transformers.integrations = fake_integrations

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", fake_import_utils)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake_modeling_utils)
    monkeypatch.setitem(sys.modules, "transformers.modeling_rope_utils", fake_rope_utils)
    monkeypatch.setitem(sys.modules, "transformers.integrations", fake_integrations)
    monkeypatch.setitem(sys.modules, "transformers.integrations.accelerate", fake_accelerate)
    monkeypatch.setattr(local_multimodal, "_install_penguin_processor_signature_compat", lambda model_path: None)
    monkeypatch.setattr(local_multimodal, "torch_dtype_for_device", lambda device_label: "float16")

    runner = local_multimodal.PenguinRunner(
        model_path="/tmp/penguin-model",
        device_label="cpu",
    )

    assert model_calls
    assert model_calls[0]["low_cpu_mem_usage"] is False
    runner.close()


def test_configure_penguin_transformers_compat_maps_meta_nested_loads_to_cpu(monkeypatch):
    class FakePreTrainedModel:
        @staticmethod
        def _load_pretrained_model(*args, **kwargs):
            return None

    fake_import_utils = types.ModuleType("transformers.utils.import_utils")
    fake_import_utils.is_flash_attn_2_available = lambda: True
    fake_import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: True
    fake_utils = types.ModuleType("transformers.utils")
    fake_utils.import_utils = fake_import_utils
    fake_utils.is_flash_attn_2_available = lambda: True
    fake_utils.is_flash_attn_greater_or_equal_2_10 = lambda: True
    fake_modeling_utils = types.ModuleType("transformers.modeling_utils")
    fake_rope_utils = types.ModuleType("transformers.modeling_rope_utils")
    fake_rope_utils.ROPE_INIT_FUNCTIONS = {"default": lambda *args, **kwargs: ("inv_freq", 1.0)}

    def _raise_meta(device_map):
        raise RuntimeError("meta device context manager")

    fake_modeling_utils.check_and_set_device_map = _raise_meta
    fake_accelerate = types.ModuleType("transformers.integrations.accelerate")
    fake_accelerate.check_and_set_device_map = _raise_meta
    fake_integrations = types.ModuleType("transformers.integrations")
    fake_integrations.accelerate = fake_accelerate
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.PreTrainedModel = FakePreTrainedModel
    fake_transformers.utils = fake_utils
    fake_transformers.modeling_utils = fake_modeling_utils
    fake_transformers.modeling_rope_utils = fake_rope_utils
    fake_transformers.integrations = fake_integrations

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", fake_import_utils)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake_modeling_utils)
    monkeypatch.setitem(sys.modules, "transformers.modeling_rope_utils", fake_rope_utils)
    monkeypatch.setitem(sys.modules, "transformers.integrations", fake_integrations)
    monkeypatch.setitem(sys.modules, "transformers.integrations.accelerate", fake_accelerate)
    monkeypatch.setattr(local_multimodal, "_install_penguin_processor_signature_compat", lambda model_path: None)

    local_multimodal._configure_penguin_transformers_compat("/tmp/penguin-model")

    assert fake_modeling_utils.check_and_set_device_map(None) == {"": "cpu"}


def test_configure_penguin_transformers_compat_restores_default_rope(monkeypatch):
    class FakePreTrainedModel:
        @staticmethod
        def _load_pretrained_model(*args, **kwargs):
            return None

    fake_import_utils = types.ModuleType("transformers.utils.import_utils")
    fake_import_utils.is_flash_attn_2_available = lambda: True
    fake_import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: True
    fake_utils = types.ModuleType("transformers.utils")
    fake_utils.import_utils = fake_import_utils
    fake_utils.is_flash_attn_2_available = lambda: True
    fake_utils.is_flash_attn_greater_or_equal_2_10 = lambda: True
    fake_modeling_utils = types.ModuleType("transformers.modeling_utils")
    fake_modeling_utils.check_and_set_device_map = lambda device_map: device_map
    fake_rope_utils = types.ModuleType("transformers.modeling_rope_utils")
    fake_rope_utils.ROPE_INIT_FUNCTIONS = {}
    fake_accelerate = types.ModuleType("transformers.integrations.accelerate")
    fake_accelerate.check_and_set_device_map = lambda device_map: device_map
    fake_integrations = types.ModuleType("transformers.integrations")
    fake_integrations.accelerate = fake_accelerate
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.PreTrainedModel = FakePreTrainedModel
    fake_transformers.utils = fake_utils
    fake_transformers.modeling_utils = fake_modeling_utils
    fake_transformers.modeling_rope_utils = fake_rope_utils
    fake_transformers.integrations = fake_integrations

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", fake_import_utils)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake_modeling_utils)
    monkeypatch.setitem(sys.modules, "transformers.modeling_rope_utils", fake_rope_utils)
    monkeypatch.setitem(sys.modules, "transformers.integrations", fake_integrations)
    monkeypatch.setitem(sys.modules, "transformers.integrations.accelerate", fake_accelerate)
    monkeypatch.setattr(local_multimodal, "_install_penguin_processor_signature_compat", lambda model_path: None)

    local_multimodal._configure_penguin_transformers_compat("/tmp/penguin-model")

    assert "default" in fake_rope_utils.ROPE_INIT_FUNCTIONS
    class _Config:
        rope_parameters = {"rope_theta": 10000.0}
        hidden_size = 16
        num_attention_heads = 4

        @staticmethod
        def standardize_rope_params():
            return None

    inv_freq, scale = fake_rope_utils.ROPE_INIT_FUNCTIONS["default"](config=_Config(), device="meta")

    assert inv_freq.device.type == "cpu"
    assert scale == 1.0


def test_install_penguin_rotary_embedding_compat_adds_default_hook(monkeypatch):
    calls = []

    class VisualRotaryEmbedding:
        pass

    fake_encoder_module = types.ModuleType("demo.modeling_penguinvl_encoder")
    fake_encoder_module.VisualRotaryEmbedding = VisualRotaryEmbedding
    fake_dynamic = types.ModuleType("transformers.dynamic_module_utils")
    fake_dynamic.get_class_from_dynamic_module = lambda *args, **kwargs: object()
    fake_rope_utils = types.ModuleType("transformers.modeling_rope_utils")
    fake_rope_utils.ROPE_INIT_FUNCTIONS = {
        "default": lambda **kwargs: calls.append(dict(kwargs)) or ("inv_freq", 1.0)
    }
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.dynamic_module_utils = fake_dynamic
    fake_transformers.modeling_rope_utils = fake_rope_utils

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", fake_dynamic)
    monkeypatch.setitem(sys.modules, "transformers.modeling_rope_utils", fake_rope_utils)
    monkeypatch.setitem(sys.modules, "demo.modeling_penguinvl_encoder", fake_encoder_module)

    local_multimodal._install_penguin_rotary_embedding_compat("/tmp/penguin-model")

    result = VisualRotaryEmbedding.compute_default_rope_parameters(config="cfg", device="cpu", seq_len=32)

    assert result == ("inv_freq", 1.0)
    assert calls == [{"config": "cfg", "device": "cpu", "seq_len": 32, "layer_type": None}]


def test_configure_penguin_transformers_compat_adapts_legacy_load_pretrained_signature(monkeypatch):
    captured = {}

    class FakeLoadStateDictConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakePreTrainedModel:
        @staticmethod
        def _load_pretrained_model(model, state_dict, checkpoint_files, load_config):
            captured["model"] = model
            captured["state_dict"] = state_dict
            captured["checkpoint_files"] = checkpoint_files
            captured["load_config"] = load_config
            return "ok"

    fake_import_utils = types.ModuleType("transformers.utils.import_utils")
    fake_import_utils.is_flash_attn_2_available = lambda: True
    fake_import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: True
    fake_utils = types.ModuleType("transformers.utils")
    fake_utils.import_utils = fake_import_utils
    fake_utils.is_flash_attn_2_available = lambda: True
    fake_utils.is_flash_attn_greater_or_equal_2_10 = lambda: True
    fake_modeling_utils = types.ModuleType("transformers.modeling_utils")
    fake_modeling_utils.LoadStateDictConfig = FakeLoadStateDictConfig
    fake_modeling_utils.check_and_set_device_map = lambda device_map: device_map
    fake_modeling_utils.get_model_conversion_mapping = lambda model, key_mapping, hf_quantizer: [("mapped", key_mapping)]
    fake_rope_utils = types.ModuleType("transformers.modeling_rope_utils")
    fake_rope_utils.ROPE_INIT_FUNCTIONS = {"default": lambda *args, **kwargs: ("inv_freq", 1.0)}
    fake_accelerate = types.ModuleType("transformers.integrations.accelerate")
    fake_accelerate.check_and_set_device_map = lambda device_map: device_map
    fake_integrations = types.ModuleType("transformers.integrations")
    fake_integrations.accelerate = fake_accelerate
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.PreTrainedModel = FakePreTrainedModel
    fake_transformers.utils = fake_utils
    fake_transformers.modeling_utils = fake_modeling_utils
    fake_transformers.modeling_rope_utils = fake_rope_utils
    fake_transformers.integrations = fake_integrations

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", fake_import_utils)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake_modeling_utils)
    monkeypatch.setitem(sys.modules, "transformers.modeling_rope_utils", fake_rope_utils)
    monkeypatch.setitem(sys.modules, "transformers.integrations", fake_integrations)
    monkeypatch.setitem(sys.modules, "transformers.integrations.accelerate", fake_accelerate)
    monkeypatch.setattr(local_multimodal, "_install_penguin_processor_signature_compat", lambda model_path: None)
    monkeypatch.setattr(local_multimodal, "_install_penguin_rotary_embedding_compat", lambda model_path: None)

    local_multimodal._configure_penguin_transformers_compat("/tmp/penguin-model")

    result = FakePreTrainedModel._load_pretrained_model(
        model="demo-model",
        state_dict={"k": "v"},
        checkpoint_files=["part-1.safetensors"],
        pretrained_model_name_or_path="/tmp/penguin-model",
        ignore_mismatched_sizes=True,
        sharded_metadata={"x": 1},
        device_map={"": "cpu"},
        disk_offload_folder="/tmp/offload",
        offload_state_dict=True,
        dtype="float16",
        hf_quantizer="quant",
        keep_in_fp32_regex=None,
        device_mesh="mesh",
        key_mapping={"old": "new"},
        weights_only=False,
    )

    assert result == "ok"
    assert captured["model"] == "demo-model"
    assert captured["state_dict"] == {"k": "v"}
    assert captured["checkpoint_files"] == ["part-1.safetensors"]
    assert captured["load_config"].pretrained_model_name_or_path == "/tmp/penguin-model"
    assert captured["load_config"].ignore_mismatched_sizes is True
    assert captured["load_config"].weight_mapping == [("mapped", {"old": "new"})]
