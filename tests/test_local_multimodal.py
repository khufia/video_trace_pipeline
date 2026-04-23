import sys
import types
import json
import os

import pytest
import numpy as np
import torch

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


def test_qwen_style_runner_honors_requested_attn_implementation(monkeypatch):
    calls = []

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            del model_path, kwargs
            return object()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeAutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    def fake_qwen_style_model(model_path, device_label, *, attn_implementation=None):
        calls.append(
            {
                "model_path": model_path,
                "device_label": device_label,
                "attn_implementation": attn_implementation,
            }
        )
        return object()

    monkeypatch.setattr(local_multimodal, "_qwen_style_model", fake_qwen_style_model)

    runner = local_multimodal.QwenStyleRunner(
        model_path="/tmp/demo-model",
        device_label="cuda:0",
        attn_implementation="flash_attention_2",
    )

    assert calls == [
        {
            "model_path": "/tmp/demo-model",
            "device_label": "cuda:0",
            "attn_implementation": "flash_attention_2",
        }
    ]
    assert runner.loaded_attn_implementation == "flash_attention_2"
    runner.model = None


def test_qwen_style_runner_generate_applies_video_bounds_patch(monkeypatch):
    patched = {"called": False}

    def _mark_patched():
        patched["called"] = True

    fake_qwen = types.ModuleType("qwen_vl_utils")
    fake_qwen.process_vision_info = lambda messages, return_video_kwargs=True, return_video_metadata=True: ([], [], {})
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen)
    monkeypatch.setattr(local_multimodal, "_patch_qwen_vl_video_reader_bounds", _mark_patched)

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            del messages, tokenize, add_generation_prompt
            return "prompt"

        def __call__(self, **kwargs):
            del kwargs
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def batch_decode(self, tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            return ["Structured output"]

    class FakeModel:
        def generate(self, **kwargs):
            del kwargs
            return torch.tensor([[1, 2, 3, 4]])

    runner = local_multimodal.QwenStyleRunner.__new__(local_multimodal.QwenStyleRunner)
    runner.processor = FakeProcessor()
    runner.model = FakeModel()
    runner.device_label = "cpu"
    runner.generate_kwargs = {}

    text = runner.generate([{"role": "user", "content": []}], max_new_tokens=32)

    assert patched["called"] is True
    assert text == "Structured output"


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


def test_install_penguin_processor_kwargs_compat_adds_common_kwargs(monkeypatch):
    class FakeTextKwargs:
        __annotations__ = {"padding": object()}

    class FakeImagesKwargs:
        __annotations__ = {"merge_size": object()}

    class FakeVideosKwargs:
        __annotations__ = {}

    class FakeAudioKwargs:
        __annotations__ = {}

    class FakeChatTemplateKwargs:
        __annotations__ = {"chat_template": object()}

    class FakeKwargs:
        __annotations__ = {
            "text_kwargs": FakeTextKwargs,
            "images_kwargs": FakeImagesKwargs,
            "videos_kwargs": FakeVideosKwargs,
            "audio_kwargs": FakeAudioKwargs,
            "chat_template_kwargs": FakeChatTemplateKwargs,
        }
        _defaults = {
            "text_kwargs": {"padding": False},
            "image_kwargs": {"merge_size": None},
        }

    fake_module = types.ModuleType("demo.processing_penguinvl")
    fake_module.PenguinVLQwen3ProcessorKwargs = FakeKwargs
    fake_module.ChatTemplateKwargs = FakeChatTemplateKwargs

    class FakeProcessor:
        __module__ = "demo.processing_penguinvl"

    fake_dynamic = types.ModuleType("transformers.dynamic_module_utils")
    fake_dynamic.get_class_from_dynamic_module = lambda *args, **kwargs: FakeProcessor
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.dynamic_module_utils = fake_dynamic

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", fake_dynamic)
    monkeypatch.setitem(sys.modules, "demo.processing_penguinvl", fake_module)

    local_multimodal._install_penguin_processor_kwargs_compat("/tmp/penguin-model")

    assert "common_kwargs" in FakeKwargs.__annotations__
    assert FakeKwargs.__annotations__["common_kwargs"].__annotations__ == {}
    assert FakeKwargs._defaults["images_kwargs"] == {}
    assert FakeKwargs._defaults["common_kwargs"] == {}


def test_install_penguin_image_processor_compat_sets_default_merge_size(monkeypatch):
    class FakeImageProcessor:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    fake_dynamic = types.ModuleType("transformers.dynamic_module_utils")
    fake_dynamic.get_class_from_dynamic_module = lambda *args, **kwargs: FakeImageProcessor
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.dynamic_module_utils = fake_dynamic

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", fake_dynamic)

    local_multimodal._install_penguin_image_processor_compat("/tmp/penguin-model")

    instance = FakeImageProcessor(custom_value=7)
    assert instance.kwargs == {"custom_value": 7}
    assert instance.merge_size == 1


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
    monkeypatch.setattr(local_multimodal, "_install_penguin_processor_kwargs_compat", lambda model_path: None)
    monkeypatch.setattr(local_multimodal, "_install_penguin_image_processor_compat", lambda model_path: None)
    monkeypatch.setattr(local_multimodal, "torch_dtype_for_device", lambda device_label: "float16")

    runner = local_multimodal.PenguinRunner(
        model_path="/tmp/penguin-model",
        device_label="cpu",
    )

    assert model_calls
    assert model_calls[0]["low_cpu_mem_usage"] is False
    runner.close()


def test_penguin_runner_generate_falls_back_to_full_decode_when_trimmed_decode_is_empty():
    class FakeProcessor:
        def __call__(self, conversation=None, return_tensors=None):
            del conversation, return_tensors
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def batch_decode(self, tensor, skip_special_tokens=True):
            del skip_special_tokens
            if tensor.shape[1] == 0:
                return [""]
            return ['{"overall_summary":"Recovered output"}']

        def decode(self, tensor, skip_special_tokens=True):
            del tensor, skip_special_tokens
            return ""

    class FakeModel:
        def generate(self, **kwargs):
            del kwargs
            return torch.tensor([[1, 2, 3]])

    runner = local_multimodal.PenguinRunner.__new__(local_multimodal.PenguinRunner)
    runner.processor = FakeProcessor()
    runner.model = FakeModel()
    runner.device_label = "cpu"
    runner.dtype = torch.float32
    runner.generate_kwargs = {}

    text = runner.generate([{"role": "user", "content": []}], max_new_tokens=32)

    assert text == '{"overall_summary":"Recovered output"}'


def test_timechat_runner_generate_decodes_trimmed_tokens(monkeypatch):
    fake_utils = types.ModuleType("qwen_omni_utils")
    fake_utils.process_mm_info = lambda messages, use_audio_in_video=True: (["audio"], [], ["video"])
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", fake_utils)
    monkeypatch.setattr(local_multimodal, "_patch_qwen_omni_video_reader_bounds", lambda: None)

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            del messages, tokenize, add_generation_prompt
            return "prompt"

        def __call__(self, **kwargs):
            del kwargs
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def batch_decode(self, tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            assert tensor.shape[1] == 2
            return ["Structured output"]

        def decode(self, tensor, skip_special_tokens=True):
            del tensor, skip_special_tokens
            return ""

    class FakeModel:
        def generate(self, **kwargs):
            del kwargs
            return torch.tensor([[1, 2, 3, 4, 5]])

    runner = local_multimodal.TimeChatCaptionerRunner.__new__(local_multimodal.TimeChatCaptionerRunner)
    runner.processor = FakeProcessor()
    runner.model = FakeModel()
    runner.device_label = "cpu"
    runner.dtype = torch.float32
    runner.generate_kwargs = {}
    runner.use_audio_in_video = True
    runner._audio_in_video_runtime_disabled = False

    text = runner.generate([{"role": "user", "content": []}], max_new_tokens=32)

    assert text == "Structured output"


def test_timechat_runner_disables_audio_in_video_after_placeholder_mismatch(monkeypatch):
    fake_utils = types.ModuleType("qwen_omni_utils")
    fake_utils.process_mm_info = lambda messages, use_audio_in_video=True: (
        ["audio"] if use_audio_in_video else None,
        [],
        ["video"],
    )
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", fake_utils)
    monkeypatch.setattr(local_multimodal, "_patch_qwen_omni_video_reader_bounds", lambda: None)

    processor_calls = []
    generate_calls = []

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            del messages, tokenize, add_generation_prompt
            return "prompt"

        def __call__(self, **kwargs):
            processor_calls.append(bool(kwargs.get("use_audio_in_video")))
            if kwargs.get("use_audio_in_video"):
                return {
                    "input_ids": torch.tensor([[1, 99, 99, 99]]),
                    "feature_attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]]),
                }
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def batch_decode(self, tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            del tensor, skip_special_tokens, clean_up_tokenization_spaces
            return ["Recovered output"]

        def decode(self, tensor, skip_special_tokens=True):
            del tensor, skip_special_tokens
            return ""

    class FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace(audio_token_id=99)

        def generate(self, **kwargs):
            generate_calls.append(bool(kwargs.get("use_audio_in_video")))
            input_ids = kwargs["input_ids"]
            suffix = torch.tensor([[4, 5]], dtype=input_ids.dtype)
            return torch.cat([input_ids, suffix], dim=1)

    runner = local_multimodal.TimeChatCaptionerRunner.__new__(local_multimodal.TimeChatCaptionerRunner)
    runner.processor = FakeProcessor()
    runner.model = FakeModel()
    runner.model_path = "/tmp/timechat-model"
    runner.device_label = "cpu"
    runner.dtype = torch.float32
    runner.generate_kwargs = {}
    runner.use_audio_in_video = True
    runner._audio_in_video_runtime_disabled = False

    first = runner.generate([{"role": "user", "content": []}], max_new_tokens=32)
    second = runner.generate([{"role": "user", "content": []}], max_new_tokens=32)

    assert first == "Recovered output"
    assert second == "Recovered output"
    assert processor_calls == [True, False, False]
    assert generate_calls == [False, False]
    assert runner._audio_in_video_runtime_disabled is True


def test_timechat_runner_eager_fallback_keeps_requested_dtype(monkeypatch):
    calls = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            del model_path, kwargs
            return object()

    class FakeModel:
        def to(self, device_label):
            self.device_label = device_label
            return self

        def eval(self):
            return self

        def disable_talker(self):
            return None

    class FakeOmniModel:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls.append({"model_path": model_path, **kwargs})
            if kwargs.get("attn_implementation") != "eager":
                raise RuntimeError("backend unavailable")
            return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen2_5OmniProcessor = FakeProcessor
    fake_transformers.Qwen2_5OmniForConditionalGeneration = FakeOmniModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(local_multimodal, "torch_dtype_for_device", lambda device_label: "bf16")

    runner = local_multimodal.TimeChatCaptionerRunner(
        model_path="/tmp/timechat-model",
        device_label="cuda:0",
    )

    assert calls[-1]["attn_implementation"] == "eager"
    assert calls[-1]["torch_dtype"] == "bf16"
    assert runner.loaded_attn_implementation == "eager"
    runner.close()


def test_timechat_runner_honors_requested_attn_implementation(monkeypatch):
    calls = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            del model_path, kwargs
            return object()

    class FakeModel:
        def to(self, device_label):
            self.device_label = device_label
            return self

        def eval(self):
            return self

        def disable_talker(self):
            return None

    class FakeOmniModel:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls.append({"model_path": model_path, **kwargs})
            return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen2_5OmniProcessor = FakeProcessor
    fake_transformers.Qwen2_5OmniForConditionalGeneration = FakeOmniModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(local_multimodal, "torch_dtype_for_device", lambda device_label: "bf16")

    runner = local_multimodal.TimeChatCaptionerRunner(
        model_path="/tmp/timechat-model",
        device_label="cuda:0",
        attn_implementation="eager",
    )

    assert len(calls) == 1
    assert calls[0]["attn_implementation"] == "eager"
    assert calls[0]["torch_dtype"] == "bf16"
    assert runner.loaded_attn_implementation == "eager"
    runner.close()


def test_patch_qwen_omni_video_reader_bounds_clamps_decord_indices(monkeypatch):
    fake_qwen = types.ModuleType("qwen_omni_utils")
    fake_v25 = types.ModuleType("qwen_omni_utils.v2_5")
    fake_vision = types.ModuleType("qwen_omni_utils.v2_5.vision_process")
    fake_vision.calculate_video_frame_range = lambda ele, total_frames, video_fps: (0, total_frames, total_frames)
    fake_vision.smart_nframes = lambda ele, total_frames, video_fps: 8
    fake_vision.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    fake_vision.VIDEO_READER_BACKENDS = {"decord": object(), "torchvision": object()}
    fake_qwen.v2_5 = fake_v25
    fake_v25.vision_process = fake_vision
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", fake_qwen)
    monkeypatch.setitem(sys.modules, "qwen_omni_utils.v2_5", fake_v25)
    monkeypatch.setitem(sys.modules, "qwen_omni_utils.v2_5.vision_process", fake_vision)

    class FakeBatch:
        def __init__(self, indices):
            self.indices = list(indices)

        def asnumpy(self):
            return np.zeros((len(self.indices), 2, 2, 3), dtype=np.uint8)

    class FakeVideoReader:
        def __init__(self, video_path):
            self.video_path = video_path

        def __len__(self):
            return 1440

        def get_avg_fps(self):
            return 24.0

        def get_batch(self, indices):
            assert max(indices) == 1439
            return FakeBatch(indices)

    fake_decord = types.ModuleType("decord")
    fake_decord.VideoReader = FakeVideoReader
    monkeypatch.setitem(sys.modules, "decord", fake_decord)

    local_multimodal._patch_qwen_omni_video_reader_bounds()

    video, metadata, sample_fps = fake_vision.VIDEO_READER_BACKENDS["decord"]({"video": "/tmp/demo.mp4"})

    assert video.shape[0] == 8
    assert metadata["frames_indices"][-1] == 1439
    assert sample_fps == pytest.approx(24.0 / 180.0)


def test_patch_qwen_vl_video_reader_bounds_clamps_decord_indices(monkeypatch):
    fake_qwen = types.ModuleType("qwen_vl_utils")
    fake_vision = types.ModuleType("qwen_vl_utils.vision_process")
    fake_vision.calculate_video_frame_range = lambda ele, total_frames, video_fps: (0, total_frames, total_frames)
    fake_vision.smart_nframes = lambda ele, total_frames, video_fps: 8
    fake_vision.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    fake_vision.VIDEO_READER_BACKENDS = {"decord": object(), "torchvision": object()}
    fake_qwen.vision_process = fake_vision
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", fake_vision)

    class FakeBatch:
        def __init__(self, indices):
            self.indices = list(indices)

        def asnumpy(self):
            return np.zeros((len(self.indices), 2, 2, 3), dtype=np.uint8)

    class FakeVideoReader:
        def __init__(self, video_path):
            self.video_path = video_path

        def __len__(self):
            return 1440

        def get_avg_fps(self):
            return 24.0

        def get_batch(self, indices):
            assert max(indices) == 1439
            return FakeBatch(indices)

    fake_decord = types.ModuleType("decord")
    fake_decord.VideoReader = FakeVideoReader
    monkeypatch.setitem(sys.modules, "decord", fake_decord)

    local_multimodal._patch_qwen_vl_video_reader_bounds()

    video, metadata, sample_fps = fake_vision.VIDEO_READER_BACKENDS["decord"]({"video": "/tmp/demo.mp4"})

    assert video.shape[0] == 8
    assert metadata["frames_indices"][-1] == 1439
    assert sample_fps == pytest.approx(24.0 / 180.0)


def test_configure_audioread_ffmpeg_command_injects_imageio_binary(monkeypatch, tmp_path):
    ffmpeg_dir = tmp_path / "ffmpeg-bin"
    ffmpeg_dir.mkdir()
    ffmpeg_path = ffmpeg_dir / "ffmpeg"
    ffmpeg_path.write_text("", encoding="utf-8")

    fake_ffdec = types.ModuleType("audioread.ffdec")
    fake_ffdec.COMMANDS = ("ffmpeg", "avconv")
    fake_audioread = types.ModuleType("audioread")
    fake_audioread.ffdec = fake_ffdec
    fake_imageio_ffmpeg = types.ModuleType("imageio_ffmpeg")
    fake_imageio_ffmpeg.get_ffmpeg_exe = lambda: str(ffmpeg_path)

    monkeypatch.setitem(sys.modules, "audioread", fake_audioread)
    monkeypatch.setitem(sys.modules, "audioread.ffdec", fake_ffdec)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_imageio_ffmpeg)
    monkeypatch.setenv("PATH", "/usr/bin")

    local_multimodal._configure_audioread_ffmpeg_command()

    assert os.environ["PATH"].split(os.pathsep)[0] == str(ffmpeg_dir)
    assert fake_ffdec.COMMANDS[0] == str(ffmpeg_path.resolve())


def test_make_timechat_video_conversation_formats_video_payload():
    conversation = local_multimodal.make_timechat_video_conversation(
        "Describe the clip.",
        "/tmp/clip.mp4",
        fps=2.0,
        max_frames=160,
        max_pixels=297920,
        video_max_pixels=297920,
    )

    assert conversation == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the clip."},
                {
                    "type": "video",
                    "video": "/tmp/clip.mp4",
                    "max_pixels": 297920,
                    "max_frames": 160,
                    "fps": 2.0,
                    "video_max_pixels": 297920,
                },
            ],
        }
    ]


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
    monkeypatch.setattr(local_multimodal, "_install_penguin_processor_kwargs_compat", lambda model_path: None)
    monkeypatch.setattr(local_multimodal, "_install_penguin_image_processor_compat", lambda model_path: None)

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
    monkeypatch.setattr(local_multimodal, "_install_penguin_processor_kwargs_compat", lambda model_path: None)
    monkeypatch.setattr(local_multimodal, "_install_penguin_image_processor_compat", lambda model_path: None)

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
    monkeypatch.setattr(local_multimodal, "_install_penguin_processor_kwargs_compat", lambda model_path: None)
    monkeypatch.setattr(local_multimodal, "_install_penguin_image_processor_compat", lambda model_path: None)
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
