from __future__ import annotations

import copy
import contextlib
import functools
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .shared import cleanup_torch, move_batch_to_device, torch_dtype_for_device


_QWEN35_MIN_TRANSFORMERS = "5.2.0"
logger = logging.getLogger(__name__)


def _checkpoint_model_type(model_path: str) -> str | None:
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.exists():
        return None
    with contextlib.suppress(Exception):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_type = str(config.get("model_type") or "").strip()
        return model_type or None
    return None


def _transformers_version_meets(minimum_version: str, current_version: str | None) -> bool:
    if not current_version:
        return False
    from packaging.version import InvalidVersion, Version

    try:
        return Version(str(current_version)) >= Version(minimum_version)
    except InvalidVersion:
        return False


def _require_supported_transformers_for_checkpoint(model_path: str, *, transformers_version: str | None) -> None:
    model_type = _checkpoint_model_type(model_path)
    if model_type != "qwen3_5":
        return
    if _transformers_version_meets(_QWEN35_MIN_TRANSFORMERS, transformers_version):
        return
    raise RuntimeError(
        "Checkpoint %r uses model_type=%r, which requires transformers>=%s. "
        "Found transformers %r. Upgrade the local multimodal environment before loading Qwen3.5 checkpoints."
        % (model_path, model_type, _QWEN35_MIN_TRANSFORMERS, transformers_version or "unknown")
    )


def _configure_audioread_ffmpeg_command() -> None:
    with contextlib.suppress(Exception):
        import audioread.ffdec

        ffmpeg_binary = None
        with contextlib.suppress(Exception):
            import imageio_ffmpeg

            ffmpeg_binary = imageio_ffmpeg.get_ffmpeg_exe()
        if not ffmpeg_binary:
            return
        ffmpeg_path = str(Path(str(ffmpeg_binary)).expanduser().resolve())
        ffmpeg_dir = str(Path(ffmpeg_path).parent)
        current_path = str(os.environ.get("PATH") or "")
        path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
        if ffmpeg_dir not in path_entries:
            os.environ["PATH"] = ffmpeg_dir if not current_path else ffmpeg_dir + os.pathsep + current_path
        commands = []
        for command in (ffmpeg_path, "ffmpeg", "avconv"):
            normalized = str(command or "").strip()
            if normalized and normalized not in commands:
                commands.append(normalized)
        if commands:
            audioread.ffdec.COMMANDS = tuple(commands)


def _patch_qwen_omni_video_reader_bounds() -> None:
    with contextlib.suppress(Exception):
        import torch
        from qwen_omni_utils.v2_5 import vision_process

        if getattr(vision_process, "_video_trace_pipeline_bounds_patch", False):
            return

        def _read_video_decord_clamped(ele: Dict[str, Any]):
            import decord
            import time

            video_path = ele["video"]
            st = time.time()
            vr = decord.VideoReader(video_path)
            source_total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            start_frame, end_frame, selected_total_frames = vision_process.calculate_video_frame_range(
                ele,
                source_total_frames,
                video_fps,
            )
            nframes = vision_process.smart_nframes(
                ele,
                total_frames=selected_total_frames,
                video_fps=video_fps,
            )
            idx = torch.linspace(start_frame, end_frame, nframes).round().long()
            if source_total_frames > 0:
                idx = idx.clamp(0, source_total_frames - 1)
            idx_list = idx.tolist()
            video = vr.get_batch(idx_list).asnumpy()
            video = torch.tensor(video).permute(0, 3, 1, 2)
            vision_process.logger.info(
                "decord(clamped): %s total_frames=%s video_fps=%s time=%.3fs",
                video_path,
                selected_total_frames,
                video_fps,
                time.time() - st,
            )
            sample_fps = nframes / max(selected_total_frames, 1e-6) * video_fps
            video_metadata = dict(
                fps=video_fps,
                frames_indices=idx_list,
                total_num_frames=selected_total_frames,
                video_backend="decord",
            )
            return video, video_metadata, sample_fps

        def _read_video_torchvision_clamped(ele: Dict[str, Any]):
            import time
            import torchvision

            video_path = ele["video"]
            if vision_process.version.parse(torchvision.__version__) < vision_process.version.parse("0.19.0"):
                if "http://" in video_path or "https://" in video_path:
                    vision_process.warnings.warn(
                        "torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0."
                    )
                if "file://" in video_path:
                    video_path = video_path[7:]
            st = time.time()
            video, audio, info = vision_process.io.read_video(
                video_path,
                start_pts=ele.get("video_start", 0.0),
                end_pts=ele.get("video_end", None),
                pts_unit="sec",
                output_format="TCHW",
            )
            del audio
            total_frames = video.size(0)
            video_fps = info["video_fps"]
            vision_process.logger.info(
                "torchvision(clamped): %s total_frames=%s video_fps=%s time=%.3fs",
                video_path,
                total_frames,
                video_fps,
                time.time() - st,
            )
            nframes = vision_process.smart_nframes(
                ele,
                total_frames=total_frames,
                video_fps=video_fps,
            )
            idx = torch.linspace(0, total_frames - 1, nframes).round().long()
            if total_frames > 0:
                idx = idx.clamp(0, total_frames - 1)
            sample_fps = nframes / max(total_frames, 1e-6) * video_fps
            video = video[idx]
            video_metadata = dict(
                fps=video_fps,
                frames_indices=idx,
                total_num_frames=total_frames,
                video_backend="torchvision",
            )
            return video, video_metadata, sample_fps

        vision_process._read_video_decord = _read_video_decord_clamped
        vision_process._read_video_torchvision = _read_video_torchvision_clamped
        if isinstance(getattr(vision_process, "VIDEO_READER_BACKENDS", None), dict):
            vision_process.VIDEO_READER_BACKENDS["decord"] = _read_video_decord_clamped
            vision_process.VIDEO_READER_BACKENDS["torchvision"] = _read_video_torchvision_clamped
        vision_process._video_trace_pipeline_bounds_patch = True


def _patch_qwen_vl_video_reader_bounds() -> None:
    with contextlib.suppress(Exception):
        import torch
        from qwen_vl_utils import vision_process

        if getattr(vision_process, "_video_trace_pipeline_bounds_patch", False):
            return

        def _read_video_decord_clamped(ele: Dict[str, Any]):
            import decord
            import time

            video_path = ele["video"]
            st = time.time()
            vr = decord.VideoReader(video_path)
            source_total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            start_frame, end_frame, selected_total_frames = vision_process.calculate_video_frame_range(
                ele,
                source_total_frames,
                video_fps,
            )
            nframes = vision_process.smart_nframes(
                ele,
                total_frames=selected_total_frames,
                video_fps=video_fps,
            )
            idx = torch.linspace(start_frame, end_frame, nframes).round().long()
            if source_total_frames > 0:
                idx = idx.clamp(0, source_total_frames - 1)
            idx_list = idx.tolist()
            video = vr.get_batch(idx_list).asnumpy()
            video = torch.tensor(video).permute(0, 3, 1, 2)
            vision_process.logger.info(
                "decord(clamped): %s total_frames=%s video_fps=%s time=%.3fs",
                video_path,
                selected_total_frames,
                video_fps,
                time.time() - st,
            )
            sample_fps = nframes / max(selected_total_frames, 1e-6) * video_fps
            video_metadata = dict(
                fps=video_fps,
                frames_indices=idx_list,
                total_num_frames=selected_total_frames,
                video_backend="decord",
            )
            return video, video_metadata, sample_fps

        def _read_video_torchvision_clamped(ele: Dict[str, Any]):
            import time
            import torchvision

            video_path = ele["video"]
            if vision_process.version.parse(torchvision.__version__) < vision_process.version.parse("0.19.0"):
                if "http://" in video_path or "https://" in video_path:
                    vision_process.warnings.warn(
                        "torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0."
                    )
                if "file://" in video_path:
                    video_path = video_path[7:]
            st = time.time()
            video, audio, info = vision_process.io.read_video(
                video_path,
                start_pts=ele.get("video_start", 0.0),
                end_pts=ele.get("video_end", None),
                pts_unit="sec",
                output_format="TCHW",
            )
            del audio
            total_frames = video.size(0)
            video_fps = info["video_fps"]
            vision_process.logger.info(
                "torchvision(clamped): %s total_frames=%s video_fps=%s time=%.3fs",
                video_path,
                total_frames,
                video_fps,
                time.time() - st,
            )
            nframes = vision_process.smart_nframes(
                ele,
                total_frames=total_frames,
                video_fps=video_fps,
            )
            idx = torch.linspace(0, total_frames - 1, nframes).round().long()
            if total_frames > 0:
                idx = idx.clamp(0, total_frames - 1)
            sample_fps = nframes / max(total_frames, 1e-6) * video_fps
            video = video[idx]
            video_metadata = dict(
                fps=video_fps,
                frames_indices=idx,
                total_num_frames=total_frames,
                video_backend="torchvision",
            )
            return video, video_metadata, sample_fps

        vision_process._read_video_decord = _read_video_decord_clamped
        vision_process._read_video_torchvision = _read_video_torchvision_clamped
        if isinstance(getattr(vision_process, "VIDEO_READER_BACKENDS", None), dict):
            vision_process.VIDEO_READER_BACKENDS["decord"] = _read_video_decord_clamped
            vision_process.VIDEO_READER_BACKENDS["torchvision"] = _read_video_torchvision_clamped
        vision_process._video_trace_pipeline_bounds_patch = True


def _patch_qwen35_flash_attention_position_ids() -> None:
    with contextlib.suppress(Exception):
        from transformers.models.qwen3_5 import modeling_qwen3_5

        attention_cls = getattr(modeling_qwen3_5, "Qwen3_5Attention", None)
        if attention_cls is None or getattr(attention_cls, "_video_trace_pipeline_position_ids_patch", False):
            return

        original_forward = attention_cls.forward

        @functools.wraps(original_forward)
        def _forward(self, *args, **kwargs):
            position_ids = kwargs.get("position_ids")
            if getattr(position_ids, "ndim", 0) == 3 and getattr(position_ids, "shape", (0,))[0] >= 1:
                # Qwen3.5 uses 3D multimodal RoPE position ids, but FlashAttention's
                # packed-sequence heuristic expects 2D text positions. Feeding the
                # 3D tensor into that path can misclassify ordinary single-sample
                # requests as packed sequences and crash inside flash-attn2.
                kwargs["position_ids"] = position_ids[0]
            return original_forward(self, *args, **kwargs)

        attention_cls.forward = _forward
        attention_cls._video_trace_pipeline_position_ids_patch = True


def _qwen_style_model(model_path: str, device_label: str, *, attn_implementation: str | None = None):
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText

    _require_supported_transformers_for_checkpoint(
        model_path,
        transformers_version=getattr(transformers, "__version__", None),
    )

    dtype = torch_dtype_for_device(device_label)
    common_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        common_kwargs["attn_implementation"] = str(attn_implementation)
    load_attempts = [
        {
            **common_kwargs,
            "dtype": dtype,
        },
        dict(common_kwargs),
    ]
    last_error = None
    for kwargs in load_attempts:
        try:
            return AutoModelForImageTextToText.from_pretrained(
                model_path,
                **kwargs,
            ).to(device_label).eval()
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not load Qwen-style model.")


def _normalize_generation_controls(*, do_sample: bool, temperature: float | None) -> tuple[bool, float | None]:
    normalized_do_sample = bool(do_sample)
    normalized_temperature = None if temperature is None else float(temperature)
    if normalized_temperature is not None and normalized_temperature <= 0.0:
        normalized_do_sample = False
        normalized_temperature = None
    if not normalized_do_sample:
        normalized_temperature = None
    return normalized_do_sample, normalized_temperature


def _apply_generation_controls(model: Any, *, do_sample: bool, temperature: float | None) -> Dict[str, Any]:
    normalized_do_sample, normalized_temperature = _normalize_generation_controls(
        do_sample=do_sample,
        temperature=temperature,
    )
    generation_kwargs: Dict[str, Any] = {"do_sample": normalized_do_sample}
    if normalized_do_sample and normalized_temperature is not None:
        generation_kwargs["temperature"] = normalized_temperature

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        configured = copy.deepcopy(generation_config)
        configured.do_sample = normalized_do_sample
        if normalized_do_sample:
            if normalized_temperature is not None:
                configured.temperature = normalized_temperature
        else:
            # Clear sampling-only fields so Transformers does not warn about
            # checkpoint defaults while we are forcing deterministic decoding.
            for field_name in (
                "temperature",
                "top_k",
                "top_p",
                "min_p",
                "typical_p",
                "epsilon_cutoff",
                "eta_cutoff",
            ):
                if hasattr(configured, field_name):
                    setattr(configured, field_name, None)
        model.generation_config = configured
    return generation_kwargs


class QwenStyleRunner:
    def __init__(
        self,
        *,
        model_path: str,
        device_label: str,
        processor_use_fast: bool | None = None,
        processor_model_path: str | None = None,
        generate_do_sample: bool = False,
        generate_temperature: float | None = None,
        attn_implementation: str | None = None,
    ):
        from transformers import AutoProcessor

        self.model_path = model_path
        self.device_label = device_label
        self.requested_attn_implementation = str(attn_implementation or "").strip() or None
        self.loaded_attn_implementation = None
        processor_source = str(processor_model_path or model_path)
        processor_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if processor_use_fast is not None:
            processor_kwargs["use_fast"] = bool(processor_use_fast)
        try:
            self.processor = AutoProcessor.from_pretrained(
                processor_source,
                padding_side="left",
                **processor_kwargs,
            )
        except Exception:
            self.processor = AutoProcessor.from_pretrained(
                processor_source,
                **processor_kwargs,
            )
        _patch_qwen35_flash_attention_position_ids()
        self.model = _qwen_style_model(
            model_path,
            device_label,
            attn_implementation=self.requested_attn_implementation,
        )
        self.loaded_attn_implementation = self.requested_attn_implementation or "default"
        self.generate_kwargs = _apply_generation_controls(
            self.model,
            do_sample=generate_do_sample,
            temperature=generate_temperature,
        )

    def generate(self, messages: List[Dict[str, Any]], *, max_new_tokens: int) -> str:
        _patch_qwen_vl_video_reader_bounds()
        from qwen_vl_utils import process_vision_info

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        vision_payload = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if len(vision_payload) == 3:
            images, videos, video_kwargs = vision_payload
        else:  # pragma: no cover - defensive for future API shape changes
            images, videos = vision_payload
            video_kwargs = {}

        video_items = []
        video_metadata = None
        if videos:
            unpacked = list(videos)
            if unpacked and isinstance(unpacked[0], tuple) and len(unpacked[0]) == 2:
                video_items = [item[0] for item in unpacked]
                video_metadata = [item[1] for item in unpacked]
            else:
                video_items = unpacked

        processor_kwargs: Dict[str, Any] = {
            "text": [prompt_text],
            "padding": True,
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images
        if video_items:
            processor_kwargs["videos"] = video_items
            # TimeLens / Qwen3-VL video inputs need an explicit resize pass at
            # call time; otherwise the fetched clip tensor can stay aligned to
            # the qwen-vl-utils sampling grid instead of the model's patch grid.
            processor_kwargs["do_resize"] = True
        if video_metadata:
            processor_kwargs["video_metadata"] = video_metadata
        processor_kwargs.update(video_kwargs or {})
        batch = self.processor(**processor_kwargs)
        batch = move_batch_to_device(batch, self.device_label)
        outputs = self.model.generate(
            **batch,
            max_new_tokens=max(32, int(max_new_tokens)),
            **self.generate_kwargs,
        )
        input_ids = getattr(batch, "input_ids", None)
        if input_ids is None and isinstance(batch, dict):
            input_ids = batch.get("input_ids")
        if input_ids is not None:
            trimmed = outputs[:, input_ids.shape[1] :]
        else:
            trimmed = outputs
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return str(decoded[0] if decoded else "").strip()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            del self.model
        cleanup_torch()


def run_qwen_style_messages(
    *,
    model_path: str,
    messages: List[Dict[str, Any]],
    device_label: str,
    max_new_tokens: int,
    processor_use_fast: bool | None = None,
    processor_model_path: str | None = None,
    generate_do_sample: bool = False,
    generate_temperature: float | None = None,
    attn_implementation: str | None = None,
) -> str:
    runner = QwenStyleRunner(
        model_path=model_path,
        device_label=device_label,
        processor_use_fast=processor_use_fast,
        processor_model_path=processor_model_path,
        generate_do_sample=generate_do_sample,
        generate_temperature=generate_temperature,
        attn_implementation=attn_implementation,
    )
    try:
        return runner.generate(messages, max_new_tokens=max_new_tokens)
    finally:
        runner.close()


class TimeChatCaptionerRunner:
    def __init__(
        self,
        *,
        model_path: str,
        device_label: str,
        generate_do_sample: bool = False,
        generate_temperature: float | None = None,
        use_audio_in_video: bool = True,
        attn_implementation: str | None = None,
    ):
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.model_path = model_path
        self.device_label = device_label
        self.dtype = torch_dtype_for_device(device_label)
        self.use_audio_in_video = bool(use_audio_in_video)
        self._audio_in_video_runtime_disabled = False
        self.requested_attn_implementation = str(attn_implementation or "").strip() or None
        self.loaded_attn_implementation = None
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model = None
        if self.requested_attn_implementation:
            load_attempts = [
                {
                    "torch_dtype": self.dtype,
                    "attn_implementation": self.requested_attn_implementation,
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                }
            ]
        else:
            load_attempts = [
                {
                    "torch_dtype": self.dtype,
                    "attn_implementation": "flash_attention_2",
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                },
                {
                    "torch_dtype": self.dtype,
                    "attn_implementation": "sdpa",
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                },
                {
                    "torch_dtype": self.dtype,
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                },
                {
                    "torch_dtype": self.dtype,
                    "attn_implementation": "eager",
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                },
            ]
        last_error = None
        for kwargs in load_attempts:
            try:
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_path,
                    **kwargs,
                ).to(device_label).eval()
                self.loaded_attn_implementation = str(kwargs.get("attn_implementation") or "default")
                break
            except Exception as exc:
                last_error = exc
        if self.model is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Could not load TimeChat captioner model.")
        with contextlib.suppress(Exception):
            self.model.disable_talker()
        self.generate_kwargs = _apply_generation_controls(
            self.model,
            do_sample=generate_do_sample,
            temperature=generate_temperature,
        )

    def _build_batch(self, messages: List[Dict[str, Any]], *, use_audio_in_video: bool):
        _patch_qwen_omni_video_reader_bounds()
        from qwen_omni_utils import process_mm_info

        if use_audio_in_video:
            _configure_audioread_ffmpeg_command()
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        audios, images, videos = process_mm_info(
            messages,
            use_audio_in_video=use_audio_in_video,
        )
        return self.processor(
            text=prompt_text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )

    def _audio_placeholder_mismatch(self, batch) -> str | None:
        batch_dict = dict(batch)
        input_ids = batch_dict.get("input_ids")
        feature_attention_mask = batch_dict.get("feature_attention_mask")
        if input_ids is None or feature_attention_mask is None:
            return None

        config = getattr(self.model, "config", None)
        audio_token_id = getattr(config, "audio_token_id", None)
        if audio_token_id is None:
            audio_token_id = getattr(config, "audio_token_index", None)
        if audio_token_id is None:
            return None

        placeholder_tokens = int((input_ids == int(audio_token_id)).sum().item())
        feature_lengths = feature_attention_mask.sum(-1)
        audio_tower = getattr(self.model, "audio_tower", None)
        if audio_tower is not None and hasattr(audio_tower, "_get_feat_extract_output_lengths"):
            _, output_lengths = audio_tower._get_feat_extract_output_lengths(feature_lengths)
        else:
            input_lengths = (feature_lengths - 1) // 2 + 1
            output_lengths = (input_lengths - 2) // 2 + 1
        feature_tokens = int(output_lengths.sum().item())
        if placeholder_tokens != feature_tokens:
            return "audio placeholders=%d features=%d" % (placeholder_tokens, feature_tokens)
        return None

    def _move_batch(self, batch):
        import torch

        moved = {}
        for key, value in dict(batch).items():
            if isinstance(value, torch.Tensor):
                tensor = value.to(self.device_label)
                if torch.is_floating_point(tensor):
                    tensor = tensor.to(self.dtype)
                moved[key] = tensor
            else:
                moved[key] = value
        return moved

    def generate(self, messages: List[Dict[str, Any]], *, max_new_tokens: int) -> str:
        use_audio_in_video = bool(self.use_audio_in_video and not self._audio_in_video_runtime_disabled)
        batch = self._build_batch(messages, use_audio_in_video=use_audio_in_video)
        mismatch = self._audio_placeholder_mismatch(batch) if use_audio_in_video else None
        if mismatch:
            # Avoid a CUDA device-side assert when Qwen Omni expands more audio
            # placeholders than the extracted audio encoder features can fill.
            self._audio_in_video_runtime_disabled = True
            logger.warning(
                "TimeChat audio-in-video disabled after placeholder mismatch for %s: %s",
                self.model_path,
                mismatch,
            )
            use_audio_in_video = False
            batch = self._build_batch(messages, use_audio_in_video=False)
        moved = self._move_batch(batch)
        outputs = self.model.generate(
            **moved,
            use_audio_in_video=use_audio_in_video,
            return_audio=False,
            thinker_max_new_tokens=max(32, int(max_new_tokens)),
            talker_max_tokens=max(32, int(max_new_tokens)),
            **self.generate_kwargs,
        )
        input_ids = moved.get("input_ids")
        if input_ids is not None:
            trimmed = outputs[:, input_ids.shape[1] :]
        else:
            trimmed = outputs
        with contextlib.suppress(Exception):
            decoded = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            text = str(decoded[0] if decoded else "").strip()
            if text:
                return text
        return str(self.processor.decode(trimmed[0], skip_special_tokens=True)).strip()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            del self.model
        with contextlib.suppress(Exception):
            del self.processor
        cleanup_torch()


def run_timechat_messages(
    *,
    model_path: str,
    messages: List[Dict[str, Any]],
    device_label: str,
    max_new_tokens: int,
    generate_do_sample: bool = False,
    generate_temperature: float | None = None,
    use_audio_in_video: bool = True,
    attn_implementation: str | None = None,
) -> str:
    runner = TimeChatCaptionerRunner(
        model_path=model_path,
        device_label=device_label,
        generate_do_sample=generate_do_sample,
        generate_temperature=generate_temperature,
        use_audio_in_video=use_audio_in_video,
        attn_implementation=attn_implementation,
    )
    try:
        return runner.generate(messages, max_new_tokens=max_new_tokens)
    finally:
        runner.close()


def _penguin_flash_attn_fallback(query_states, key_states, value_states, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0, causal=False, **kwargs):
    import torch
    import torch.nn.functional as F

    del max_seqlen_q, max_seqlen_k, kwargs
    q = query_states
    k = key_states
    v = value_states
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise RuntimeError("Penguin flash-attn fallback expected [tokens, heads, dim] tensors.")

    q_offsets = cu_seqlens_q.tolist() if hasattr(cu_seqlens_q, "tolist") else list(cu_seqlens_q)
    k_offsets = cu_seqlens_k.tolist() if hasattr(cu_seqlens_k, "tolist") else list(cu_seqlens_k)
    if len(q_offsets) != len(k_offsets):
        raise RuntimeError("Mismatched cu_seqlens for Penguin flash-attn fallback.")

    outputs = []
    for seg_index in range(len(q_offsets) - 1):
        q_start, q_end = int(q_offsets[seg_index]), int(q_offsets[seg_index + 1])
        k_start, k_end = int(k_offsets[seg_index]), int(k_offsets[seg_index + 1])
        q_seg = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_seg = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        v_seg = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        if k_seg.shape[1] != q_seg.shape[1]:
            if k_seg.shape[1] != v_seg.shape[1] or q_seg.shape[1] % k_seg.shape[1] != 0:
                raise RuntimeError(
                    "Penguin flash-attn fallback could not align query and key/value heads: "
                    f"q={q_seg.shape[1]} kv={k_seg.shape[1]}"
                )
            # Penguin's encoder uses grouped-query attention, but the local SDPA
            # fallback expects matching head counts.
            repeat_factor = q_seg.shape[1] // k_seg.shape[1]
            k_seg = k_seg.repeat_interleave(repeat_factor, dim=1)
            v_seg = v_seg.repeat_interleave(repeat_factor, dim=1)
        attn_seg = F.scaled_dot_product_attention(
            q_seg,
            k_seg,
            v_seg,
            attn_mask=None,
            dropout_p=float(dropout_p or 0.0),
            is_causal=bool(causal),
        )
        outputs.append(attn_seg.squeeze(0).transpose(0, 1))

    if not outputs:
        return q.new_zeros(q.shape)
    return torch.cat(outputs, dim=0)


def _install_penguin_attention_fallback() -> None:
    for module_name, module in list(sys.modules.items()):
        if not module_name.endswith("modeling_penguinvl_encoder"):
            continue
        if getattr(module, "flash_attn_varlen_func", None) is None:
            module.flash_attn_varlen_func = _penguin_flash_attn_fallback


def _install_penguin_processor_signature_compat(model_path: str) -> None:
    config_path = Path(model_path).expanduser() / "preprocessor_config.json"
    if not config_path.exists():
        return
    with contextlib.suppress(Exception):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        class_reference = str(((payload.get("auto_map") or {}).get("AutoProcessor")) or "").strip()
        if not class_reference:
            return
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        processor_class = get_class_from_dynamic_module(
            class_reference,
            pretrained_model_name_or_path=str(Path(model_path).expanduser()),
            local_files_only=True,
        )
        original_method = getattr(processor_class, "_get_arguments_from_pretrained", None)
        if original_method is None:
            return
        original_func = getattr(original_method, "__func__", original_method)
        if getattr(original_func, "_penguin_processor_signature_compat", False):
            return
        if "processor_dict" in inspect.signature(original_func).parameters:
            return

        @classmethod
        def _compat_get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
            return original_func(cls, pretrained_model_name_or_path, **kwargs)

        _compat_get_arguments_from_pretrained.__func__._penguin_processor_signature_compat = True
        processor_class._get_arguments_from_pretrained = _compat_get_arguments_from_pretrained


def _install_penguin_processor_kwargs_compat(model_path: str) -> None:
    with contextlib.suppress(Exception):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        processor_class = get_class_from_dynamic_module(
            "processing_penguinvl.PenguinVLQwen3Processor",
            pretrained_model_name_or_path=str(Path(model_path).expanduser()),
            local_files_only=True,
        )
        module = sys.modules.get(processor_class.__module__)
        if module is None:
            module = __import__(processor_class.__module__, fromlist=["PenguinVLQwen3ProcessorKwargs"])
        kwargs_class = getattr(module, "PenguinVLQwen3ProcessorKwargs", None)
        if kwargs_class is None or getattr(kwargs_class, "_penguin_processor_kwargs_compat", False):
            return

        class _EmptyCommonKwargs(object):
            __annotations__ = {}

        annotations = dict(getattr(kwargs_class, "__annotations__", {}) or {})
        annotations.setdefault("common_kwargs", _EmptyCommonKwargs)
        annotations.setdefault("chat_template_kwargs", getattr(module, "ChatTemplateKwargs", _EmptyCommonKwargs))
        kwargs_class.__annotations__ = annotations

        defaults = dict(getattr(kwargs_class, "_defaults", {}) or {})
        if "images_kwargs" not in defaults:
            defaults["images_kwargs"] = dict(defaults.get("image_kwargs") or {})
        if defaults.get("images_kwargs", {}).get("merge_size") is None:
            defaults["images_kwargs"] = {
                key: value
                for key, value in dict(defaults.get("images_kwargs") or {}).items()
                if key != "merge_size"
            }
        defaults.setdefault("text_kwargs", {})
        defaults.setdefault("images_kwargs", {})
        defaults.setdefault("videos_kwargs", {})
        defaults.setdefault("audio_kwargs", {})
        defaults.setdefault("chat_template_kwargs", {})
        defaults.setdefault("common_kwargs", {})
        kwargs_class._defaults = defaults
        kwargs_class._penguin_processor_kwargs_compat = True


def _install_penguin_image_processor_compat(model_path: str) -> None:
    with contextlib.suppress(Exception):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        image_processor_class = get_class_from_dynamic_module(
            "image_processing_penguinvl.PenguinVLImageProcessor",
            pretrained_model_name_or_path=str(Path(model_path).expanduser()),
            local_files_only=True,
        )
        original_init = getattr(image_processor_class, "__init__", None)
        original_func = getattr(original_init, "__func__", original_init)
        if original_func is None or getattr(original_func, "_penguin_image_processor_compat", False):
            return

        def _compat_init(self, *args, **kwargs):
            merge_size = kwargs.pop("merge_size", None)
            original_func(self, *args, **kwargs)
            if not hasattr(self, "merge_size"):
                self.merge_size = 1 if merge_size is None else merge_size

        _compat_init._penguin_image_processor_compat = True
        image_processor_class.__init__ = _compat_init


def _install_penguin_rotary_embedding_compat(model_path: str) -> None:
    with contextlib.suppress(Exception):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        import transformers.modeling_rope_utils as hf_rope_utils

        get_class_from_dynamic_module(
            "modeling_penguinvl_encoder.PenguinVLVisionEncoderModel",
            pretrained_model_name_or_path=str(Path(model_path).expanduser()),
            local_files_only=True,
        )
        for module_name, module in list(sys.modules.items()):
            if not module_name.endswith("modeling_penguinvl_encoder"):
                continue
            rotary_class = getattr(module, "VisualRotaryEmbedding", None)
            if rotary_class is None or hasattr(rotary_class, "compute_default_rope_parameters"):
                continue

            @staticmethod
            def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
                return hf_rope_utils.ROPE_INIT_FUNCTIONS["default"](
                    config=config,
                    device=device,
                    seq_len=seq_len,
                    layer_type=layer_type,
                )

            rotary_class.compute_default_rope_parameters = _compute_default_rope_parameters


def _configure_penguin_transformers_compat(model_path: str):
    from transformers import PreTrainedModel
    from transformers.integrations import accelerate as hf_accelerate
    import transformers.modeling_utils as hf_modeling_utils
    import transformers.modeling_rope_utils as hf_rope_utils
    from transformers.utils import import_utils as hf_import_utils
    import transformers.utils as hf_utils

    if "default" not in hf_rope_utils.ROPE_INIT_FUNCTIONS:
        def _compat_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
            import torch

            if hasattr(config, "standardize_rope_params"):
                config.standardize_rope_params()
            rope_parameters = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
            base = rope_parameters["rope_theta"]
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * partial_rotary_factor)
            target_device = device
            if target_device is None or str(target_device) == "meta":
                target_device = "cpu"
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, device=target_device, dtype=torch.float) / dim)
            )
            return inv_freq, 1.0

        hf_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compat_default_rope_parameters

    # Penguin's remote-code encoder tries to import flash-attn whenever
    # transformers reports the package as available. On this cluster the
    # installed flash-attn extension is ABI-incompatible with the pinned torch
    # wheel, so force the model onto a non-flash attention path.
    hf_import_utils.is_flash_attn_2_available = lambda: False
    hf_import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
    hf_utils.is_flash_attn_2_available = lambda: False
    hf_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
    _install_penguin_processor_kwargs_compat(model_path)
    _install_penguin_image_processor_compat(model_path)
    if not getattr(PreTrainedModel._load_pretrained_model, "_penguin_offload_compat", False):
        original_load_pretrained_model = PreTrainedModel._load_pretrained_model

        def _compat_load_pretrained_model(*args, **kwargs):
            kwargs.pop("offload_state_dict", None)
            if "load_config" not in kwargs and (kwargs or len(args) > 4):
                model = kwargs.pop("model", args[0] if args else None)
                state_dict = kwargs.pop("state_dict", args[1] if len(args) > 1 else None)
                checkpoint_files = kwargs.pop("checkpoint_files", args[2] if len(args) > 2 else None)
                pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
                ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
                sharded_metadata = kwargs.pop("sharded_metadata", None)
                device_map = kwargs.pop("device_map", None)
                disk_offload_folder = kwargs.pop("disk_offload_folder", None)
                dtype = kwargs.pop("dtype", None)
                hf_quantizer = kwargs.pop("hf_quantizer", None)
                device_mesh = kwargs.pop("device_mesh", None)
                weights_only = kwargs.pop("weights_only", True)
                key_mapping = kwargs.pop("key_mapping", None)
                kwargs.pop("keep_in_fp32_regex", None)
                load_config = hf_modeling_utils.LoadStateDictConfig(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    sharded_metadata=sharded_metadata,
                    device_map=device_map,
                    disk_offload_folder=disk_offload_folder,
                    dtype=dtype,
                    hf_quantizer=hf_quantizer,
                    device_mesh=device_mesh,
                    weights_only=weights_only,
                    weight_mapping=hf_modeling_utils.get_model_conversion_mapping(model, key_mapping, hf_quantizer),
                )
                return original_load_pretrained_model(model, state_dict, checkpoint_files, load_config)
            return original_load_pretrained_model(*args, **kwargs)

        _compat_load_pretrained_model._penguin_offload_compat = True
        PreTrainedModel._load_pretrained_model = _compat_load_pretrained_model
    if not getattr(hf_modeling_utils.check_and_set_device_map, "_penguin_meta_device_compat", False):
        original_check_and_set_device_map = hf_modeling_utils.check_and_set_device_map

        def _compat_check_and_set_device_map(device_map):
            try:
                return original_check_and_set_device_map(device_map)
            except RuntimeError as exc:
                if device_map is None and "meta device context manager" in str(exc):
                    return {"": "cpu"}
                raise

        _compat_check_and_set_device_map._penguin_meta_device_compat = True
        hf_modeling_utils.check_and_set_device_map = _compat_check_and_set_device_map
        hf_accelerate.check_and_set_device_map = _compat_check_and_set_device_map
    _install_penguin_processor_signature_compat(model_path)
    _install_penguin_rotary_embedding_compat(model_path)


class PenguinRunner:
    def __init__(
        self,
        *,
        model_path: str,
        device_label: str,
        generate_do_sample: bool = False,
        generate_temperature: float | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model_path = model_path
        self.device_label = device_label
        self.dtype = torch_dtype_for_device(device_label)
        _configure_penguin_transformers_compat(model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=self.dtype,
                attn_implementation="sdpa",
                local_files_only=True,
                low_cpu_mem_usage=False,
            ).to(device_label).eval()
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation="eager",
                local_files_only=True,
                low_cpu_mem_usage=False,
            ).to(device_label).eval()
        self.generate_kwargs = _apply_generation_controls(
            self.model,
            do_sample=generate_do_sample,
            temperature=generate_temperature,
        )
        _install_penguin_attention_fallback()

    def generate(self, conversation: List[Dict[str, Any]], *, max_new_tokens: int) -> str:
        import torch

        batch = self.processor(conversation=conversation, return_tensors="pt")
        moved = {}
        for key, value in dict(batch).items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device_label)
                if key == "pixel_values" and str(self.device_label).startswith("cuda"):
                    moved[key] = moved[key].to(self.dtype)
            else:
                moved[key] = value
        outputs = self.model.generate(
            **moved,
            max_new_tokens=max(32, int(max_new_tokens)),
            **self.generate_kwargs,
        )
        input_ids = moved.get("input_ids")
        if input_ids is not None:
            trimmed = outputs[:, input_ids.shape[1] :]
        else:
            trimmed = outputs
        with contextlib.suppress(Exception):
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)
            text = str(decoded[0] if decoded else "").strip()
            if text:
                return text
        with contextlib.suppress(Exception):
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
            text = str(decoded[0] if decoded else "").strip()
            if text:
                return text
        return str(self.processor.decode(outputs[0], skip_special_tokens=True)).strip()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            del self.model
        with contextlib.suppress(Exception):
            del self.processor
        cleanup_torch()


def run_penguin_messages(
    *,
    model_path: str,
    conversation: List[Dict[str, Any]],
    device_label: str,
    max_new_tokens: int,
    generate_do_sample: bool = False,
    generate_temperature: float | None = None,
) -> str:
    runner = PenguinRunner(
        model_path=model_path,
        device_label=device_label,
        generate_do_sample=generate_do_sample,
        generate_temperature=generate_temperature,
    )
    try:
        return runner.generate(conversation, max_new_tokens=max_new_tokens)
    finally:
        runner.close()


def make_qwen_image_messages(prompt: str, image_paths: Sequence[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for image_path in image_paths:
        content.append({"type": "image", "image": str(image_path)})
    content.append({"type": "text", "text": str(prompt or "").strip()})
    return [{"role": "user", "content": content}]


def make_qwen_video_message(prompt: str, video_path: str, fps: float = 2.0) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "min_pixels": 64 * 32 * 32,
                    "total_pixels": 14336 * 32 * 32,
                    "fps": float(fps),
                },
                {
                    "type": "text",
                    "text": str(prompt or "").strip(),
                },
            ],
        }
    ]


def make_timechat_video_conversation(
    prompt: str,
    video_path: str,
    *,
    fps: float = 2.0,
    max_frames: int = 160,
    max_pixels: int = 297920,
    video_max_pixels: int = 297920,
) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": str(prompt or "").strip(),
                },
                {
                    "type": "video",
                    "video": str(video_path),
                    "max_pixels": int(max_pixels),
                    "max_frames": int(max_frames),
                    "fps": float(fps),
                    "video_max_pixels": int(video_max_pixels),
                },
            ],
        }
    ]


def make_penguin_conversation(prompt: str, image_paths: Sequence[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for image_path in image_paths:
        content.append({"type": "image", "image": {"image_path": str(image_path)}})
    content.append({"type": "text", "text": str(prompt or "").strip()})
    return [
        {"role": "system", "content": "You are a careful multimodal extraction model."},
        {"role": "user", "content": content},
    ]
