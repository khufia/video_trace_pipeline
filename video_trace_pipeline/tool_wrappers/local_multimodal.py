from __future__ import annotations

import copy
import contextlib
import sys
from typing import Any, Dict, List, Sequence

from .shared import cleanup_torch, move_batch_to_device, torch_dtype_for_device


def _qwen_style_model(model_path: str, device_label: str):
    import torch
    from transformers import AutoModelForImageTextToText

    dtype = torch_dtype_for_device(device_label)
    with contextlib.suppress(Exception):
        return AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(device_label).eval()
    return AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device_label).eval()


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
    ):
        from transformers import AutoProcessor

        self.model_path = model_path
        self.device_label = device_label
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
        self.model = _qwen_style_model(model_path, device_label)
        self.generate_kwargs = _apply_generation_controls(
            self.model,
            do_sample=generate_do_sample,
            temperature=generate_temperature,
        )

    def generate(self, messages: List[Dict[str, Any]], *, max_new_tokens: int) -> str:
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
) -> str:
    runner = QwenStyleRunner(
        model_path=model_path,
        device_label=device_label,
        processor_use_fast=processor_use_fast,
        processor_model_path=processor_model_path,
        generate_do_sample=generate_do_sample,
        generate_temperature=generate_temperature,
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


def _configure_penguin_transformers_compat():
    from transformers import PreTrainedModel
    from transformers.utils import import_utils as hf_import_utils
    import transformers.utils as hf_utils

    # Penguin's remote-code encoder tries to import flash-attn whenever
    # transformers reports the package as available. On this cluster the
    # installed flash-attn extension is ABI-incompatible with the pinned torch
    # wheel, so force the model onto a non-flash attention path.
    hf_import_utils.is_flash_attn_2_available = lambda: False
    hf_import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
    hf_utils.is_flash_attn_2_available = lambda: False
    hf_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
    if not getattr(PreTrainedModel._load_pretrained_model, "_penguin_offload_compat", False):
        original_load_pretrained_model = PreTrainedModel._load_pretrained_model

        def _compat_load_pretrained_model(*args, **kwargs):
            kwargs.pop("offload_state_dict", None)
            return original_load_pretrained_model(*args, **kwargs)

        _compat_load_pretrained_model._penguin_offload_compat = True
        PreTrainedModel._load_pretrained_model = _compat_load_pretrained_model


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
        _configure_penguin_transformers_compat()
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
                low_cpu_mem_usage=True,
            ).to(device_label).eval()
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation="eager",
                local_files_only=True,
                low_cpu_mem_usage=True,
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
            return str(decoded[0] if decoded else "").strip()
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


def make_penguin_conversation(prompt: str, image_paths: Sequence[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for image_path in image_paths:
        content.append({"type": "image", "image": {"image_path": str(image_path)}})
    content.append({"type": "text", "text": str(prompt or "").strip()})
    return [
        {"role": "system", "content": "You are a careful multimodal extraction model."},
        {"role": "user", "content": content},
    ]
