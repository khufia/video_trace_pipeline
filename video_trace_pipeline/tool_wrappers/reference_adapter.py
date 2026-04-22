from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..tools.media import get_video_duration
from .shared import device_index, resolve_model_path, resolve_aux_model_path, resolved_device_label, tool_cache_root


def _prime_reference_cache_env() -> None:
    hf_home_raw = str(os.environ.get("HF_HOME") or "").strip()
    if hf_home_raw:
        hf_home = Path(hf_home_raw).expanduser().resolve()
    else:
        hf_home = (Path.home() / ".cache" / "huggingface").resolve()
        os.environ.setdefault("HF_HOME", str(hf_home))
    hub_dir = hf_home / "hub"
    runtime_root = hf_home.parent
    os.environ.setdefault("VDR_RUNTIME_ROOT", str(runtime_root))
    os.environ.setdefault("NEXUS_RUNTIME_ROOT", str(runtime_root))
    os.environ.setdefault("VDR_SHARED_HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_dir))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _reference_eval_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "cot_trace_generator_clean" / "VideoDeepResearch" / "eval"


def _ensure_reference_imports() -> Path:
    eval_dir = _reference_eval_dir()
    if not eval_dir.exists():
        raise RuntimeError("Reference eval directory is missing: %s" % eval_dir)
    search_roots = [eval_dir.resolve(), eval_dir.resolve().parent]
    for root in reversed(search_roots):
        text_path = str(root)
        if text_path not in sys.path:
            sys.path.insert(0, text_path)
    return eval_dir


def _install_transformers_generic_compat() -> None:
    try:
        from transformers.utils import generic as hf_generic
    except Exception:
        return
    if hasattr(hf_generic, "check_model_inputs"):
        return

    def check_model_inputs(func=None, *args, **kwargs):
        if func is None:
            def decorator(inner):
                return inner

            return decorator
        return func

    hf_generic.check_model_inputs = check_model_inputs


_prime_reference_cache_env()
_ensure_reference_imports()
_install_transformers_generic_compat()

from refiner_tools import RefinerToolsMixin  # type: ignore  # noqa: E402
from refiner_utils import RefinerUtilsMixin  # type: ignore  # noqa: E402


class ReferenceHarness(RefinerUtilsMixin, RefinerToolsMixin):
    def __init__(
        self,
        *,
        task: Dict[str, Any],
        runtime: Dict[str, Any],
        clip_duration_s: float,
        embedder_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
    ):
        self.video_path = str(task.get("video_path") or "")
        self.question = str(task.get("question") or "")
        self.answer = None
        self.options = list(task.get("options") or [])
        self.use_subtitle = False
        self.subtitles = ""
        self.clip_duration = max(1, int(round(float(clip_duration_s or 30.0))))
        self.duration = float(get_video_duration(self.video_path) or 0.0)
        self.dataset_folder = str(tool_cache_root(runtime, "reference", str(task.get("video_id") or "video")))
        self.dense_frame_fps = 1.0
        self._dense_frame_fps_override = 1.0
        self.use_clip_retrieval = False
        self.dense_segment_half_width = 0.5
        self.retrieval_top_k = 8
        self.dense_frame_embed_batch = max(1, int((runtime.get("extra") or {}).get("candidate_frames") or 8))
        self.use_retrieved_context = False
        self.segment_size_s = float(self.clip_duration)
        self._segment_captions_cache = []
        self._segment_index = []
        self._video_caption_summary = ""
        self._refinement_prev_step_results = {}
        self._refinement_prev_step_tools = {}
        self._temporal_grounder_video_info_cache = None
        self._temporal_grounder_qwen_clip_embeddings_cache = None
        self._temporal_grounder_embedder_class = None
        self._temporal_grounder_reranker_class = None
        self._frame_embedder = None
        self._chart_model = None
        self._chart_tokenizer = None
        self._grounding_dino_processor = None
        self._grounding_dino_model = None
        self._grounding_dino_loaded_model_name = None
        self._grounding_dino_loaded_device = None
        self.vlm_model_name = ""
        self.local_vlm_model_name = ""
        self.vlm_api_base = []
        self.vlm_api_keys = ["EMPTY"]
        self.vlm_tensor_parallel_size = 1
        self.vlm_server = None
        self.processor = None
        self._dense_captioner_vlm_server = None
        self._dense_captioner_processor = None
        self.refinement_debug_root = None
        self._refinement_debug_session_base = None
        self._refinement_debug_iter_dir = None
        self._refinement_debug_vlm_outputs_dir = None
        self._refinement_debug_vlm_input_basename = None

        device_label = resolved_device_label(runtime)
        device_idx = device_index(device_label)
        os.environ["FRAME_EMBEDDER_DEVICE_INDEX"] = str(device_idx if device_idx is not None else 0)
        os.environ["TEMPORAL_GROUNDER_DEVICE_INDEX"] = str(device_idx if device_idx is not None else 0)
        os.environ["CLAP_DEVICE"] = device_label
        os.environ["WHISPERX_DEVICE"] = device_label

        self.chart_device = device_label
        self.asr_device = device_label
        self.asr_compute_type = None
        self.spatial_grounder_backend = "vlm"
        self.spatial_grounder_model_name = ""
        self.spatial_grounder_device = device_label
        self.spatial_grounder_box_threshold = 0.25
        self.spatial_grounder_iou_threshold = 0.8

        self.temporal_grounder_backend = "qwen"
        self.temporal_grounder_model_name = resolve_model_path(
            embedder_model or str(runtime.get("model_name") or "Qwen/Qwen3-VL-Embedding-8B"),
            runtime,
        )
        self.temporal_grounder_reranker_model_name = (
            resolve_model_path(reranker_model, runtime)
            if reranker_model
            else resolve_aux_model_path(runtime, "reranker_model")
            or self.temporal_grounder_model_name
        )
        self.temporal_grounder_sample_fps = 1.0
        self.temporal_grounder_max_frames = 32
        self.temporal_grounder_batch_size = 4
        self.temporal_grounder_stride_seconds = max(1.0, float(self.clip_duration) / 2.0)
        self.temporal_grounder_device_index = device_idx if device_idx is not None else 0

    def _setup_environment(self):
        pass

    def _use_vlm_remote_api(self) -> bool:
        return False

    def _ensure_dense_frames(self) -> None:
        from video_utils import timestamp_to_clip_path  # type: ignore

        timestamp_to_clip_path(
            self.dataset_folder,
            0.0,
            float(self.duration),
            self.video_path,
            fps=float(getattr(self, "dense_frame_fps", 1.0)),
        )
