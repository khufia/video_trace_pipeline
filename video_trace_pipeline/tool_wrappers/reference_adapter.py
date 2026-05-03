from __future__ import annotations

import gc
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional in tests
    import cv2
except Exception:  # pragma: no cover - optional in tests
    cv2 = None

try:  # pragma: no cover - optional in tests
    import numpy as np
except Exception:  # pragma: no cover - optional in tests
    np = None

from ..tools.media import get_video_duration
from ..runtime_devices import cuda_device_map_primary_label, parse_cuda_device_map
from .shared import device_index, resolve_aux_model_path, resolve_model_path, resolved_device_label, tool_cache_root


_CLAP_MODULE = None
_CLAP_RUNTIME: Dict[str, Any] = {}


def _prime_local_cache_env() -> None:
    hf_home_raw = str(os.environ.get("HF_HOME") or "").strip()
    if hf_home_raw:
        hf_home = Path(hf_home_raw).expanduser().resolve()
    else:
        hf_home = (Path.home() / ".cache" / "huggingface").resolve()
        os.environ.setdefault("HF_HOME", str(hf_home))
    hub_dir = hf_home / "hub"
    os.environ.setdefault("HF_HUB_CACHE", str(hub_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_dir))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _install_transformers_generic_compat() -> None:
    try:
        from transformers.utils import generic as hf_generic
    except Exception:
        return
    if hasattr(hf_generic, "check_model_inputs"):
        return

    def check_model_inputs(func=None, *args, **kwargs):
        del args, kwargs
        if func is None:
            def decorator(inner):
                return inner

            return decorator
        return func

    hf_generic.check_model_inputs = check_model_inputs


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for dense frame extraction")


def _timestamp_to_clip_path(
    dataset_folder: str,
    begin_time_stamp: float,
    end_time_stamp: float,
    video_path: str,
    *,
    fps: float = 2.0,
) -> tuple[list[str], list[float]]:
    _require_cv2()
    video_id = Path(str(video_path)).stem
    frame_folder = Path(dataset_folder) / "dense_frames" / video_id
    frame_folder.mkdir(parents=True, exist_ok=True)

    start_s = max(0.0, float(begin_time_stamp or 0.0))
    end_s = max(start_s, float(end_time_stamp or start_s))
    sample_fps = max(0.1, float(fps or 1.0))
    if (end_s - start_s) < 1.0:
        start_s = max(0.0, start_s - 0.5)
        end_s = end_s + 0.5

    span_s = max(0.0, end_s - start_s)
    num_frames = max(1, int(span_s * sample_fps))
    time_points = [start_s + idx * (1.0 / sample_fps) for idx in range(num_frames)]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open video for dense frame extraction: %s" % video_path)
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 0.0:
        capture.release()
        raise RuntimeError("Could not determine FPS for video: %s" % video_path)

    frame_paths: List[str] = []
    timestamps: List[float] = []
    for timestamp_s in time_points:
        frame_path = frame_folder / ("frame_%.2f.png" % float(timestamp_s))
        if not frame_path.exists():
            frame_index = max(0, int(round(float(timestamp_s) * video_fps)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            cv2.imwrite(str(frame_path), frame)
        frame_paths.append(str(frame_path))
        timestamps.append(float(timestamp_s))
    capture.release()
    return frame_paths, timestamps


def _as_float_tensor(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float32)


def _normalize_embeddings(value):
    import torch

    tensor = _as_float_tensor(value)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor / tensor.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)


_prime_local_cache_env()
_install_transformers_generic_compat()


class ReferenceHarness:
    def __init__(
        self,
        *,
        task: Dict[str, Any],
        runtime: Dict[str, Any],
        clip_duration_s: float,
        embedder_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
    ):
        self._runtime = dict(runtime or {})
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
        extra = dict(runtime.get("extra") or {})
        self.dense_frame_embed_batch = max(
            1,
            int(extra.get("dense_frame_embed_batch") or extra.get("candidate_frames") or 8),
        )
        self.segment_size_s = float(self.clip_duration)
        self._temporal_grounder_embedder_class = None
        self._temporal_grounder_reranker_class = None
        self._frame_embedder = None
        self._frame_embedder_attn_override = None
        self._frame_embedder_requested_attn_implementation = str(extra.get("attn_implementation") or "").strip() or None
        self._frame_embedder_device_map = str(extra.get("device_map") or "").strip() or None
        self._frame_embedder_device_indices = parse_cuda_device_map(self._frame_embedder_device_map)
        self._frame_embedder_loaded_attn_implementation = None
        self._frame_embedder_diagnostics: Dict[str, Any] = {
            "requested_attn_implementation": self._frame_embedder_requested_attn_implementation or "default",
            "loaded_attn_implementation": None,
            "device_map": self._frame_embedder_device_map,
            "fallback_attn_implementations": [],
            "embed_call_count": 0,
            "retry_count": 0,
            "batch_attempt_count": 0,
            "batch_failure_count": 0,
            "single_item_fallback_count": 0,
            "zero_embedding_fallback_count": 0,
            "dense_frame_embed_batch": self.dense_frame_embed_batch,
            "cache_load_error": None,
            "cache_build_error": None,
            "cache_write_error": None,
            "query_error": None,
            "frame_error": None,
            "cache_load_s": 0.0,
            "cache_build_s": 0.0,
            "query_embed_s": 0.0,
            "frame_embed_s": 0.0,
        }

        device_label = resolved_device_label(runtime)
        frame_device_label = cuda_device_map_primary_label(self._frame_embedder_device_map, device_label)
        device_idx = device_index(frame_device_label)
        os.environ["FRAME_EMBEDDER_DEVICE_INDEX"] = str(device_idx if device_idx is not None else 0)
        os.environ["TEMPORAL_GROUNDER_DEVICE_INDEX"] = str(device_idx if device_idx is not None else 0)
        os.environ["CLAP_DEVICE"] = frame_device_label

        self.chart_device = frame_device_label
        self.asr_device = frame_device_label
        self.asr_compute_type = None
        self.temporal_grounder_backend = "qwen"
        self.temporal_grounder_model_name = resolve_model_path(
            embedder_model or str(runtime.get("model_name") or "Qwen/Qwen3-VL-Embedding-8B"),
            runtime,
            prefer_runtime_resolved=not bool(embedder_model),
        )
        self.temporal_grounder_reranker_model_name = (
            resolve_model_path(reranker_model, runtime, prefer_runtime_resolved=False)
            if reranker_model
            else resolve_aux_model_path(runtime, "reranker_model")
            or self.temporal_grounder_model_name
        )
        self.temporal_grounder_device_index = device_idx if device_idx is not None else 0

    def _setup_environment(self):
        return None

    def _use_vlm_remote_api(self) -> bool:
        return False

    def _list_dense_frame_paths(self, dataset_folder: str, video_path: str) -> tuple[list[str], str]:
        dense_dir = Path(dataset_folder) / "dense_frames" / Path(video_path).stem
        if not dense_dir.is_dir():
            return [], str(dense_dir)
        files = [path for path in dense_dir.iterdir() if path.name.startswith("frame_") and path.suffix.lower() == ".png"]
        files.sort(key=lambda path: float(path.stem.replace("frame_", "")))
        return [str(path) for path in files], str(dense_dir)

    @staticmethod
    def _timestamp_from_dense_frame_path(path: str) -> float:
        return float(Path(path).stem.replace("frame_", ""))

    def _ensure_dense_frames(self) -> None:
        _timestamp_to_clip_path(
            self.dataset_folder,
            0.0,
            float(self.duration),
            self.video_path,
            fps=float(getattr(self, "dense_frame_fps", 1.0)),
        )

    def _dense_frame_embed_cache_paths(self):
        video_id = Path(str(self.video_path)).stem
        cache_dir = Path(self.dataset_folder) / "dense_frames" / video_id
        return cache_dir / "qwen_frame_embeddings.pt", cache_dir / "qwen_frame_paths.json"

    def _frame_embedder_device_index(self) -> int:
        raw = str(os.environ.get("FRAME_EMBEDDER_DEVICE_INDEX", "") or "").strip()
        if raw.isdigit():
            return int(raw)
        try:
            import torch

            count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            count = 0
        return 1 if count > 1 else 0

    @contextmanager
    def _frame_embedder_inference_context(self):
        try:
            import torch
        except Exception:
            yield
            return
        embed_dev = self._frame_embedder_device_index()
        if not torch.cuda.is_available():
            yield
            return
        use_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        with torch.cuda.device(embed_dev):
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    yield
            else:
                yield

    def _temporal_grounder_model_kwargs(self) -> Dict[str, Any]:
        try:
            import torch
        except Exception:
            torch = None
        use_bf16 = bool(
            torch is not None
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        kwargs: Dict[str, Any] = {"local_files_only": True}
        if use_bf16:
            kwargs["torch_dtype"] = torch.bfloat16
        attn_implementation = str(self._frame_embedder_attn_override or self._frame_embedder_requested_attn_implementation or "").strip()
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        return kwargs

    @staticmethod
    def _cuda_device_map_max_memory(device_indices: List[int]) -> Dict[int, int]:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA device_map requires visible CUDA devices."
            )
        device_count = torch.cuda.device_count()
        missing = [index for index in list(device_indices or []) if index < 0 or index >= device_count]
        if missing:
            raise RuntimeError(
                "device_map requested CUDA index/indices %s, but only %d CUDA device(s) are visible."
                % (", ".join(str(item) for item in missing), device_count)
            )
        return {
            int(index): int(torch.cuda.get_device_properties(int(index)).total_memory)
            for index in list(device_indices or [])
        }

    def _load_sharded_cuda_frame_embedder(self, Embedder, model_kwargs: Dict[str, Any]):
        module = sys.modules.get(getattr(Embedder, "__module__", ""))
        Model = getattr(module, "Qwen3VLForEmbedding", None) if module is not None else None
        Processor = getattr(module, "Qwen3VLProcessor", None) if module is not None else None
        if Model is None or Processor is None:
            raise RuntimeError("Qwen3-VL embedding helper does not expose sharded-load classes.")

        embedder = Embedder.__new__(Embedder)
        embedder.max_length = int(getattr(module, "MAX_LENGTH", 8192))
        embedder.min_pixels = int(getattr(module, "MIN_PIXELS", 4096))
        embedder.max_pixels = int(getattr(module, "MAX_PIXELS", 1800 * 32 * 32))
        embedder.total_pixels = int(getattr(module, "MAX_TOTAL_PIXELS", 10 * 768 * 32 * 32))
        embedder.fps = float(getattr(module, "FPS", 1))
        embedder.num_frames = int(getattr(module, "MAX_FRAMES", 64))
        embedder.max_frames = int(getattr(module, "MAX_FRAMES", 64))
        embedder.default_instruction = "Represent the user's input."

        kwargs = dict(model_kwargs)
        kwargs["device_map"] = "balanced"
        kwargs["max_memory"] = self._cuda_device_map_max_memory(self._frame_embedder_device_indices or [])
        embedder.model = Model.from_pretrained(
            str(self.temporal_grounder_model_name),
            trust_remote_code=True,
            **kwargs,
        )
        embedder.processor = Processor.from_pretrained(
            str(self.temporal_grounder_model_name),
            padding_side="right",
        )
        embedder.model.eval()
        return embedder

    def _frame_embedding_cache_ready(self) -> bool:
        emb_path, paths_path = self._dense_frame_embed_cache_paths()
        return emb_path.is_file() and paths_path.is_file()

    def _frame_embedder_runtime_metadata(self) -> Dict[str, Any]:
        metadata = dict(self._frame_embedder_diagnostics)
        metadata["loaded_attn_implementation"] = self._frame_embedder_loaded_attn_implementation or "unknown"
        metadata["embedding_cache_ready"] = self._frame_embedding_cache_ready()
        return metadata

    @staticmethod
    def _frame_embedder_error_text(exc: Exception) -> str:
        return "%s: %s" % (exc.__class__.__name__, exc)

    def _record_frame_embedder_error(self, key: str, exc: Exception) -> None:
        if key not in self._frame_embedder_diagnostics or self._frame_embedder_diagnostics.get(key) is None:
            self._frame_embedder_diagnostics[key] = self._frame_embedder_error_text(exc)

    @staticmethod
    def _should_retry_without_flash_attention(exc: Exception) -> bool:
        text = str(exc or "")
        return "FlashAttention" in text or "flash_attention" in text or "dim=0" in text or "zero dimension" in text

    def _fallback_frame_embedder_attn_implementations(self) -> List[Optional[str]]:
        current = str(self._frame_embedder_loaded_attn_implementation or self._frame_embedder_attn_override or self._frame_embedder_requested_attn_implementation or "").strip() or "default"
        candidates: List[Optional[str]] = ["sdpa", "eager", None]
        fallbacks: List[Optional[str]] = []
        for candidate in candidates:
            normalized = str(candidate or "default")
            if normalized == current:
                continue
            if normalized in [str(item or "default") for item in fallbacks]:
                continue
            fallbacks.append(candidate)
        return fallbacks

    def _reload_frame_embedder(self, attn_implementation: Optional[str]) -> None:
        self._release_frame_embedder()
        self._frame_embedder_attn_override = attn_implementation
        self._frame_embedder_loaded_attn_implementation = None
        if attn_implementation is not None:
            normalized = str(attn_implementation or "default")
            recorded = [str(item or "default") for item in self._frame_embedder_diagnostics["fallback_attn_implementations"]]
            if normalized not in recorded:
                self._frame_embedder_diagnostics["fallback_attn_implementations"].append(normalized)

    def _embed_with_frame_embedder(self, samples: List[Dict[str, Any]], *, phase: str):
        last_error: Optional[Exception] = None
        attempt_specs: List[Optional[str]] = [self._frame_embedder_attn_override]
        for fallback_attn in self._fallback_frame_embedder_attn_implementations():
            if fallback_attn not in attempt_specs:
                attempt_specs.append(fallback_attn)
        for attempt_index, attn_implementation in enumerate(attempt_specs):
            if attempt_index > 0:
                self._frame_embedder_diagnostics["retry_count"] += 1
                self._reload_frame_embedder(attn_implementation)
            embedder = self._get_or_load_frame_embedder()
            started_at = time.perf_counter()
            self._frame_embedder_diagnostics["embed_call_count"] += 1
            try:
                with self._frame_embedder_inference_context():
                    embs = _normalize_embeddings(embedder.process(samples))
                elapsed_s = time.perf_counter() - started_at
                if phase == "query":
                    self._frame_embedder_diagnostics["query_embed_s"] += elapsed_s
                else:
                    self._frame_embedder_diagnostics["frame_embed_s"] += elapsed_s
                return embs
            except Exception as exc:
                elapsed_s = time.perf_counter() - started_at
                if phase == "query":
                    self._frame_embedder_diagnostics["query_embed_s"] += elapsed_s
                    self._record_frame_embedder_error("query_error", exc)
                else:
                    self._frame_embedder_diagnostics["frame_embed_s"] += elapsed_s
                    self._record_frame_embedder_error("frame_error", exc)
                last_error = exc
                if not self._should_retry_without_flash_attention(exc):
                    break
        if last_error is not None:
            raise last_error
        raise RuntimeError("Frame embedder did not produce embeddings.")

    def _load_model_helper_class(
        self,
        model_dir: str,
        script_name: str,
        class_name: str,
        cache_attr: str,
        *,
        patch_sample_frames: bool = False,
    ):
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached

        root = Path(model_dir)
        module_path = root / "scripts" / script_name
        if not module_path.exists():
            module_path = root / script_name
        if not module_path.exists():
            raise RuntimeError("Missing helper script: %s" % module_path)

        module_name = "vtp_%s_%s" % (class_name.lower(), abs(hash(str(module_path.resolve()))))
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load helper script: %s" % module_path)

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        klass = getattr(module, class_name, None)
        if klass is None:
            raise RuntimeError("Helper script %s does not define %s" % (module_path, class_name))
        if patch_sample_frames and not hasattr(klass, "_sample_frames") and hasattr(module, "sample_frames"):
            klass._sample_frames = staticmethod(module.sample_frames)
        setattr(self, cache_attr, klass)
        return klass

    def _get_or_load_frame_embedder(self):
        if self._frame_embedder is not None:
            return self._frame_embedder

        Embedder = self._load_model_helper_class(
            self.temporal_grounder_model_name,
            "qwen3_vl_embedding.py",
            "Qwen3VLEmbedder",
            "_temporal_grounder_embedder_class",
        )
        model_kwargs = self._temporal_grounder_model_kwargs()
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required for frame embedding") from exc

        embed_dev_idx = self._frame_embedder_device_index()
        device_context = torch.cuda.device(embed_dev_idx) if torch.cuda.is_available() else nullcontext()
        with device_context:
            if self._frame_embedder_device_indices:
                self._frame_embedder = self._load_sharded_cuda_frame_embedder(Embedder, model_kwargs)
            else:
                self._frame_embedder = Embedder(
                    model_name_or_path=str(self.temporal_grounder_model_name),
                    **model_kwargs,
                )
        self._frame_embedder_loaded_attn_implementation = str(
            self._frame_embedder_attn_override or self._frame_embedder_requested_attn_implementation or "default"
        )
        self._frame_embedder_diagnostics["loaded_attn_implementation"] = self._frame_embedder_loaded_attn_implementation
        return self._frame_embedder

    def _release_frame_embedder(self):
        try:
            import torch
        except Exception:
            self._frame_embedder = None
            return

        embedder = self._frame_embedder
        if embedder is None:
            return
        embed_dev_idx = self._frame_embedder_device_index()
        model = getattr(embedder, "model", None)
        if isinstance(model, torch.nn.Module):
            try:
                model.cpu()
            except Exception:
                pass
        for attr in ("model", "processor", "tokenizer", "score_linear"):
            if hasattr(embedder, attr):
                try:
                    setattr(embedder, attr, None)
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            device_indices = [embed_dev_idx]
            if self._frame_embedder_device_indices:
                device_indices = list(self._frame_embedder_device_indices)
            for device_index_value in device_indices:
                with torch.cuda.device(device_index_value):
                    torch.cuda.empty_cache()
        self._frame_embedder = None
        self._frame_embedder_loaded_attn_implementation = None

    def _precompute_frame_embeddings_cache(self, frame_items: list) -> bool:
        import torch

        emb_path, paths_path = self._dense_frame_embed_cache_paths()
        if emb_path.is_file() and paths_path.is_file():
            return True
        if not frame_items:
            return False

        batch_size = max(1, int(getattr(self, "dense_frame_embed_batch", 8)))
        all_embs = []
        started_at = time.perf_counter()
        try:
            for offset in range(0, len(frame_items), batch_size):
                batch = frame_items[offset : offset + batch_size]
                samples = [{"video": [item["frame_path"]]} for item in batch]
                try:
                    self._frame_embedder_diagnostics["batch_attempt_count"] += 1
                    embs = self._embed_with_frame_embedder(samples, phase="frame_batch")
                except Exception as exc:
                    self._frame_embedder_diagnostics["batch_failure_count"] += 1
                    self._record_frame_embedder_error("frame_error", exc)
                    fallback = []
                    for item in batch:
                        self._frame_embedder_diagnostics["single_item_fallback_count"] += 1
                        try:
                            emb = self._embed_with_frame_embedder(
                                [{"video": [item["frame_path"]]}],
                                phase="frame_single",
                            )
                        except Exception:
                            emb = torch.zeros(1, 4096, dtype=torch.float32)
                            self._frame_embedder_diagnostics["zero_embedding_fallback_count"] += 1
                        fallback.append(emb)
                    embs = torch.cat(fallback, dim=0)
                all_embs.append(embs.cpu())
            frame_embs = torch.cat(all_embs, dim=0).float()
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(frame_embs, emb_path)
            paths_path.write_text(json.dumps([item["frame_path"] for item in frame_items]), encoding="utf-8")
            self._frame_embedder_diagnostics["cache_build_s"] += time.perf_counter() - started_at
            return True
        except Exception as exc:
            self._frame_embedder_diagnostics["cache_build_s"] += time.perf_counter() - started_at
            self._record_frame_embedder_error("cache_build_error", exc)
            return False

    def _qwen_score_frames(self, query: str, frame_items: list, top_k: int, *, persist_cache: bool = True) -> list:
        import torch

        if not frame_items:
            return []

        emb_path, paths_path = self._dense_frame_embed_cache_paths()
        cached_frame_embs = None
        cached_paths_index: Dict[str, int] = {}
        if self._frame_embedding_cache_ready():
            cache_load_started_at = time.perf_counter()
            try:
                saved_paths = json.loads(paths_path.read_text(encoding="utf-8"))
                cached_frame_embs = torch.load(emb_path, map_location="cpu").float()
                cached_paths_index = {str(path): idx for idx, path in enumerate(saved_paths)}
            except Exception as exc:
                cached_frame_embs = None
                cached_paths_index = {}
                self._record_frame_embedder_error("cache_load_error", exc)
            finally:
                self._frame_embedder_diagnostics["cache_load_s"] += time.perf_counter() - cache_load_started_at

        q_emb = self._embed_with_frame_embedder(
            [{"text": query, "instruction": "Retrieve frames relevant to the user's query."}],
            phase="query",
        )

        ordered_embs: List[Optional[torch.Tensor]] = [None] * len(frame_items)
        if cached_frame_embs is not None:
            missing_items = []
            missing_indices = []
            for idx, item in enumerate(frame_items):
                cache_idx = cached_paths_index.get(str(item["frame_path"]))
                if cache_idx is None:
                    missing_items.append(item)
                    missing_indices.append(idx)
                    continue
                ordered_embs[idx] = cached_frame_embs[cache_idx].float()
            if missing_items:
                batch_size = max(1, int(getattr(self, "dense_frame_embed_batch", 8)))
                for offset in range(0, len(missing_items), batch_size):
                    batch = missing_items[offset : offset + batch_size]
                    batch_indices = missing_indices[offset : offset + batch_size]
                    try:
                        self._frame_embedder_diagnostics["batch_attempt_count"] += 1
                        embs = self._embed_with_frame_embedder(
                            [{"video": [item["frame_path"]]} for item in batch],
                            phase="frame_batch",
                        )
                    except Exception as exc:
                        self._frame_embedder_diagnostics["batch_failure_count"] += 1
                        self._record_frame_embedder_error("frame_error", exc)
                        embs = torch.zeros(len(batch), q_emb.shape[-1], dtype=torch.float32)
                        self._frame_embedder_diagnostics["zero_embedding_fallback_count"] += len(batch)
                    for row_idx, emb in enumerate(embs):
                        ordered_embs[batch_indices[row_idx]] = emb.float()
            frame_embs = torch.stack(
                [emb if emb is not None else torch.zeros(q_emb.shape[-1], dtype=torch.float32) for emb in ordered_embs],
                dim=0,
            )
        else:
            batch_size = max(1, int(getattr(self, "dense_frame_embed_batch", 8)))
            batches = []
            for offset in range(0, len(frame_items), batch_size):
                batch = frame_items[offset : offset + batch_size]
                try:
                    self._frame_embedder_diagnostics["batch_attempt_count"] += 1
                    embs = self._embed_with_frame_embedder(
                        [{"video": [item["frame_path"]]} for item in batch],
                        phase="frame_batch",
                    )
                except Exception as exc:
                    self._frame_embedder_diagnostics["batch_failure_count"] += 1
                    self._record_frame_embedder_error("frame_error", exc)
                    embs = torch.zeros(len(batch), q_emb.shape[-1], dtype=torch.float32)
                    self._frame_embedder_diagnostics["zero_embedding_fallback_count"] += len(batch)
                batches.append(embs)
            frame_embs = torch.cat(batches, dim=0).float()
            if persist_cache:
                try:
                    emb_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(frame_embs.cpu(), emb_path)
                    paths_path.write_text(json.dumps([item["frame_path"] for item in frame_items]), encoding="utf-8")
                except Exception as exc:
                    self._record_frame_embedder_error("cache_write_error", exc)

        sims = (q_emb @ frame_embs.T).squeeze(0)
        if sims.dim() == 0:
            sims = sims.unsqueeze(0)
        scored = [
            {
                "frame_path": item["frame_path"],
                "timestamp": item["timestamp"],
                "relevance_score": float(score),
            }
            for item, score in zip(frame_items, sims.tolist())
        ]
        scored.sort(key=lambda item: (-item["relevance_score"], item["timestamp"]))
        return scored[: max(1, int(top_k or 1))]

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _get_time_range(self, start_time=None, end_time=None):
        start = self._safe_float(start_time, 0.0)
        end = self._safe_float(end_time, float(self.duration))
        start = max(0.0, min(start, float(self.duration)))
        end = max(0.0, min(end, float(self.duration)))
        if end <= start:
            end = min(float(self.duration), start + max(1.0, float(self.clip_duration)))
        return start, end

    @staticmethod
    def _normalize_audio_label(text: str) -> str:
        cleaned = str(text or "").strip().lower()
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in cleaned)
        return " ".join(cleaned.split())

    def _audio_grounder_query_mode(self, query: str) -> str:
        q = self._normalize_audio_label(query)
        if not q:
            return "inventory"
        broad_markers = (
            "distinct sound",
            "different sound",
            "sound effect",
            "sound effects",
            "audio cue",
            "audio cues",
            "non speech",
            "how many sounds",
            "what sounds",
            "which sounds",
            "various sounds",
        )
        if any(marker in q for marker in broad_markers):
            return "inventory"
        return "targeted"

    @staticmethod
    def _merge_audio_grounder_events(events):
        merged = []
        ordered = sorted(events or [], key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
        for event in ordered:
            label = str(event.get("event_label") or "").strip()
            start = float(event.get("start", 0.0))
            end = float(event.get("end", start))
            confidence = float(event.get("confidence", 0.0) or 0.0)
            if merged and merged[-1]["event_label"] == label and start <= float(merged[-1]["end"]) + 1e-6:
                merged[-1]["end"] = max(float(merged[-1]["end"]), end)
                merged[-1]["confidence"] = max(float(merged[-1]["confidence"]), confidence)
                continue
            merged.append(
                {
                    "event_label": label,
                    "start": start,
                    "end": end,
                    "confidence": confidence,
                }
            )
        return merged

    def _resolve_ffmpeg_binary(self) -> str:
        return shutil.which("ffmpeg") or "ffmpeg"

    def _load_audio_window_with_ffmpeg(self, start_time=None, end_time=None, sample_rate: int = 48000):
        if np is None:
            raise RuntimeError("numpy is required for audio grounding")
        start, end = self._get_time_range(start_time, end_time)
        duration = max(0.05, float(end - start))
        command = [
            self._resolve_ffmpeg_binary(),
            "-nostdin",
            "-threads",
            "0",
            "-ss",
            str(float(start)),
            "-i",
            self.video_path,
            "-t",
            str(duration),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(int(sample_rate)),
            "-",
        ]
        completed = subprocess.run(command, capture_output=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or b"").decode(errors="ignore").strip()
            raise RuntimeError("Failed to extract audio window with ffmpeg: %s" % detail)
        audio = np.frombuffer(completed.stdout, np.int16).flatten().astype(np.float32) / 32768.0
        return audio, int(sample_rate), float(start), float(end)

    def _write_mono_wav(self, wav_path: str, audio_data, sample_rate: int) -> None:
        if np is None:
            raise RuntimeError("numpy is required for audio grounding")
        if audio_data is None:
            audio = np.asarray([], dtype=np.float32)
        else:
            audio = np.asarray(audio_data, dtype=np.float32).flatten()
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)
        with wave.open(wav_path, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            handle.writeframes(pcm.tobytes())

    def _clap_torch_device(self):
        import torch

        raw = str(os.environ.get("CLAP_DEVICE") or "").strip()
        if not raw or raw.lower() == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if raw.isdigit():
            raw = "cuda:%s" % raw
        try:
            device = torch.device(raw)
        except Exception:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                return torch.device("cpu")
            if device.index is not None and device.index >= torch.cuda.device_count():
                return torch.device("cuda:0")
        return device

    def _clap_checkpoint_path(self) -> Optional[Path]:
        candidates: List[Path] = []
        explicit = str(os.environ.get("CLAP_CKPT_PATH") or "").strip()
        if explicit:
            candidates.append(Path(explicit).expanduser())

        hf_roots = []
        explicit_hf = str(self._runtime.get("hf_cache") or "").strip()
        if explicit_hf:
            hf_roots.append(Path(explicit_hf).expanduser())
        for env_name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
            raw = str(os.environ.get(env_name) or "").strip()
            if raw:
                hf_roots.append(Path(raw).expanduser())
        seen = set()
        for root in hf_roots:
            normalized = str(root)
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(root / "assets" / "laion_clap" / "630k-audioset-best.pt")
            candidates.append(root / "630k-audioset-best.pt")
            snapshots_dir = root / "hub" / "models--lukewys--laion_clap" / "snapshots"
            if snapshots_dir.is_dir():
                for snapshot in sorted(path for path in snapshots_dir.iterdir() if path.is_dir()):
                    candidates.append(snapshot / "630k-audioset-best.pt")

        torch_home = str(os.environ.get("TORCH_HOME") or "").strip()
        if torch_home:
            candidates.append(Path(torch_home).expanduser() / "hub" / "checkpoints" / "630k-audioset-best.pt")

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _get_or_load_clap_module(self):
        global _CLAP_MODULE, _CLAP_RUNTIME

        import torch
        from laion_clap import CLAP_Module

        if _CLAP_MODULE is not None:
            return _CLAP_MODULE

        clap_device = self._clap_torch_device()
        clap_ckpt = self._clap_checkpoint_path()
        if clap_ckpt is None and str(os.environ.get("HF_HUB_OFFLINE") or "").strip() == "1":
            raise RuntimeError("LAION-CLAP checkpoint 630k-audioset-best.pt is not available in the local cache.")

        module = CLAP_Module(enable_fusion=False, device=str(clap_device))
        if clap_ckpt is not None:
            module.load_ckpt(ckpt=str(clap_ckpt), verbose=False)
        else:
            module.load_ckpt(verbose=False)
        _CLAP_MODULE = module
        _CLAP_RUNTIME = {
            "device": str(clap_device),
            "checkpoint": str(clap_ckpt) if clap_ckpt is not None else None,
        }
        return _CLAP_MODULE

    def _audio_grounder_clap(self, arguments: dict) -> dict:
        try:
            import torch
            from laion_clap import CLAP_Module  # noqa: F401
        except Exception:
            return {
                "query": str(arguments.get("query", "")).strip(),
                "query_mode": self._audio_grounder_query_mode(str(arguments.get("query", "")).strip()),
                "events": [],
                "audio_summary": "laion_clap or deps not available",
                "backend": "none",
            }
        if np is None:
            return {
                "query": str(arguments.get("query", "")).strip(),
                "query_mode": self._audio_grounder_query_mode(str(arguments.get("query", "")).strip()),
                "events": [],
                "audio_summary": "numpy is not available",
                "backend": "none",
            }

        query = str(arguments.get("query", "")).strip()
        query_mode = self._audio_grounder_query_mode(query)
        if not query:
            return {
                "query": query,
                "query_mode": query_mode,
                "events": [],
                "audio_summary": "empty query",
                "backend": "none",
            }

        try:
            module = self._get_or_load_clap_module()
            audio_data, sample_rate, start_s, end_s = self._load_audio_window_with_ffmpeg(
                arguments.get("start_time"),
                arguments.get("end_time"),
                sample_rate=48000,
            )
            duration = len(audio_data) / float(sample_rate)
            window_s = float(os.environ.get("CLAP_WINDOW_SEC", "2.0") or 2.0)
            hop_s = float(os.environ.get("CLAP_HOP_SEC", "1.0") or 1.0)
            threshold = float(os.environ.get("CLAP_SIM_THRESHOLD", "0.25") or 0.25)

            with torch.no_grad():
                text_embedding = module.get_text_embedding([query], use_tensor=False)
            text_embedding = np.asarray(text_embedding).reshape(-1).flatten()

            events = []
            offset_s = 0.0
            while offset_s + window_s <= duration + 1e-6:
                i0 = int(offset_s * sample_rate)
                i1 = int(min(len(audio_data), (offset_s + window_s) * sample_rate))
                if i1 <= i0:
                    break
                chunk = audio_data[i0:i1].astype(np.float32)
                chunk_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                try:
                    self._write_mono_wav(chunk_path, chunk, sample_rate)
                    with torch.no_grad():
                        audio_embedding = module.get_audio_embedding_from_filelist([chunk_path], use_tensor=False)
                    audio_embedding = np.asarray(audio_embedding).reshape(-1).flatten()
                finally:
                    try:
                        os.unlink(chunk_path)
                    except OSError:
                        pass
                similarity = float(
                    np.dot(audio_embedding, text_embedding)
                    / (np.linalg.norm(audio_embedding) * np.linalg.norm(text_embedding) + 1e-8)
                )
                if similarity >= threshold:
                    events.append(
                        {
                            "event_label": query,
                            "start": float(start_s + offset_s),
                            "end": float(start_s + min(offset_s + window_s, duration)),
                            "confidence": min(1.0, max(0.0, similarity)),
                        }
                    )
                offset_s += hop_s

            merged_events = self._merge_audio_grounder_events(events)
            return {
                "query": query,
                "query_mode": query_mode,
                "events": merged_events,
                "raw_event_count": len(events),
                "audio_summary": "CLAP scan %d peaks; merged %d events; query=%r" % (len(events), len(merged_events), query),
                "backend": "laion_clap",
                "audio_status": "ok",
                "audio_error": None,
                "audio_fallback_used": False,
                "clap_device": _CLAP_RUNTIME.get("device"),
                "clap_checkpoint": _CLAP_RUNTIME.get("checkpoint"),
                "requested_range": {"start": float(start_s), "end": float(end_s)},
            }
        except Exception as exc:
            return {
                "query": query,
                "query_mode": query_mode,
                "events": [],
                "audio_summary": str(exc),
                "backend": "none",
                "audio_status": "unavailable",
                "audio_error": str(exc),
                "audio_fallback_used": False,
            }
