from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..common import read_json, write_json, write_text
from ..tools.base import ToolExecutionContext
from ..storage import WorkspaceManager


class DenseCaptionPreprocessor(object):
    def __init__(self, workspace: WorkspaceManager, tool_registry, models_config):
        self.workspace = workspace
        self.tool_registry = tool_registry
        self.models_config = models_config

    def get_or_build(self, task, clip_duration_s: float) -> Dict[str, object]:
        dense_cfg = self.models_config.tools.get("dense_captioner")
        model_id = dense_cfg.model if dense_cfg and dense_cfg.model else "internal_dense_captioner"
        prompt_version = dense_cfg.prompt_version if dense_cfg else "v1"
        video_fingerprint = self.workspace.video_fingerprint(task.video_path)
        cache_dir = self.workspace.preprocess_dir(
            video_fingerprint_value=video_fingerprint,
            model_id=model_id,
            clip_duration_s=clip_duration_s,
            prompt_version=prompt_version,
        )
        manifest_path = cache_dir / "manifest.json"
        segments_path = cache_dir / "segments.json"
        summary_path = cache_dir / "summary.txt"
        if manifest_path.exists() and segments_path.exists() and summary_path.exists():
            return {
                "cache_hit": True,
                "cache_dir": self.workspace.relative_path(cache_dir),
                "manifest": read_json(manifest_path),
                "segments": read_json(segments_path),
                "summary": summary_path.read_text(encoding="utf-8"),
                "video_fingerprint": video_fingerprint,
            }

        class _PreprocessRun(object):
            def __init__(self, base_dir: Path):
                self.tools_dir = base_dir

        preprocess_run = _PreprocessRun(cache_dir / "_tool_scratch")
        preprocess_context = ToolExecutionContext(
            workspace=self.workspace,
            run=preprocess_run,
            task=task,
            models_config=self.models_config,
            llm_client=self.tool_registry.llm_client,
            evidence_lookup=None,
            preprocess_bundle=None,
        )
        result = self.tool_registry.build_dense_caption_cache(task, clip_duration_s, preprocess_context)
        manifest = {
            "video_fingerprint": video_fingerprint,
            "clip_duration_s": float(clip_duration_s),
            "model_id": model_id,
            "prompt_version": prompt_version,
            "segment_count": len(result.get("segments") or []),
        }
        write_json(manifest_path, manifest)
        write_json(segments_path, result.get("segments") or [])
        write_text(summary_path, str(result.get("summary") or ""))
        return {
            "cache_hit": False,
            "cache_dir": self.workspace.relative_path(cache_dir),
            "manifest": manifest,
            "segments": result.get("segments") or [],
            "summary": str(result.get("summary") or ""),
            "video_fingerprint": video_fingerprint,
        }
