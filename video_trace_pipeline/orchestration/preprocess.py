from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from filelock import FileLock

from ..common import has_meaningful_text, read_json, write_json, write_text
from ..tools.base import ToolExecutionContext
from ..tools.specs import tool_implementation
from ..storage import WorkspaceManager


def _segment_summary_text(segment: Dict[str, Any]) -> str:
    if not isinstance(segment, dict):
        return ""
    caption_summary = str(segment.get("caption_summary") or "").strip()
    if has_meaningful_text(caption_summary):
        return caption_summary
    payload = segment.get("dense_caption")
    if not isinstance(payload, dict):
        return ""
    overall_summary = str(payload.get("overall_summary") or "").strip()
    if has_meaningful_text(overall_summary):
        return overall_summary
    parts = []
    for item in list(payload.get("captions") or []):
        if not isinstance(item, dict):
            continue
        for key in ("visual", "audio", "on_screen_text"):
            value = str(item.get(key) or "").strip()
            if has_meaningful_text(value):
                parts.append(value)
        for key in ("actions", "objects", "attributes"):
            values = [str(value).strip() for value in list(item.get(key) or []) if str(value).strip()]
            if values:
                parts.append("%s: %s" % (key, ", ".join(values)))
    return " ".join(parts).strip()


def _normalize_bundle(base_dir: Path, bundle: Dict[str, Any]) -> Dict[str, Any] | None:
    normalized = dict(bundle or {})
    manifest = dict(normalized.get("manifest") or {})
    summary = str(normalized.get("summary") or "").strip()
    if has_meaningful_text(summary):
        normalized["summary"] = summary
        return normalized
    segments = list(normalized.get("segments") or [])
    if not segments:
        summary_status = str(manifest.get("summary_status") or "").strip()
        if summary_status.startswith("unavailable"):
            normalized["summary"] = ""
            normalized["manifest"] = manifest
            return normalized
        return None
    derived_summary = " ".join(
        text for text in (_segment_summary_text(segment) for segment in segments) if has_meaningful_text(text)
    ).strip()
    if not has_meaningful_text(derived_summary):
        summary_status = str(manifest.get("summary_status") or "").strip()
        if summary_status.startswith("unavailable"):
            normalized["summary"] = ""
            normalized["manifest"] = manifest
            return normalized
        return None
    write_text(base_dir / "summary.txt", derived_summary)
    if manifest.get("summary_status") != "available":
        manifest["summary_status"] = "available"
        write_json(base_dir / "manifest.json", manifest)
    normalized["summary"] = derived_summary
    normalized["manifest"] = manifest
    return normalized


class DenseCaptionPreprocessor(object):
    def __init__(self, workspace: WorkspaceManager, tool_registry, models_config):
        self.workspace = workspace
        self.tool_registry = tool_registry
        self.models_config = models_config

    def get_or_build(self, task, clip_duration_s: float) -> Dict[str, object]:
        dense_cfg = self.models_config.tools.get("dense_captioner")
        implementation = tool_implementation("dense_captioner")
        model_name = dense_cfg.model if dense_cfg and dense_cfg.model else "dense_captioner"
        model_id = "%s__%s" % (implementation, model_name)
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

        def _bundle_if_complete(base_dir: Path):
            candidate_manifest = base_dir / "manifest.json"
            candidate_segments = base_dir / "segments.json"
            candidate_summary = base_dir / "summary.txt"
            if candidate_manifest.exists() and candidate_segments.exists() and candidate_summary.exists():
                return {
                    "manifest": read_json(candidate_manifest),
                    "segments": read_json(candidate_segments),
                    "summary": candidate_summary.read_text(encoding="utf-8"),
                }
            return None

        lock = FileLock(str(cache_dir / ".lock"))
        with lock:
            bundle = _bundle_if_complete(cache_dir)
            if bundle is not None:
                bundle = _normalize_bundle(cache_dir, bundle)
            if bundle is None:
                legacy_model_ids = [
                    "command_dense_captioner__%s" % model_name,
                ]
                for legacy_model_id in legacy_model_ids:
                    if legacy_model_id == model_id:
                        continue
                    legacy_dir = self.workspace.preprocess_dir(
                        video_fingerprint_value=video_fingerprint,
                        model_id=legacy_model_id,
                        clip_duration_s=clip_duration_s,
                        prompt_version=prompt_version,
                    )
                    legacy_bundle = _bundle_if_complete(legacy_dir)
                    if legacy_bundle is None:
                        continue
                    legacy_bundle = _normalize_bundle(legacy_dir, legacy_bundle)
                    if legacy_bundle is None:
                        continue
                    migrated_manifest = dict(legacy_bundle["manifest"] or {})
                    migrated_manifest["model_id"] = model_id
                    write_json(manifest_path, migrated_manifest)
                    write_json(segments_path, legacy_bundle["segments"] or [])
                    write_text(summary_path, str(legacy_bundle["summary"] or ""))
                    bundle = {
                        "manifest": migrated_manifest,
                        "segments": legacy_bundle["segments"] or [],
                        "summary": str(legacy_bundle["summary"] or ""),
                    }
                    break
            if bundle is not None:
                return {
                    "cache_hit": True,
                    "cache_dir": self.workspace.relative_path(cache_dir),
                    "manifest": bundle["manifest"],
                    "segments": bundle["segments"],
                    "summary": bundle["summary"],
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
            built_segments = result.get("segments") or []
            built_summary = str(result.get("summary") or "").strip()
            if not has_meaningful_text(built_summary):
                built_summary = " ".join(
                    text for text in (_segment_summary_text(segment) for segment in built_segments) if has_meaningful_text(text)
                ).strip()
            summary_status = "available" if has_meaningful_text(built_summary) else "unavailable_low_signal"
            manifest = {
                "video_fingerprint": video_fingerprint,
                "clip_duration_s": float(clip_duration_s),
                "model_id": model_id,
                "prompt_version": prompt_version,
                "segment_count": len(built_segments),
                "summary_status": summary_status,
            }
            write_json(manifest_path, manifest)
            write_json(segments_path, built_segments)
            write_text(summary_path, built_summary if summary_status == "available" else "")
            return {
                "cache_hit": False,
                "cache_dir": self.workspace.relative_path(cache_dir),
                "manifest": manifest,
                "segments": built_segments,
                "summary": built_summary if summary_status == "available" else "",
                "video_fingerprint": video_fingerprint,
            }
