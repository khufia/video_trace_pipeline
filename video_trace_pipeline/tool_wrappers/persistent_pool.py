from __future__ import annotations

from typing import Iterable

from .local_multimodal import PenguinRunner, QwenStyleRunner


_QWEN35_SHARED_TOOLS = frozenset({"generic_purpose", "spatial_grounder"})
_TOOL_ALIASES = {
    "temporal_grounder": "visual_temporal_grounder",
    "visual_grounder": "visual_temporal_grounder",
    "visual-temporal-grounder": "visual_temporal_grounder",
    "dense_caption": "dense_captioner",
    "dense-captioner": "dense_captioner",
    "generic": "generic_purpose",
    "generic-purpose": "generic_purpose",
    "spatial": "spatial_grounder",
}


def normalize_persist_tool_name(tool_name: str) -> str:
    normalized = str(tool_name or "").strip()
    if not normalized:
        return ""
    return _TOOL_ALIASES.get(normalized, normalized)


class PersistentModelPool(object):
    def __init__(self, enabled_tools: Iterable[str] | None = None):
        normalized = []
        seen = set()
        for item in list(enabled_tools or []):
            name = normalize_persist_tool_name(str(item or ""))
            if not name or name in seen:
                continue
            seen.add(name)
            normalized.append(name)
        self.enabled_tools = set(normalized)
        self._qwen_style_runners = {}
        self._penguin_runners = {}

    def _share_scope(self, tool_name: str) -> str:
        normalized = normalize_persist_tool_name(tool_name)
        if normalized in _QWEN35_SHARED_TOOLS and self.enabled_tools.intersection(_QWEN35_SHARED_TOOLS):
            return "qwen35_shared"
        return normalized

    def should_persist(self, tool_name: str) -> bool:
        normalized = normalize_persist_tool_name(tool_name)
        if normalized in _QWEN35_SHARED_TOOLS and self.enabled_tools.intersection(_QWEN35_SHARED_TOOLS):
            return True
        return normalized in self.enabled_tools

    def qwen_style_key(
        self,
        *,
        tool_name: str,
        model_path: str,
        device_label: str,
        processor_use_fast: bool | None = None,
        processor_model_path: str | None = None,
        generate_do_sample: bool = False,
        generate_temperature: float | None = None,
    ):
        return (
            self._share_scope(tool_name),
            str(model_path),
            str(device_label),
            None if processor_use_fast is None else bool(processor_use_fast),
            str(processor_model_path or ""),
            bool(generate_do_sample),
            None if generate_temperature is None else float(generate_temperature),
        )

    def penguin_key(
        self,
        *,
        tool_name: str,
        model_path: str,
        device_label: str,
        generate_do_sample: bool = False,
        generate_temperature: float | None = None,
    ):
        return (
            self._share_scope(tool_name),
            str(model_path),
            str(device_label),
            bool(generate_do_sample),
            None if generate_temperature is None else float(generate_temperature),
        )

    def acquire_qwen_style_runner(
        self,
        *,
        tool_name: str,
        model_path: str,
        device_label: str,
        processor_use_fast: bool | None = None,
        processor_model_path: str | None = None,
        generate_do_sample: bool = False,
        generate_temperature: float | None = None,
    ):
        if not self.should_persist(tool_name):
            return None
        key = self.qwen_style_key(
            tool_name=tool_name,
            model_path=model_path,
            device_label=device_label,
            processor_use_fast=processor_use_fast,
            processor_model_path=processor_model_path,
            generate_do_sample=generate_do_sample,
            generate_temperature=generate_temperature,
        )
        runner = self._qwen_style_runners.get(key)
        if runner is None:
            runner = QwenStyleRunner(
                model_path=model_path,
                device_label=device_label,
                processor_use_fast=processor_use_fast,
                processor_model_path=processor_model_path,
                generate_do_sample=generate_do_sample,
                generate_temperature=generate_temperature,
            )
            self._qwen_style_runners[key] = runner
        return runner

    def acquire_penguin_runner(
        self,
        *,
        tool_name: str,
        model_path: str,
        device_label: str,
        generate_do_sample: bool = False,
        generate_temperature: float | None = None,
    ):
        if not self.should_persist(tool_name):
            return None
        key = self.penguin_key(
            tool_name=tool_name,
            model_path=model_path,
            device_label=device_label,
            generate_do_sample=generate_do_sample,
            generate_temperature=generate_temperature,
        )
        runner = self._penguin_runners.get(key)
        if runner is None:
            runner = PenguinRunner(
                model_path=model_path,
                device_label=device_label,
                generate_do_sample=generate_do_sample,
                generate_temperature=generate_temperature,
            )
            self._penguin_runners[key] = runner
        return runner

    def close(self) -> None:
        closed = set()
        for runner in list(self._qwen_style_runners.values()) + list(self._penguin_runners.values()):
            runner_id = id(runner)
            if runner is None or runner_id in closed:
                continue
            closed.add(runner_id)
            try:
                runner.close()
            except Exception:
                pass
        self._qwen_style_runners.clear()
        self._penguin_runners.clear()
