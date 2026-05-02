from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from .common import assign_path, ensure_dir, read_json, sanitize_for_json, traverse_path, write_json
from .config import tool_runtime

LIST_ARGUMENT_FIELDS = frozenset({"anchors", "captions", "clips", "evidence", "frames", "lines", "reads", "regions", "segments", "texts", "transcript_segments", "transcripts", "ocr_results"})


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _merge_dependency_values(existing: Any, new_value: Any, target_field: str) -> Any:
    leaf = str(target_field or "").split(".")[-1].lower()
    if leaf in LIST_ARGUMENT_FIELDS:
        return _coerce_list(existing) + _coerce_list(new_value)
    if existing is None:
        return new_value
    if isinstance(existing, list):
        return existing + _coerce_list(new_value)
    if isinstance(new_value, list):
        return [existing] + new_value
    return new_value


def _record_step_id(record: dict[str, Any]) -> str:
    step = dict(record.get("step") or {})
    return str(step.get("id") or "").strip()


def _source_record(ref_step_id: str, previous_steps: list[dict[str, Any]]) -> dict[str, Any]:
    for record in reversed(list(previous_steps or [])):
        if _record_step_id(record) == str(ref_step_id):
            return record
    raise KeyError("Missing step output for %s" % ref_step_id)


def _runtime_ref_path(output: Any) -> str:
    text = str(output or "").strip()
    if not text:
        return ""
    return "output.%s" % text


def _coerce_ref(ref: Any) -> dict[str, Any]:
    if isinstance(ref, dict):
        from_step = str(ref.get("from_step") or "").strip()
        output = str(ref.get("output") or "").strip()
        if not from_step or not output:
            raise ValueError("request_refs entries must contain from_step and output")
        return {
            "from_step": from_step,
            "output": output,
        }
    raise TypeError("request_refs entries must be objects")


def resolve_request_refs(step_request: dict[str, Any], request_refs: dict[str, Any], previous_steps: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = dict(step_request or {})
    for target_field, refs in dict(request_refs or {}).items():
        for ref in list(refs or []):
            ref = _coerce_ref(ref)
            source = _source_record(str(ref.get("from_step") or ""), previous_steps)
            path = _runtime_ref_path(ref.get("output"))
            value = traverse_path(source.get("result"), path)
            if value is None:
                raise KeyError("Could not resolve %s from step %s" % (path, ref.get("from_step")))
            existing = traverse_path(resolved, str(target_field))
            assign_path(resolved, str(target_field), _merge_dependency_values(existing, value, str(target_field)))
    return resolved


def _configured_command(tool_name: str, models: dict[str, Any]) -> list[str]:
    config = dict((models.get("tools") or {}).get(tool_name) or {})
    command = config.get("command") or (config.get("extra") or {}).get("command")
    if not command:
        command = [sys.executable, "-m", "video_trace_pipeline_simple.tools.%s" % tool_name]
    command = [str(item) for item in command]
    if command and Path(command[0]).name in {"python", "python3"}:
        command[0] = sys.executable
    return command


def call_tool_script(tool_name: str, payload: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    command = _configured_command(tool_name, models)
    run_dir = Path(str((payload.get("runtime") or {}).get("run_dir") or ".")).expanduser().resolve()
    tmp_dir = ensure_dir(run_dir / "_tmp")
    with tempfile.TemporaryDirectory(prefix="%s_" % tool_name, dir=str(tmp_dir)) as temp_dir:
        input_path = Path(temp_dir) / "request.json"
        output_path = Path(temp_dir) / "result.json"
        write_json(input_path, payload)
        completed = subprocess.run(
            command + ["--input", str(input_path), "--output", str(output_path)],
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if not output_path.exists():
            return {
                "ok": False,
                "tool": tool_name,
                "output": {},
                "artifacts": [],
                "error": {
                    "type": "ToolSubprocessError",
                    "message": "tool did not write an output file",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                    "command": command,
                },
                "metadata": {},
            }
        result = read_json(output_path)
        if completed.returncode != 0 and result.get("ok") is not False:
            result = {
                "ok": False,
                "tool": tool_name,
                "output": result if isinstance(result, dict) else {},
                "artifacts": [],
                "error": {
                    "type": "ToolSubprocessError",
                    "message": "tool exited with code %s" % completed.returncode,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                    "command": command,
                },
                "metadata": {},
            }
        return sanitize_for_json(result)


def run_tool_step(
    step: dict[str, Any],
    task: dict[str, Any],
    previous_steps: list[dict[str, Any]],
    profile: dict[str, Any],
    models: dict[str, Any],
    run_dir: str | Path,
    on_request: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if "request" not in step:
        raise KeyError("Plan step %s is missing request" % (step.get("id") or "<unknown>"))
    if "request_refs" not in step:
        raise KeyError("Plan step %s is missing request_refs" % (step.get("id") or "<unknown>"))
    step_request = step.get("request") or {}
    request_refs = step.get("request_refs") or {}
    request = resolve_request_refs(dict(step_request or {}), dict(request_refs or {}), previous_steps)
    if on_request is not None:
        on_request(sanitize_for_json(request))
    payload = {
        "tool": step["tool"],
        "task": task,
        "request": request,
        "runtime": tool_runtime(profile, models, step["tool"], run_dir),
        "context": {"previous_steps": sanitize_for_json(previous_steps)},
    }
    result = call_tool_script(step["tool"], payload, models)
    return {"step": step, "request": request, "payload": payload, "result": result}


def run_control_tool(name: str, request: dict[str, Any], profile: dict[str, Any], models: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    payload = {
        "tool": name,
        "task": request.get("task") or {},
        "request": request,
        "runtime": tool_runtime(profile, models, name, run_dir),
        "context": request.get("context") or {},
    }
    result = call_tool_script(name, payload, models)
    if not result.get("ok"):
        error = result.get("error") or {}
        raise RuntimeError("%s failed: %s" % (name, error.get("message") or error))
    return result
