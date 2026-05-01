from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .common import write_json
from .config import enabled_plan_tools, load_models, load_profile, redacted_runtime
from .outputs import build_final_result, make_output_dir, round_dir, step_dir, write_summary
from .plan_verify import verify_plan
from .preprocess import preprocess
from .task import load_task
from .task_state import initial_state, update_state
from .tool_runner import run_control_tool, run_tool_step

try:
    from rich.console import Console
except Exception:  # pragma: no cover - rich is optional at import time
    Console = None

_CONSOLE: Any = None


def _display_text(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    _ = limit
    return text


def _progress_enabled(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "no_show_progress", False))


def _color_enabled(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "no_color", False)):
        return False
    explicit = os.environ.get("SIMPLE_PIPELINE_COLOR")
    if explicit is not None:
        return str(explicit).strip() != "0"
    if os.environ.get("NO_COLOR"):
        return False
    return True


def _console(args: argparse.Namespace):
    global _CONSOLE
    if Console is None:
        return None
    if _CONSOLE is None:
        _CONSOLE = Console(force_terminal=_color_enabled(args), color_system="truecolor", soft_wrap=True)
    return _CONSOLE


def _log(args: argparse.Namespace, message: str = "", style: str | None = None) -> None:
    if _progress_enabled(args):
        console = _console(args) if _color_enabled(args) else None
        if console is not None:
            console.print(message, style=style, markup=False)
        else:
            print(message, flush=True)


def _format_seconds(value: Any) -> str:
    if value is None:
        return "?"
    try:
        text = "%.2f" % float(value)
    except Exception:
        return str(value)
    text = text.rstrip("0").rstrip(".")
    return "%ss" % text


def _format_time_range(value: dict[str, Any]) -> str:
    if value.get("start_s") is not None or value.get("end_s") is not None:
        return "%s-%s" % (_format_seconds(value.get("start_s")), _format_seconds(value.get("end_s")))
    if value.get("timestamp_s") is not None:
        return "@%s" % _format_seconds(value.get("timestamp_s"))
    return "time=?"


def _format_clips(values: list[Any]) -> str:
    clips = [dict(item or {}) for item in values if isinstance(item, dict)]
    if not clips:
        return "none"
    return ", ".join(_format_time_range(item) for item in clips)


def _format_anchors(values: list[Any]) -> str:
    anchors = [dict(item or {}) for item in values if isinstance(item, dict)]
    if not anchors:
        return "none"
    parts = []
    for anchor in anchors:
        time_value = anchor.get("time_s") if anchor.get("time_s") is not None else anchor.get("timecode")
        radius = anchor.get("radius_s")
        reference = anchor.get("reference") or "video"
        suffix = ""
        if radius is not None:
            suffix = " radius=%s" % _format_seconds(radius)
        parts.append("%s ref=%s%s" % (_format_seconds(time_value) if isinstance(time_value, (int, float)) else str(time_value), reference, suffix))
    return ", ".join(parts)


def _format_media_value(key: str, value: Any) -> str:
    if isinstance(value, list):
        if key == "frames":
            stamps = [_format_seconds(item.get("timestamp_s")) for item in value if isinstance(item, dict) and item.get("timestamp_s") is not None]
            return "frames=%d%s" % (len(value), " @ " + ", ".join(stamps) if stamps else "")
        if key in {"segments", "captions", "transcript_segments"}:
            ranges = [_format_time_range(item) for item in value if isinstance(item, dict)]
            return "%s=%d%s" % (key, len(value), " | " + ", ".join(ranges) if ranges else "")
        if key == "texts":
            texts = [str(item) for item in value]
            return "texts=%d%s" % (len(value), " | " + " | ".join(texts) if texts else "")
        return "%s=%d" % (key, len(value))
    if isinstance(value, dict):
        return "%s=%d fields" % (key, len(value))
    return "%s=%s" % (key, value)


def _log_request_summary(args: argparse.Namespace, request: dict[str, Any]) -> None:
    query = str(request.get("query") or "").strip()
    _log(args, "request: query=%s" % (query or "<empty>"), "cyan")
    temporal_scope = dict(request.get("temporal_scope") or {})
    clips = list(temporal_scope.get("clips") or [])
    anchors = list(temporal_scope.get("anchors") or [])
    if clips or anchors:
        parts = []
        if clips:
            parts.append("clips: %s" % _format_clips(clips))
        if anchors:
            parts.append("anchors: %s" % _format_anchors(anchors))
        _log(args, "  temporal: %s" % " | ".join(parts))
    media = dict(request.get("media") or {})
    if media:
        parts = [_format_media_value(str(key), value) for key, value in sorted(media.items())]
        _log(args, "  media: %s" % " | ".join(parts))
    options = dict(request.get("options") or {})
    if options:
        parts = ["%s=%s" % (key, value) for key, value in sorted(options.items())]
        _log(args, "  options: %s" % " | ".join(parts))


def _log_segments(args: argparse.Namespace, output: dict[str, Any]) -> None:
    segments = [dict(item or {}) for item in list(output.get("segments") or []) if isinstance(item, dict)]
    _log(args, "output: segments=%d" % len(segments), "green")
    for index, segment in enumerate(segments, start=1):
        details = [_format_time_range(segment)]
        if segment.get("label"):
            details.append("label=%s" % segment.get("label"))
        if segment.get("confidence") is not None:
            details.append("confidence=%s" % segment.get("confidence"))
        _log(args, "  segment %d: %s" % (index, " | ".join(details)))
        if segment.get("summary"):
            _log(args, "    summary: %s" % segment.get("summary"))
        elif segment.get("rationale"):
            _log(args, "    summary: %s" % segment.get("rationale"))
    if output.get("summary"):
        _log(args, "  summary: %s" % output.get("summary"))


def _log_tool_output_summary(args: argparse.Namespace, tool: str, output: dict[str, Any]) -> None:
    if tool in {"visual_temporal_grounder", "audio_temporal_grounder"}:
        _log_segments(args, output)
        return
    if tool == "frame_retriever":
        frames = [dict(item or {}) for item in list(output.get("frames") or []) if isinstance(item, dict)]
        _log(args, "output: frames=%d" % len(frames), "green")
        for index, frame in enumerate(frames, start=1):
            path = frame.get("relpath") or frame.get("frame_path") or ""
            _log(args, "  frame %d: %s | %s" % (index, _format_seconds(frame.get("timestamp_s")), path))
        return
    if tool == "ocr":
        _log(args, "output: text=%s" % (output.get("text") or "<empty>"), "green")
        lines = [dict(item or {}) for item in list(output.get("lines") or []) if isinstance(item, dict)]
        if lines:
            _log(args, "  lines=%d" % len(lines))
            for index, line in enumerate(lines, start=1):
                confidence = "" if line.get("confidence") is None else " | confidence=%s" % line.get("confidence")
                _log(args, "  line %d: %s%s" % (index, line.get("text") or "", confidence))
        return
    if tool == "asr":
        spans = [dict(item or {}) for item in list(output.get("transcript_segments") or []) if isinstance(item, dict)]
        _log(args, "output: transcript_segments=%d" % len(spans), "green")
        for index, span in enumerate(spans, start=1):
            speaker = " | speaker=%s" % span.get("speaker") if span.get("speaker") else ""
            _log(args, "  span %d: %s%s | %s" % (index, _format_time_range(span), speaker, span.get("text") or ""))
        return
    if tool == "dense_captioner":
        captions = [dict(item or {}) for item in list(output.get("captions") or []) if isinstance(item, dict)]
        _log(args, "output: captions=%d" % len(captions), "green")
        for index, caption in enumerate(captions, start=1):
            _log(args, "  caption %d: %s | %s" % (index, _format_time_range(caption), caption.get("caption") or ""))
        return
    if tool == "spatial_grounder":
        regions = [dict(item or {}) for item in list(output.get("regions") or []) if isinstance(item, dict)]
        _log(args, "output: regions=%d" % len(regions), "green")
        for index, region in enumerate(regions, start=1):
            _log(args, "  region %d: label=%s | bbox=%s | confidence=%s" % (index, region.get("label"), region.get("bbox"), region.get("confidence")))
        if output.get("spatial_description"):
            _log(args, "  summary: %s" % output.get("spatial_description"))
        return
    if tool == "multimodal_reasoner":
        _log(args, "output: answer=%s | confidence=%s" % (output.get("answer") or "", output.get("confidence")), "green")
        if output.get("reasoning"):
            _log(args, "  reasoning: %s" % output.get("reasoning"))
        for index, item in enumerate(list(output.get("evidence") or []), start=1):
            _log(args, "  evidence %d: %s" % (index, item))
        return
    _log(args, "output: %s" % (_result_counts(output) or "no structured output"), "green")


def _result_counts(output: dict[str, Any]) -> str:
    parts = []
    for key in ("anchors", "captions", "clips", "evidence", "frames", "regions", "lines", "reads", "transcript_segments", "claim_results", "unresolved_gaps", "segments"):
        value = output.get(key)
        if isinstance(value, list):
            parts.append("%s=%d" % (key, len(value)))
    if output.get("answer"):
        parts.append("answer=present")
    if output.get("text"):
        parts.append("text_chars=%d" % len(str(output.get("text") or "")))
    if output.get("summary"):
        parts.append("summary=present")
    if output.get("reasoning"):
        parts.append("reasoning=present")
    return " | ".join(parts)


def _save_control_result(base: Path, name: str, request: dict[str, Any], result: dict[str, Any], payload_key: str) -> None:
    write_json(base / ("%s_request.json" % name), request)
    write_json(base / ("%s_result.json" % name), result)
    raw_text = str((result.get("metadata") or {}).get("raw_text") or "")
    if raw_text:
        (base / ("%s_raw.txt" % name)).write_text(raw_text, encoding="utf-8")
    if payload_key:
        write_json(base / ("%s.json" % payload_key), (result.get("output") or {}).get(payload_key) or (result.get("output") or {}))


def _fail_final(
    *,
    task: dict[str, Any],
    run_dir: Path,
    preprocess_result: dict[str, Any],
    previous_steps: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    latest_trace: dict[str, Any] | None,
    latest_audit: dict[str, Any] | None,
    state: dict[str, Any] | None,
    rounds_executed: int,
    exc: BaseException,
) -> dict[str, Any]:
    final = build_final_result(
        task,
        run_dir,
        preprocess_result,
        previous_steps,
        observations,
        latest_trace,
        latest_audit,
        state,
        rounds_executed,
        ok=False,
        error={"type": exc.__class__.__name__, "message": str(exc)},
    )
    write_summary(run_dir, final)
    return final


def run_task(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    models = load_models(args.models)
    task = load_task(args, profile)
    run_dir = make_output_dir(args.output_dir, profile, task)

    write_json(run_dir / "task.json", task)
    write_json(run_dir / "runtime.json", redacted_runtime(profile, models))

    _log(args, "")
    _log(args, "Run %s | benchmark=%s | max_rounds=%s" % (task.get("sample_key"), task.get("benchmark"), int(args.max_rounds)), "bold cyan")
    _log(args, "question: %s" % _display_text(task.get("question"), 260))
    if task.get("options"):
        _log(args, "options: %s" % " | ".join(str(item) for item in list(task.get("options") or [])))
    _log(args, "run_dir: %s" % run_dir)

    previous_steps: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    state = initial_state(task) if args.use_task_state else None
    latest_trace: dict[str, Any] | None = None
    latest_audit: dict[str, Any] | None = None
    preprocess_result: dict[str, Any] = {}
    rounds_executed = 0

    try:
        _log(args, "")
        _log(args, "Preprocess", "bold cyan")
        preprocess_result = preprocess(task, profile, models, run_dir, use_cache=not args.no_preprocess_cache)
        write_json(run_dir / "preprocess.json", preprocess_result)
        _log(
            args,
            "preprocess ok=%s | duration_s=%s | segments=%d | artifacts=%d"
            % (
                bool(preprocess_result.get("ok")),
                preprocess_result.get("video_duration_s"),
                len(list(preprocess_result.get("segments") or [])),
                len(list(preprocess_result.get("artifacts") or [])),
            ),
        )
        preprocess_metadata = dict(preprocess_result.get("metadata") or {})
        if preprocess_metadata.get("source") or preprocess_metadata.get("cache_path"):
            _log(
                args,
                "preprocess source=%s | cache=%s"
                % (
                    preprocess_metadata.get("source") or "unknown",
                    preprocess_metadata.get("cache_path") or "",
                ),
            )
        preprocess_warnings = [str(item) for item in list(preprocess_metadata.get("warnings") or []) if str(item).strip()]
        if preprocess_warnings:
            _log(args, "preprocess warnings: %s" % " | ".join(preprocess_warnings[:4]), "yellow")

        for round_idx in range(1, int(args.max_rounds) + 1):
            rounds_executed = round_idx
            rdir = round_dir(run_dir, round_idx)
            _log(args, "")
            _log(args, "Round %02d" % round_idx, "bold green")

            planner_request = {
                "task": task,
                "context": {
                    "preprocess": preprocess_result,
                    "previous_steps": previous_steps,
                    "latest_trace": latest_trace,
                    "latest_audit": latest_audit,
                    "task_state": state,
                    "available_tools": enabled_plan_tools(models),
                },
            }
            _log(args, "Planner start", "bold")
            planner_result = run_control_tool("planner", planner_request, profile, models, run_dir)
            plan = dict((planner_result.get("output") or {}).get("plan") or {})
            _save_control_result(rdir, "planner", planner_request, planner_result, "plan")
            _log(args, "Planner strategy: %s" % _display_text(plan.get("strategy"), 260))
            for step in list(plan.get("steps") or []):
                _log(
                    args,
                    "  step %s | %s | %s"
                    % (
                        step.get("id"),
                        step.get("tool"),
                        _display_text(step.get("purpose"), 180),
                    ),
                )

            plan_errors = verify_plan(plan, enabled_plan_tools(models), previous_steps)
            if plan_errors:
                _log(args, "Planner repair start | errors=%s" % "; ".join(plan_errors), "bold yellow")
                repair_request = {
                    "task": task,
                    "context": {
                        **planner_request["context"],
                        "plan_errors": plan_errors,
                        "rejected_plan": plan,
                    },
                }
                planner_result = run_control_tool("planner", repair_request, profile, models, run_dir)
                plan = dict((planner_result.get("output") or {}).get("plan") or {})
                _save_control_result(rdir, "planner_repair", repair_request, planner_result, "plan")
                plan_errors = verify_plan(plan, enabled_plan_tools(models), previous_steps)
                _log(args, "Planner repaired strategy: %s" % _display_text(plan.get("strategy"), 260))
            if plan_errors:
                write_json(rdir / "plan_errors.json", plan_errors)
                raise ValueError("Planner returned invalid plan: %s" % "; ".join(plan_errors))

            if args.dry_run_plan:
                _log(args, "Dry-run plan requested; stopping after planner.", "yellow")
                latest_trace = {
                    "answer": "",
                    "confidence": None,
                    "reasoning": "Dry-run stopped after planner.",
                    "evidence": [],
                    "trace_steps": [],
                    "open_questions": [],
                }
                latest_audit = {"verdict": "DRY_RUN", "confidence": None, "findings": [], "missing_information": [], "feedback": ""}
                break

            for step_index, step in enumerate(list(plan.get("steps") or []), start=1):
                step = dict(step)
                _log(args, "")
                _log(args, "Tool %02d start | %s | id=%s" % (step_index, step.get("tool"), step.get("id")), "bold magenta")
                _log(args, "purpose: %s" % _display_text(step.get("purpose"), 240))
                record = run_tool_step(
                    step,
                    task,
                    preprocess_result,
                    previous_steps,
                    profile,
                    models,
                    run_dir,
                    on_request=lambda request: _log_request_summary(args, request),
                )
                record["round"] = round_idx
                record["observations"] = []
                tdir = step_dir(run_dir, round_idx, step_index, step["tool"])
                write_json(tdir / "request.json", record["request"])
                write_json(tdir / "payload.json", record["payload"])
                write_json(tdir / "result.json", record["result"])
                result = dict(record.get("result") or {})
                output = dict(result.get("output") or {})
                status = "ok" if result.get("ok") else "error"
                _log(args, "Tool %02d end | %s | %s" % (step_index, status, _result_counts(output) or "no structured output"), "green" if status == "ok" else "red")
                _log_tool_output_summary(args, str(step.get("tool") or ""), output)
                if result.get("error"):
                    error = dict(result.get("error") or {})
                    _log(args, "tool error: %s | %s" % (error.get("type"), error.get("message")), "red")
                previous_steps.append(record)
            write_json(rdir / "observations.json", [])

            _log(args, "")
            _log(args, "Synthesizer start", "bold")
            synth_request = {
                "task": task,
                "context": {
                    "preprocess": preprocess_result,
                    "previous_steps": previous_steps,
                    "plan": plan,
                    "latest_trace": latest_trace,
                    "latest_audit": latest_audit,
                    "observations": observations,
                    "task_state": state,
                },
            }
            synth_result = run_control_tool("synthesizer", synth_request, profile, models, run_dir)
            latest_trace = dict((synth_result.get("output") or {}).get("trace") or {})
            _save_control_result(rdir, "synthesizer", synth_request, synth_result, "trace")
            _log(
                args,
                "Synthesizer end | answer=%s | confidence=%s | reason=%s"
                % (
                    _display_text(latest_trace.get("answer"), 120),
                    latest_trace.get("confidence"),
                    _display_text(latest_trace.get("reasoning"), 240),
                ),
            )

            _log(args, "Auditor start", "bold")
            audit_request = {
                "task": task,
                "context": {
                    "trace": latest_trace,
                    "preprocess": preprocess_result,
                    "previous_steps": previous_steps,
                    "observations": observations,
                    "task_state": state,
                },
            }
            audit_result = run_control_tool("auditor", audit_request, profile, models, run_dir)
            latest_audit = dict((audit_result.get("output") or {}).get("audit") or {})
            _save_control_result(rdir, "auditor", audit_request, audit_result, "audit")
            _log(
                args,
                "Auditor end | verdict=%s | confidence=%s | feedback=%s"
                % (
                    latest_audit.get("verdict"),
                    latest_audit.get("confidence"),
                    _display_text(latest_audit.get("feedback"), 260),
                ),
            )
            missing = list(latest_audit.get("missing_information") or [])
            if missing:
                _log(args, "missing: %s" % " | ".join(_display_text(item, 120) for item in missing[:6]))

            if args.use_task_state:
                state = update_state(task, state or {}, previous_steps, observations, latest_trace, latest_audit)
                write_json(run_dir / "task_state.json", state)
                _log(
                    args,
                    "Task state updated | facts=%d | open_questions=%d | failures=%d"
                    % (
                        len(list((state or {}).get("known_facts") or [])),
                        len(list((state or {}).get("open_questions") or [])),
                        len(list((state or {}).get("tool_failures") or [])),
                    ),
                )

            if str(latest_audit.get("verdict") or "").upper() == "PASS":
                _log(args, "Stopping: auditor PASS", "bold green")
                break

        final = build_final_result(
            task,
            run_dir,
            preprocess_result,
            previous_steps,
            observations,
            latest_trace,
            latest_audit,
            state,
            rounds_executed,
            ok=True,
        )
        write_summary(run_dir, final)
        _log(args, "")
        _log(
            args,
            "Final | ok=%s | answer=%s | audit=%s | rounds=%s"
            % (final.get("ok"), _display_text(final.get("final_answer"), 160), final.get("audit_verdict"), final.get("rounds_executed")),
            "bold green" if final.get("ok") else "bold red",
        )
        _log(args, "final_result: %s" % (run_dir / "final_result.json"))
        return final
    except BaseException as exc:
        _log(args, "")
        _log(args, "Run failed | %s: %s" % (exc.__class__.__name__, _display_text(str(exc), 360)), "bold red")
        return _fail_final(
            task=task,
            run_dir=run_dir,
            preprocess_result=preprocess_result,
            previous_steps=previous_steps,
            observations=observations,
            latest_trace=latest_trace,
            latest_audit=latest_audit,
            state=state,
            rounds_executed=rounds_executed,
            exc=exc,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple side-by-side video trace pipeline")
    parser.add_argument("--profile", required=True, help="Machine profile YAML")
    parser.add_argument("--models", default="video_trace_pipeline_simple/configs/models.yaml", help="Simple models YAML")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--options-json", default=None)
    parser.add_argument("--sample-key", default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--gold-answer", default=None)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--use-task-state", action="store_true")
    parser.add_argument("--no-preprocess-cache", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--no-show-progress", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    return parser


def main() -> None:
    final = run_task(build_arg_parser().parse_args())
    write_json(Path(final["run_dir"]) / "final_result.json", final)
    print(final["run_dir"])


if __name__ == "__main__":
    main()
