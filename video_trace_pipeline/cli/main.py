from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from ..benchmarks import get_benchmark_adapter
from ..common import sanitize_path_component, short_hash
from ..config import load_machine_profile, load_models_config
from ..diagnostics import dataset_report, model_report, package_report, summarize_status
from ..orchestration import PipelineRunner
from ..renderers import export_trace_for_benchmark, write_run_debug_bundle
from ..schemas import TaskSpec, TracePackage
from ..storage import EvidenceLedger, RunContext, WorkspaceManager
from ..tool_wrappers.persistent_pool import normalize_persist_tool_name
from .progress import LiveRunReporter

app = typer.Typer(help="Video Trace Pipeline CLI")
console = Console()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_runner(
    profile: str,
    models: str,
    workspace_root: Optional[str] = None,
    persist_tool_models: Optional[List[str]] = None,
    preload_persisted_models: bool = False,
) -> PipelineRunner:
    machine_profile = load_machine_profile(profile, workspace_root=workspace_root)
    models_config = load_models_config(models)
    return PipelineRunner(
        machine_profile,
        models_config,
        persist_tool_models=persist_tool_models,
        preload_persisted_models=preload_persisted_models,
    )


def _parse_options(options_text: Optional[str]) -> List[str]:
    raw = str(options_text or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, dict):
        return [str(value).strip() for _, value in sorted(parsed.items()) if str(value).strip()]
    if "||" in raw:
        return [item.strip() for item in raw.split("||") if item.strip()]
    return [raw]


def _parse_steps(steps_text: Optional[str]) -> Optional[List[str]]:
    raw = str(steps_text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        values = [str(item).strip() for item in parsed if str(item).strip()]
        return values or None
    if isinstance(parsed, dict):
        values = [str(value).strip() for _, value in sorted(parsed.items()) if str(value).strip()]
        return values or None
    if "||" in raw:
        values = [item.strip() for item in raw.split("||") if item.strip()]
        return values or None
    return [raw]


def _parse_tool_names(tool_names_text: Optional[str]) -> List[str]:
    raw = str(tool_names_text or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        values = [str(item).strip() for item in parsed if str(item).strip()]
    elif isinstance(parsed, dict):
        values = [str(value).strip() for _, value in sorted(parsed.items()) if str(value).strip()]
    elif "||" in raw:
        values = [item.strip() for item in raw.split("||") if item.strip()]
    else:
        values = [raw]
    normalized = []
    seen = set()
    for item in values:
        name = normalize_persist_tool_name(item)
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def _read_inputs_sample(inputs_json: str, input_index: int) -> dict:
    path = Path(inputs_json).expanduser().resolve()
    if not path.exists():
        raise typer.BadParameter("Inputs JSON path does not exist: %s" % path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise typer.BadParameter("Expected a JSON list in inputs file: %s" % path)
    if input_index < 0 or input_index >= len(payload):
        raise typer.BadParameter("input_index %s out of range for %s samples in %s" % (input_index, len(payload), path))
    sample = payload[input_index]
    if not isinstance(sample, dict):
        raise typer.BadParameter("Sample at index %s is not a JSON object in %s" % (input_index, path))
    return sample


def _build_direct_task(
    video_path: str,
    question: str,
    benchmark: Optional[str] = None,
    options_text: Optional[str] = None,
    sample_key: Optional[str] = None,
    video_id: Optional[str] = None,
    question_id: Optional[str] = None,
    gold_answer: Optional[str] = None,
    initial_trace_steps_text: Optional[str] = None,
) -> TaskSpec:
    video_path_obj = Path(video_path).expanduser().resolve()
    if not video_path_obj.exists():
        raise typer.BadParameter("Video path does not exist: %s" % video_path_obj)
    question_text = str(question or "").strip()
    if not question_text:
        raise typer.BadParameter("`question` is required when using `video_path`.")
    resolved_video_id = str(video_id or video_path_obj.stem).strip() or video_path_obj.stem
    resolved_sample_key = str(sample_key or "").strip()
    if not resolved_sample_key:
        resolved_sample_key = "%s__%s" % (
            sanitize_path_component(video_path_obj.stem or "video"),
            short_hash(question_text, 12),
        )
    return TaskSpec(
        benchmark=str(benchmark or "adhoc").strip() or "adhoc",
        sample_key=resolved_sample_key,
        question=question_text,
        options=_parse_options(options_text),
        video_path=str(video_path_obj),
        video_id=resolved_video_id,
        question_id=str(question_id).strip() if question_id is not None and str(question_id).strip() else None,
        gold_answer=str(gold_answer).strip() if gold_answer is not None and str(gold_answer).strip() else None,
        initial_trace_steps=_parse_steps(initial_trace_steps_text),
        metadata={"source": "direct_cli"},
    )


def _load_tasks(
    runner: PipelineRunner,
    benchmark: Optional[str],
    index: Optional[int],
    limit: Optional[int],
    video_path: Optional[str] = None,
    question: Optional[str] = None,
    options_text: Optional[str] = None,
    sample_key: Optional[str] = None,
    video_id: Optional[str] = None,
    question_id: Optional[str] = None,
    gold_answer: Optional[str] = None,
    initial_trace_steps_text: Optional[str] = None,
    inputs_json: Optional[str] = None,
    input_index: Optional[int] = None,
):
    if str(inputs_json or "").strip():
        if input_index is None:
            raise typer.BadParameter("`input_index` is required when using `inputs_json`.")
        sample = _read_inputs_sample(inputs_json, input_index)
        return [
            _build_direct_task(
                video_path=str(sample.get("video_path") or ""),
                question=str(sample.get("question") or ""),
                benchmark=benchmark,
                options_text=json.dumps(sample.get("options") or [], ensure_ascii=False),
                sample_key=sample_key,
                video_id=video_id or sample.get("video_id"),
                question_id=question_id or sample.get("question_id"),
                gold_answer=gold_answer or sample.get("answer"),
                initial_trace_steps_text=json.dumps(sample.get("initial_trace_steps") or [], ensure_ascii=False),
            )
        ]
    if str(video_path or "").strip():
        return [
            _build_direct_task(
                video_path=video_path,
                question=question or "",
                benchmark=benchmark,
                options_text=options_text,
                sample_key=sample_key,
                video_id=video_id,
                question_id=question_id,
                gold_answer=gold_answer,
                initial_trace_steps_text=initial_trace_steps_text,
            )
        ]
    if not benchmark:
        raise typer.BadParameter("Either `benchmark` or `video_path` must be provided.")
    if benchmark not in runner.profile.datasets:
        raise typer.BadParameter("Benchmark %s is not configured in the machine profile" % benchmark)
    adapter = get_benchmark_adapter(benchmark, runner.profile.datasets[benchmark])
    return adapter.select(index=index, limit=limit)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _print_section(title: str):
    console.print("")
    console.print("[bold]%s[/bold]" % title)


def _display_model_status(item):
    environment_status = str(item.get("status"))
    plan_status = str(item.get("plan_status") or "")
    if environment_status not in {"ok", "configured", "disabled"}:
        return environment_status
    if plan_status and plan_status not in {"planned", "supported_extension", "custom_extension", "not_checked"}:
        return plan_status
    return environment_status


def _status_style(status: str) -> str:
    if status in {"ok", "configured", "disabled"}:
        return "green"
    if status in {
        "model_mismatch",
        "backend_mismatch",
        "implementation_mismatch",
        "extension_backend_mismatch",
    }:
        return "yellow"
    return "red"


@app.command("check-env")
def check_env(
    profile: str = typer.Option(..., help="Machine profile YAML"),
    models: str = typer.Option("configs/models.yaml", help="Models config YAML"),
    benchmark: Optional[str] = typer.Option(None, help="Optional benchmark name to validate"),
    workspace_root: Optional[str] = typer.Option(None, help="Override workspace root"),
    include_whisperx: bool = typer.Option(True, help="Check WhisperX extra requirements"),
    include_vllm: bool = typer.Option(False, help="Check optional vLLM requirements"),
    strict: bool = typer.Option(False, help="Exit non-zero if anything important is missing"),
):
    machine_profile = load_machine_profile(profile, workspace_root=workspace_root)
    models_config = load_models_config(models)

    requirement_files = [REPO_ROOT / "requirements.txt", REPO_ROOT / "requirements-local-tools.txt"]
    if include_whisperx:
        requirement_files.append(REPO_ROOT / "requirements-whisperx.txt")
    if include_vllm:
        requirement_files.append(REPO_ROOT / "requirements-vllm.txt")

    packages = package_report(requirement_files, optional_packages={"pytest", "vllm"})
    datasets = dataset_report(machine_profile, benchmark=benchmark)
    models_payload = model_report(machine_profile, models_config)

    _print_section("Packages")
    for item in packages:
        if item["requirement"] is None:
            console.print("[red]missing file[/red]", item["requirement_file"])
            continue
        status_style = (
            "green"
            if item["status"] in {"ok", "optional_missing", "optional_version_mismatch"}
            else "yellow"
            if item["status"] == "version_mismatch"
            else "red"
        )
        console.print(
            "[%s]%s[/%s] %s installed=%s"
            % (
                status_style,
                item["status"],
                status_style,
                item["requirement"],
                item["installed_version"] or "missing",
            )
        )

    _print_section("Datasets")
    for item in datasets:
        status_style = "green" if item["status"] == "ok" else "red"
        console.print("[%s]%s[/%s] %s" % (status_style, item["status"], status_style, item["benchmark"]))
        if item.get("annotations"):
            console.print("  annotations=%s" % item["annotations"])
        if item.get("videos_dir"):
            console.print("  videos_dir=%s" % item["videos_dir"])

    _print_section("Models And Endpoints")
    for item in models_payload:
        display_status = _display_model_status(item)
        status_style = _status_style(display_status)
        label = "%s:%s" % (item["kind"], item["name"])
        if item["kind"] == "tool":
            console.print("[%s]%s[/%s] %s implementation=%s model=%s" % (
                status_style,
                display_status,
                status_style,
                label,
                item.get("implementation"),
                item.get("model"),
            ))
        else:
            console.print("[%s]%s[/%s] %s backend=%s model=%s" % (
                status_style,
                display_status,
                status_style,
                label,
                item.get("backend"),
                item.get("model"),
            ))
        if display_status != item["status"]:
            console.print("  environment_status=%s" % item["status"])
        if item.get("resolved_path"):
            console.print("  resolved_model_path=%s" % item["resolved_path"])
        if item.get("model_resolution_status"):
            console.print("  model_cache_status=%s" % item["model_resolution_status"])
        if item.get("plan_status"):
            console.print("  plan_alignment=%s" % item["plan_status"])
        if item.get("expected_backend"):
            console.print("  expected_backend=%s" % item["expected_backend"])
        if item.get("expected_implementation"):
            console.print("  expected_implementation=%s" % item["expected_implementation"])
        if item.get("expected_model"):
            console.print("  expected_model=%s" % item["expected_model"])
        for aux_model in item.get("auxiliary_models") or []:
            console.print(
                "  auxiliary_model[%s]=%s status=%s%s"
                % (
                    aux_model.get("field"),
                    aux_model.get("requested_model") or aux_model.get("normalized_model"),
                    aux_model.get("status"),
                    " path=%s" % aux_model["resolved_path"] if aux_model.get("resolved_path") else "",
                )
            )
        if item.get("endpoint"):
            console.print("  endpoint=%s" % item["endpoint"])
        if item.get("wrapper_status"):
            console.print("  wrapper_status=%s" % item["wrapper_status"])
        if item.get("module"):
            console.print("  wrapper_module=%s" % item["module"])

    packages_ok = summarize_status(packages, ok_statuses={"ok", "optional_missing", "optional_version_mismatch"}) == "ok"
    datasets_ok = summarize_status(datasets, ok_statuses={"ok"}) == "ok"
    environment_ok = summarize_status(models_payload) == "ok"
    plan_ok = summarize_status(
        models_payload,
        ok_statuses={"planned", "supported_extension", "custom_extension", "not_checked"},
        status_field="plan_status",
    ) == "ok"
    execution_ready = packages_ok and datasets_ok and environment_ok
    overall = execution_ready and plan_ok

    _print_section("Summary")
    console.print("packages=%s" % ("ok" if packages_ok else "needs_attention"))
    console.print("datasets=%s" % ("ok" if datasets_ok else "needs_attention"))
    console.print("environment=%s" % ("ok" if environment_ok else "needs_attention"))
    console.print("execution_ready=%s" % ("ok" if execution_ready else "needs_attention"))
    console.print("plan_alignment=%s" % ("ok" if plan_ok else "needs_attention"))
    if strict and not execution_ready:
        raise typer.Exit(code=1)


@app.command()
def preprocess(
    profile: str = typer.Option(..., help="Machine profile YAML"),
    models: str = typer.Option("configs/models.yaml", help="Models config YAML"),
    benchmark: Optional[str] = typer.Option(None, help="Benchmark name"),
    index: Optional[int] = typer.Option(None, help="Single sample index"),
    limit: Optional[int] = typer.Option(None, help="Limit number of tasks"),
    video_path: Optional[str] = typer.Option(None, help="Direct video path for an ad hoc run"),
    question: Optional[str] = typer.Option(None, help="Question for the direct video input"),
    options_json: Optional[str] = typer.Option(None, help="Options as a JSON list or `||`-separated string"),
    sample_key: Optional[str] = typer.Option(None, help="Optional sample key override for direct video input"),
    video_id: Optional[str] = typer.Option(None, help="Optional video id override for direct video input"),
    question_id: Optional[str] = typer.Option(None, help="Optional question id override for direct video input"),
    gold_answer: Optional[str] = typer.Option(None, help="Optional gold answer for direct video input"),
    initial_trace_steps_json: Optional[str] = typer.Option(None, help="Initial trace steps as JSON list or `||`-separated string"),
    inputs_json: Optional[str] = typer.Option(None, help="JSON file containing a list of direct-input samples"),
    input_index: Optional[int] = typer.Option(None, help="Sample index inside `inputs_json`"),
    clip_duration: Optional[float] = typer.Option(None, help="Dense-caption clip duration in seconds"),
    workspace_root: Optional[str] = typer.Option(None, help="Override workspace root"),
    persist_tool_models: Optional[str] = typer.Option(
        None,
        "--persist-tool-models",
        help="Tool names as a JSON list or `||`-separated string to keep loaded on GPU during this CLI run",
    ),
    preload_persisted_models: bool = typer.Option(
        False,
        "--preload-persisted-models/--no-preload-persisted-models",
        help="Eagerly load persisted tool models at startup, in parallel across distinct devices",
    ),
):
    runner = _load_runner(
        profile=profile,
        models=models,
        workspace_root=workspace_root,
        persist_tool_models=_parse_tool_names(persist_tool_models),
        preload_persisted_models=preload_persisted_models,
    )
    try:
        runner.preload_models()
        tasks = _load_tasks(
            runner,
            benchmark=benchmark,
            index=index,
            limit=limit,
            video_path=video_path,
            question=question,
            options_text=options_json,
            sample_key=sample_key,
            video_id=video_id,
            question_id=question_id,
            gold_answer=gold_answer,
            initial_trace_steps_text=initial_trace_steps_json,
            inputs_json=inputs_json,
            input_index=input_index,
        )
        for task in tasks:
            output = runner.preprocessor.get_or_build(task, clip_duration_s=clip_duration)
            console.print(
                "[green]preprocess[/green]",
                task.sample_key,
                "cache_hit=%s" % output["cache_hit"],
                "cache_dir=%s" % output["cache_dir"],
            )
    finally:
        runner.close()


@app.command()
def run(
    profile: str = typer.Option(..., help="Machine profile YAML"),
    models: str = typer.Option("configs/models.yaml", help="Models config YAML"),
    benchmark: Optional[str] = typer.Option(None, help="Benchmark name"),
    mode: str = typer.Option("generate", help="generate or refine"),
    index: Optional[int] = typer.Option(None, help="Single sample index"),
    limit: Optional[int] = typer.Option(None, help="Limit number of tasks"),
    video_path: Optional[str] = typer.Option(None, help="Direct video path for an ad hoc run"),
    question: Optional[str] = typer.Option(None, help="Question for the direct video input"),
    options_json: Optional[str] = typer.Option(None, help="Options as a JSON list or `||`-separated string"),
    sample_key: Optional[str] = typer.Option(None, help="Optional sample key override for direct video input"),
    video_id: Optional[str] = typer.Option(None, help="Optional video id override for direct video input"),
    question_id: Optional[str] = typer.Option(None, help="Optional question id override for direct video input"),
    gold_answer: Optional[str] = typer.Option(None, help="Optional gold answer for direct video input"),
    initial_trace_steps_json: Optional[str] = typer.Option(None, help="Initial trace steps as JSON list or `||`-separated string"),
    inputs_json: Optional[str] = typer.Option(None, help="JSON file containing a list of direct-input samples"),
    input_index: Optional[int] = typer.Option(None, help="Sample index inside `inputs_json`"),
    clip_duration: Optional[float] = typer.Option(None, help="Dense-caption clip duration in seconds"),
    max_rounds: int = typer.Option(3, help="Maximum generation/refinement rounds"),
    initial_trace_path: Optional[str] = typer.Option(None, help="Optional initial trace package JSON"),
    results_name: Optional[str] = typer.Option(None, help="Optional repo-local results directory name"),
    workspace_root: Optional[str] = typer.Option(None, help="Override workspace root"),
    persist_tool_models: Optional[str] = typer.Option(
        None,
        "--persist-tool-models",
        help="Tool names as a JSON list or `||`-separated string to keep loaded on GPU during this CLI run",
    ),
    preload_persisted_models: bool = typer.Option(
        False,
        "--preload-persisted-models/--no-preload-persisted-models",
        help="Eagerly load persisted tool models at startup, in parallel across distinct devices",
    ),
    show_progress: bool = typer.Option(True, "--show-progress/--no-show-progress", help="Print live planner/tool/trace/auditor updates"),
):
    runner = _load_runner(
        profile=profile,
        models=models,
        workspace_root=workspace_root,
        persist_tool_models=_parse_tool_names(persist_tool_models),
        preload_persisted_models=preload_persisted_models,
    )
    try:
        progress_reporter = LiveRunReporter(console) if show_progress else None
        tasks = _load_tasks(
            runner,
            benchmark=benchmark,
            index=index,
            limit=limit,
            video_path=video_path,
            question=question,
            options_text=options_json,
            sample_key=sample_key,
            video_id=video_id,
            question_id=question_id,
            gold_answer=gold_answer,
            initial_trace_steps_text=initial_trace_steps_json,
            inputs_json=inputs_json,
            input_index=input_index,
        )
        for task in tasks:
            result = runner.run_task(
                task=task,
                mode=mode,
                max_rounds=max_rounds,
                clip_duration_s=clip_duration,
                initial_trace_path=initial_trace_path,
                results_name=results_name,
                progress_reporter=progress_reporter,
            )
            if result.get("exported_results_dir"):
                console.print(
                    "[green]run[/green]",
                    task.sample_key,
                    "->",
                    result["run_dir"],
                    "exported=",
                    result["exported_results_dir"],
                )
            else:
                console.print("[green]run[/green]", task.sample_key, "->", result["run_dir"])
    finally:
        runner.close()


@app.command()
def audit(
    profile: str = typer.Option(..., help="Machine profile YAML"),
    models: str = typer.Option("configs/models.yaml", help="Models config YAML"),
    run_dir: str = typer.Option(..., help="Run directory produced by `vtp run`"),
    workspace_root: Optional[str] = typer.Option(None, help="Override workspace root"),
):
    runner = _load_runner(profile=profile, models=models, workspace_root=workspace_root)
    try:
        run_path = Path(run_dir).expanduser().resolve()
        manifest = _read_json(run_path / "run_manifest.json")
        trace_package = _read_json(run_path / "trace" / "trace_package.json")
        task_payload = manifest["task"]
        task_payload["video_path"] = task_payload.get("video_path") or "<redacted>"
        task = TaskSpec.parse_obj(task_payload)
        evidence_ledger = EvidenceLedger(
            RunContext(
                workspace_root=run_path.parents[3],
                benchmark=manifest["benchmark"],
                sample_key=manifest["sample_key"],
                run_id=manifest["run_id"],
            )
        )
        evidence_summary = {
            "evidence_entry_count": len(evidence_ledger.entries()),
            "observation_count": len(evidence_ledger.observations()),
            "recent_observations": evidence_ledger.observations()[-20:],
        }
        raw, report = runner.auditor.audit(task, trace_package, evidence_summary)
        (run_path / "auditor" / "manual_audit_raw.txt").write_text(raw, encoding="utf-8")
        (run_path / "auditor" / "manual_audit_report.json").write_text(
            json.dumps(report.dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        console.print("[green]audit[/green]", run_path)
    finally:
        runner.close()


@app.command()
def export(
    run_dir: str = typer.Option(..., help="Run directory produced by `vtp run`"),
):
    run_path = Path(run_dir).expanduser().resolve()
    manifest = _read_json(run_path / "run_manifest.json")
    trace_package = _read_json(run_path / "trace" / "trace_package.json")
    task_payload = manifest["task"]
    task_payload["video_path"] = task_payload.get("video_path") or "<redacted>"
    task = TaskSpec.parse_obj(task_payload)
    export_payload = export_trace_for_benchmark(manifest["benchmark"], task, trace_package)
    output_path = run_path / "results" / "benchmark_export.json"
    output_path.write_text(json.dumps(export_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console.print("[green]export[/green]", output_path)


@app.command("debug-run")
def debug_run(
    run_dir: str = typer.Option(..., help="Run directory produced by `vtp run`"),
    output_dir: Optional[str] = typer.Option(
        None,
        help="Optional output directory for the debug bundle. Defaults to <run_dir>/debug",
    ),
):
    report_path = write_run_debug_bundle(run_dir, output_dir=output_dir)
    console.print("[green]debug-run[/green]", report_path)


def main():
    app()


if __name__ == "__main__":
    main()
