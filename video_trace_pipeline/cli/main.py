from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ..benchmarks import get_benchmark_adapter
from ..config import load_machine_profile, load_models_config
from ..orchestration import PipelineRunner
from ..renderers import export_trace_for_benchmark
from ..schemas import TaskSpec, TracePackage
from ..storage import EvidenceLedger, RunContext, WorkspaceManager

app = typer.Typer(help="Video Trace Pipeline CLI")
console = Console()


def _load_runner(profile: str, models: str, workspace_root: Optional[str] = None) -> PipelineRunner:
    machine_profile = load_machine_profile(profile, workspace_root=workspace_root)
    models_config = load_models_config(models)
    return PipelineRunner(machine_profile, models_config)


def _load_tasks(runner: PipelineRunner, benchmark: str, index: Optional[int], limit: Optional[int]):
    if benchmark not in runner.profile.datasets:
        raise typer.BadParameter("Benchmark %s is not configured in the machine profile" % benchmark)
    adapter = get_benchmark_adapter(benchmark, runner.profile.datasets[benchmark])
    return adapter.select(index=index, limit=limit)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@app.command()
def preprocess(
    profile: str = typer.Option(..., help="Machine profile YAML"),
    models: str = typer.Option("configs/models.yaml", help="Models config YAML"),
    benchmark: str = typer.Option(..., help="Benchmark name"),
    index: Optional[int] = typer.Option(None, help="Single sample index"),
    limit: Optional[int] = typer.Option(None, help="Limit number of tasks"),
    clip_duration: float = typer.Option(30.0, help="Dense-caption clip duration in seconds"),
    workspace_root: Optional[str] = typer.Option(None, help="Override workspace root"),
):
    runner = _load_runner(profile=profile, models=models, workspace_root=workspace_root)
    try:
        tasks = _load_tasks(runner, benchmark=benchmark, index=index, limit=limit)
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
    benchmark: str = typer.Option(..., help="Benchmark name"),
    mode: str = typer.Option("generate", help="generate or refine"),
    index: Optional[int] = typer.Option(None, help="Single sample index"),
    limit: Optional[int] = typer.Option(None, help="Limit number of tasks"),
    clip_duration: float = typer.Option(30.0, help="Dense-caption clip duration in seconds"),
    max_rounds: int = typer.Option(2, help="Maximum generation/refinement rounds"),
    initial_trace_path: Optional[str] = typer.Option(None, help="Optional initial trace package JSON"),
    workspace_root: Optional[str] = typer.Option(None, help="Override workspace root"),
):
    runner = _load_runner(profile=profile, models=models, workspace_root=workspace_root)
    try:
        tasks = _load_tasks(runner, benchmark=benchmark, index=index, limit=limit)
        for task in tasks:
            result = runner.run_task(
                task=task,
                mode=mode,
                max_rounds=max_rounds,
                clip_duration_s=clip_duration,
                initial_trace_path=initial_trace_path,
            )
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


def main():
    app()


if __name__ == "__main__":
    main()
