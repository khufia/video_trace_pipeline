import json

from video_trace_pipeline.common import append_jsonl, ensure_dir, write_json, write_text
from video_trace_pipeline.renderers import build_run_debug_payload, write_run_debug_bundle


def test_write_run_debug_bundle_creates_readable_report_and_csv(tmp_path):
    run_dir = tmp_path / "runs" / "videomathqa" / "sample1" / "20260423T000000Z_test"
    ensure_dir(run_dir / "planner")
    ensure_dir(run_dir / "synthesizer")
    ensure_dir(run_dir / "auditor")
    ensure_dir(run_dir / "tools" / "01_generic_purpose")
    ensure_dir(run_dir / "evidence")
    ensure_dir(run_dir / "trace")
    ensure_dir(run_dir / "results")

    write_json(
        run_dir / "run_manifest.json",
        {
            "benchmark": "videomathqa",
            "sample_key": "sample1",
            "run_id": "20260423T000000Z_test",
            "mode": "generate",
            "task": {
                "question": "How many triangles are shown?",
                "options": ["A. 24", "B. 28"],
                "gold_answer": "B",
                "video_id": "video1",
                "question_id": "267",
            },
        },
    )
    write_json(
        run_dir / "results" / "final_result.json",
        {
            "audit_report": {"verdict": "FAIL", "feedback": "Need a grounded whole-figure count."},
            "rounds_executed": 1,
        },
    )
    write_json(
        run_dir / "trace" / "trace_package.json",
        {
            "task_key": "sample1",
            "mode": "generate",
            "evidence_entries": [],
            "inference_steps": [
                {
                    "step_id": 1,
                    "text": "The figure has three squares.",
                    "supporting_observation_ids": [],
                    "time_start_s": 12.0,
                    "time_end_s": 15.0,
                }
            ],
            "final_answer": "",
            "benchmark_renderings": {},
            "metadata": {},
        },
    )
    write_text(run_dir / "trace" / "trace_readable.md", "# Trace\n")
    write_text(run_dir / "evidence" / "evidence_readable.md", "# Evidence\n")
    append_jsonl(
        run_dir / "evidence" / "evidence_index.jsonl",
        [{"evidence_id": "ev_1", "tool_name": "generic_purpose", "evidence_text": "Three squares.", "observation_ids": []}],
    )
    append_jsonl(
        run_dir / "evidence" / "atomic_observations.jsonl",
        [{"observation_id": "obs_1", "source_tool": "generic_purpose", "subject": "figure", "predicate": "shows"}],
    )
    write_json(
        run_dir / "planner" / "round_01_plan.json",
        {
            "strategy": "Inspect the final figure.",
            "steps": [
                {
                    "step_id": 1,
                    "tool_name": "generic_purpose",
                    "purpose": "Count the triangles.",
                    "arguments": {"query": "count triangles"},
                }
            ],
            "refinement_instructions": "",
        },
    )
    write_text(run_dir / "planner" / "round_01_raw.txt", "planner raw")
    write_json(
        run_dir / "synthesizer" / "round_01_trace_package.json",
        {
            "task_key": "sample1",
            "mode": "generate",
            "evidence_entries": [],
            "inference_steps": [
                {
                    "step_id": 1,
                    "text": "The figure has three squares.",
                    "supporting_observation_ids": [],
                    "time_start_s": 12.0,
                    "time_end_s": 15.0,
                }
            ],
            "final_answer": "",
            "benchmark_renderings": {},
            "metadata": {},
        },
    )
    write_text(run_dir / "synthesizer" / "round_01_raw.txt", "trace raw")
    write_json(
        run_dir / "auditor" / "round_01_report.json",
        {
            "verdict": "FAIL",
            "confidence": 0.9,
            "feedback": "Need a grounded whole-figure count.",
            "missing_information": ["Whole-figure total"],
            "findings": [
                {
                    "severity": "high",
                    "category": "grounding",
                    "message": "The answer only covers a partial case.",
                    "evidence_ids": ["ev_1"],
                }
            ],
        },
    )
    write_text(run_dir / "auditor" / "round_01_raw.txt", "audit raw")
    write_json(
        run_dir / "tools" / "01_generic_purpose" / "request_full.json",
        {"tool_name": "generic_purpose", "query": "count triangles", "frames": [{"timestamp_s": 99.0}]},
    )
    write_json(
        run_dir / "tools" / "01_generic_purpose" / "result.json",
        {
            "tool_name": "generic_purpose",
            "ok": True,
            "cache_hit": False,
            "data": {"answer": "The figure has three squares."},
            "metadata": {"observation_ids": ["obs_1"]},
        },
    )
    write_json(run_dir / "tools" / "01_generic_purpose" / "timing.json", {"execution_mode": "executed", "duration_s": 3.5})
    write_text(run_dir / "tools" / "01_generic_purpose" / "summary.md", "The model counted the single-square case only.")
    write_json(run_dir / "tools" / "01_generic_purpose" / "artifact_refs.json", {"artifact_refs": []})

    payload = build_run_debug_payload(run_dir)
    assert payload["result"]["latest_audit_verdict"] == "FAIL"
    assert payload["tool_steps"][0]["tool_name"] == "generic_purpose"
    assert payload["rounds"][0]["trace_inference_steps"][0]["time_start_s"] == 12.0

    report_path = write_run_debug_bundle(run_dir)

    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Run Debug Report" in report_text
    assert "How many triangles are shown?" in report_text
    assert "Round 01" in report_text
    assert "generic_purpose" in report_text
    assert "plan summary" in report_text
    assert "audit readable" in report_text

    csv_text = (run_dir / "debug" / "tool_steps.csv").read_text(encoding="utf-8")
    assert "tool_dir,tool_name,ok,cache_hit" in csv_text
    assert "01_generic_purpose,generic_purpose,True,False" in csv_text

    root_readme = (run_dir / "README.md").read_text(encoding="utf-8")
    assert "Run Overview" in root_readme
    assert "results/final_result_readable.md" in root_readme
    assert "[12s to 15s]" in root_readme
    assert (run_dir / "planner" / "round_01_summary.md").exists()
    assert (run_dir / "synthesizer" / "round_01_trace_readable.md").exists()
    assert (run_dir / "auditor" / "round_01_report_readable.md").exists()
    assert (run_dir / "results" / "final_result_readable.md").exists()
    assert "The answer only covers a partial case." in (run_dir / "auditor" / "round_01_report_readable.md").read_text(
        encoding="utf-8"
    )

    overview = json.loads((run_dir / "debug" / "overview.json").read_text(encoding="utf-8"))
    assert overview["evidence_summary"]["evidence_entry_count"] == 1
