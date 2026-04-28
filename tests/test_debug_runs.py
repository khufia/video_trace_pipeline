import json

from video_trace_pipeline.common import append_jsonl, ensure_dir, write_json
from video_trace_pipeline.renderers import build_run_debug_payload, write_run_debug_bundle


def test_write_run_debug_bundle_writes_json_only_payloads(tmp_path):
    run_dir = tmp_path / "runs" / "video1" / "20260423T000000Z_test"
    ensure_dir(run_dir / "round_01" / "tools" / "01_generic_purpose")
    ensure_dir(run_dir / "evidence")
    ensure_dir(run_dir / "debug")

    write_json(
        run_dir / "run_manifest.json",
        {
            "benchmark": "videomathqa",
            "sample_key": "sample1",
            "video_id": "video1",
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
        run_dir / "final_result.json",
        {
            "audit_report": {"verdict": "FAIL", "feedback": "Need a grounded whole-figure count."},
            "rounds_executed": 1,
        },
    )
    write_json(
        run_dir / "trace_package.json",
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
    append_jsonl(
        run_dir / "evidence" / "evidence_index.jsonl",
        [{"evidence_id": "ev_1", "tool_name": "generic_purpose", "evidence_text": "Three squares.", "observation_ids": []}],
    )
    append_jsonl(
        run_dir / "evidence" / "atomic_observations.jsonl",
        [{"observation_id": "obs_1", "source_tool": "generic_purpose", "subject": "figure", "predicate": "shows"}],
    )
    write_json(
        run_dir / "round_01" / "planner_request.json",
        {"system_prompt": "planner", "user_prompt": "plan the task"},
    )
    write_json(
        run_dir / "round_01" / "planner_plan.json",
        {
            "strategy": "Inspect the final figure.",
            "steps": [
                {
                    "step_id": 1,
                    "tool_name": "generic_purpose",
                    "purpose": "Count the triangles.",
                    "inputs": {"query": "count triangles"},
                }
            ],
            "refinement_instructions": "",
        },
    )
    write_json(
        run_dir / "round_01" / "synthesizer_request.json",
        {"system_prompt": "synth", "user_prompt": "write the trace"},
    )
    write_json(
        run_dir / "round_01" / "synthesizer_trace_package.json",
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
    write_json(
        run_dir / "round_01" / "auditor_request.json",
        {"system_prompt": "audit", "user_prompt": "audit the trace"},
    )
    write_json(
        run_dir / "round_01" / "auditor_report.json",
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
    write_json(
        run_dir / "round_01" / "tools" / "01_generic_purpose" / "request.json",
        {"tool_name": "generic_purpose", "query": "count triangles", "frames": [{"timestamp_s": 99.0}]},
    )
    write_json(
        run_dir / "round_01" / "tools" / "01_generic_purpose" / "result.json",
        {
            "tool_name": "generic_purpose",
            "ok": True,
            "cache_hit": False,
            "data": {"answer": "The figure has three squares."},
            "metadata": {"observation_ids": ["obs_1"]},
        },
    )
    write_json(run_dir / "round_01" / "tools" / "01_generic_purpose" / "timing.json", {"execution_mode": "executed", "duration_s": 3.5})
    write_json(run_dir / "round_01" / "tools" / "01_generic_purpose" / "runtime.json", {"model_name": "gpt-5.4"})
    write_json(run_dir / "round_01" / "tools" / "01_generic_purpose" / "artifact_refs.json", [])
    write_json(run_dir / "round_01" / "tools" / "01_generic_purpose" / "observations.json", [{"observation_id": "obs_1"}])

    payload = build_run_debug_payload(run_dir)
    assert payload["result"]["latest_audit_verdict"] == "FAIL"
    assert payload["tool_steps"][0]["tool_name"] == "generic_purpose"
    assert payload["rounds"][0]["trace_inference_steps"][0]["time_start_s"] == 12.0

    overview_path = write_run_debug_bundle(run_dir)

    assert overview_path.exists()
    overview = json.loads(overview_path.read_text(encoding="utf-8"))
    assert overview["evidence_summary"]["evidence_entry_count"] == 1
    assert overview["rounds"][0]["planner_plan_path"] == "round_01/planner_plan.json"
    assert overview["tool_steps"][0]["request_path"] == "round_01/tools/01_generic_purpose/request.json"
    assert (run_dir / "debug" / "tool_steps.json").exists()
    assert (run_dir / "debug" / "rounds.json").exists()
