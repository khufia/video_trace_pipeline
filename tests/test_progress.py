from rich.console import Console

from video_trace_pipeline.cli.progress import LiveRunReporter
from video_trace_pipeline.tool_wrappers.timelens_runner import _candidate_window_indices


def test_live_reporter_prints_confidence_summary_and_observation_confidence():
    console = Console(record=True, width=160)
    reporter = LiveRunReporter(console)

    reporter.on_tool_end(
        round_index=1,
        step_id=1,
        tool_name="visual_temporal_grounder",
        result_payload={
            "ok": True,
            "cache_hit": False,
            "summary": "The queried event appears at: 10.00s-15.00s.",
            "metadata": {
                "confidence": 0.91,
                "confidence_avg": 0.83,
                "confidence_count": 3,
            },
        },
        observations=[
            {
                "atomic_text": '"chart" is present from 10.00s to 15.00s.',
                "confidence": 0.91,
            }
        ],
        step_dir="workspace/runs/demo/tools/01_visual_temporal_grounder",
    )

    rendered = console.export_text()
    assert "confidence=max=0.9100 avg=0.8300 n=3" in rendered
    assert "[0.9100]" in rendered


def test_candidate_window_indices_expands_neighbors_and_caps_budget():
    windows = [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0), (180.0, 240.0)]
    scored_frames = [
        {"timestamp": 65.0, "relevance_score": 0.9},
        {"timestamp": 190.0, "relevance_score": 0.8},
    ]

    indices = _candidate_window_indices(
        scored_frames,
        windows,
        neighbor_radius=1,
        max_windows=4,
    )

    assert indices == [0, 1, 2, 3]


def test_progress_distinguishes_summary_context_from_plan_use_summary():
    console = Console(record=True, width=160)
    reporter = LiveRunReporter(console)

    reporter.on_round_start(
        round_index=1,
        planning_mode="refine",
        use_summary=True,
        retrieved_count=0,
    )
    reporter.on_planner(
        round_index=1,
        plan_payload={
            "strategy": "Use direct evidence only.",
            "use_summary": False,
            "steps": [],
            "refinement_instructions": "",
        },
    )

    rendered = console.export_text()
    assert "summary_context=True" in rendered
    assert "plan_use_summary: False" in rendered
