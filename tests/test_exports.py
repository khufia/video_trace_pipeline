from types import SimpleNamespace

from video_trace_pipeline.renderers import export_trace_for_benchmark, render_trace_markdown


def test_export_trace_for_adhoc_is_flattened():
    task = SimpleNamespace(sample_key="sample-1", question_id=None, video_id="video-1")
    trace_package = {
        "inference_steps": [{"text": "step one"}],
        "evidence_entries": [{"tool_name": "ocr", "evidence_text": "42", "observation_ids": ["obs-1"]}],
        "final_answer": "B",
    }

    payload = export_trace_for_benchmark("adhoc", task, trace_package)

    assert payload["sample_key"] == "sample-1"
    assert payload["answer"] == "B"
    assert payload["trace"]["inference_steps"] == ["step one"]
    assert payload["trace"]["evidence"][0]["tool_name"] == "ocr"
    assert "trace_package" not in payload


def test_export_trace_for_adhoc_preserves_temporal_fields():
    task = SimpleNamespace(sample_key="sample-1", question_id=None, video_id="video-1")
    trace_package = {
        "inference_steps": [{"text": "step one"}],
        "evidence_entries": [
            {
                "tool_name": "asr",
                "evidence_text": "speaker said hello",
                "observation_ids": ["obs-1"],
                "time_start_s": 2.377,
                "time_end_s": 22.323,
            }
        ],
        "final_answer": "B",
    }

    payload = export_trace_for_benchmark("adhoc", task, trace_package)

    assert payload["trace"]["evidence"][0]["time_start_s"] == 2.377
    assert payload["trace"]["evidence"][0]["time_end_s"] == 22.323


def test_render_trace_markdown_includes_temporal_anchor():
    rendered = render_trace_markdown(
        {
            "final_answer": "B",
            "inference_steps": [{"step_id": 1, "text": "step one"}],
            "evidence_entries": [
                {
                    "evidence_id": "ev_01",
                    "tool_name": "asr",
                    "evidence_text": "speaker said hello",
                    "time_start_s": 2.377,
                    "time_end_s": 22.323,
                    "observation_ids": ["obs-1"],
                }
            ],
        }
    )

    assert "- Time: 2.377s to 22.323s" in rendered
    assert "- Tool:" not in rendered


def test_render_trace_markdown_preserves_disjoint_temporal_intervals():
    rendered = render_trace_markdown(
        {
            "final_answer": "B",
            "inference_steps": [
                {
                    "step_id": 1,
                    "text": "step one",
                    "time_intervals": [
                        {"start_s": 2.377, "end_s": 22.323},
                        {"start_s": 22.323, "end_s": 41.847},
                    ],
                }
            ],
            "evidence_entries": [
                {
                    "evidence_id": "ev_01",
                    "tool_name": "visual_temporal_grounder",
                    "evidence_text": "candidate intervals",
                    "time_intervals": [
                        {"start_s": 158.0, "end_s": 168.0},
                        {"start_s": 207.0, "end_s": 213.0},
                        {"start_s": 300.0, "end_s": 301.0},
                    ],
                    "observation_ids": ["obs-1"],
                }
            ],
        }
    )

    assert "1. step one [2.377s to 41.847s]" not in rendered
    assert "1. step one [2.377s to 22.323s; 22.323s to 41.847s]" in rendered
    assert "- Time: 158s to 168s; 207s to 213s; 300s to 301s" in rendered
