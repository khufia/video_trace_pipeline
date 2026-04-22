from types import SimpleNamespace

from video_trace_pipeline.renderers import export_trace_for_benchmark


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
