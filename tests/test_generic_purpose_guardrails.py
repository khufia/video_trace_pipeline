from video_trace_pipeline.schemas import GenericPurposeRequest
from video_trace_pipeline.tool_wrappers import qwen35vl_runner
from video_trace_pipeline.tools.process_adapters import GenericPurposeProcessAdapter


def test_generic_purpose_prompt_includes_task_question_and_options():
    prompt = qwen35vl_runner._build_prompt(
        {"query": "Pick the option that is not mentioned."},
        {
            "question": "Which option is not mentioned?",
            "options": ["A. Safety is important.", "B. No door."],
        },
        transcript_text="Transcript goes here.",
        evidence_lines=[],
        text_contexts=[],
        media_lines=["Image 1 | artifact_id=frame_demo | timestamp=10s | relpath=artifacts/demo/frames/frame_demo.png"],
    )

    assert "TASK QUESTION:" in prompt
    assert "Which option is not mentioned?" in prompt
    assert "ANSWER OPTIONS:" in prompt
    assert "- A. Safety is important." in prompt
    assert "- B. No door." in prompt
    assert "numeric score from 0.0 to 1.0" in prompt
    assert "Do not rely on scene priors" in prompt
    assert "answer indeterminate instead of guessing" in prompt
    assert "visible object presence alone does not prove that state" in prompt
    assert "collapse near-synonymous labels" in prompt
    assert "best matches the directly grounded phenomenon" in prompt
    assert "identify the earliest validated candidate first" in prompt
    assert "compare full surface forms" in prompt
    assert "count whole named entities or repeated surface phrases" in prompt
    assert "longest repeated matching name/phrase" in prompt
    assert "use the longer repeated phrase as the boundary" in prompt
    assert "not an answer key" in prompt
    assert "verify that attribute directly" in prompt
    assert "INPUT MEDIA:" in prompt
    assert "artifact_id=frame_demo" in prompt
    assert "latest stable complete image" in prompt
    assert "Keep answer short" in prompt
    assert "one short sentence in analysis" in prompt
    assert "Do not put phrases like 'Thinking Process'" in prompt
    assert '"answer":"B. Example Store, 20 percentage points"' in prompt
    assert '"answer":"C. Example phrase"' in prompt


def test_generic_purpose_uses_image_artifacts_from_evidence_records(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    artifact_path = workspace_root / "cache" / "artifacts" / "demo" / "frame.png"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"fake image")
    captured = {}

    class FakeRunner(object):
        def __init__(self, **kwargs):
            captured["runner_kwargs"] = dict(kwargs)

        def generate(self, messages, max_new_tokens):
            captured["messages"] = messages
            captured["max_new_tokens"] = max_new_tokens
            return '{"answer":"A","supporting_points":[],"confidence":0.7,"analysis":"ok"}'

        def close(self):
            return None

    monkeypatch.setattr(qwen35vl_runner, "QwenStyleRunner", FakeRunner)
    monkeypatch.setattr(qwen35vl_runner, "resolve_model_path", lambda *args, **kwargs: "/tmp/fake-model")

    result = qwen35vl_runner.execute_payload(
        {
            "request": {
                "tool_name": "generic_purpose",
                "query": "Inspect the previously retrieved frames.",
            },
            "task": {
                "question": "What is visible?",
                "options": [],
            },
            "runtime": {
                "model_name": "Qwen/Qwen3.5-9B",
                "workspace_root": str(workspace_root),
                "extra": {},
            },
            "evidence_records": [
                {
                    "evidence_id": "ev_01_demo",
                    "artifact_refs": [
                        {
                            "artifact_id": "art_01",
                            "kind": "frame",
                            "relpath": "cache/artifacts/demo/frame.png",
                            "metadata": {
                                "source_path": str(artifact_path),
                                "timestamp_s": 12.0,
                                "video_id": "sample1",
                            },
                        }
                    ],
                }
            ],
        }
    )

    assert result["answer"] == "A"
    assert captured["messages"][0]["content"][0]["type"] == "image"
    assert captured["messages"][0]["content"][0]["image"] == str(artifact_path.resolve())
    prompt_text = captured["messages"][0]["content"][-1]["text"]
    assert "INPUT MEDIA:" in prompt_text
    assert "artifact_id=art_01" in prompt_text
    assert "timestamp=12s" in prompt_text


def test_generic_purpose_adapter_omits_low_signal_output(monkeypatch):
    adapter = GenericPurposeProcessAdapter(name="generic_purpose", model_name="Qwen/Qwen3.5-9B")

    def fake_run_json(context, request_payload):
        del context, request_payload
        raw = '{"answer":"!!!!!!!!!!!!!!!!","supporting_points":[],"confidence":null,"analysis":"!!!!!!!!!!!!!!!!"}'
        return {
            "answer": "!!!!!!!!!!!!!!!!",
            "supporting_points": [],
            "confidence": None,
            "analysis": "!!!!!!!!!!!!!!!!",
        }, raw

    monkeypatch.setattr(adapter, "_run_json", fake_run_json)

    result = adapter.execute(GenericPurposeRequest(tool_name="generic_purpose", query="Test query"), context=None)

    assert result.data["answer"] == ""
    assert result.data["analysis"] == ""
    assert result.summary == "generic_purpose produced a low-signal response and it was omitted from evidence."
    assert result.metadata["low_signal_output"] is True


def test_generic_purpose_adapter_coerces_labeled_confidence(monkeypatch):
    adapter = GenericPurposeProcessAdapter(name="generic_purpose", model_name="Qwen/Qwen3.5-9B")

    def fake_run_json(context, request_payload):
        del context, request_payload
        raw = '{"answer":"2","supporting_points":["Two bottles are visible."],"confidence":"High","analysis":"Two bottles are visible on the table."}'
        return {
            "answer": "2",
            "supporting_points": ["Two bottles are visible."],
            "confidence": "High",
            "analysis": "Two bottles are visible on the table.",
        }, raw

    monkeypatch.setattr(adapter, "_run_json", fake_run_json)

    result = adapter.execute(GenericPurposeRequest(tool_name="generic_purpose", query="Count the bottles"), context=None)

    assert result.data["confidence"] == 0.85
    assert result.metadata["confidence"] == 0.85
    assert result.metadata["confidence_kind"] == "answer_confidence"
