from types import SimpleNamespace

from video_trace_pipeline.schemas import VerifierOutput
from video_trace_pipeline.tools.process_adapters import VerifierProcessAdapter
from video_trace_pipeline.tool_wrappers import verifier_runner


def test_verifier_unparseable_output_degrades_to_unknown(monkeypatch, tmp_path):
    class FakeRunner:
        def __init__(self, **kwargs):
            del kwargs

        def generate(self, messages, *, max_new_tokens):
            del messages, max_new_tokens
            return '{"claim_results": [{"claim_id": "claim_1", "verdict": "supported"'

        def close(self):
            pass

    monkeypatch.setattr(verifier_runner, "QwenStyleRunner", FakeRunner)
    monkeypatch.setattr(verifier_runner, "resolve_model_path", lambda model_name, runtime: "/tmp/model")

    result = verifier_runner.execute_payload(
        {
            "request": {
                "query": "verify the claim",
                "claims": [
                    {
                        "claim_id": "claim_1",
                        "text": "The answer-critical claim is true.",
                        "claim_type": "visual",
                    }
                ],
                "text_contexts": ["some context"],
            },
            "task": {"question": "What happened?"},
            "runtime": {
                "model_name": "demo-model",
                "device": "cpu",
                "workspace_root": str(tmp_path),
                "scratch_dir": str(tmp_path / "scratch"),
                "extra": {"max_new_tokens": 32},
            },
        }
    )

    assert result["claim_results"][0]["claim_id"] == "claim_1"
    assert result["claim_results"][0]["verdict"] == "unknown"
    assert result["unresolved_gaps"][0].startswith("verifier_unparseable_output:")
    VerifierOutput.model_validate(result)


def test_verifier_normalizes_structured_unresolved_gaps(monkeypatch, tmp_path):
    class FakeRunner:
        def __init__(self, **kwargs):
            del kwargs

        def generate(self, messages, *, max_new_tokens):
            del messages, max_new_tokens
            return """
            {
              "claim_results": [
                {"claim_id": "claim_1", "verdict": "unknown", "rationale": "not enough evidence"}
              ],
              "unresolved_gaps": [
                {"gap": "missing_frame", "details": "need earlier frame", "raw_prefix": "abc"}
              ]
            }
            """

        def close(self):
            pass

    monkeypatch.setattr(verifier_runner, "QwenStyleRunner", FakeRunner)
    monkeypatch.setattr(verifier_runner, "resolve_model_path", lambda model_name, runtime: "/tmp/model")

    result = verifier_runner.execute_payload(
        {
            "request": {
                "query": "verify the claim",
                "claims": [
                    {
                        "claim_id": "claim_1",
                        "text": "The answer-critical claim is true.",
                        "claim_type": "visual",
                    }
                ],
                "text_contexts": ["some context"],
            },
            "task": {"question": "What happened?"},
            "runtime": {
                "model_name": "demo-model",
                "device": "cpu",
                "workspace_root": str(tmp_path),
                "scratch_dir": str(tmp_path / "scratch"),
                "extra": {"max_new_tokens": 32},
            },
        }
    )

    assert result["unresolved_gaps"] == ["missing_frame: need earlier frame raw_prefix=abc"]
    VerifierOutput.model_validate(result)


def test_verifier_repairs_placeholder_claim_id_to_requested_id(monkeypatch, tmp_path):
    class FakeRunner:
        def __init__(self, **kwargs):
            del kwargs

        def generate(self, messages, *, max_new_tokens):
            prompt = messages[0]["content"][0]["text"]
            assert "<copy_input_claim_id_exactly>" in prompt
            assert '"claim_id":"claim_1"' not in prompt
            del max_new_tokens
            return """
            <think>some hidden text</think>
            ```json
            {
              "claim_results": [
                {"claim_id": "claim_1", "verdict": "supported", "rationale": "placeholder id from model"}
              ],
              "unresolved_gaps": []
            }
            ```
            """

        def close(self):
            pass

    monkeypatch.setattr(verifier_runner, "QwenStyleRunner", FakeRunner)
    monkeypatch.setattr(verifier_runner, "resolve_model_path", lambda model_name, runtime: "/tmp/model")

    result = verifier_runner.execute_payload(
        {
            "request": {
                "query": "verify the claim",
                "claims": [
                    {
                        "claim_id": "opt_02_claim",
                        "text": "The answer-critical claim is true.",
                        "claim_type": "visual",
                    }
                ],
                "text_contexts": ["some context"],
            },
            "task": {"question": "What happened?"},
            "runtime": {
                "model_name": "demo-model",
                "device": "cpu",
                "workspace_root": str(tmp_path),
                "scratch_dir": str(tmp_path / "scratch"),
                "extra": {"max_new_tokens": 32},
            },
        }
    )

    assert result["claim_results"][0]["claim_id"] == "opt_02_claim"
    VerifierOutput.model_validate(result)


def test_verifier_prompt_includes_audio_clip_candidates_even_with_no_images(monkeypatch, tmp_path):
    class FakeRunner:
        def __init__(self, **kwargs):
            del kwargs

        def generate(self, messages, *, max_new_tokens):
            del max_new_tokens
            prompt = messages[0]["content"][0]["text"]
            assert "INPUT CLIPS / AUDIO EVENT CANDIDATES:" in prompt
            assert "audio_temporal_grounder" in prompt
            assert "event_label=relaxing sigh sound" in prompt
            assert "do not say no audio evidence exists" in prompt
            return """
            {
              "claim_results": [
                {"claim_id": "claim_audio", "verdict": "unknown", "rationale": "needs stronger evidence"}
              ],
              "unresolved_gaps": []
            }
            """

        def close(self):
            pass

    monkeypatch.setattr(verifier_runner, "QwenStyleRunner", FakeRunner)
    monkeypatch.setattr(verifier_runner, "resolve_model_path", lambda model_name, runtime: "/tmp/model")
    monkeypatch.setattr(verifier_runner, "sample_request_frames", lambda *args, **kwargs: [])

    result = verifier_runner.execute_payload(
        {
            "request": {
                "query": "verify the audio count",
                "claims": [
                    {
                        "claim_id": "claim_audio",
                        "text": "There is one distinct red-sauce sound.",
                        "claim_type": "count",
                    }
                ],
                "clips": [
                    {
                        "video_id": "video_1",
                        "start_s": 9.0,
                        "end_s": 10.0,
                        "metadata": {
                            "tool_backend": "audio_temporal_grounder",
                            "event_label": "relaxing sigh sound",
                        },
                    }
                ],
            },
            "task": {"question": "How many sounds are heard when using the red sauce?"},
            "runtime": {
                "model_name": "demo-model",
                "device": "cpu",
                "workspace_root": str(tmp_path),
                "scratch_dir": str(tmp_path / "scratch"),
                "extra": {"max_new_tokens": 32},
            },
        }
    )

    assert result["claim_results"][0]["claim_id"] == "claim_audio"
    VerifierOutput.model_validate(result)


def test_verifier_prompt_has_mcq_comparative_mode(monkeypatch, tmp_path):
    class FakeRunner:
        def __init__(self, **kwargs):
            del kwargs

        def generate(self, messages, *, max_new_tokens):
            del max_new_tokens
            prompt = messages[0]["content"][0]["text"]
            assert "VERIFICATION MODE: mcq_comparative" in prompt
            assert "Support the option whose core discriminator is best grounded" in prompt
            assert "Transcript text alone is not sufficient" in prompt
            return """
            {
              "claim_results": [
                {"claim_id": "opt_01_claim", "verdict": "supported", "rationale": "best supported"}
              ],
              "unresolved_gaps": []
            }
            """

        def close(self):
            pass

    monkeypatch.setattr(verifier_runner, "QwenStyleRunner", FakeRunner)
    monkeypatch.setattr(verifier_runner, "resolve_model_path", lambda model_name, runtime: "/tmp/model")

    result = verifier_runner.execute_payload(
        {
            "request": {
                "query": "compare options",
                "verification_mode": "mcq_comparative",
                "claims": [
                    {
                        "claim_id": "opt_01_claim",
                        "text": "Option A is best supported.",
                        "claim_type": "speaker_tone",
                    }
                ],
                "text_contexts": ["the speaker changes from calm to angry"],
            },
            "task": {"question": "What was the woman's tone?", "options": ["A", "B"]},
            "runtime": {
                "model_name": "demo-model",
                "device": "cpu",
                "workspace_root": str(tmp_path),
                "scratch_dir": str(tmp_path / "scratch"),
                "extra": {"max_new_tokens": 32},
            },
        }
    )

    assert result["claim_results"][0]["verdict"] == "supported"
    VerifierOutput.model_validate(result)


def test_verifier_adapter_auto_uses_comparative_mode_for_mcq_task():
    class CapturingVerifierAdapter(VerifierProcessAdapter):
        def _run_json(self, context, request_payload):
            del context
            self.captured_request_payload = request_payload
            return (
                {
                    "claim_results": [
                        {
                            "claim_id": "claim_1",
                            "verdict": "supported",
                            "rationale": "Comparative verifier selected the best option.",
                        }
                    ]
                },
                "raw verifier output",
            )

    adapter = CapturingVerifierAdapter(name="verifier", model_name="demo-model")
    request = adapter.parse_request(
        {
            "query": "compare options",
            "claims": [{"claim_id": "claim_1", "text": "A is best supported.", "claim_type": "option_mapping"}],
            "text_contexts": ["context"],
        }
    )
    context = SimpleNamespace(task=SimpleNamespace(options=["A", "B"]))

    result = adapter.execute(request, context=context)

    assert adapter.captured_request_payload["verification_mode"] == "mcq_comparative"
    assert result.data["claim_results"][0]["verdict"] == "supported"


def test_verifier_process_adapter_degrades_schema_failure_to_unknown():
    class BadVerifierAdapter(VerifierProcessAdapter):
        def _run_json(self, context, request_payload):
            del context, request_payload
            return (
                {
                    "claim_results": [
                        {
                            "verdict": "supported",
                            "rationale": "Malformed verifier payload.",
                        }
                    ]
                },
                "raw malformed verifier output",
            )

    adapter = BadVerifierAdapter(name="verifier", model_name="demo-model")
    request = adapter.parse_request(
        {
            "query": "verify the claim",
            "claims": [{"claim_id": "claim_1", "text": "The claim is true.", "claim_type": "visual"}],
            "text_contexts": ["context"],
        }
    )

    result = adapter.execute(request, context=object())

    assert result.ok is True
    assert result.data["claim_results"][0]["claim_id"] == "claim_1"
    assert result.data["claim_results"][0]["verdict"] == "unknown"
    assert result.metadata["schema_validation_failed"] is True
    assert result.data["unresolved_gaps"][0].startswith("verifier_schema_validation_error:")
