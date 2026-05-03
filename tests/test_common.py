from video_trace_pipeline.common import extract_json_object


def test_extract_json_object_prefers_valid_fenced_json_after_thinking():
    raw = """
    <think>
    I might emit a malformed object first: {"claim_results": [
    </think>

    ```json
    {"claim_results": [{"claim_id": "claim_a", "verdict": "unknown"}], "unresolved_gaps": []}
    ```
    """

    parsed = extract_json_object(raw)

    assert parsed["claim_results"][0]["claim_id"] == "claim_a"
    assert parsed["unresolved_gaps"] == []


def test_extract_json_object_uses_last_balanced_candidate_when_prefix_has_bad_json():
    raw = 'draft {"bad": true trailing text final {"ok": true, "items": [1, 2]}'

    parsed = extract_json_object(raw)

    assert parsed == {"ok": True, "items": [1, 2]}
