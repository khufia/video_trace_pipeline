from video_trace_pipeline.orchestration.task_state import update_task_state_after_execution
from video_trace_pipeline.schemas import AnswerCandidate, TaskClaimResult, TaskState


def test_task_state_reducer_absorbs_verifier_claim_and_state_updates():
    state = TaskState(
        task_key="sample1",
        claim_results=[
            TaskClaimResult(
                claim_id="claim_price",
                option="A",
                text="The visible price is 42.",
                claim_type="ocr",
                status="unverified",
            )
        ],
        answer_candidates=[AnswerCandidate(option="A", unknown_claim_ids=["claim_price"])],
    )
    execution_records = [
        {
            "tool_name": "verifier",
            "request": {"query": "verify price"},
            "result": {
                "ok": True,
                "data": {
                    "claim_results": [
                        {
                            "claim_id": "claim_price",
                            "verdict": "supported",
                            "supporting_observation_ids": ["obs_price"],
                            "supporting_evidence_ids": ["ev_ocr"],
                            "rationale": "OCR and frame context support 42.",
                        }
                    ],
                    "evidence_updates": [
                        {
                            "evidence_id": "ev_ocr",
                            "previous_status": "candidate",
                            "new_status": "validated",
                            "claim_id": "claim_price",
                            "reason": "Verified against localized OCR.",
                        }
                    ],
                    "counter_updates": [{"counter_id": "count_prices", "target": "price values", "count": 1, "status": "validated"}],
                    "referent_updates": [{"referent_id": "price_tag", "description": "localized price tag", "status": "validated"}],
                    "ocr_occurrence_updates": [{"occurrence_id": "ocr_42", "raw_text": "42", "normalized_value": 42}],
                    "unresolved_gaps": [],
                },
            },
            "evidence_entry": {"evidence_id": "ev_verifier", "observation_ids": ["obs_verifier"]},
            "observations": [{"observation_id": "obs_verifier", "atomic_text": "Verifier supported the price claim."}],
        }
    ]

    updated = update_task_state_after_execution(state, execution_records, round_index=1)

    claim = updated.claim_results[0]
    assert claim.status == "validated"
    assert "ev_ocr" in claim.supporting_evidence_ids
    assert "obs_price" in claim.supporting_observation_ids
    assert updated.answer_candidates[0].status == "supported"
    assert any(item.evidence_id == "ev_ocr" and item.new_status == "validated" for item in updated.evidence_status_updates)
    assert updated.counter_records[0].count == 1
    assert updated.referent_slots[0].referent_id == "price_tag"
    assert updated.ocr_occurrences[0].normalized_value == 42
