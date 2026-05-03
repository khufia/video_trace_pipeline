from video_trace_pipeline.orchestration.task_state import build_initial_task_state, update_task_state_after_execution
from video_trace_pipeline.schemas import AnswerCandidate, TaskClaimResult, TaskSpec, TaskState


def test_initial_task_state_creates_open_counter_for_count_questions():
    task = TaskSpec(
        benchmark="adhoc",
        sample_key="sample_count",
        question="How many distinct sounds are heard while the machine is being used?",
        options=["one", "two", "three"],
        video_path="video.mp4",
    )

    state = build_initial_task_state(task, {"planner_segments": []})

    assert state.counter_records[0].counter_id == "task_count"
    assert state.counter_records[0].status == "open"
    assert "duplicates" in state.counter_records[0].exclusion_rule


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


def test_task_state_merges_counter_updates_for_same_counter_id():
    state = TaskState(
        task_key="sample1",
        counter_records=[
            {
                "counter_id": "task_count",
                "target": "distinct sounds",
                "inclusion_rule": "count supported sound types",
                "accepted_observation_ids": ["obs_a"],
                "candidates": [{"label": "bang", "status": "accepted"}],
                "count": 1,
                "status": "candidate",
            }
        ],
    )
    execution_records = [
        {
            "tool_name": "verifier",
            "result": {
                "ok": True,
                "data": {
                    "claim_results": [],
                    "counter_updates": [
                        {
                            "counter_id": "task_count",
                            "target": "distinct sounds",
                            "accepted_observation_ids": ["obs_b"],
                            "rejected_observation_ids": ["obs_dup"],
                            "candidates": [
                                {"label": "bang", "status": "accepted"},
                                {"label": "echo", "status": "rejected", "reason": "duplicate ambience"},
                            ],
                            "count": 2,
                            "status": "validated",
                        }
                    ],
                },
            },
        }
    ]

    updated = update_task_state_after_execution(state, execution_records, round_index=1)

    assert len(updated.counter_records) == 1
    counter = updated.counter_records[0]
    assert counter.accepted_observation_ids == ["obs_a", "obs_b"]
    assert counter.rejected_observation_ids == ["obs_dup"]
    assert counter.count == 2
    assert counter.status == "validated"
    assert len(counter.candidates) == 2
    assert counter.candidates[0]["canonical_label"] == "bang"


def test_task_state_prunes_resolved_claim_questions():
    state = TaskState(
        task_key="sample1",
        claim_results=[
            TaskClaimResult(
                claim_id="claim_a",
                option="A",
                text="Option A is correct.",
                status="unverified",
            ),
            TaskClaimResult(
                claim_id="claim_b",
                option="B",
                text="Option B is correct.",
                status="unverified",
            ),
        ],
        answer_candidates=[
            AnswerCandidate(option="A", unknown_claim_ids=["claim_a"]),
            AnswerCandidate(option="B", unknown_claim_ids=["claim_b"]),
        ],
        open_questions=["Option A is correct.", "Option B is correct.", "shared missing discriminator"],
    )
    execution_records = [
        {
            "tool_name": "verifier",
            "result": {
                "ok": True,
                "data": {
                    "claim_results": [
                        {"claim_id": "claim_a", "verdict": "supported", "rationale": "A is supported."},
                        {"claim_id": "claim_b", "verdict": "refuted", "rationale": "B is contradicted."},
                    ]
                },
            },
            "evidence_entry": {"evidence_id": "ev_verifier"},
        }
    ]

    updated = update_task_state_after_execution(state, execution_records, round_index=1)

    assert updated.open_questions == ["shared missing discriminator"]
    assert updated.answer_candidates[0].status == "supported"
    assert updated.answer_candidates[1].status == "rejected"
    assert updated.ready_for_synthesis is True


def test_task_state_counter_merges_synonym_mentions_by_canonical_label():
    state = TaskState(
        task_key="sample1",
        counter_records=[
            {
                "counter_id": "task_count",
                "target": "distinct sounds",
                "candidates": [
                    {
                        "canonical_label": "relaxing sigh",
                        "raw_mentions": ["relaxing sigh"],
                        "status": "accepted",
                    }
                ],
                "count": 1,
            }
        ],
    )
    execution_records = [
        {
            "tool_name": "verifier",
            "result": {
                "ok": True,
                "data": {
                    "claim_results": [],
                    "counter_updates": [
                        {
                            "counter_id": "task_count",
                            "target": "distinct sounds",
                            "candidates": [
                                {
                                    "canonical_label": "relaxing sigh",
                                    "raw_mentions": ["soothing sigh"],
                                    "status": "accepted",
                                    "dedupe_rationale": "same product sigh category",
                                }
                            ],
                            "count": 1,
                            "status": "validated",
                        }
                    ],
                },
            },
        }
    ]

    updated = update_task_state_after_execution(state, execution_records, round_index=1)

    counter = updated.counter_records[0]
    assert len(counter.candidates) == 1
    assert counter.candidates[0]["raw_mentions"] == ["relaxing sigh", "soothing sigh"]
    assert counter.count == 1
