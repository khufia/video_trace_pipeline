from types import SimpleNamespace
import json

from video_trace_pipeline.orchestration.planner_retrieval import PlannerContextRetriever, query_terms_from_task_and_audit
from video_trace_pipeline.schemas import MachineProfile, TaskSpec
from video_trace_pipeline.storage import WorkspaceManager


class _Ledger(object):
    def __init__(self):
        self._entries = [
            {
                "evidence_id": "ev_validated_irrelevant",
                "tool_name": "frame_retriever",
                "status": "validated",
                "evidence_text": "A generic frame was retrieved.",
                "observation_ids": ["obs_validated_irrelevant"],
            },
            {
                "evidence_id": "ev_asr_relevant",
                "tool_name": "asr",
                "status": "provisional",
                "evidence_text": "Come to Phil's Ammu Nation today.",
                "observation_ids": ["obs_asr_relevant"],
            },
        ]
        self._observations = [
            {
                "observation_id": "obs_validated_irrelevant",
                "evidence_id": "ev_validated_irrelevant",
                "evidence_status": "validated",
                "source_tool": "frame_retriever",
                "atomic_text": "A candidate frame was retrieved.",
            },
            {
                "observation_id": "obs_asr_relevant",
                "evidence_id": "ev_asr_relevant",
                "evidence_status": "provisional",
                "source_tool": "asr",
                "atomic_text": "The ASR transcript says: Come to Phil's Ammu Nation today.",
            },
        ]

    def entries(self):
        return list(self._entries)

    def observations(self):
        return list(self._observations)

    def lookup_records(self, ids):
        requested = {str(item) for item in list(ids or [])}
        records = []
        for entry in self._entries:
            if entry["evidence_id"] in requested:
                records.append(
                    {
                        "record_id": entry["evidence_id"],
                        "evidence_id": entry["evidence_id"],
                        "tool_name": entry["tool_name"],
                        "evidence_text": entry["evidence_text"],
                        "atomic_text": entry["evidence_text"],
                        "status": entry["status"],
                        "observation_ids": entry["observation_ids"],
                    }
                )
        for observation in self._observations:
            if observation["observation_id"] in requested:
                records.append(dict(observation))
        return records

    def retrieve(self, query_terms=None, evidence_status=None, limit=50, **kwargs):
        del kwargs
        terms = [term.lower() for term in list(query_terms or [])]
        results = []
        for item in self._observations:
            if evidence_status and item.get("evidence_status") != evidence_status:
                continue
            haystack = str(item).lower()
            if terms and not any(term in haystack for term in terms):
                continue
            results.append(dict(item))
        return results[:limit]


def test_query_terms_filter_stopwords_so_relevant_preprocess_segment_wins():
    task = SimpleNamespace(
        question="When the sound appears in the video, how many empty beer bottles are there on the table?",
        options=[],
    )

    terms = query_terms_from_task_and_audit(task)

    assert "empty" in terms
    assert "bottles" in terms
    assert "sound" not in terms
    assert "when" not in terms
    assert "the" not in terms
    assert "video" not in terms


def test_retriever_supplements_validated_evidence_and_ranks_preprocess(tmp_path):
    workspace = WorkspaceManager(MachineProfile(workspace_root=str(tmp_path / "workspace")))
    task = TaskSpec(
        benchmark="adhoc",
        sample_key="video_13",
        video_id="video_13",
        question="When the sound 'come to bill's ammunition' appears, how many empty beer bottles are on the table?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    preprocess_bundle = {
        "planner_segments": [
            {
                "segment_id": "seg_generic",
                "start_s": 0.0,
                "end_s": 30.0,
                "dense_caption": {"overall_summary": "The video opens with a rating screen."},
            },
            {
                "segment_id": "seg_sound",
                "start_s": 120.0,
                "end_s": 150.0,
                "dense_caption": {"overall_summary": "A table and bottles are visible."},
                "asr": {"transcript_spans": [{"text": "Come to Phil's Ammu Nation today."}]},
            },
        ],
        "raw_segments": [],
        "asr_transcripts": [
            {
                "transcript_id": "tx_sound",
                "text": "Come to Phil's Ammu Nation today.",
            }
        ],
    }

    retrieved = PlannerContextRetriever(workspace).retrieve(
        task=task,
        preprocess_bundle=preprocess_bundle,
        evidence_ledger=_Ledger(),
        audit_report={"missing_information": ["validated sound moment"]},
        current_trace=None,
        mode="refine",
        limit=5,
    )

    assert [item["observation_id"] for item in retrieved["observations"]] == ["obs_asr_relevant"]
    assert retrieved["preprocess_matches"]["planner_segments"][0]["segment_id"] == "seg_sound"
    assert retrieved["preprocess_matches"]["asr_transcripts"][0]["transcript_id"] == "tx_sound"
    assert retrieved["audit_gaps"] == ["validated sound moment"]


def test_retriever_compacts_artifact_context_to_relevant_text(tmp_path):
    workspace = WorkspaceManager(MachineProfile(workspace_root=str(tmp_path / "workspace")))
    task = TaskSpec(
        benchmark="adhoc",
        sample_key="video_13",
        video_id="video_13",
        question="How many empty beer bottles are on the table?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    artifact_context_path = workspace.artifacts_root / "video_13" / "artifact_context.jsonl"
    artifact_context_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "artifact_id": "frame_132.00",
        "artifact_type": "frame",
        "relpath": "artifacts/video_13/frames/frame_132.00.png",
        "contains": [
            "A generic background wall is visible.",
            "The table contains two beer bottles.",
            "Both beer bottles appear to contain liquid, so they are not clearly empty.",
            "A distant poster is visible.",
        ],
        "linked_observations": [
            {"observation_id": "obs_bg", "text": "The background wall is visible."},
            {"observation_id": "obs_bottle", "text": "The beer bottles appear to contain liquid."},
        ],
        "linked_evidence": [
            {
                "evidence_id": "ev_generic",
                "tool_name": "generic_purpose",
                "summary": "The bottle state was inspected.",
                "observation_texts": [
                    "The wall is visible.",
                    "The left beer bottle has light liquid.",
                    "The right beer bottle has amber liquid.",
                    "A game controller is visible.",
                    "The bottles are not clearly empty.",
                ],
            }
        ],
    }
    artifact_context_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    retrieved = PlannerContextRetriever(workspace).retrieve(
        task=task,
        preprocess_bundle={},
        evidence_ledger=_Ledger(),
        audit_report={"missing_information": ["empty bottle state"]},
        current_trace=None,
        mode="refine",
        limit=5,
    )

    artifact = retrieved["artifact_context"][0]
    assert artifact["artifact_id"] == "frame_132.00"
    assert "Both beer bottles appear to contain liquid" in " ".join(artifact["contains"])
    assert all("wall" not in item.lower() for item in artifact["contains"])
    assert artifact["linked_observations"] == [
        {"observation_id": "obs_bottle", "text": "The beer bottles appear to contain liquid."}
    ]
    assert artifact["linked_evidence"][0]["observation_texts"] == [
        "The left beer bottle has light liquid.",
        "The right beer bottle has amber liquid.",
        "The bottles are not clearly empty.",
    ]


def test_retrieval_catalog_lists_available_sources(tmp_path):
    workspace = WorkspaceManager(MachineProfile(workspace_root=str(tmp_path / "workspace")))
    task = TaskSpec(
        benchmark="adhoc",
        sample_key="video_13",
        video_id="video_13",
        question="What price label is visible?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    preprocess_bundle = {
        "planner_segments": [
            {
                "segment_id": "seg_price",
                "start_s": 0.0,
                "end_s": 30.0,
                "dense_caption": {
                    "overall_summary": "A product shelf and visible price label appear.",
                    "captions": [{"visual": "A price label is visible.", "attributes": ["view: close"]}],
                    "clips": [{"video_id": "video_13", "start_s": 0.0, "end_s": 30.0}],
                },
                "asr": {"transcript_spans": [{"start_s": 1.0, "end_s": 2.0, "text": "This one is forty two."}]},
            }
        ],
        "raw_segments": [{"segment_id": "raw_1"}],
        "asr_transcripts": [{"transcript_id": "tx_1", "text": "forty two"}],
        "dense_caption_segments": [{"caption_id": "cap_1", "overall_summary": "price label"}],
    }
    artifact_context_path = workspace.artifacts_root / "video_13" / "artifact_context.jsonl"
    artifact_context_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_context_path.write_text(
        json.dumps({"artifact_id": "frame_001", "artifact_type": "frame", "contains": ["A price label is visible."]}) + "\n",
        encoding="utf-8",
    )

    catalog = PlannerContextRetriever(workspace).build_catalog(
        task=task,
        preprocess_bundle=preprocess_bundle,
        evidence_ledger=_Ledger(),
        audit_report=None,
        current_trace=None,
    )

    assert catalog["preprocess"]["planner_segment_count"] == 1
    assert catalog["preprocess"]["segments"][0]["segment_id"] == "seg_price"
    assert catalog["preprocess"]["segments"][0]["asr_span_count"] == 1
    assert catalog["evidence_store"]["evidence_entry_count"] == 2
    assert catalog["artifact_context"]["record_count"] == 1


def test_retrieve_for_requests_returns_requested_preprocess_and_existing_evidence(tmp_path):
    workspace = WorkspaceManager(MachineProfile(workspace_root=str(tmp_path / "workspace")))
    task = TaskSpec(
        benchmark="adhoc",
        sample_key="video_13",
        video_id="video_13",
        question="When the Ammu Nation line plays, what is on the table?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    preprocess_bundle = {
        "planner_segments": [
            {
                "segment_id": "seg_sound",
                "start_s": 120.0,
                "end_s": 150.0,
                "dense_caption": {"overall_summary": "A table with beer bottles is visible."},
                "asr": {"transcript_spans": [{"start_s": 130.0, "end_s": 132.0, "text": "Come to Phil's Ammu Nation today."}]},
            }
        ],
        "raw_segments": [],
        "asr_transcripts": [
            {
                "transcript_id": "tx_sound",
                "start_s": 130.0,
                "end_s": 132.0,
                "text": "Come to Phil's Ammu Nation today.",
            }
        ],
        "dense_caption_segments": [],
    }
    artifact_context_path = workspace.artifacts_root / "video_13" / "artifact_context.jsonl"
    artifact_context_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_context_path.write_text(
        json.dumps(
            {
                "artifact_id": "frame_132",
                "artifact_type": "frame",
                "time": {"timestamp_s": 132.0},
                "contains": ["The table contains beer bottles near the Ammu Nation line."],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    retrieved = PlannerContextRetriever(workspace).retrieve_for_requests(
        task=task,
        preprocess_bundle=preprocess_bundle,
        evidence_ledger=_Ledger(),
        requests=[
            {
                "request_id": "asr_line",
                "target": "asr_transcripts",
                "need": "Ammu Nation transcript line",
                "query": "Ammu Nation Phil",
                "time_range": {"start_s": 120.0, "end_s": 150.0},
                "limit": 5,
            },
            {
                "request_id": "existing_asr",
                "target": "existing_evidence",
                "need": "Existing ASR evidence for Ammu Nation",
                "query": "Ammu Nation",
                "source_tools": ["asr"],
                "limit": 5,
            },
            {
                "request_id": "artifact_table",
                "target": "artifact_context",
                "need": "Table content near 132s",
                "query": "table beer bottles",
                "time_range": {"start_s": 131.0, "end_s": 133.0},
                "limit": 5,
            },
        ],
        audit_report=None,
        current_trace=None,
    )

    assert retrieved["preprocess_matches"]["asr_transcripts"][0]["transcript_id"] == "tx_sound"
    assert retrieved["evidence"][0]["evidence_id"] == "ev_asr_relevant"
    assert retrieved["observations"][0]["observation_id"] == "obs_asr_relevant"
    assert retrieved["artifact_context"][0]["artifact_id"] == "frame_132"
