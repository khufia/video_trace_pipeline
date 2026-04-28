from types import SimpleNamespace

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
                "atomic_text": "A candidate frame was retrieved.",
            },
            {
                "observation_id": "obs_asr_relevant",
                "evidence_id": "ev_asr_relevant",
                "evidence_status": "provisional",
                "atomic_text": "The ASR transcript says: Come to Phil's Ammu Nation today.",
            },
        ]

    def entries(self):
        return list(self._entries)

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
