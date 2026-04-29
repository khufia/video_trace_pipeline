from video_trace_pipeline.orchestration.executor import PlanExecutor
from video_trace_pipeline.schemas import (
    AgentConfig,
    ASRRequest,
    ClipRef,
    ExecutionPlan,
    FrameRetrieverRequest,
    GenericPurposeRequest,
    MachineProfile,
    ModelsConfig,
    OCRRequest,
    PlanStep,
    SpatialGrounderRequest,
    TaskSpec,
    ToolConfig,
    ToolResult,
    VerifierRequest,
    VisualTemporalGrounderRequest,
)
from video_trace_pipeline.storage import EvidenceLedger, SharedEvidenceCache, WorkspaceManager
from video_trace_pipeline.tools import ObservationExtractor
from video_trace_pipeline.tools.base import ToolExecutionContext


class _Adapter(object):
    def __init__(self, name, request_model, output):
        self.name = name
        self.request_model = request_model
        self.output = output
        self.calls = []

    def parse_request(self, inputs):
        payload = dict(inputs or {})
        payload.setdefault("tool_name", self.name)
        return self.request_model.parse_obj(payload)

    def execute(self, request, context):
        del context
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        self.calls.append(payload)
        result_data = self.output(payload)
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=result_data,
            raw_output_text="raw %s" % self.name,
            summary="%s summary" % self.name,
        )


class _Registry(object):
    def __init__(self):
        self.adapters = {
            "visual_temporal_grounder": _Adapter(
                "visual_temporal_grounder",
                VisualTemporalGrounderRequest,
                lambda request: {
                    "query": request["query"],
                    "clips": [{"video_id": "video1", "start_s": 10.0, "end_s": 20.0}],
                },
            ),
            "frame_retriever": _Adapter(
                "frame_retriever",
                FrameRetrieverRequest,
                lambda request: {
                    "query": request.get("query"),
                    "frames": [
                        {
                            "video_id": "video1",
                            "timestamp_s": 12.0,
                            "clip": request["clips"][0],
                            "metadata": {"relevance_score": 0.9},
                        }
                    ],
                },
            ),
            "spatial_grounder": _Adapter(
                "spatial_grounder",
                SpatialGrounderRequest,
                lambda request: {
                    "query": request["query"],
                    "frames": request["frames"],
                    "regions": [
                        {
                            "frame": request["frames"][0],
                            "bbox": [0.1, 0.2, 0.3, 0.4],
                            "label": "price label",
                        }
                    ],
                    "spatial_description": "The price label is in the upper right.",
                },
            ),
            "ocr": _Adapter(
                "ocr",
                OCRRequest,
                lambda request: {
                    "query": request.get("query"),
                    "text": "PRICE 42",
                    "lines": [{"text": "PRICE 42", "confidence": 0.9}],
                    "reads": [{"regions": request.get("regions", []), "text": "PRICE 42"}],
                },
            ),
            "generic_purpose": _Adapter(
                "generic_purpose",
                GenericPurposeRequest,
                lambda request: {
                    "answer": "The grounded text says PRICE 42.",
                    "analysis": "Used explicit OCR/text/transcript context.",
                    "supporting_points": list(request.get("text_contexts") or []),
                },
            ),
            "verifier": _Adapter(
                "verifier",
                VerifierRequest,
                lambda request: {
                    "claim_results": [
                        {
                            "claim_id": request["claims"][0]["claim_id"],
                            "verdict": "supported",
                            "confidence": 0.9,
                            "supporting_observation_ids": [],
                            "supporting_evidence_ids": request.get("evidence_ids", []),
                            "refuting_observation_ids": [],
                            "refuting_evidence_ids": [],
                            "time_intervals": [],
                            "artifact_refs": [],
                            "rationale": "OCR and frame context support the claim.",
                            "coverage": {"checked_inputs": ["frames", "ocr_results"], "missing_inputs": [], "sampling_summary": "checked one frame"},
                        }
                    ],
                    "new_observations": [],
                    "evidence_updates": [],
                    "unresolved_gaps": [],
                },
            ),
            "asr": _Adapter(
                "asr",
                ASRRequest,
                lambda request: {
                    "clips": request["clips"],
                    "transcripts": [
                        {
                            "transcript_id": "tx_1",
                            "clip": request["clips"][0],
                            "segments": [{"start_s": 0.0, "end_s": 1.0, "text": "hello world"}],
                        }
                    ],
                },
            ),
        }

    def get_adapter(self, tool_name):
        return self.adapters[tool_name]


def _models_config():
    tool_names = [
        "visual_temporal_grounder",
        "frame_retriever",
        "spatial_grounder",
        "ocr",
        "generic_purpose",
        "verifier",
        "asr",
        "dense_captioner",
    ]
    return ModelsConfig(
        agents={"atomicizer": AgentConfig(backend="openai", model="gpt-5.4")},
        tools={name: ToolConfig(enabled=True, model="test-model", prompt_version="test") for name in tool_names},
    )


def _context(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    workspace = WorkspaceManager(MachineProfile(workspace_root=str(tmp_path / "workspace")))
    task = TaskSpec(
        benchmark="adhoc",
        sample_key="sample1",
        video_id="video1",
        question="What is the price?",
        options=["42", "24"],
        video_path=str(video),
    )
    run = workspace.create_run(task)
    return workspace, task, run, ToolExecutionContext(
        workspace=workspace,
        run=run,
        task=task,
        models_config=_models_config(),
        preprocess_bundle={},
    )


def _executor(workspace, registry):
    return PlanExecutor(
        tool_registry=registry,
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )


def test_tool_chain_passes_structured_outputs_between_tools(tmp_path):
    workspace, task, run, context = _context(tmp_path)
    registry = _Registry()
    executor = _executor(workspace, registry)
    ledger = EvidenceLedger(run)
    plan = ExecutionPlan(
        strategy="Ground text through visual chain.",
        steps=[
            PlanStep(step_id=1, tool_name="visual_temporal_grounder", purpose="Find price moment.", inputs={"query": "price label", "top_k": 1}),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve price frame.",
                inputs={"query": "readable price frame", "num_frames": 1},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
            PlanStep(
                step_id=3,
                tool_name="spatial_grounder",
                purpose="Find price label region.",
                inputs={"query": "price label"},
                input_refs={"frames": [{"step_id": 2, "field_path": "frames"}]},
            ),
            PlanStep(
                step_id=4,
                tool_name="ocr",
                purpose="Read price label.",
                inputs={"query": "read exact price text"},
                input_refs={"regions": [{"step_id": 3, "field_path": "regions"}]},
            ),
            PlanStep(
                step_id=5,
                tool_name="generic_purpose",
                purpose="Answer from OCR text.",
                inputs={"query": "Which option matches the OCR text?"},
                input_refs={
                    "text_contexts": [{"step_id": 4, "field_path": "text"}],
                    "frames": [{"step_id": 2, "field_path": "frames"}],
                },
            ),
        ],
    )

    records = executor.execute_plan(plan, context, ledger, video_fingerprint="vid", round_index=1)

    assert len(records) == 5
    assert registry.adapters["frame_retriever"].calls[0]["clips"][0]["start_s"] == 10.0
    assert registry.adapters["spatial_grounder"].calls[0]["frames"][0]["timestamp_s"] == 12.0
    assert registry.adapters["ocr"].calls[0]["regions"][0]["label"] == "price label"
    assert registry.adapters["generic_purpose"].calls[0]["text_contexts"] == ["PRICE 42"]
    assert registry.adapters["generic_purpose"].calls[0]["frames"][0]["timestamp_s"] == 12.0


def test_ocr_to_verifier_passes_structured_regions_frames_and_reads(tmp_path):
    workspace, task, run, context = _context(tmp_path)
    registry = _Registry()
    executor = _executor(workspace, registry)
    ledger = EvidenceLedger(run)
    plan = ExecutionPlan(
        strategy="Ground, read, and verify price claim.",
        steps=[
            PlanStep(step_id=1, tool_name="visual_temporal_grounder", purpose="Find price moment.", inputs={"query": "price label", "top_k": 1}),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve price frame.",
                inputs={"query": "readable price frame", "num_frames": 1},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
            PlanStep(
                step_id=3,
                tool_name="spatial_grounder",
                purpose="Find price label region.",
                inputs={"query": "price label"},
                input_refs={"frames": [{"step_id": 2, "field_path": "frames"}]},
            ),
            PlanStep(
                step_id=4,
                tool_name="ocr",
                purpose="Read price label.",
                inputs={"query": "read exact price text"},
                input_refs={"regions": [{"step_id": 3, "field_path": "regions"}]},
            ),
            PlanStep(
                step_id=5,
                tool_name="verifier",
                purpose="Verify the OCR price claim.",
                inputs={
                    "query": "verify exact price claim",
                    "claims": [{"claim_id": "claim_price", "text": "The visible price is 42.", "claim_type": "ocr"}],
                },
                input_refs={
                    "frames": [{"step_id": 2, "field_path": "frames"}],
                    "regions": [{"step_id": 3, "field_path": "regions"}],
                    "ocr_results": [{"step_id": 4, "field_path": "reads"}],
                },
            ),
        ],
    )

    records = executor.execute_plan(plan, context, ledger, video_fingerprint="vid", round_index=1)

    assert len(records) == 5
    verifier_request = registry.adapters["verifier"].calls[0]
    assert verifier_request["frames"][0]["timestamp_s"] == 12.0
    assert verifier_request["regions"][0]["label"] == "price label"
    assert verifier_request["ocr_results"][0]["text"] == "PRICE 42"
    assert records[-1]["result"]["data"]["claim_results"][0]["verdict"] == "supported"


def test_asr_to_generic_passes_transcripts_never_text_contexts(tmp_path):
    workspace, task, run, context = _context(tmp_path)
    registry = _Registry()
    executor = _executor(workspace, registry)
    ledger = EvidenceLedger(run)
    plan = ExecutionPlan(
        strategy="Use transcript structurally.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="asr",
                purpose="Transcribe clip.",
                inputs={"clips": [ClipRef(video_id="video1", start_s=0.0, end_s=2.0).model_dump()]},
            ),
            PlanStep(
                step_id=2,
                tool_name="generic_purpose",
                purpose="Answer from transcript.",
                inputs={"query": "Answer from transcript."},
                input_refs={"transcripts": [{"step_id": 1, "field_path": "transcripts"}]},
            ),
        ],
    )

    executor.execute_plan(plan, context, ledger, video_fingerprint="vid", round_index=1)

    generic_request = registry.adapters["generic_purpose"].calls[0]
    assert generic_request["transcripts"][0]["transcript_id"] == "tx_1"
    assert generic_request["text_contexts"] == []


def test_asr_transcript_clip_wildcard_feeds_frame_retriever(tmp_path):
    workspace, task, run, context = _context(tmp_path)
    registry = _Registry()
    executor = _executor(workspace, registry)
    ledger = EvidenceLedger(run)
    plan = ExecutionPlan(
        strategy="Use ASR transcript clip for visual follow-up.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="asr",
                purpose="Transcribe clip.",
                inputs={"clips": [ClipRef(video_id="video1", start_s=0.0, end_s=2.0).model_dump()]},
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve frames for the transcript clip.",
                inputs={"query": "frames around spoken line", "num_frames": 2},
                input_refs={"clips": [{"step_id": 1, "field_path": "transcripts[].clip"}]},
            ),
        ],
    )

    records = executor.execute_plan(plan, context, ledger, video_fingerprint="vid", round_index=1)

    assert records[1]["result"]["ok"] is True
    assert registry.adapters["frame_retriever"].calls[0]["clips"][0]["start_s"] == 0.0
    assert registry.adapters["frame_retriever"].calls[0]["clips"][0]["end_s"] == 2.0


def test_invalid_wiring_records_failure_before_downstream_tool_call(tmp_path):
    workspace, task, run, context = _context(tmp_path)
    registry = _Registry()
    executor = _executor(workspace, registry)
    ledger = EvidenceLedger(run)
    plan = ExecutionPlan(
        strategy="Bad wiring.",
        steps=[
            PlanStep(step_id=1, tool_name="visual_temporal_grounder", purpose="Find clip.", inputs={"query": "price", "top_k": 1}),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Bad source field.",
                inputs={"query": "frame", "num_frames": 1},
                input_refs={"clips": [{"step_id": 1, "field_path": "not_a_field"}]},
            ),
        ],
    )

    records = executor.execute_plan(plan, context, ledger, video_fingerprint="vid", round_index=1)

    assert len(records) == 2
    assert records[1]["result"]["ok"] is False
    assert records[1]["result"]["data"]["error_type"] == "unresolved_dependency"
    assert registry.adapters["frame_retriever"].calls == []


def test_empty_resolved_media_records_invalid_request_before_tool_call(tmp_path):
    workspace, task, run, context = _context(tmp_path)
    registry = _Registry()
    registry.adapters["spatial_grounder"].output = lambda request: {
        "query": request["query"],
        "frames": request["frames"],
        "regions": [],
        "spatial_description": "No matching region found.",
    }
    executor = _executor(workspace, registry)
    ledger = EvidenceLedger(run)
    plan = ExecutionPlan(
        strategy="OCR only if spatial grounding finds a region.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find clip.",
                inputs={"query": "price", "top_k": 1},
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve frame.",
                inputs={"query": "price frame", "num_frames": 1},
                input_refs={"clips": [{"step_id": 1, "field_path": "clips"}]},
            ),
            PlanStep(
                step_id=3,
                tool_name="spatial_grounder",
                purpose="Find price region.",
                inputs={"query": "price label"},
                input_refs={"frames": [{"step_id": 2, "field_path": "frames"}]},
            ),
            PlanStep(
                step_id=4,
                tool_name="ocr",
                purpose="Read price label.",
                inputs={"query": "read exact price"},
                input_refs={"regions": [{"step_id": 3, "field_path": "regions"}]},
            ),
        ],
    )

    records = executor.execute_plan(plan, context, ledger, video_fingerprint="vid", round_index=1)

    assert len(records) == 4
    assert records[3]["result"]["ok"] is False
    assert records[3]["result"]["data"]["error_type"] == "invalid_request"
    assert records[3]["request"]["resolved_arguments"]["regions"] == []
    assert registry.adapters["ocr"].calls == []
