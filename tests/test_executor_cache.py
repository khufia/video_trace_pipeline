from video_trace_pipeline.orchestration.executor import PlanExecutor, augment_dependency_output, hydrate_arguments_with_task_context
from video_trace_pipeline.schemas import (
    AgentConfig,
    ASRRequest,
    EvidenceEntry,
    ExecutionPlan,
    GenericPurposeRequest,
    FrameRef,
    FrameRetrieverRequest,
    MachineProfile,
    ModelsConfig,
    PlanStep,
    SpatialGrounderRequest,
    TaskSpec,
    ToolConfig,
    ToolResult,
    VisualTemporalGrounderRequest,
)
from video_trace_pipeline.storage import EvidenceLedger, RunContext, SharedEvidenceCache, WorkspaceManager
from video_trace_pipeline.tools import ObservationExtractor
from video_trace_pipeline.tools.base import ToolAdapter, ToolExecutionContext


class FakeAdapter(object):
    request_model = VisualTemporalGrounderRequest

    def __init__(self):
        self.calls = 0

    def parse_request(self, arguments):
        return self.request_model.parse_obj(arguments)

    def execute(self, request, context):
        self.calls += 1
        return ToolResult(
            tool_name="visual_temporal_grounder",
            ok=True,
            data={
                "query": request.query,
                "clips": [
                    {
                        "video_id": context.task.video_id or context.task.sample_key,
                        "start_s": 0.0,
                        "end_s": 5.0,
                        "metadata": {"confidence": 0.9},
                    }
                ],
            },
            summary="The event appears in the first clip.",
        )


class FakeRegistry(object):
    def __init__(self, adapters):
        self.adapters = adapters

    def get_adapter(self, tool_name):
        return self.adapters[tool_name]


class FakeFrameRetrieverAdapter(ToolAdapter):
    name = "frame_retriever"
    request_model = FrameRetrieverRequest

    def __init__(self):
        self.calls = 0
        self.last_clip = None
        self.last_clips = []
        self.last_frame = None

    def parse_request(self, arguments):
        request = super().parse_request(arguments)
        self.last_clips = list(request.clips or ([request.clip] if request.clip is not None else []))
        self.last_clip = request.clip or (self.last_clips[0] if self.last_clips else None)
        return request

    def execute(self, request, context):
        self.calls += 1
        clips = list(request.clips or ([request.clip] if request.clip is not None else []))
        frames = []
        for index, clip in enumerate(clips or [None]):
            frame = FrameRef(
                video_id=context.task.video_id or context.task.sample_key,
                timestamp_s=2.5 + float(index),
                clip=clip,
                metadata={"source_path": "frame_%02d.png" % index},
            )
            frames.append(frame)
        self.last_frame = frames[0]
        return ToolResult(
            tool_name="frame_retriever",
            ok=True,
            data={"frames": [frame.dict() for frame in frames]},
            summary="Frame retrieval succeeded.",
        )


class FakeSpatialGrounderAdapter(object):
    name = "spatial_grounder"
    request_model = SpatialGrounderRequest

    def __init__(self):
        self.calls = 0
        self.last_frame = None

    def parse_request(self, arguments):
        request = self.request_model.parse_obj(arguments)
        self.last_frame = request.frame
        return request

    def execute(self, request, context):
        self.calls += 1
        region = {
            "frame": request.frame.dict(),
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "label": "target",
            "metadata": {"confidence": 0.9},
        }
        return ToolResult(
            tool_name="spatial_grounder",
            ok=True,
            data={
                "frame": request.frame.dict(),
                "detections": [{"label": "target", "bbox": [10.0, 20.0, 30.0, 40.0], "confidence": 0.9}],
                "regions": [region],
                "region": region,
                "spatial_description": "Target localized.",
            },
            summary="Spatial grounding succeeded.",
        )


class FakeASRAdapter(ToolAdapter):
    name = "asr"
    request_model = ASRRequest

    def __init__(self):
        self.calls = 0
        self.last_request = None

    def parse_request(self, arguments):
        request = super().parse_request(arguments)
        self.last_request = request
        return request

    def execute(self, request, context):
        self.calls += 1
        clip = request.clip
        return ToolResult(
            tool_name="asr",
            ok=True,
            data={
                "clip": clip.dict(),
                "text": "hello world",
                "segments": [],
                "backend": "fake_asr",
            },
            summary="ASR succeeded.",
        )


class FailingASRAdapter(ToolAdapter):
    name = "asr"
    request_model = ASRRequest

    def __init__(self):
        self.calls = 0

    def execute(self, request, context):
        self.calls += 1
        return ToolResult(
            tool_name="asr",
            ok=False,
            data={"error": "ASR unavailable."},
            summary="ASR unavailable.",
            metadata={"error": "ASR unavailable."},
        )


class FakeGenericPurposeAdapter(ToolAdapter):
    name = "generic_purpose"
    request_model = GenericPurposeRequest

    def __init__(self):
        self.calls = 0
        self.last_request = None

    def parse_request(self, arguments):
        request = super().parse_request(arguments)
        self.last_request = request
        return request

    def execute(self, request, context):
        self.calls += 1
        return ToolResult(
            tool_name="generic_purpose",
            ok=True,
            data={
                "answer": "A",
                "frames": [item.dict() for item in list(request.frames or [])],
                "evidence_ids": list(request.evidence_ids or []),
                "text_contexts": list(request.text_contexts or []),
            },
            summary="Generic reasoning succeeded.",
        )


class MultiClipTemporalAdapter(object):
    request_model = VisualTemporalGrounderRequest

    def parse_request(self, arguments):
        return self.request_model.parse_obj(arguments)

    def execute(self, request, context):
        return ToolResult(
            tool_name="visual_temporal_grounder",
            ok=True,
            data={
                "query": request.query,
                "clips": [
                    {
                        "video_id": context.task.video_id or context.task.sample_key,
                        "start_s": 0.0,
                        "end_s": 5.0,
                        "metadata": {"confidence": 0.9},
                    },
                    {
                        "video_id": context.task.video_id or context.task.sample_key,
                        "start_s": 10.0,
                        "end_s": 15.0,
                        "metadata": {"confidence": 0.8},
                    },
                ],
            },
            summary="The event appears in two clips.",
        )


def _models_config():
    return ModelsConfig(
        agents={
            "planner": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_synthesizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_auditor": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "atomicizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
        },
        tools={
            "visual_temporal_grounder": ToolConfig(
                enabled=True,
                prompt_version="tool_v1",
            ),
            "frame_retriever": ToolConfig(
                enabled=True,
                prompt_version="tool_v1",
            ),
            "spatial_grounder": ToolConfig(
                enabled=True,
                prompt_version="tool_v1",
            ),
            "asr": ToolConfig(
                enabled=True,
                prompt_version="tool_v1",
            ),
            "generic_purpose": ToolConfig(
                enabled=True,
                prompt_version="tool_v1",
            ),
        },
    )


def test_executor_reuses_shared_cache(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    adapter = FakeAdapter()
    frame_adapter = FakeFrameRetrieverAdapter()
    executor = PlanExecutor(
        tool_registry=FakeRegistry(
            {
                "visual_temporal_grounder": adapter,
                "frame_retriever": frame_adapter,
            }
        ),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="When does the event happen?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    (tmp_path / "video.mp4").write_bytes(b"video-bytes")
    run = workspace.create_run(task)
    ledger = EvidenceLedger(run)
    context = ToolExecutionContext(
        workspace=workspace,
        run=run,
        task=task,
        models_config=_models_config(),
        preprocess_bundle={"segments": []},
    )
    plan = ExecutionPlan(
        strategy="Locate the event.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find the event.",
                arguments={"tool_name": "visual_temporal_grounder", "query": "goal"},
                input_refs=[],
                depends_on=[],
            )
        ],
        refinement_instructions="",
    )
    executor.execute_plan(plan, context, ledger, video_fingerprint="abc123")
    executor.execute_plan(plan, context, ledger, video_fingerprint="abc123")
    assert adapter.calls == 1


def test_executor_writes_runtime_file_for_process_style_adapters(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    adapter = FakeAdapter()
    adapter._runtime_payload = lambda context: {
        "model_name": "TencentARC/TimeLens-8B",
        "resolved_model_path": "/tmp/timelens-model",
    }
    executor = PlanExecutor(
        tool_registry=FakeRegistry({"visual_temporal_grounder": adapter}),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="When does the event happen?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    (tmp_path / "video.mp4").write_bytes(b"video-bytes")
    run = workspace.create_run(task)
    ledger = EvidenceLedger(run)
    context = ToolExecutionContext(
        workspace=workspace,
        run=run,
        task=task,
        models_config=_models_config(),
        preprocess_bundle={"segments": []},
    )
    plan = ExecutionPlan(
        strategy="Locate the event.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find the event.",
                arguments={"tool_name": "visual_temporal_grounder", "query": "goal"},
                input_refs=[],
                depends_on=[],
            )
        ],
        refinement_instructions="",
    )

    executor.execute_plan(plan, context, ledger, video_fingerprint="abc123")

    runtime_file = run.tool_step_dir(1, "visual_temporal_grounder") / "runtime.json"
    request_full_file = run.tool_step_dir(1, "visual_temporal_grounder") / "request_full.json"
    result_file = run.tool_step_dir(1, "visual_temporal_grounder") / "result.json"
    timing_file = run.tool_step_dir(1, "visual_temporal_grounder") / "timing.json"
    assert request_full_file.exists()
    assert '"query": "goal"' in request_full_file.read_text(encoding="utf-8")
    assert runtime_file.exists()
    assert '"resolved_model_path": "/tmp/timelens-model"' in runtime_file.read_text(encoding="utf-8")
    assert result_file.exists()
    assert '"timing"' in result_file.read_text(encoding="utf-8")
    assert timing_file.exists()
    assert '"execution_mode": "executed"' in timing_file.read_text(encoding="utf-8")


def test_executor_resolves_clip_alias_and_numeric_path(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    temporal_adapter = FakeAdapter()
    frame_adapter = FakeFrameRetrieverAdapter()
    executor = PlanExecutor(
        tool_registry=FakeRegistry(
            {
                "visual_temporal_grounder": temporal_adapter,
                "frame_retriever": frame_adapter,
            }
        ),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="What appears in the grounded segment?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    (tmp_path / "video.mp4").write_bytes(b"video-bytes")
    run = workspace.create_run(task)
    ledger = EvidenceLedger(run)
    context = ToolExecutionContext(
        workspace=workspace,
        run=run,
        task=task,
        models_config=_models_config(),
        preprocess_bundle={"segments": []},
    )

    alias_plan = ExecutionPlan(
        strategy="Ground then retrieve a frame.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find the relevant segment.",
                arguments={"tool_name": "visual_temporal_grounder", "query": "goal"},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Get a frame from the first clip.",
                arguments={"tool_name": "frame_retriever", "query": "goal"},
                input_refs=[{"target_field": "clip", "source": {"step_id": 1, "field_path": "clip"}}],
                depends_on=[1],
            ),
        ],
        refinement_instructions="",
    )
    executor.execute_plan(alias_plan, context, ledger, video_fingerprint="abc123")
    assert frame_adapter.last_clip.start_s == 0.0
    assert frame_adapter.last_clip.end_s == 5.0

    numeric_plan = ExecutionPlan(
        strategy="Ground then retrieve a frame.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find the relevant segment.",
                arguments={"tool_name": "visual_temporal_grounder", "query": "goal"},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Get a frame from the first clip.",
                arguments={"tool_name": "frame_retriever", "query": "goal"},
                input_refs=[{"target_field": "clip", "source": {"step_id": 1, "field_path": "clips.0"}}],
                depends_on=[1],
            ),
        ],
        refinement_instructions="",
    )
    executor.execute_plan(numeric_plan, context, ledger, video_fingerprint="xyz789")
    assert frame_adapter.last_clip.start_s == 0.0
    assert frame_adapter.last_clip.end_s == 5.0

    bracket_plan = ExecutionPlan(
        strategy="Ground then retrieve a frame.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find the relevant segment.",
                arguments={"tool_name": "visual_temporal_grounder", "query": "goal"},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Get a frame from the first planner-selected segment.",
                arguments={"tool_name": "frame_retriever", "query": "goal"},
                input_refs=[{"target_field": "clip", "source": {"step_id": 1, "field_path": "segments[0]"}}],
                depends_on=[1],
            ),
        ],
        refinement_instructions="",
    )
    executor.execute_plan(bracket_plan, context, ledger, video_fingerprint="segpath456")
    assert frame_adapter.last_clip.start_s == 0.0
    assert frame_adapter.last_clip.end_s == 5.0


def test_executor_autofills_frame_from_dependency(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    temporal_adapter = FakeAdapter()
    frame_adapter = FakeFrameRetrieverAdapter()
    spatial_adapter = FakeSpatialGrounderAdapter()
    executor = PlanExecutor(
        tool_registry=FakeRegistry(
            {
                "visual_temporal_grounder": temporal_adapter,
                "frame_retriever": frame_adapter,
                "spatial_grounder": spatial_adapter,
            }
        ),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="Where is the highlighted object?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    (tmp_path / "video.mp4").write_bytes(b"video-bytes")
    run = workspace.create_run(task)
    ledger = EvidenceLedger(run)
    context = ToolExecutionContext(
        workspace=workspace,
        run=run,
        task=task,
        models_config=_models_config(),
        preprocess_bundle={"segments": []},
    )
    plan = ExecutionPlan(
        strategy="Ground the segment, get a frame, then localize the object.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find the relevant segment.",
                arguments={"tool_name": "visual_temporal_grounder", "query": "highlighted object"},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Get a representative frame.",
                arguments={"tool_name": "frame_retriever", "query": "highlighted object"},
                input_refs=[{"target_field": "clip", "source": {"step_id": 1, "field_path": "clip"}}],
                depends_on=[1],
            ),
            PlanStep(
                step_id=3,
                tool_name="spatial_grounder",
                purpose="Localize the object in the retrieved frame.",
                arguments={"tool_name": "spatial_grounder", "query": "Locate the highlighted object"},
                input_refs=[],
                depends_on=[2],
            ),
        ],
        refinement_instructions="",
    )
    executor.execute_plan(plan, context, ledger, video_fingerprint="framefill123")
    assert spatial_adapter.calls == 1
    assert spatial_adapter.last_frame is not None
    assert spatial_adapter.last_frame.timestamp_s == 2.5


def test_executor_passes_clip_lists_into_frame_retriever(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    temporal_adapter = MultiClipTemporalAdapter()
    frame_adapter = FakeFrameRetrieverAdapter()
    executor = PlanExecutor(
        tool_registry=FakeRegistry(
            {
                "visual_temporal_grounder": temporal_adapter,
                "frame_retriever": frame_adapter,
            }
        ),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="What appears in the grounded segments?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    (tmp_path / "video.mp4").write_bytes(b"video-bytes")
    run = workspace.create_run(task)
    ledger = EvidenceLedger(run)
    context = ToolExecutionContext(
        workspace=workspace,
        run=run,
        task=task,
        models_config=_models_config(),
        preprocess_bundle={"segments": []},
    )
    plan = ExecutionPlan(
        strategy="Ground the video and inspect each matched clip.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find all matching segments.",
                arguments={"tool_name": "visual_temporal_grounder", "query": "goal"},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Retrieve a representative frame from each matched clip.",
                arguments={"tool_name": "frame_retriever", "query": "goal", "top_k": 2},
                input_refs=[{"target_field": "clip", "source": {"step_id": 1, "field_path": "clips"}}],
                depends_on=[1],
            ),
        ],
        refinement_instructions="",
    )

    executor.execute_plan(plan, context, ledger, video_fingerprint="multiclip123")

    assert frame_adapter.calls == 1
    assert len(frame_adapter.last_clips) == 2
    assert frame_adapter.last_clips[0].start_s == 0.0
    assert frame_adapter.last_clips[1].start_s == 10.0


def test_hydrate_arguments_with_task_context_fills_clip_shape():
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="What is visible?",
        options=[],
        video_path="/tmp/video.mp4",
    )

    hydrated = hydrate_arguments_with_task_context({"clip": {"time_start_s": 0.0, "time_end_s": 5.0}}, task)

    assert hydrated["clip"]["video_id"] == "sample1"
    assert hydrated["clip"]["start_s"] == 0.0
    assert hydrated["clip"]["end_s"] == 5.0


def test_hydrate_arguments_with_task_context_expands_full_video_clip_shorthand():
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="What text is visible?",
        options=[],
        video_path="/tmp/video.mp4",
        metadata={"video_duration": 12.5},
    )

    hydrated = hydrate_arguments_with_task_context({"clip": "full_video", "query": "scoreboard"}, task)

    assert hydrated["clip"]["video_id"] == "sample1"
    assert hydrated["clip"]["start_s"] == 0.0
    assert hydrated["clip"]["end_s"] == 12.5
    assert hydrated["query"] == "scoreboard"


def test_hydrate_arguments_with_task_context_expands_full_video_clip_list_shorthand():
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="What text is visible?",
        options=[],
        video_path="/tmp/video.mp4",
        metadata={"video_duration_s": 12.5},
    )

    hydrated = hydrate_arguments_with_task_context({"clips": ["full_video"]}, task)

    assert len(hydrated["clips"]) == 1
    assert hydrated["clips"][0]["video_id"] == "sample1"
    assert hydrated["clips"][0]["start_s"] == 0.0
    assert hydrated["clips"][0]["end_s"] == 12.5


def test_executor_defaults_asr_to_full_video_when_clip_is_missing(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    adapter = FakeASRAdapter()
    executor = PlanExecutor(
        tool_registry=FakeRegistry({"asr": adapter}),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="What is said?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
        metadata={"video_duration_s": 12.5},
    )
    (tmp_path / "video.mp4").write_bytes(b"video-bytes")
    run = workspace.create_run(task)
    ledger = EvidenceLedger(run)
    context = ToolExecutionContext(
        workspace=workspace,
        run=run,
        task=task,
        models_config=_models_config(),
        preprocess_bundle={"segments": []},
    )
    plan = ExecutionPlan(
        strategy="Transcribe the video.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="asr",
                purpose="Transcribe the speech in the video.",
                arguments={"speaker_attribution": False},
                input_refs=[],
                depends_on=[],
            )
        ],
        refinement_instructions="",
    )

    executor.execute_plan(plan, context, ledger, video_fingerprint="asrfullvideo123")

    assert adapter.calls == 1
    assert adapter.last_request is not None
    assert adapter.last_request.clip is not None
    assert adapter.last_request.clip.video_id == "sample1"
    assert adapter.last_request.clip.start_s == 0.0
    assert adapter.last_request.clip.end_s == 12.5


def test_executor_does_not_reuse_failed_cached_results(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    failing_adapter = FailingASRAdapter()
    failing_executor = PlanExecutor(
        tool_registry=FakeRegistry({"asr": failing_adapter}),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="What is said?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
        metadata={"video_duration_s": 12.5},
    )
    (tmp_path / "video.mp4").write_bytes(b"video-bytes")
    plan = ExecutionPlan(
        strategy="Transcribe the video.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="asr",
                purpose="Transcribe the speech in the video.",
                arguments={"speaker_attribution": False},
                input_refs=[],
                depends_on=[],
            )
        ],
        refinement_instructions="",
    )
    first_run = workspace.create_run(task)
    first_ledger = EvidenceLedger(first_run)
    first_context = ToolExecutionContext(
        workspace=workspace,
        run=first_run,
        task=task,
        models_config=_models_config(),
        preprocess_bundle={"segments": []},
    )

    first_results = failing_executor.execute_plan(plan, first_context, first_ledger, video_fingerprint="asrretry123")

    assert failing_adapter.calls == 1
    assert first_results[0]["result"]["ok"] is False
    assert first_results[0]["result"]["cache_hit"] is False

    success_adapter = FakeASRAdapter()
    success_executor = PlanExecutor(
        tool_registry=FakeRegistry({"asr": success_adapter}),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )
    second_run = workspace.create_run(task)
    second_ledger = EvidenceLedger(second_run)
    second_context = ToolExecutionContext(
        workspace=workspace,
        run=second_run,
        task=task,
        models_config=_models_config(),
        preprocess_bundle={"segments": []},
    )

    second_results = success_executor.execute_plan(plan, second_context, second_ledger, video_fingerprint="asrretry123")

    assert success_adapter.calls == 1
    assert second_results[0]["result"]["ok"] is True
    assert second_results[0]["result"]["cache_hit"] is False


def test_augment_dependency_output_promotes_analysis_and_best_frame():
    payload = {
        "frames": [{"video_id": "sample1", "timestamp_s": 2.5}],
        "regions": [{"frame": {"video_id": "sample1", "timestamp_s": 2.5}, "label": "puzzle", "bbox": [1, 2, 3, 4]}],
        "response": "Counted 18 triangles.",
    }

    augmented = augment_dependency_output(payload)

    assert augmented["best_frame"]["timestamp_s"] == 2.5
    assert augmented["puzzle_bbox"]["label"] == "puzzle"
    assert augmented["analysis"] == "Counted 18 triangles."


def test_executor_merges_duplicate_input_refs_for_plural_targets(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    executor = PlanExecutor(
        tool_registry=FakeRegistry(
            {
                "visual_temporal_grounder": MultiClipTemporalAdapter(),
                "frame_retriever": FakeFrameRetrieverAdapter(),
            }
        ),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )
    step = PlanStep(
        step_id=7,
        tool_name="frame_retriever",
        purpose="Merge duplicated bindings.",
        arguments={"query": "goal"},
        input_refs=[
            {"target_field": "clips", "source": {"step_id": 1, "field_path": "clips"}},
            {"target_field": "clips", "source": {"step_id": 2, "field_path": "clips"}},
        ],
        depends_on=[1, 2],
    )
    step_outputs = {
        1: {"clips": [{"video_id": "sample1", "start_s": 0.0, "end_s": 5.0}]},
        2: {"clips": [{"video_id": "sample1", "start_s": 10.0, "end_s": 15.0}]},
    }

    resolved = executor._resolve_arguments(step, step_outputs)

    assert len(resolved["clips"]) == 2
    assert resolved["clips"][0]["start_s"] == 0.0
    assert resolved["clips"][1]["start_s"] == 10.0

    text_step = PlanStep(
        step_id=8,
        tool_name="frame_retriever",
        purpose="Merge scalar strings into a plural target.",
        arguments={"query": "goal"},
        input_refs=[
            {"target_field": "text_contexts", "source": {"step_id": 1, "field_path": "text"}},
            {"target_field": "text_contexts", "source": {"step_id": 2, "field_path": "text"}},
        ],
        depends_on=[1, 2],
    )
    text_outputs = {
        1: {"text": "line one"},
        2: {"text": "line two"},
    }

    resolved_text = executor._resolve_arguments(text_step, text_outputs)

    assert resolved_text["text_contexts"] == ["line one", "line two"]


def test_resolve_arguments_wraps_single_scalar_binding_for_plural_target(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    executor = PlanExecutor(
        tool_registry=FakeRegistry({}),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )
    text_step = PlanStep(
        step_id=9,
        tool_name="generic_purpose",
        purpose="Keep OCR text list-shaped for the request model.",
        arguments={"query": "goal"},
        input_refs=[
            {"target_field": "text_contexts", "source": {"step_id": 1, "field_path": "text"}},
        ],
        depends_on=[1],
    )
    text_outputs = {
        1: {"text": "line one"},
    }

    resolved_text = executor._resolve_arguments(text_step, text_outputs)

    assert resolved_text["text_contexts"] == ["line one"]


def test_executor_autofills_frame_bundle_into_generic_purpose_from_dependency(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    frame_adapter = FakeFrameRetrieverAdapter()
    generic_adapter = FakeGenericPurposeAdapter()
    executor = PlanExecutor(
        tool_registry=FakeRegistry(
            {
                "frame_retriever": frame_adapter,
                "generic_purpose": generic_adapter,
            }
        ),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="What is visible in the retrieved frame bundle?",
        options=[],
        video_path=str(tmp_path / "video.mp4"),
    )
    (tmp_path / "video.mp4").write_bytes(b"video-bytes")
    run = workspace.create_run(task)
    ledger = EvidenceLedger(run)
    context = ToolExecutionContext(
        workspace=workspace,
        run=run,
        task=task,
        models_config=_models_config(),
        preprocess_bundle={"segments": []},
    )
    plan = ExecutionPlan(
        strategy="Retrieve frames and inspect them.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="frame_retriever",
                purpose="Get the relevant frame bundle.",
                arguments={
                    "tool_name": "frame_retriever",
                    "query": "living room table",
                    "clip": {"video_id": "sample1", "start_s": 0.0, "end_s": 5.0},
                },
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="generic_purpose",
                purpose="Inspect the retrieved frames.",
                arguments={"tool_name": "generic_purpose", "query": "Inspect the retrieved frames."},
                input_refs=[],
                depends_on=[1],
            ),
        ],
        refinement_instructions="",
    )

    executor.execute_plan(plan, context, ledger, video_fingerprint="genericframes123")

    assert generic_adapter.calls == 1
    assert len(generic_adapter.last_request.frames) == 1
    assert generic_adapter.last_request.frames[0].timestamp_s == 2.5


def test_executor_seeds_recent_evidence_ids_for_context_free_reuse_followup(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    generic_adapter = FakeGenericPurposeAdapter()
    executor = PlanExecutor(
        tool_registry=FakeRegistry({"generic_purpose": generic_adapter}),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=_models_config(),
    )
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="How many bottles are clearly empty?",
        options=["A", "B", "C"],
        video_path=str(tmp_path / "video.mp4"),
    )
    (tmp_path / "video.mp4").write_bytes(b"video-bytes")
    run = workspace.create_run(task)
    ledger = EvidenceLedger(run)
    ledger.append(
        EvidenceEntry(
            evidence_id="ev_01_demo",
            tool_name="frame_retriever",
            evidence_text="Retrieved living-room coffee-table frames.",
            observation_ids=[],
        ),
        [],
    )
    ledger.append(
        EvidenceEntry(
            evidence_id="ev_02_demo",
            tool_name="asr",
            evidence_text='Speaker says "Come to Phil\'s MU Nation today."',
            observation_ids=[],
        ),
        [],
    )
    context = ToolExecutionContext(
        workspace=workspace,
        run=run,
        task=task,
        models_config=_models_config(),
        preprocess_bundle={"segments": []},
    )
    plan = ExecutionPlan(
        strategy="Reuse prior grounded evidence.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="generic_purpose",
                purpose="Determine, from the previously retrieved living-room frames, how many bottles are actually empty.",
                arguments={
                    "tool_name": "generic_purpose",
                    "query": "Inspect the retrieved frames from the earlier living-room scene and decide how many bottles are actually empty.",
                },
                input_refs=[],
                depends_on=[],
            ),
        ],
        refinement_instructions="",
    )

    executor.execute_plan(plan, context, ledger, video_fingerprint="reusefollowup123")

    assert generic_adapter.calls == 1
    assert generic_adapter.last_request.evidence_ids == ["ev_01_demo", "ev_02_demo"]
