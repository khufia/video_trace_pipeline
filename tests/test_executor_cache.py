from video_trace_pipeline.orchestration.executor import PlanExecutor, augment_dependency_output, hydrate_arguments_with_task_context
from video_trace_pipeline.schemas import (
    AgentConfig,
    GenericPurposeRequest,
    ExecutionPlan,
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


class FakeFrameRetrieverAdapter(object):
    request_model = FrameRetrieverRequest

    def __init__(self):
        self.calls = 0
        self.last_clip = None
        self.last_frame = None

    def parse_request(self, arguments):
        request = self.request_model.parse_obj(arguments)
        self.last_clip = request.clip
        return request

    def execute(self, request, context):
        self.calls += 1
        frame = FrameRef(
            video_id=context.task.video_id or context.task.sample_key,
            timestamp_s=2.5,
            clip=request.clip,
            metadata={"source_path": "frame.png"},
        )
        self.last_frame = frame
        return ToolResult(
            tool_name="frame_retriever",
            ok=True,
            data={"frames": [frame.dict()]},
            summary="Frame retrieval succeeded.",
        )


class FakeSpatialGrounderAdapter(object):
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


class FakeGenericAdapter(ToolAdapter):
    name = "generic_purpose"
    request_model = GenericPurposeRequest

    def __init__(self):
        self.calls = 0
        self.last_frame = None

    def parse_request(self, arguments):
        request = super().parse_request(arguments)
        self.last_frame = request.frame
        return request

    def execute(self, request, context):
        self.calls += 1
        return ToolResult(
            tool_name="generic_purpose",
            ok=True,
            data={"answer": "placeholder"},
            summary="Generic reasoning succeeded.",
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
                backend="internal_temporal_grounder",
                prompt_version="tool_v1",
            ),
            "frame_retriever": ToolConfig(
                enabled=True,
                backend="internal_frame_retriever",
                prompt_version="tool_v1",
            ),
            "spatial_grounder": ToolConfig(
                enabled=True,
                backend="internal_spatial_grounder",
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


def test_executor_maps_image_alias_to_frame(tmp_path):
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
                input_refs=[{"target_field": "image", "source": {"step_id": 2, "field_path": "frames[0]"}}],
                depends_on=[2],
            ),
        ],
        refinement_instructions="",
    )
    executor.execute_plan(plan, context, ledger, video_fingerprint="imagealias123")
    assert spatial_adapter.calls == 1
    assert spatial_adapter.last_frame is not None
    assert spatial_adapter.last_frame.timestamp_s == 2.5


def test_executor_maps_image_region_alias_to_frame(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    workspace = WorkspaceManager(profile)
    temporal_adapter = FakeAdapter()
    frame_adapter = FakeFrameRetrieverAdapter()
    spatial_adapter = FakeSpatialGrounderAdapter()
    generic_adapter = FakeGenericAdapter()
    executor = PlanExecutor(
        tool_registry=FakeRegistry(
            {
                "visual_temporal_grounder": temporal_adapter,
                "frame_retriever": frame_adapter,
                "spatial_grounder": spatial_adapter,
                "generic_purpose": generic_adapter,
            }
        ),
        evidence_cache=SharedEvidenceCache(workspace),
        extractor=ObservationExtractor(),
        models_config=ModelsConfig(
            agents=_models_config().agents,
            tools={
                **_models_config().tools,
                "generic_purpose": ToolConfig(
                    enabled=True,
                    backend="openai_multimodal",
                    prompt_version="tool_v1",
                ),
            },
        ),
    )
    task = TaskSpec(
        benchmark="videomathqa",
        sample_key="sample1",
        question="How many shapes are visible?",
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
        strategy="Ground the segment, get a frame, localize a region, then analyze it.",
        use_summary=True,
        steps=[
            PlanStep(
                step_id=1,
                tool_name="visual_temporal_grounder",
                purpose="Find the relevant segment.",
                arguments={"tool_name": "visual_temporal_grounder", "query": "shapes"},
                input_refs=[],
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                tool_name="frame_retriever",
                purpose="Get a representative frame.",
                arguments={"tool_name": "frame_retriever", "query": "shapes"},
                input_refs=[{"target_field": "clip", "source": {"step_id": 1, "field_path": "clip"}}],
                depends_on=[1],
            ),
            PlanStep(
                step_id=3,
                tool_name="spatial_grounder",
                purpose="Localize the target region.",
                arguments={"tool_name": "spatial_grounder", "query": "Locate the shape cluster"},
                input_refs=[{"target_field": "frame", "source": {"step_id": 2, "field_path": "frames[0]"}}],
                depends_on=[2],
            ),
            PlanStep(
                step_id=4,
                tool_name="generic_purpose",
                purpose="Analyze the localized region.",
                arguments={"tool_name": "generic_purpose", "task": "Count the shapes."},
                input_refs=[{"target_field": "image_region", "source": {"step_id": 3, "field_path": "region"}}],
                depends_on=[3],
            ),
        ],
        refinement_instructions="",
    )
    executor.execute_plan(plan, context, ledger, video_fingerprint="imageregion123")
    assert generic_adapter.calls == 1
    assert generic_adapter.last_frame is not None
    assert generic_adapter.last_frame.timestamp_s == 2.5


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
