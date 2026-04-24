from video_trace_pipeline.agents.planner import PlannerAgent
from video_trace_pipeline.orchestration.pipeline import PipelineRunner
from video_trace_pipeline.schemas import (
    AgentConfig,
    AuditReport,
    ExecutionPlan,
    InferenceStep,
    MachineProfile,
    ModelsConfig,
    TaskSpec,
    TracePackage,
    ToolConfig,
)
from video_trace_pipeline.storage import EvidenceLedger


class FakePreprocessor(object):
    def resolve_preprocess_settings(self, clip_duration_s):
        return {"clip_duration_s": float(clip_duration_s or 60.0)}

    def get_or_build(self, task, clip_duration_s):
        return {
            "cache_hit": False,
            "cache_dir": "cache/preprocess/fake",
            "manifest": {},
            "segments": [],
            "summary": "A short summary.",
            "video_fingerprint": "vid123",
        }


class FakePlanner(object):
    def plan(self, **kwargs):
        from video_trace_pipeline.schemas import ExecutionPlan

        return "{}", ExecutionPlan(strategy="No tools needed.", use_summary=True, steps=[], refinement_instructions="")


class FakeExecutor(object):
    def execute_plan(self, **kwargs):
        return []


class FakeSynthesizer(object):
    def synthesize(self, task, mode, evidence_entries, observations, current_trace, refinement_instructions):
        trace = TracePackage(
            task_key=task.sample_key,
            mode=mode,
            evidence_entries=[],
            inference_steps=[
                InferenceStep(
                    step_id=1,
                    text="The summary directly answers the question.",
                    supporting_observation_ids=[],
                    answer_relevance="high",
                )
            ],
            final_answer="A",
            benchmark_renderings={},
        )
        return "{}", trace


class FakeAuditor(object):
    def audit(self, task, trace_package, evidence_summary):
        return "{}", AuditReport(verdict="PASS", confidence=0.9, scores={"support": 1.0}, findings=[], feedback="")


class FakeLLMClient(object):
    def __init__(self):
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return (
            ExecutionPlan(strategy="Plan from prompt.", use_summary=True, steps=[], refinement_instructions=""),
            "{}",
        )


def _models_config():
    return ModelsConfig(
        agents={
            "planner": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_synthesizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_auditor": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "atomicizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
        },
        tools={"dense_captioner": ToolConfig(enabled=True, model="yaolily/TimeChat-Captioner-GRPO-7B")},
    )


def test_pipeline_runner_writes_final_outputs(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    runner = PipelineRunner(profile, _models_config())
    runner.workspace.package_results_root = tmp_path / "repo_results"
    runner.workspace.package_results_root.mkdir(parents=True, exist_ok=True)
    runner.preprocessor = FakePreprocessor()
    runner.planner = FakePlanner()
    runner.executor = FakeExecutor()
    runner.synthesizer = FakeSynthesizer()
    runner.auditor = FakeAuditor()
    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What is the answer?",
        options=["A", "B"],
        video_path=str(tmp_path / "video.mp4"),
    )
    (tmp_path / "video.mp4").write_bytes(b"video")
    result = runner.run_task(
        task,
        mode="generate",
        max_rounds=1,
        clip_duration_s=30.0,
        results_name="rur_refiner_inputs",
    )
    final_result_path = tmp_path / "workspace" / result["run_dir"] / "results" / "final_result.json"
    readme_path = tmp_path / "workspace" / result["run_dir"] / "README.md"
    final_result_readable_path = tmp_path / "workspace" / result["run_dir"] / "results" / "final_result_readable.md"
    debug_report_path = tmp_path / "workspace" / result["run_dir"] / "debug" / "README.md"
    assert final_result_path.exists()
    assert readme_path.exists()
    assert final_result_readable_path.exists()
    assert debug_report_path.exists()
    assert result["trace_package"]["final_answer"] == "A"
    exported_dir = tmp_path / "repo_results" / "rur_refiner_inputs"
    assert exported_dir.exists()
    assert result["exported_results_dir"].endswith("/sample1")


def test_pipeline_runner_preloads_models_before_preprocess(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    runner = PipelineRunner(
        profile,
        _models_config(),
        persist_tool_models=["dense_captioner"],
        preload_persisted_models=True,
    )
    runner.workspace.package_results_root = tmp_path / "repo_results"
    runner.workspace.package_results_root.mkdir(parents=True, exist_ok=True)

    call_order = []

    class OrderedPreprocessor(object):
        def resolve_preprocess_settings(self, clip_duration_s):
            return {"clip_duration_s": float(clip_duration_s or 60.0)}

        def get_or_build(self, task, clip_duration_s):
            call_order.append("preprocess")
            return {
                "cache_hit": False,
                "cache_dir": "cache/preprocess/fake",
                "manifest": {},
                "segments": [],
                "summary": "A short summary.",
                "video_fingerprint": "vid123",
            }

    runner.preprocessor = OrderedPreprocessor()
    runner.planner = FakePlanner()
    runner.executor = FakeExecutor()
    runner.synthesizer = FakeSynthesizer()
    runner.auditor = FakeAuditor()

    def fake_preload():
        call_order.append("preload")
        return {
            "enabled": True,
            "requested_tools": ["dense_captioner"],
            "loaded_models": [
                {
                    "tool_name": "dense_captioner",
                    "runner_type": "timechat",
                    "model_name": "yaolily/TimeChat-Captioner-GRPO-7B",
                    "resolved_model_path": "/models/timechat",
                    "device_label": "cuda:1",
                }
            ],
            "parallel_workers": 1,
            "shared_tools": [],
        }

    runner.tool_registry.preload_persistent_models = fake_preload

    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What is the answer?",
        options=["A", "B"],
        video_path=str(tmp_path / "video.mp4"),
    )
    (tmp_path / "video.mp4").write_bytes(b"video")

    runner.run_task(
        task,
        mode="generate",
        max_rounds=1,
        clip_duration_s=30.0,
        results_name="rur_refiner_inputs",
    )

    assert call_order[:2] == ["preload", "preprocess"]


def test_planner_agent_uses_prompt_based_request():
    llm_client = FakeLLMClient()
    planner = PlannerAgent(llm_client=llm_client, agent_config=AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"))
    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="Among the shown charts, which option has the largest percentage difference?",
        options=["A. 10%", "B. 25%", "C. 40%"],
        video_path="video.mp4",
    )
    tool_catalog = {
        "visual_temporal_grounder": {"request_fields": ["query", "top_k"]},
        "frame_retriever": {},
        "ocr": {},
        "generic_purpose": {},
        "spatial_grounder": {},
    }

    raw, plan = planner.plan(
        task=task,
        mode="refine",
        summary_text="summary",
        compact_rounds=[],
        retrieved_observations=[],
        audit_feedback={"feedback": "Need direct readings of the values."},
        tool_catalog=tool_catalog,
    )
    assert raw == "{}"
    assert plan.strategy == "Plan from prompt."
    assert llm_client.calls
    assert llm_client.calls[0]["system_prompt"].startswith("You are the Planner")
    assert "AVAILABLE_TOOLS:" in llm_client.calls[0]["user_prompt"]


def test_planner_agent_repairs_zero_based_input_refs_before_validation():
    class FakeLLMClientWithZeroBasedRef(object):
        def complete_json(self, **kwargs):
            del kwargs
            return (
                {
                    "strategy": "Plan from prompt.",
                    "use_summary": True,
                    "steps": [
                        {
                            "step_id": 1,
                            "tool_name": "visual_temporal_grounder",
                            "purpose": "Find the right clip.",
                            "arguments": {"query": "Find the chart."},
                            "input_refs": [],
                            "depends_on": [],
                        },
                        {
                            "step_id": 2,
                            "tool_name": "frame_retriever",
                            "purpose": "Grab a frame.",
                            "arguments": {"query": "Best frame."},
                            "input_refs": [
                                {"target_field": "clip", "source": {"step_id": 0, "field_path": "clips"}}
                            ],
                            "depends_on": [0],
                        },
                    ],
                    "refinement_instructions": "",
                },
                "{}",
            )

    planner = PlannerAgent(
        llm_client=FakeLLMClientWithZeroBasedRef(),
        agent_config=AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
    )
    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="Among the shown charts, which option has the largest percentage difference?",
        options=["A. 10%", "B. 25%", "C. 40%"],
        video_path="video.mp4",
    )

    raw, plan = planner.plan(
        task=task,
        mode="refine",
        summary_text="summary",
        compact_rounds=[],
        retrieved_observations=[],
        audit_feedback=None,
        tool_catalog={},
    )

    assert raw == "{}"
    assert plan.steps[1].input_refs[0].source.step_id == 1
    assert plan.steps[1].depends_on == [1]


def test_pipeline_evidence_summary_is_stable_across_runs(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    runner = PipelineRunner(profile, _models_config())
    task = TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        question="What is the answer?",
        options=["A", "B"],
        video_path=str(tmp_path / "video.mp4"),
    )
    (tmp_path / "video.mp4").write_bytes(b"video")

    run_a = runner.workspace.create_run(task)
    run_b = runner.workspace.create_run(task)
    summary_a = runner._build_evidence_summary(EvidenceLedger(run_a))
    summary_b = runner._build_evidence_summary(EvidenceLedger(run_b))

    assert "database_path" not in summary_a
    assert summary_a == summary_b
