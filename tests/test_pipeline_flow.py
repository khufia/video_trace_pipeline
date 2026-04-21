from video_trace_pipeline.orchestration.pipeline import PipelineRunner
from video_trace_pipeline.schemas import (
    AgentConfig,
    AuditReport,
    InferenceStep,
    MachineProfile,
    ModelsConfig,
    TaskSpec,
    TracePackage,
    ToolConfig,
)


class FakePreprocessor(object):
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


def _models_config():
    return ModelsConfig(
        agents={
            "planner": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_synthesizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_auditor": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "atomicizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
        },
        tools={"dense_captioner": ToolConfig(enabled=True, backend="internal_dense_captioner", model="gpt-5.4", endpoint="default")},
    )


def test_pipeline_runner_writes_final_outputs(tmp_path):
    profile = MachineProfile(workspace_root=str(tmp_path / "workspace"))
    runner = PipelineRunner(profile, _models_config())
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
    result = runner.run_task(task, mode="generate", max_rounds=1, clip_duration_s=30.0)
    final_result_path = tmp_path / "workspace" / result["run_dir"] / "results" / "final_result.json"
    assert final_result_path.exists()
    assert result["trace_package"]["final_answer"] == "A"
