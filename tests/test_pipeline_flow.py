import pytest

from video_trace_pipeline.orchestration.pipeline import PipelineRunner
from video_trace_pipeline.schemas import (
    AgentConfig,
    AuditReport,
    InferenceStep,
    MachineProfile,
    ModelsConfig,
    PlannerAction,
    TaskSpec,
    ToolConfig,
    TracePackage,
)


class FakePlanner(object):
    def __init__(self, action=None):
        self.calls = []
        self.completed_requests = []
        if isinstance(action, list):
            self.actions = list(action)
        elif action is not None:
            self.actions = [action]
        else:
            self.actions = []

    def build_request(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"endpoint_name": "default", "model_name": "gpt-5.4", "system_prompt": "planner", "user_prompt": "prompt"}

    def complete_request(self, request):
        self.completed_requests.append(dict(request))
        action = self.actions.pop(0) if self.actions else PlannerAction(action_type="synthesize", rationale="No more tools needed.", synthesis_instructions="Write the trace from current evidence.")
        return "{}", action


class FakeExecutor(object):
    def __init__(self, records=None):
        self.calls = []
        self.records = list(records or [])

    def execute_plan(self, **kwargs):
        self.calls.append(dict(kwargs))
        return list(self.records)


class FakeSynthesizer(object):
    def __init__(self):
        self.calls = []

    def build_request(
        self,
        task,
        mode,
        round_evidence_entries,
        round_observations,
        current_trace,
        refinement_instructions,
        audit_feedback=None,
        preprocess_context=None,
    ):
        self.calls.append(locals())
        return {"endpoint_name": "default", "model_name": "gpt-5.4", "system_prompt": "synth", "user_prompt": "prompt"}

    def complete_request(self, request):
        del request
        trace = TracePackage(
            task_key="sample1",
            mode="generate",
            evidence_entries=[],
            inference_steps=[
                InferenceStep(
                    step_id=1,
                    text="The evidence supports option A.",
                    supporting_observation_ids=[],
                    answer_relevance="high",
                )
            ],
            final_answer="A",
            benchmark_renderings={},
        )
        return "{}", trace


class FakeAuditor(object):
    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first
        self.count = 0

    def build_request(self, task, trace_package, evidence_summary, preprocess_context=None):
        self.calls.append({"task": task, "trace_package": trace_package, "evidence_summary": evidence_summary, "preprocess_context": preprocess_context})
        return {"endpoint_name": "default", "model_name": "gpt-5.4", "system_prompt": "audit", "user_prompt": "prompt"}

    def complete_request(self, request):
        del request
        self.count += 1
        if self.fail_first and self.count == 1:
            return "{}", AuditReport(
                verdict="FAIL",
                confidence=0.8,
                scores={"completeness": 2},
                findings=[],
                feedback="Need exact price.",
                missing_information=["exact visible price label"],
            )
        return "{}", AuditReport(verdict="PASS", confidence=0.9, scores={"completeness": 5}, findings=[], feedback="")


class BrokenPreprocessor(object):
    def is_enabled(self):
        return True

    def resolve_preprocess_settings(self, clip_duration_s=None):
        return {"clip_duration_s": clip_duration_s or 30.0}

    def get_or_build(self, task, clip_duration_s=None):
        del task, clip_duration_s
        return {"cache_dir": "broken-cache"}


def _models_config():
    return ModelsConfig(
        agents={
            "planner": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_synthesizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "trace_auditor": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
            "atomicizer": AgentConfig(backend="openai", model="gpt-5.4", endpoint="default"),
        },
        tools={"dense_captioner": ToolConfig(enabled=True, model="test-model")},
    )


def _runner(tmp_path, auditor=None, planner=None):
    runner = PipelineRunner(MachineProfile(workspace_root=str(tmp_path / "workspace")), _models_config())
    runner.planner = planner or FakePlanner()
    runner.executor = FakeExecutor()
    runner.synthesizer = FakeSynthesizer()
    runner.auditor = auditor or FakeAuditor()
    return runner


def _task(tmp_path, initial_trace_steps=None):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    return TaskSpec(
        benchmark="omnivideobench",
        sample_key="sample1",
        video_id="video_1",
        question="What is the answer?",
        options=["A", "B"],
        video_path=str(video),
        initial_trace_steps=initial_trace_steps,
    )


def test_pipeline_runner_builds_first_planner_round_without_preprocess(tmp_path):
    runner = _runner(tmp_path)

    result = runner.run_task(_task(tmp_path), mode="generate", max_rounds=1, clip_duration_s=30.0)

    assert result["trace_package"]["final_answer"] == "A"
    call = runner.planner.calls[0]
    assert "planner_segments" not in call
    assert "preprocess_planning_memory" not in call
    assert "compact_rounds" not in call
    assert call["mode"] == "generate"
    assert call["audit_feedback"] is None
    assert "tool_catalog" in call
    assert "action_history" in call


def test_refine_planner_receives_audit_feedback_before_plan(tmp_path):
    runner = _runner(tmp_path, auditor=FakeAuditor(fail_first=True))
    task = _task(tmp_path, initial_trace_steps=["The old trace says the answer is B."])

    runner.run_task(task, mode="refine", max_rounds=1, clip_duration_s=30.0)

    assert runner.planner.calls
    call = runner.planner.calls[0]
    assert call["mode"] == "refine"
    assert call["audit_feedback"]["missing_information"] == ["exact visible price label"]
    assert "planner_context" not in call


def test_pipeline_writes_final_outputs(tmp_path):
    runner = _runner(tmp_path)
    result = runner.run_task(_task(tmp_path), mode="generate", max_rounds=1, clip_duration_s=30.0)

    final_result_path = tmp_path / "workspace" / result["run_dir"] / "final_result.json"
    planner_request_path = tmp_path / "workspace" / result["run_dir"] / "round_01" / "planner_request.json"
    planner_action_path = tmp_path / "workspace" / result["run_dir"] / "round_01" / "planner_action.json"
    assert final_result_path.exists()
    assert planner_request_path.exists()
    assert planner_action_path.exists()


def test_pipeline_executes_one_tool_action_before_synthesis(tmp_path):
    planner = FakePlanner(
        action=[
            PlannerAction(
                action_type="tool_call",
                rationale="Need one extraction.",
                tool_name="generic_purpose",
                tool_request={"tool_name": "generic_purpose", "query": "Inspect the evidence.", "text_contexts": ["context"]},
                expected_observation="one extracted answer",
            ),
            PlannerAction(action_type="synthesize", rationale="Evidence collected.", synthesis_instructions="Write the trace."),
        ]
    )
    runner = _runner(tmp_path, planner=planner)
    runner.executor.records = [
        {
            "step_id": 1,
            "tool_name": "generic_purpose",
            "purpose": "one extracted answer",
            "request": {"tool_name": "generic_purpose", "query": "Inspect the evidence.", "text_contexts": ["context"]},
            "result": {"ok": True, "data": {"clips": [{"video_id": "video_1", "start_s": 1.0, "end_s": 2.0}]}, "summary": "Found clip."},
            "evidence_entry": {"evidence_id": "ev_1", "tool_name": "generic_purpose", "observation_ids": ["obs_1"]},
            "observations": [{"observation_id": "obs_1", "atomic_text": "The clip supports A."}],
        }
    ]

    result = runner.run_task(_task(tmp_path), mode="generate", max_rounds=1, clip_duration_s=30.0)

    assert len(planner.completed_requests) == 2
    assert len(runner.executor.calls) == 1
    assert result["rounds_executed"] == 1
    assert result["trace_package"]["final_answer"] == "A"
    history = planner.calls[1]["action_history"]
    assert history[0]["tool_outputs"][0]["result"]["data"]["clips"][0]["start_s"] == 1.0
    assert result["action_history"][0]["tool_outputs"][0]["observations"][0]["atomic_text"] == "The clip supports A."


def test_pipeline_fails_loudly_when_enabled_preprocess_lacks_planner_segments(tmp_path):
    runner = _runner(tmp_path)
    runner.preprocessor = BrokenPreprocessor()

    with pytest.raises(RuntimeError, match="planner_segments"):
        runner.run_task(_task(tmp_path), mode="generate", max_rounds=1, clip_duration_s=30.0)
