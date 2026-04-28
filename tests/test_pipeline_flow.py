from types import SimpleNamespace

from video_trace_pipeline.orchestration.pipeline import PipelineRunner
from video_trace_pipeline.schemas import (
    AgentConfig,
    AuditReport,
    ExecutionPlan,
    InferenceStep,
    MachineProfile,
    ModelsConfig,
    TaskSpec,
    ToolConfig,
    TracePackage,
)


class FakePreprocessor(object):
    def resolve_preprocess_settings(self, clip_duration_s):
        return {"clip_duration_s": float(clip_duration_s or 60.0)}

    def get_or_build(self, task, clip_duration_s):
        del clip_duration_s
        return {
            "cache_hit": True,
            "cache_dir": "workspace/preprocess/video_1",
            "manifest": {"video_id": task.video_id, "planner_segment_count": 1},
            "raw_segments": [],
            "asr_transcripts": [],
            "dense_caption_segments": [],
            "planner_segments": [
                {
                    "segment_id": "seg_001",
                    "start_s": 0.0,
                    "end_s": 12.0,
                    "dense_caption": {
                        "overall_summary": "A short rich summary.",
                        "clips": [{"video_id": task.video_id, "start_s": 0.0, "end_s": 12.0}],
                        "captions": [{"start_s": 0.0, "end_s": 12.0, "visual": "A price label is shown."}],
                    },
                    "asr": {"transcript_spans": [{"start_s": 1.0, "end_s": 2.0, "text": "The price is forty two."}]},
                }
            ],
            "video_fingerprint": "vid123",
        }


class FakePlanner(object):
    def __init__(self):
        self.calls = []

    def build_request(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"endpoint_name": "default", "model_name": "gpt-5.4", "system_prompt": "planner", "user_prompt": "prompt"}

    def complete_request(self, request):
        del request
        return "{}", ExecutionPlan(strategy="No tools needed.", steps=[], refinement_instructions="")


class FakeExecutor(object):
    def execute_plan(self, **kwargs):
        return []


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

    def build_request(self, task, trace_package, evidence_summary):
        self.calls.append({"task": task, "trace_package": trace_package, "evidence_summary": evidence_summary})
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


def _runner(tmp_path, auditor=None):
    runner = PipelineRunner(MachineProfile(workspace_root=str(tmp_path / "workspace")), _models_config())
    runner.preprocessor = FakePreprocessor()
    runner.planner = FakePlanner()
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


def test_pipeline_runner_passes_full_rich_preprocess_to_first_planner_round(tmp_path):
    runner = _runner(tmp_path)

    result = runner.run_task(_task(tmp_path), mode="generate", max_rounds=1, clip_duration_s=30.0)

    assert result["trace_package"]["final_answer"] == "A"
    call = runner.planner.calls[0]
    assert call["planner_segments"][0]["dense_caption"]["overall_summary"] == "A short rich summary."
    assert call["planner_segments"][0]["asr"]["transcript_spans"][0]["text"] == "The price is forty two."
    assert "preprocess_planning_memory" not in call
    assert "compact_rounds" not in call


def test_refine_planner_receives_retrieved_context_before_plan(tmp_path):
    runner = _runner(tmp_path, auditor=FakeAuditor(fail_first=True))
    task = _task(tmp_path, initial_trace_steps=["The old trace says the answer is B."])

    runner.run_task(task, mode="refine", max_rounds=1, clip_duration_s=30.0)

    assert runner.planner.calls
    call = runner.planner.calls[0]
    assert call["mode"] == "refine"
    assert call["retrieved_context"]["audit_gaps"] == ["exact visible price label"]
    assert "planner_context" not in call


def test_pipeline_writes_final_outputs(tmp_path):
    runner = _runner(tmp_path)
    result = runner.run_task(_task(tmp_path), mode="generate", max_rounds=1, clip_duration_s=30.0)

    final_result_path = tmp_path / "workspace" / result["run_dir"] / "final_result.json"
    planner_request_path = tmp_path / "workspace" / result["run_dir"] / "round_01" / "planner_request.json"
    assert final_result_path.exists()
    assert planner_request_path.exists()
