import json

from video_trace_pipeline.cli.main import _load_tasks
from video_trace_pipeline.schemas import MachineProfile


class _Runner(object):
    def __init__(self):
        self.profile = MachineProfile(workspace_root="/tmp/workspace")


def test_load_tasks_supports_inputs_json(tmp_path):
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(
        json.dumps(
            [
                {
                    "video_path": str(tmp_path / "video.mp4"),
                    "question": "What happens?",
                    "options": ["A", "B"],
                    "initial_trace_steps": ["step 1", "step 2"],
                    "answer": "A",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "video.mp4").write_bytes(b"video")
    tasks = _load_tasks(
        _Runner(),
        benchmark=None,
        index=None,
        limit=None,
        inputs_json=str(inputs_path),
        input_index=0,
    )
    assert len(tasks) == 1
    assert tasks[0].question == "What happens?"
    assert tasks[0].options == ["A", "B"]
    assert tasks[0].initial_trace_steps == ["step 1", "step 2"]
    assert tasks[0].gold_answer == "A"
