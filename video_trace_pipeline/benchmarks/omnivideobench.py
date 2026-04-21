from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from ..schemas import TaskSpec
from .base import BenchmarkAdapter, make_sample_key, parse_steps_field


def _load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class OmniVideoBenchAdapter(BenchmarkAdapter):
    benchmark_name = "omnivideobench"

    def load_tasks(self) -> List[TaskSpec]:
        annotation_path = Path(self.config.annotations).expanduser().resolve()
        payload = _load_json_or_jsonl(annotation_path)
        if isinstance(payload, dict):
            items = [payload]
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError("Unsupported OmniVideoBench annotation payload in %s" % annotation_path)

        tasks = []
        root = Path(self.config.root).expanduser().resolve()
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            raw_video_path = (
                item.get("video_path")
                or item.get("video")
                or item.get("video_file")
                or item.get("video_name")
            )
            if not raw_video_path:
                continue
            video_path_obj = Path(str(raw_video_path))
            if not video_path_obj.is_absolute():
                video_path_obj = (root / self.config.videos_subdir / video_path_obj).resolve()
            question_id = item.get("question_id")
            task = TaskSpec(
                benchmark=self.benchmark_name,
                sample_key=make_sample_key(
                    str(video_path_obj),
                    question,
                    question_id=str(question_id) if question_id is not None else None,
                ),
                question=question,
                options=list(item.get("options") or []),
                video_path=str(video_path_obj),
                video_id=Path(str(video_path_obj)).stem,
                question_id=str(question_id) if question_id is not None else None,
                gold_answer=str(item.get("answer", "")).strip() or None,
                initial_trace_steps=parse_steps_field(item.get("initial_trace_steps")),
                metadata={"source_index": index, "raw": item},
            )
            tasks.append(task)
        return tasks
