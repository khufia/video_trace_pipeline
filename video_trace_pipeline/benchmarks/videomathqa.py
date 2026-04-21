from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..schemas import TaskSpec
from .base import BenchmarkAdapter, make_sample_key, parse_steps_field


class VideoMathQAAdapter(BenchmarkAdapter):
    benchmark_name = "videomathqa"

    def load_tasks(self) -> List[TaskSpec]:
        annotation_path = Path(self.config.annotations).expanduser().resolve()
        dataset_root = Path(self.config.root).expanduser().resolve()
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Expected a JSON array in %s" % annotation_path)
        tasks = []
        videos_dir = dataset_root / self.config.videos_subdir
        for index, item in enumerate(raw):
            video_id = str(item.get("videoID", "")).strip()
            if not video_id:
                continue
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            video_path = str((videos_dir / ("%s.mp4" % video_id)).resolve())
            question_id = item.get("question_id")
            task = TaskSpec(
                benchmark=self.benchmark_name,
                sample_key=make_sample_key(video_path, question, question_id=str(question_id) if question_id is not None else None),
                question=question,
                options=list(item.get("options") or []),
                video_path=video_path,
                video_id=video_id,
                question_id=str(question_id) if question_id is not None else None,
                gold_answer=str(item.get("answer", "")).strip() or None,
                initial_trace_steps=parse_steps_field(item.get("steps")),
                metadata={"source_index": index, "raw": item},
            )
            tasks.append(task)
        return tasks
