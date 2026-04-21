from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..schemas import TaskSpec
from .base import BenchmarkAdapter, make_sample_key, parse_steps_field

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov")
OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def _find_video_path(videos_dir: Path, video_id: str) -> Path:
    for extension in VIDEO_EXTENSIONS:
        candidate = videos_dir / ("%s%s" % (video_id, extension))
        if candidate.is_file():
            return candidate.resolve()
    return (videos_dir / ("%s.mp4" % video_id)).resolve()


def _build_options(item) -> List[str]:
    options = []
    for choice_index, letter in enumerate(OPTION_LETTERS):
        field = "answer_choice_%d" % choice_index
        choice = str(item.get(field, "")).strip()
        if choice:
            options.append("%s. %s" % (letter, choice))
    return options


def _resolve_gold_answer(item) -> str:
    answer = str(item.get("answer", "")).strip()
    if answer:
        return answer
    answer_id = item.get("answer_id")
    options = _build_options(item)
    if isinstance(answer_id, int) and 0 <= answer_id < len(options):
        return options[answer_id]
    if isinstance(answer_id, str) and answer_id.isdigit():
        index = int(answer_id)
        if 0 <= index < len(options):
            return options[index]
    return ""


class MinervaAdapter(BenchmarkAdapter):
    benchmark_name = "minerva"

    def load_tasks(self) -> List[TaskSpec]:
        annotation_path = Path(self.config.annotations).expanduser().resolve()
        dataset_root = Path(self.config.root).expanduser().resolve()
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Expected a JSON array in %s" % annotation_path)
        videos_dir = dataset_root / self.config.videos_subdir
        tasks = []
        for index, item in enumerate(raw):
            video_id = str(item.get("video_id", "")).strip()
            question = str(item.get("question", "")).strip()
            if not video_id or not question:
                continue
            key = str(item.get("key", "")).strip() or "%s:%s" % (video_id, index)
            video_path = str(_find_video_path(videos_dir, video_id))
            task = TaskSpec(
                benchmark=self.benchmark_name,
                sample_key=make_sample_key(video_path, question, question_id=key),
                question=question,
                options=_build_options(item),
                video_path=video_path,
                video_id=video_id,
                question_id=key,
                gold_answer=_resolve_gold_answer(item) or None,
                initial_trace_steps=parse_steps_field(item.get("reasoning")),
                metadata={"source_index": index, "raw": item},
            )
            tasks.append(task)
        return tasks
