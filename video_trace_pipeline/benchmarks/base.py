from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from ..common import sanitize_path_component
from ..schemas import DatasetConfig, TaskSpec


def parse_steps_field(value) -> Optional[List[str]]:
    if value is None:
        return None
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except Exception:
            return [stripped]
    if isinstance(parsed, list):
        result = [str(item).strip() for item in parsed if str(item).strip()]
        return result or None
    if isinstance(parsed, dict):
        result = []
        for key, item in sorted(parsed.items(), key=lambda pair: str(pair[0])):
            del key
            text = str(item).strip()
            if text:
                result.append(text)
        return result or None
    text = str(parsed).strip()
    return [text] if text else None


def make_sample_key(video_path: str, question: str, question_id: Optional[str] = None) -> str:
    stem = Path(video_path).stem or "video"
    if question_id:
        return "%s__qid%s" % (sanitize_path_component(stem), sanitize_path_component(str(question_id)))
    digest = hashlib.sha256(str(question or "").encode("utf-8")).hexdigest()[:12]
    return "%s__%s" % (sanitize_path_component(stem), digest)


class BenchmarkAdapter(object):
    benchmark_name = ""

    def __init__(self, config: DatasetConfig):
        self.config = config

    def load_tasks(self) -> List[TaskSpec]:
        raise NotImplementedError

    def select(self, index: Optional[int] = None, limit: Optional[int] = None) -> List[TaskSpec]:
        tasks = self.load_tasks()
        if index is not None:
            if index < 0 or index >= len(tasks):
                raise IndexError("index %s out of range for %s tasks" % (index, len(tasks)))
            tasks = [tasks[index]]
        if limit is not None:
            tasks = tasks[:limit]
        return tasks
