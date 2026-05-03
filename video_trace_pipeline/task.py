from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from .common import read_json, read_jsonl, sanitize_path_component, short_hash

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov")
OPTION_LETTERS = ("A", "B", "C", "D", "E")


def _parse_steps_field(value: Any) -> list[str] | None:
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
        steps = [str(item).strip() for item in parsed if str(item).strip()]
        return steps or None
    if isinstance(parsed, dict):
        steps = [str(item).strip() for _, item in sorted(parsed.items(), key=lambda pair: str(pair[0])) if str(item).strip()]
        return steps or None
    text = str(parsed).strip()
    return [text] if text else None


def _make_sample_key(video_path: str, question: str, question_id: str | None = None) -> str:
    stem = Path(video_path).stem or "video"
    if question_id:
        return "%s__qid%s" % (sanitize_path_component(stem), sanitize_path_component(question_id))
    return "%s__%s" % (sanitize_path_component(stem), short_hash(question, 12))


def _find_video_path(videos_dir: Path, video_id: str) -> Path:
    for extension in VIDEO_EXTENSIONS:
        candidate = videos_dir / ("%s%s" % (video_id, extension))
        if candidate.is_file():
            return candidate.resolve()
    return (videos_dir / ("%s.mp4" % video_id)).resolve()


def _task_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    video_path = str(payload.get("video_path") or "").strip()
    question = str(payload.get("question") or "").strip()
    if not video_path:
        raise ValueError("Task requires `video_path`.")
    if not question:
        raise ValueError("Task requires `question`.")
    video_path_obj = Path(video_path).expanduser().resolve()
    if not video_path_obj.exists():
        raise ValueError("Video path does not exist: %s" % video_path_obj)
    video_id = str(payload.get("video_id") or video_path_obj.stem).strip() or video_path_obj.stem
    benchmark = str(payload.get("benchmark") or "adhoc").strip() or "adhoc"
    question_id = str(payload.get("question_id") or "").strip() or None
    sample_key = str(payload.get("sample_key") or "").strip() or _make_sample_key(str(video_path_obj), question, question_id)
    return {
        "benchmark": benchmark,
        "sample_key": sample_key,
        "question_id": question_id,
        "video_id": video_id,
        "video_path": str(video_path_obj),
        "question": question,
        "options": list(payload.get("options") or []),
        "gold_answer": payload.get("gold_answer", payload.get("answer")),
        "initial_trace": payload.get("initial_trace"),
        "initial_trace_steps": _parse_steps_field(payload.get("initial_trace_steps") or payload.get("steps") or payload.get("reasoning")),
        "metadata": dict(payload.get("metadata") or {}),
    }


def _task_from_input_json(path: str, index: int | None) -> dict[str, Any]:
    payload = read_json(path)
    if isinstance(payload, dict):
        return _task_from_payload(payload)
    if not isinstance(payload, list):
        raise ValueError("Input JSON must contain either one task object or a list of task objects.")
    selected_index = int(index or 0)
    if selected_index < 0 or selected_index >= len(payload):
        raise IndexError("index %s out of range for %s tasks in %s" % (selected_index, len(payload), path))
    item = payload[selected_index]
    if not isinstance(item, dict):
        raise ValueError("Input JSON item at index %s is not an object." % selected_index)
    task = _task_from_payload(item)
    metadata = dict(task.get("metadata") or {})
    metadata.setdefault("source_index", selected_index)
    metadata.setdefault("source_input_json", str(Path(path).expanduser().resolve()))
    task["metadata"] = metadata
    return task


def _load_options_json(path: str | None) -> list[str]:
    if not path:
        return []
    payload = read_json(path)
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    if isinstance(payload, dict):
        return [str(value).strip() for _, value in sorted(payload.items()) if str(value).strip()]
    raise ValueError("Options JSON must contain a list or object.")


def _load_annotation_items(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    else:
        payload = read_json(path)
        rows = payload if isinstance(payload, list) else [payload]
    return [dict(item) for item in rows if isinstance(item, dict)]


def _build_minerva_options(item: dict[str, Any]) -> list[str]:
    options = []
    for index, letter in enumerate(OPTION_LETTERS):
        text = str(item.get("answer_choice_%d" % index, "")).strip()
        if text:
            options.append("%s. %s" % (letter, text))
    return options


def _load_benchmark_tasks(benchmark: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = profile.get("datasets") or {}
    if benchmark not in datasets:
        raise ValueError("Benchmark %s is not configured in the machine profile." % benchmark)
    config = dict(datasets[benchmark] or {})
    root = Path(str(config.get("root") or "")).expanduser().resolve()
    annotations = Path(str(config.get("annotations") or "")).expanduser().resolve()
    videos_dir = root / str(config.get("videos_subdir") or "videos")
    items = _load_annotation_items(annotations)
    tasks = []
    for index, item in enumerate(items):
        if benchmark == "videomathqa":
            video_id = str(item.get("videoID") or item.get("video_id") or "").strip()
            question = str(item.get("question") or "").strip()
            if not video_id or not question:
                continue
            question_id = item.get("question_id")
            video_path = videos_dir / ("%s.mp4" % video_id)
            tasks.append(
                _task_from_payload(
                    {
                        "benchmark": benchmark,
                        "sample_key": _make_sample_key(str(video_path), question, str(question_id) if question_id is not None else None),
                        "question": question,
                        "options": list(item.get("options") or []),
                        "video_path": str(video_path),
                        "video_id": video_id,
                        "question_id": str(question_id) if question_id is not None else None,
                        "gold_answer": str(item.get("answer") or "").strip() or None,
                        "initial_trace_steps": item.get("steps"),
                        "metadata": {"source_index": index, "raw": item},
                    }
                )
            )
        elif benchmark == "minerva":
            video_id = str(item.get("video_id") or "").strip()
            question = str(item.get("question") or "").strip()
            if not video_id or not question:
                continue
            key = str(item.get("key") or "%s:%s" % (video_id, index)).strip()
            video_path = _find_video_path(videos_dir, video_id)
            answer = str(item.get("answer") or "").strip()
            if not answer and str(item.get("answer_id") or "").isdigit():
                options = _build_minerva_options(item)
                answer_index = int(item["answer_id"])
                if 0 <= answer_index < len(options):
                    answer = options[answer_index]
            tasks.append(
                _task_from_payload(
                    {
                        "benchmark": benchmark,
                        "sample_key": _make_sample_key(str(video_path), question, key),
                        "question": question,
                        "options": _build_minerva_options(item),
                        "video_path": str(video_path),
                        "video_id": video_id,
                        "question_id": key,
                        "gold_answer": answer or None,
                        "initial_trace_steps": item.get("reasoning"),
                        "metadata": {"source_index": index, "raw": item},
                    }
                )
            )
        else:
            question = str(item.get("question") or "").strip()
            raw_video_path = item.get("video_path") or item.get("video") or item.get("video_file") or item.get("video_name")
            if not question or not raw_video_path:
                continue
            video_path = Path(str(raw_video_path))
            if not video_path.is_absolute():
                root_candidate = (root / video_path).resolve()
                videos_candidate = (videos_dir / video_path).resolve()
                video_path = root_candidate if root_candidate.exists() else videos_candidate
            question_id = item.get("question_id")
            tasks.append(
                _task_from_payload(
                    {
                        "benchmark": benchmark,
                        "sample_key": _make_sample_key(str(video_path), question, str(question_id) if question_id is not None else None),
                        "question": question,
                        "options": list(item.get("options") or []),
                        "video_path": str(video_path),
                        "video_id": video_path.stem,
                        "question_id": str(question_id) if question_id is not None else None,
                        "gold_answer": str(item.get("answer") or "").strip() or None,
                        "initial_trace_steps": item.get("initial_trace_steps"),
                        "metadata": {"source_index": index, "raw": item},
                    }
                )
            )
    return tasks


def load_task(args: Namespace, profile: dict[str, Any]) -> dict[str, Any]:
    if getattr(args, "input_json", None):
        return _task_from_input_json(args.input_json, getattr(args, "index", None))
    if getattr(args, "question", None) and getattr(args, "video_path", None):
        return _task_from_payload(
            {
                "benchmark": getattr(args, "benchmark", None) or "adhoc",
                "sample_key": getattr(args, "sample_key", None),
                "question": args.question,
                "video_path": args.video_path,
                "video_id": getattr(args, "video_id", None),
                "question_id": getattr(args, "question_id", None),
                "gold_answer": getattr(args, "gold_answer", None),
                "options": _load_options_json(getattr(args, "options_json", None)),
                "metadata": {"source": "direct_cli"},
            }
        )
    benchmark = getattr(args, "benchmark", None)
    if not benchmark:
        raise ValueError("Provide --benchmark, --input-json, or --question with --video-path.")
    tasks = _load_benchmark_tasks(str(benchmark), profile)
    index = getattr(args, "index", None)
    if index is None:
        index = 0
    if int(index) < 0 or int(index) >= len(tasks):
        raise IndexError("index %s out of range for %s tasks" % (index, len(tasks)))
    return tasks[int(index)]
