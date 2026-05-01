from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id() -> str:
    return "%s_%s" % (utc_now_compact(), uuid.uuid4().hex[:8])


def short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:length]


def sanitize_path_component(value: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-") or "item"
    return text[:max_len]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path_obj = Path(path).expanduser()
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: str | Path, text: str) -> None:
    path_obj = Path(path).expanduser()
    ensure_dir(path_obj.parent)
    path_obj.write_text(str(text or ""), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[Any]:
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        return []
    rows = []
    with path_obj.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def apply_env_map(values: dict[str, Any] | None) -> None:
    for key, value in dict(values or {}).items():
        if value is not None:
            os.environ[str(key)] = str(value)


def sanitize_for_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return sanitize_for_json(value.model_dump(mode="json"))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def deep_copy(value: Any) -> Any:
    return copy.deepcopy(value)


def stable_json_dumps(value: Any) -> str:
    return json.dumps(sanitize_for_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_payload(value: Any, length: int = 24) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()[:length]


def fingerprint_file(path: str | Path) -> str:
    file_path = Path(path).expanduser()
    stat = file_path.stat()
    hasher = hashlib.sha256()
    hasher.update(str(stat.st_size).encode("utf-8"))
    hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    with file_path.open("rb") as handle:
        first = handle.read(1024 * 1024)
        if stat.st_size > 2 * 1024 * 1024:
            handle.seek(max(0, stat.st_size - 1024 * 1024))
        last = handle.read(1024 * 1024)
    hasher.update(first)
    hasher.update(last)
    return hasher.hexdigest()


def traverse_path(obj: Any, field_path: str) -> Any:
    tokens = [token for token in str(field_path or "").split(".") if token]

    def walk(current: Any, index: int) -> Any:
        if index >= len(tokens):
            return current
        if current is None:
            return None
        token = tokens[index]
        if token == "[]":
            if not isinstance(current, list):
                return None
            values = [walk(item, index + 1) for item in current]
            values = [item for item in values if item is not None]
            return values or None
        match = re.match(r"^([A-Za-z0-9_]+)\[\]$", token)
        if match:
            if not isinstance(current, dict):
                return None
            current = current.get(match.group(1))
            if not isinstance(current, list):
                return None
            values = [walk(item, index + 1) for item in current]
            values = [item for item in values if item is not None]
            return values or None
        if re.fullmatch(r"\d+", token):
            if not isinstance(current, list):
                return None
            item_index = int(token)
            if item_index < 0 or item_index >= len(current):
                return None
            return walk(current[item_index], index + 1)
        if isinstance(current, dict):
            return walk(current.get(token), index + 1)
        return walk(getattr(current, token, None), index + 1)

    return walk(obj, 0)


def assign_path(obj: dict[str, Any], field_path: str, value: Any) -> None:
    tokens = [token for token in str(field_path or "").split(".") if token]
    if not tokens:
        raise ValueError("field_path must be non-empty")
    current = obj
    for token in tokens[:-1]:
        child = current.get(token)
        if not isinstance(child, dict):
            child = {}
            current[token] = child
        current = child
    current[tokens[-1]] = value


def _parse_json_dict(candidate: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(str(candidate or "").strip())
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _balanced_json_candidates(raw: str) -> Iterable[str]:
    starts = [index for index, char in enumerate(raw) if char == "{"]
    for start in reversed(starts):
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield raw[start : index + 1]
                    break


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    parsed = _parse_json_dict(raw)
    if parsed is not None:
        return parsed
    final_match = re.search(r"<final>(.*?)</final>", raw, flags=re.DOTALL | re.IGNORECASE)
    if final_match:
        parsed = extract_json_object(final_match.group(1))
        if parsed is not None:
            return parsed
    for block in reversed(re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)):
        parsed = extract_json_object(block)
        if parsed is not None:
            return parsed
    if "</think>" in raw.lower():
        parsed = extract_json_object(re.split(r"</think>", raw, flags=re.IGNORECASE)[-1])
        if parsed is not None:
            return parsed
    for candidate in _balanced_json_candidates(raw):
        parsed = _parse_json_dict(candidate)
        if parsed is not None:
            return parsed
    return None
