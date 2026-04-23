from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9._-])(?P<path>/(?:[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*))")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def utc_now_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def make_run_id() -> str:
    return "%s_%s" % (utc_now_compact(), uuid.uuid4().hex[:8])


def short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_payload(value: Any, length: int = 24) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()[:length]


def sanitize_path_component(value: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-") or "item"
    return text[:max_len]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text or "", encoding="utf-8")


def read_jsonl(path: Path) -> list:
    items = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            items.append(json.loads(stripped))
    return items


def deep_copy(value: Any) -> Any:
    return copy.deepcopy(value)


def relative_to_root(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def guess_media_type(path: str) -> Optional[str]:
    suffix = Path(path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return "image"
    if suffix in {".mp4", ".mkv", ".mov", ".webm"}:
        return "video"
    if suffix in {".wav", ".mp3", ".flac"}:
        return "audio"
    if suffix in {".json"}:
        return "application/json"
    if suffix in {".txt", ".md", ".jsonl"}:
        return "text/plain"
    return None


def fingerprint_file(path: str) -> str:
    file_path = Path(path)
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


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = raw[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def is_low_signal_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True

    visible_chars = [char for char in raw if not char.isspace()]
    if not visible_chars:
        return True

    alnum_chars = [char.lower() for char in raw if char.isalnum()]
    if not alnum_chars:
        return True

    if len(visible_chars) >= 8 and len(set(visible_chars)) == 1:
        return True

    if len(alnum_chars) >= 8 and len(set(alnum_chars)) == 1:
        return True

    return False


def has_meaningful_text(text: str) -> bool:
    raw = str(text or "").strip()
    if is_low_signal_text(raw):
        return False

    words = _WORD_RE.findall(raw)
    if not words:
        return False

    alnum_count = sum(len(word) for word in words)
    if alnum_count >= 6:
        return True

    return len(words) >= 2


def traverse_path(obj: Any, field_path: str) -> Any:
    current = obj
    for token in str(field_path or "").split("."):
        if current is None:
            return None
        if re.fullmatch(r"\d+", token):
            index = int(token)
            if isinstance(current, list) and 0 <= index < len(current):
                current = current[index]
                continue
            return None
        match = re.match(r"^([A-Za-z0-9_]+)\[(\d+)\]$", token)
        if match:
            key = match.group(1)
            index = int(match.group(2))
            current = current.get(key) if isinstance(current, dict) else None
            if isinstance(current, list) and index < len(current):
                current = current[index]
            else:
                return None
        else:
            if isinstance(current, dict):
                current = current.get(token)
            elif isinstance(current, list):
                lowered = str(token).strip().lower()
                exact_match = None
                fuzzy_match = None
                for item in current:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("label") or item.get("name") or "").strip().lower()
                    if not label:
                        continue
                    if label == lowered:
                        exact_match = item
                        break
                    if fuzzy_match is None and lowered in label:
                        fuzzy_match = item
                current = exact_match or fuzzy_match
            else:
                return None
    return current


def assign_path(target: Dict[str, Any], field_path: str, value: Any) -> Dict[str, Any]:
    parts = str(field_path or "").split(".")
    cursor = target
    for idx, part in enumerate(parts):
        is_last = idx == len(parts) - 1
        if is_last:
            cursor[part] = value
            break
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    return target


def apply_env_map(env_overrides: Dict[str, str]) -> None:
    for key, value in sorted((env_overrides or {}).items()):
        os.environ[str(key)] = str(value)


def _repo_relative_path(value: str) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw.startswith("/"):
        return None
    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except Exception:
        resolved = candidate
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except Exception:
        return None


def _sanitize_string_value(value: str) -> str:
    text = str(value or "")
    if text.startswith("/"):
        return _repo_relative_path(text) or "<redacted-path>"

    def _replace(match):
        path_text = str(match.group("path") or "")
        return _repo_relative_path(path_text) or "<redacted-path>"

    return _ABSOLUTE_PATH_RE.sub(_replace, text)


def sanitize_for_persistence(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in {"source_path", "video_path", "frame_path", "path"}:
                continue
            cleaned[key] = sanitize_for_persistence(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize_for_persistence(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_persistence(item) for item in value]
    if isinstance(value, str):
        return _sanitize_string_value(value)
    return value
