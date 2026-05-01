from __future__ import annotations

from typing import Any


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _append_dict_result(lines: list[dict[str, Any]], result: dict[str, Any]) -> None:
    texts = list(result.get("rec_texts") or [])
    scores = list(result.get("rec_scores") or [])
    boxes = list(result.get("rec_polys") or result.get("dt_polys") or [])
    for index, text in enumerate(texts):
        text = str(text or "").strip()
        if not text:
            continue
        confidence = scores[index] if index < len(scores) else None
        bbox = boxes[index] if index < len(boxes) else None
        lines.append({"text": text, "bbox": _jsonable(bbox), "confidence": confidence})


def _append_legacy_result(lines: list[dict[str, Any]], result: Any) -> None:
    for block in result or []:
        for item in block or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            text = item[1][0] if isinstance(item[1], (list, tuple)) and item[1] else ""
            confidence = item[1][1] if isinstance(item[1], (list, tuple)) and len(item[1]) > 1 else None
            if str(text).strip():
                lines.append({"text": str(text).strip(), "bbox": _jsonable(item[0]), "confidence": confidence})


def paddle_ocr_lines(raw: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        _append_dict_result(lines, raw)
        return lines
    if isinstance(raw, list):
        for result in raw:
            if isinstance(result, dict):
                _append_dict_result(lines, result)
            else:
                _append_legacy_result(lines, [result])
        return lines
    return lines


def text_from_lines(lines: list[dict[str, Any]]) -> str:
    return "\n".join(str(line.get("text") or "").strip() for line in lines if str(line.get("text") or "").strip())
