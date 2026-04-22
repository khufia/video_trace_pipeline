from __future__ import annotations

import json
import sys
from typing import Any, Dict


def load_request() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise SystemExit("Expected a JSON payload on stdin.")
    try:
        payload = json.loads(raw)
    except Exception as exc:  # pragma: no cover - defensive wrapper guard
        raise SystemExit("Failed to parse stdin JSON: %s" % exc)
    if not isinstance(payload, dict):
        raise SystemExit("Expected the stdin payload to be a JSON object.")
    return payload


def fail_stub(wrapper_name: str, integration_hint: str, expected_response: Dict[str, Any]) -> None:
    message = {
        "status": "not_implemented",
        "wrapper": wrapper_name,
        "message": integration_hint,
        "expected_response_example": expected_response,
    }
    sys.stderr.write(json.dumps(message, ensure_ascii=False, indent=2) + "\n")
    raise SystemExit(2)


def emit_json(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")


def fail_runtime(message: str, *, extra: Dict[str, Any] | None = None, exit_code: int = 1) -> None:
    payload = {"status": "error", "message": str(message or "").strip() or "unknown wrapper error"}
    if extra:
        payload["details"] = dict(extra)
    sys.stderr.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    raise SystemExit(exit_code)
