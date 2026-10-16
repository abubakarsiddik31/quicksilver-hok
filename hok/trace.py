"""One JSON-lines trace file per Honor of Kings match: ``traces/hok-<unix time>-<pid>.jsonl``.

Folder: ``HOK_TRACE_DIR`` (default ``traces``); off: ``HOK_TRACE=0``. Lines are
``{"t": <unix time>, "kind": ..., ...}``:

- ``snapshot``  the full unmodified server state, first view and every ~30 s
- ``state``     a compact sample about once a second: clock, every hero of ours and every enemy
                seen, structures, standing orders, pending seats
- ``orders``    each order sent, with where it came from (``floor`` now, the General later)
- ``result``    the server's answer to each order (a refusal shows its error)
- ``fallback``  every time the floor had to decide instead of the planner, and why

There is no HoK simulator, so these traces are the only offline material. Writing never raises.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

_path: str | None = None
_failed = False


def _target() -> str | None:
    global _path, _failed
    if _failed or os.environ.get("HOK_TRACE", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    if _path is None:
        folder = os.environ.get("HOK_TRACE_DIR") or "traces"
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            _failed = True
            return None
        _path = os.path.join(folder, f"hok-{int(time.time())}-{os.getpid()}.jsonl")
    return _path


def write(kind: str, **data: Any) -> None:
    path = _target()
    if path is None:
        return
    try:
        line = json.dumps(
            {"t": round(time.time(), 2), "kind": kind, **data}, default=str, separators=(",", ":")
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001 - tracing must never cost a match
        pass


def current_path() -> str | None:
    return _path
