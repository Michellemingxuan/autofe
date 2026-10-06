"""The event log a run or an evaluation keeps, and running code into it.

Every step is a record: numbered, timestamped, appended to ``events.jsonl``
and passed to any listener - the CLI prints them, the server streams them.
A discovery run (:class:`agent.session.Session`) and an evaluation
(:class:`agent.evaluate.Evaluation`) both keep one.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

from agent.execution import CodeResult, run_code

__all__ = ["EventLog", "finite", "read_events", "run_logged"]


def finite(value: Any) -> Any:
    """NaN and inf become None: JSON has no spelling for them, and a browser's
    JSON.parse rejects the whole event when Python writes a bare NaN."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return finite(value.item())
        except (TypeError, ValueError):
            return value
    return value


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [finite(json.loads(line)) for line in open(path) if line.strip()]


class EventLog:
    def __init__(self, folder: Path, log_id: str):
        self.folder = folder
        self.log_id = log_id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / "events.jsonl"
        self.events: list[dict[str, Any]] = read_events(self.path)
        self.listeners: list[Callable[[dict[str, Any]], None]] = []
        self._lock = threading.Lock()

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            seq = self.events[-1]["seq"] + 1 if self.events else 1
            record = finite({"seq": seq, "event": event, "run_id": self.log_id,
                             "ts": round(time.time(), 3), **payload})
            self.events.append(record)
            with open(self.path, "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        for listener in list(self.listeners):
            listener(record)
        return record

    def next_code_id(self) -> str:
        started = sum(1 for e in self.events
                      if e["event"] == "code_status" and e["state"] == "running")
        return f"c{started + 1:03d}"


def run_logged(log: EventLog, code: str, mode: str, *, engine: str, id_col: str,
               base_path: Path, timeout_s: float, intent: str, level: str | None = None,
               title: str = "", **kwargs: Any) -> tuple[str, CodeResult]:
    """Run a script and record it: one event as it starts, one when it ends."""
    code_id = log.next_code_id()
    common = {"code_id": code_id, "intent": intent, "level": level, "mode": mode, "title": title}
    log.emit("code_status", **common, code=code, state="running")
    result = run_code(code, mode, workdir=log.folder / "code", tag=code_id, engine=engine,
                      id_col=id_col, base_path=base_path, timeout_s=timeout_s, **kwargs)
    log.emit("code_status", **common, state="ok" if result.ok else "error",
             stdout=result.stdout, error=result.error, result=result.result,
             elapsed_s=result.elapsed_s)
    return code_id, result
