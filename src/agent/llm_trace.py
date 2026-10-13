"""Every model call, recorded: what went in, what came out, how long, how many tokens.

A run's events say what the agent *did* (a tool call, a screened feature);
this says what the model was *shown and answered* at each step - the
lifecycle under the events. One SQLite file for every run
(``<agent.run_dir>/llm_traces.db``, or ``agent.trace_db``), one row per call:

    run_id, stage, round   which run, and where in it: ideas, propose, linkage,
                           themes - and the round of ideas it belongs to
    call                   its number within the run (1, 2, 3 ...)
    model, timings         started_at, ended_at, duration_ms
    tokens                 input, output, cached and reasoning tokens - the
                           backend's count, or an estimate (``tokens_estimated``)
                           when it gives none (safechain)
    input_json             the messages the model received - after the filter
                           that trims old tool results, so exactly what it saw
    system_sha             its instructions (the brief), kept once per text in
                           ``prompts`` - the same brief serves a whole run
    output_json            what it answered: text, tool calls, a structured
                           answer
    tool_calls             the tools it called, by name
    outcome                ok, or unfinished (the call never returned: an error,
                           a stop)

The rows are written by SDK hooks (:class:`TraceHooks`), bound to the session
and the stage when the runner starts - so nothing depends on the context the
SDK runs the call in. Like AgenticSys_v2's node trace, recording never breaks
a call: every write swallows its errors. On a cloud-synced folder (Google
Drive, iCloud) SQLite's WAL mode is refused; the store falls back to the
default journal, which is fine at one row per call.

    PYTHONPATH=src python -m agent.llm_trace outputs/agent/llm_traces.db <run_id>
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

from agents import RunHooks

__all__ = ["TraceStore", "TraceHooks", "store_for", "trace_path", "MAX_IO_BYTES"]

# A payload above this is kept as a stub with its size - a whole conversation
# can grow large, and the trace must not become the run's biggest file.
MAX_IO_BYTES = 2_000_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_call (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id             TEXT NOT NULL,
  kind               TEXT,
  stage              TEXT NOT NULL,
  round              INTEGER,
  call               INTEGER NOT NULL,
  agent              TEXT,
  model              TEXT,
  started_at         REAL NOT NULL,
  ended_at           REAL,
  duration_ms        INTEGER,
  input_items        INTEGER,
  input_chars        INTEGER,
  input_tokens       INTEGER,
  cached_tokens      INTEGER,
  output_chars       INTEGER,
  output_tokens      INTEGER,
  reasoning_tokens   INTEGER,
  tokens_estimated   INTEGER DEFAULT 0,
  system_sha         TEXT,
  input_json         TEXT,
  output_json        TEXT,
  tool_calls         TEXT,
  outcome            TEXT
);
CREATE INDEX IF NOT EXISTS idx_call_run ON llm_call(run_id, call);
CREATE TABLE IF NOT EXISTS prompts (
  sha   TEXT PRIMARY KEY,
  text  TEXT NOT NULL
);
"""

_log = logging.getLogger(__name__)


class TraceStore:
    """The SQLite file and its write lock. Reads take no lock. Every method
    swallows its errors: a trace that cannot be written is logged once, and the
    run goes on."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.Lock()
        self._warned = False
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            # A cloud-synced folder refuses WAL's memory-mapped files.
            self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.executescript(_SCHEMA)

    def _failed(self, what: str, error: Exception) -> None:
        if not self._warned:
            self._warned = True
            _log.warning("llm trace: %s failed (%s) - the run goes on untraced", what, error)

    def begin(self, **row: Any) -> int:
        try:
            with self._lock:
                cur = self._conn.execute(
                    f"INSERT INTO llm_call ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                    tuple(row.values()))
                return int(cur.lastrowid or -1)
        except Exception as error:  # noqa: BLE001 - recording never breaks a call
            self._failed("insert", error)
            return -1

    def finish(self, row_id: int, **values: Any) -> None:
        if row_id < 0 or not values:
            return
        try:
            with self._lock:
                self._conn.execute(
                    f"UPDATE llm_call SET {', '.join(f'{k} = ?' for k in values)} WHERE id = ?",
                    (*values.values(), row_id))
        except Exception as error:  # noqa: BLE001
            self._failed("update", error)

    def prompt(self, text: str) -> str | None:
        """Keep a system prompt once; its hash goes on each row."""
        if not text:
            return None
        sha = hashlib.sha256(text.encode()).hexdigest()[:16]
        try:
            with self._lock:
                self._conn.execute("INSERT OR IGNORE INTO prompts (sha, text) VALUES (?, ?)", (sha, text))
        except Exception as error:  # noqa: BLE001
            self._failed("prompt", error)
        return sha

    def next_call(self, run_id: str) -> int:
        try:
            row = self._conn.execute("SELECT MAX(call) FROM llm_call WHERE run_id = ?",
                                     (run_id,)).fetchone()
            return int(row[0] or 0) + 1
        except Exception:  # noqa: BLE001
            return 1

    # ----------------------------------------------------------------- reads
    def calls(self, run_id: str) -> list[dict[str, Any]]:
        """A run's calls, in order, without their payloads."""
        cols = ("id", "run_id", "kind", "stage", "round", "call", "agent", "model", "started_at",
                "ended_at", "duration_ms", "input_items", "input_chars", "input_tokens",
                "cached_tokens", "output_chars", "output_tokens", "reasoning_tokens",
                "tokens_estimated", "system_sha", "tool_calls", "outcome")
        try:
            rows = self._conn.execute(
                f"SELECT {', '.join(cols)}, substr(output_json, 1, 400) FROM llm_call "
                "WHERE run_id = ? ORDER BY call", (run_id,)).fetchall()
        except Exception as error:  # noqa: BLE001
            self._failed("read", error)
            return []
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            d["tool_calls"] = json.loads(d["tool_calls"]) if d["tool_calls"] else []
            d["output_preview"] = _preview(r[-1])
            out.append(d)
        return out

    def call(self, row_id: int) -> dict[str, Any] | None:
        """One call in full: its input, its output and its system prompt."""
        try:
            cur = self._conn.execute("SELECT * FROM llm_call WHERE id = ?", (row_id,))
            row = cur.fetchone()
            if row is None:
                return None
            d = dict(zip([c[0] for c in cur.description], row))
            prompt = self._conn.execute("SELECT text FROM prompts WHERE sha = ?",
                                        (d.get("system_sha"),)).fetchone()
        except Exception as error:  # noqa: BLE001
            self._failed("read", error)
            return None
        for key in ("input_json", "output_json", "tool_calls"):
            try:
                d[key] = json.loads(d[key]) if d[key] else None
            except ValueError:
                pass
        d["system_prompt"] = prompt[0] if prompt else ""
        return d

    def delete_run(self, run_id: str) -> None:
        try:
            with self._lock:
                self._conn.execute("DELETE FROM llm_call WHERE run_id = ?", (run_id,))
        except Exception as error:  # noqa: BLE001
            self._failed("delete", error)


def _preview(output: str | None) -> str:
    """The first words of an answer, or its tool calls - for the list view."""
    return (output or "")[:200]


_STORES: dict[str, TraceStore] = {}


def trace_path(cfg: Any) -> str:
    """Where a config's trace lives: agent.trace_db, else <run_dir>/llm_traces.db."""
    return getattr(cfg.agent, "trace_db", None) or str(Path(cfg.agent.run_dir) / "llm_traces.db")


def store_for(cfg: Any) -> TraceStore | None:
    """The trace store of a workspace's config - one per file, for the process.
    None when it cannot be opened: the run then goes on untraced."""
    path = trace_path(cfg)
    if path not in _STORES:
        try:
            _STORES[path] = TraceStore(path)
        except Exception as error:  # noqa: BLE001
            _log.warning("llm trace: cannot open %s (%s) - runs go untraced", path, error)
            return None
    return _STORES[path]


# ------------------------------------------------------------------- hooks
def _items(items: Any) -> list[Any]:
    out = []
    for item in items or []:
        if isinstance(item, dict):
            out.append(item)
        elif hasattr(item, "model_dump"):
            try:
                out.append(item.model_dump())
            except Exception:  # noqa: BLE001
                out.append(str(item))
        else:
            out.append(str(item))
    return out


def _capped(payload: Any) -> tuple[str, int]:
    text = json.dumps(payload, default=str)
    if len(text) > MAX_IO_BYTES:
        return json.dumps({"_elided": f"payload above {MAX_IO_BYTES} bytes", "size": len(text)}), len(text)
    return text, len(text)


def _estimate(text: str) -> int:
    """A token estimate for a backend that reports none: about 4 characters a token."""
    return max(1, len(text) // 4)


class TraceHooks(RunHooks):
    """Writes one row per model call of a runner, bound to the run and the stage
    when the runner starts. A stack pairs each start with its end, as the SDK
    can nest them."""

    def __init__(self, store: TraceStore | None, run_id: str, stage: str, *,
                 kind: str = "", round: int | None = None):
        self.store, self.run_id, self.stage = store, run_id, stage
        self.kind, self.round = kind, round
        self._inflight: list[tuple[int, float, str]] = []

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:  # noqa: ANN001
        if self.store is None:
            return
        try:
            payload, chars = _capped(_items(input_items))
            row_id = self.store.begin(
                run_id=self.run_id, kind=self.kind, stage=self.stage, round=self.round,
                call=self.store.next_call(self.run_id), agent=getattr(agent, "name", ""),
                model=str(getattr(getattr(agent, "model", None), "model", "") or ""),
                started_at=time.time(), input_items=len(input_items or []), input_chars=chars,
                system_sha=self.store.prompt(system_prompt or ""), input_json=payload,
                outcome="unfinished")
            self._inflight.append((row_id, time.perf_counter(), payload))
        except Exception:  # noqa: BLE001 - recording never breaks a call
            pass

    async def on_llm_end(self, context, agent, response) -> None:  # noqa: ANN001
        if self.store is None or not self._inflight:
            return
        row_id, started, payload = self._inflight.pop()
        try:
            items = _items(response.to_input_items() if hasattr(response, "to_input_items") else [])
            output, chars = _capped(items)
            calls = [i.get("name") for i in items if isinstance(i, dict)
                     and i.get("type") == "function_call"]
            usage = getattr(response, "usage", None)
            tokens_in = getattr(usage, "input_tokens", None) if usage else None
            tokens_out = getattr(usage, "output_tokens", None) if usage else None
            details_in = getattr(usage, "input_tokens_details", None) if usage else None
            details_out = getattr(usage, "output_tokens_details", None) if usage else None
            estimated = not tokens_in and not tokens_out
            if estimated:                    # safechain reports no usage
                tokens_in, tokens_out = _estimate(payload), _estimate(output)
            self.store.finish(
                row_id, ended_at=time.time(),
                duration_ms=int((time.perf_counter() - started) * 1000),
                input_tokens=tokens_in, output_tokens=tokens_out,
                cached_tokens=getattr(details_in, "cached_tokens", None),
                reasoning_tokens=getattr(details_out, "reasoning_tokens", None),
                tokens_estimated=int(estimated), output_chars=chars, output_json=output,
                tool_calls=json.dumps(calls), outcome="ok")
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> None:
    """Print a run's calls: one line each, then totals."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        raise SystemExit("usage: python -m agent.llm_trace <llm_traces.db> <run_id>")
    store = TraceStore(args[0])
    calls = store.calls(args[1])
    for c in calls:
        tools = ", ".join(c["tool_calls"]) or "text"
        print(f"{c['call']:>3}  {c['stage']:<8} r{c['round'] or '-':<3} {c['duration_ms'] or 0:>7} ms  "
              f"in {c['input_tokens'] or 0:>7}  out {c['output_tokens'] or 0:>6}  {c['outcome']:<10} {tools}")
    tin = sum(c["input_tokens"] or 0 for c in calls)
    tout = sum(c["output_tokens"] or 0 for c in calls)
    print(f"\n{len(calls)} calls · {tin:,} input tokens · {tout:,} output tokens"
          + (" (estimated)" if any(c["tokens_estimated"] for c in calls) else ""))


if __name__ == "__main__":
    main()
