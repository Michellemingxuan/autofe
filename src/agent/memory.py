"""What earlier directions already proposed, so a new one builds on them.

Every run under ``agent.run_dir`` leaves its features (``ledger.json``) and its
data requests (``data_requests.json``). A new run is briefed with them up front
- what was tried, what verified, which CAS columns are already asked for - and
scope() marks each requested column, so the agent plans around them before it
proposes. Two checks remain as a backstop, refusing before anything is spent:

* **A feature identical in values** to an earlier one: on the screen rows they
  share, a rank correlation of at least ``SAME_FEATURE`` (0.99). The values of
  every screened feature are kept for this (``values/<name>.parquet``). A
  variation is not identical - a 30-day and a 90-day sum of the same events
  correlate well below that - so it is screened as usual.
* **A data request for the same data**: the same CAS table and the same columns,
  leaving aside the identifiers and the partition date every request selects.
  A request the analyst rejected does not block: their note may ask for a
  narrower or a revised pull of the same data.

A deleted intent (``deleted`` in the ledger) or a deleted run is forgotten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from agent.session import Session
    from agent.workspace import Workspace

__all__ = ["SAME_FEATURE", "prior_features", "prior_requests", "requested_columns", "open_raw_columns",
           "scope_left",
           "memory_brief", "save_values", "same_feature", "same_request"]

SAME_FEATURE = 0.99
_NOT_RUNS = {"linkage", "evaluations", "replaced", "uploads", "shots"}


def _runs(ws: "Workspace") -> list[Path]:
    root = Path(ws.cfg.agent.run_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name not in _NOT_RUNS)


def _direction(folder: Path) -> str:
    events = folder / "events.jsonl"
    if events.exists():
        for line in open(events):
            if '"run_started"' in line:
                return json.loads(line).get("direction", "")
    return ""


def prior_features(ws: "Workspace", skip_run: str | None = None) -> list[dict[str, Any]]:
    """Every screened, undeleted feature of the other runs, oldest first."""
    out = []
    for folder in _runs(ws):
        ledger = folder / "ledger.json"
        if folder.name == skip_run or not ledger.exists():
            continue
        direction = _direction(folder)
        for e in json.loads(ledger.read_text()):
            if not e.get("deleted"):
                out.append({**{k: e.get(k) for k in ("name", "description", "level", "sources",
                                                     "verified", "delta", "reason")},
                            "run_id": folder.name, "direction": direction})
    return out


def prior_requests(ws: "Workspace", skip_run: str | None = None) -> list[dict[str, Any]]:
    """Every data request of the other runs, with how it ended."""
    out = []
    for folder in _runs(ws):
        path = folder / "data_requests.json"
        if folder.name == skip_run or not path.exists():
            continue
        for r in json.loads(path.read_text()):
            out.append({**{k: r.get(k) for k in ("source_name", "tables", "columns", "gap",
                                                 "data", "approved", "note")},
                        "status": _status(r), "run_id": folder.name})
    return out


def _status(r: dict[str, Any]) -> str:
    if r.get("status"):
        return r["status"]                              # a data-request run: proposed/kept/dropped
    if "approved" in r:
        return "approved" if r["approved"] else "rejected by the analyst"
    return "proposed"


def _requested(ws: "Workspace", records: list[dict[str, Any]],
               run_id: str | None = None) -> dict[tuple[str, str], list[str]]:
    """(table, column) -> the requests asking for it; keys, and rejected pulls, aside."""
    out: dict[tuple[str, str], list[str]] = {}
    for r in records:
        if r.get("approved") is False:
            continue
        where = "this run" if r["run_id"] == run_id else r["run_id"]
        for table in r.get("tables") or []:
            for column in _payload(ws, [table], r.get("columns") or []):
                out.setdefault((table.split(".")[-1].lower(), column), []).append(
                    f"{r['source_name']} ({r['status']}, {where})")
    return out


def requested_columns(session: "Session") -> dict[tuple[str, str], list[str]]:
    """The CAS columns already asked for - (table, column) -> the requests that ask for
    them - in earlier runs and this one; keys, and pulls the analyst rejected, aside."""
    earlier = prior_requests(session.ws, session.run_id) + [
        {**r, "status": _status(r), "run_id": session.run_id} for r in session.data_requests]
    return _requested(session.ws, earlier, session.run_id)


def _open(ws: "Workspace", requested: dict[tuple[str, str], list[str]]) -> list[str]:
    scope = ws.scope()
    if not len(scope):
        return []
    keys: dict[str, set[str]] = {}                  # per table, worked out once
    out = []
    for r in scope[scope["status"] == "unused_raw"].itertuples(index=False):
        table, column = str(r.table).lower(), str(r.variable).lower()
        if table not in keys:
            p = ws.table_profile(str(r.table))
            keys[table] = {c.lower() for c in (*p["identifiers"], *p["partition"])}
        if column not in keys[table] and (table, column) not in requested \
                and not column.endswith("pkey"):
            out.append(f"{table}.{column}")
    return out


def open_raw_columns(session: "Session",
                     requested: dict[tuple[str, str], list[str]] | None = None) -> list[str]:
    """The unused_raw CAS columns no request asks for yet - keys aside."""
    return _open(session.ws, requested_columns(session) if requested is None else requested)


def scope_left(ws: "Workspace") -> list[str]:
    """The unused_raw CAS columns no run has asked for yet - before a run starts."""
    return _open(ws, _requested(ws, prior_requests(ws)))


# ------------------------------------------------------------------ features
def save_values(session: "Session", name: str, values: np.ndarray) -> None:
    """Keep a screened feature's values on the screen rows, for later runs."""
    folder = session.run_dir / "values"
    folder.mkdir(exist_ok=True)
    pd.DataFrame({session.ws.id_col: session.ws.screen[session.ws.id_col].astype(str).to_numpy(),
                  "value": np.asarray(values, dtype=float)}).to_parquet(folder / f"{name}.parquet")


def same_feature(session: "Session", name: str, values: np.ndarray) -> str | None:
    """The earlier feature this one is identical to, in words - or None."""
    ws = session.ws
    new = pd.Series(np.asarray(values, dtype=float),
                    index=ws.screen[ws.id_col].astype(str).to_numpy())
    if new.notna().sum() < 50 or new.nunique() < 2:
        return None
    deleted = {e["name"] for e in session.ledger if e.get("deleted")}
    for folder in _runs(ws):
        stored = folder / "values"
        if not stored.exists():
            continue
        ledger = folder / "ledger.json"
        gone = deleted if folder.name == session.run_id else \
            {e["name"] for e in json.loads(ledger.read_text()) if e.get("deleted")} \
            if ledger.exists() else set()
        for path in sorted(stored.glob("*.parquet")):
            if path.stem in gone or (folder.name == session.run_id and path.stem == name):
                continue
            old = pd.read_parquet(path).set_index(ws.id_col)["value"]
            both = pd.concat([new.rename("a"), old.rename("b")], axis=1, join="inner").dropna()
            if len(both) < 50 or both["b"].nunique() < 2:
                continue
            rho = both["a"].rank().corr(both["b"].rank())
            if abs(rho) >= SAME_FEATURE:
                where = "this run" if folder.name == session.run_id else f"run {folder.name}"
                return (f"identical to `{path.stem}` from {where} (rank correlation {rho:+.3f} "
                        f"on the screen rows)")
    return None


# ------------------------------------------------------------------ requests
def _payload(ws: "Workspace", tables: list[str], columns: list[str]) -> frozenset[str]:
    """The data a request asks for: its columns, minus the keys every request selects."""
    keys: set[str] = set()
    for t in tables:
        p = ws.table_profile(t.split(".")[-1])
        keys |= {c.lower() for c in (*p["identifiers"], *p["partition"])}
    return frozenset(c.lower() for c in columns) - keys


def same_request(session: "Session", tables: list[str], columns: list[str]) -> str | None:
    """The earlier request asking for the same data, in words - or None."""
    ws = session.ws
    want_tables = {t.split(".")[-1].lower() for t in tables}
    want = _payload(ws, tables, columns)
    if not want:
        return None
    earlier = [{**r, "status": _status(r), "run_id": session.run_id}
               for r in session.data_requests] + prior_requests(ws, session.run_id)
    for r in earlier:
        if r.get("approved") is False:
            continue                                    # rejected: a revised pull may follow
        theirs = {t.split(".")[-1].lower() for t in r.get("tables") or []}
        if theirs == want_tables and _payload(ws, r["tables"], r.get("columns") or []) == want:
            where = "this run" if r["run_id"] == session.run_id else f"run {r['run_id']}"
            return (f"the same data as `{r['source_name']}` from {where} ({r.get('status')}): "
                    f"{', '.join(sorted(want))} of {', '.join(sorted(want_tables))}")
    return None
