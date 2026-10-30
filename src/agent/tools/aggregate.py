"""Summarise a long source before it is joined: a named aggregate features can read.

The agent writes ``aggregate(spark, raw)`` over the raw source - one row per key
and period (``agent.aggregates``). It runs once on the source as discovery reads
it, is joined through the source's confirmed linkage and checked point in time;
from then on, any feature of the run reads it as ``sources["<name>"]``, as it
would a linked source - but with a row per period, not per event. Defining one
costs no attempt: it is a step towards features, like a linkage.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from agent._child import _schema
from agent.aggregates import PERIODS, materialise_aggregate
from agent.session import Session
from agent.tools.linkage import _check, engine_of

__all__ = ["define_aggregate", "MAX_AGGREGATES"]

# Aggregates a run may define: each is a pass over a whole source.
MAX_AGGREGATES = 8


def define_aggregate(session: Session, name: str, source: str, period: str,
                     description: str, code: str, window_days: int | None = None,
                     key: str | list[str] | None = None) -> dict[str, Any]:
    ws = session.ws
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
        return {"ok": False, "error": f"{name!r} is not a valid name (letters, digits, _)"}
    if name in ws.sources() or name in ws.base_features or name in session.aggregates:
        return {"ok": False, "error": f"{name!r} is taken - by a source, a model column or "
                                      "an aggregate of this run; choose another name"}
    if len(session.aggregates) >= MAX_AGGREGATES:
        return {"ok": False, "error": (
            f"this run has {MAX_AGGREGATES} aggregates already: "
            f"{sorted(session.aggregates)} - build features on them")}
    if period not in PERIODS:
        return {"ok": False, "error": f"period must be one of {list(PERIODS)}"}
    if window_days is not None and (period != "transaction" or int(window_days) <= 0):
        return {"ok": False, "error": "window_days is for period='transaction': a positive "
                                      "number of days before each as-of date"}
    keys = [k.strip() for k in (key.split(",") if isinstance(key, str) else key or [])
            if k.strip()]
    if period == "transaction" and keys:
        return {"ok": False, "error": "key is for period='day', 'week' or 'month' - by "
                                      "transaction the runner groups by the id"}
    if (problem := session.allowed(source)):
        return {"ok": False, "error": problem}
    info = ws.linkage_info(source)
    if info is None:
        return {"ok": False, "error": (f"{source!r} has no confirmed linkage - an aggregate "
                                       "joins through it; call propose_linkage first")}
    raw_path = session.build_paths().get(source)
    if raw_path is None:
        return {"ok": False, "error": f"{source!r} has no data to aggregate"}
    linkage = ws.linkage_path(source).read_text()
    # Checked before anything runs, against the source's own columns: by day, week or
    # month the aggregate reads the source before the linkage - not the columns
    # link() makes, nor the model's id.
    columns, time_column = _schema(raw_path), info.get("time_column")
    if period != "transaction":
        keys = keys or list(info.get("key") or [])     # the linkage's, unless it names one
        if not keys:
            return {"ok": False, "error": (
                f"by {period}, name the key the aggregate groups by with period_end - the "
                f"column(s) of {source!r} the linkage joins on (no key was recorded for "
                f"it). {source!r} has: {columns[:40]}")}
        if unknown := [k for k in keys if k not in columns]:
            return {"ok": False, "error": (
                f"the key {unknown} is not a column of the raw {source!r} - the aggregate "
                "reads the source before the linkage, so it groups by the source's own "
                "column(s)" + (f": the linkage joins on {info['key']}, use key="
                               f"{','.join(info['key'])!r}" if info.get("key") else "")
                + f". {source!r} has: {columns[:40]}")}
        if time_column not in columns:
            return {"ok": False, "error": (
                f"the linkage's time column {time_column!r} is not a column of the raw "
                f"{source!r} - link() makes it - so {source!r} cannot be summarised by "
                f"{period}. Use period='transaction', which reads the linked rows.")}
    if not time_column:
        return {"ok": False, "error": f"the linkage for {source!r} does not record its time column"}
    # The time column goes with it: a linkage recorded in a run has no header to read it from.
    spec = {"source": source, "period": period, "time_column": time_column,
            "rule": info.get("rule") or "strict",
            "window_days": int(window_days) if window_days else None, "key": keys or None,
            "description": description.strip(), "code": code}

    def run(code: str, mode: str, **kwargs: Any):
        return session.run(code, mode, intent=f"aggregate:{name}", **kwargs)

    try:
        summary, linked = materialise_aggregate(
            run, name, spec, linkage, raw_path, engine=session.local_engine,
            linkage_engine=info.get("engine") or engine_of(linkage))
    except ValueError as error:
        return {"ok": False, "error": str(error)}
    if not summary.ok:
        return {**summary.for_agent(), "hint": _hint(source, info, columns)}
    if not linked.ok:
        return {"ok": False, "error": (
            f"the aggregate ran, but the linkage for {source!r} could not join it: "
            f"{linked.short_error}. Keep the columns the linkage joins on, under their "
            "names."), "linkage": linkage}
    rows = pd.read_parquet(linked.out_path)
    per_id = period == "transaction"
    # By transaction the runner checked each joined row against its as-of date.
    checks = {"head": rows.head(5).to_string(index=False, max_colwidth=25)} if per_id \
        else _check(session, rows, time_column, spec["rule"])
    if checks.get("error"):
        return {"ok": False, **checks}

    columns = [c for c in rows.columns if c not in (ws.id_col, "as_of")]
    source_rows = (summary.result or {}).get("source_rows")
    entry = {**spec, "name": name, "columns": columns, "path": linked.out_path,
             "source_rows": source_rows, "rows": (summary.result or {}).get("n_rows"),
             "linked_rows": (summary.result or {}).get("linked_rows") if per_id else int(len(rows)),
             "rows_per_id": round(len(rows) / max(rows[ws.id_col].nunique(), 1), 1)}
    session.aggregates[name] = entry
    folder = session.run_dir / "aggregates"
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.py").write_text(code)
    (session.run_dir / "aggregates.json").write_text(json.dumps(
        {n: {k: v for k, v in a.items() if k != "path"} for n, a in session.aggregates.items()},
        indent=2, default=str))
    session.emit("aggregate_defined", **{k: v for k, v in entry.items() if k not in ("code", "path")})
    if per_id:
        window = f" in the {spec['window_days']} days" if spec["window_days"] else ""
        shape = (f"{source_rows or 0:,} {source} rows -> {entry['linked_rows'] or 0:,} joined{window} "
                 f"before each as-of date -> {entry['rows'] or 0:,} rows, one per id")
        reads = (f"features read it as sources[{name!r}]: one row per id - map its columns "
                 "onto base by the id and combine them with base columns. Ids with no "
                 "events have no row. Its features are L2.")
    else:
        shape = (f"{source_rows or 0:,} {source} rows -> {entry['rows'] or 0:,} {period}ly rows "
                 f"-> {entry['linked_rows']:,} linked rows, {entry['rows_per_id']} per id")
        reads = (f"features read it as sources[{name!r}]: one row per id and {period} "
                 f"before its as-of date, `{time_column}` the {period}'s last day. Count "
                 "windows back from as_of as usual, then combine with base columns. Its "
                 "features are L2.")
    return {"ok": True, "name": name, "summary": shape, "columns": columns,
            "head": checks.get("head"), "next": reads}


def _hint(source: str, info: dict[str, Any], columns: list[str]) -> str:
    """The two views an aggregate may read, by their actual columns."""
    linked = info.get("linked_columns") or []
    return (f"by transaction, aggregate(spark, rows) gets each id's {source!r} events as the "
            f"linkage joins them - columns {linked[:30] or '(id, as_of and what link() keeps)'}: "
            f"return one row per id. By day, week or month it gets the raw {source!r} - "
            f"columns {columns[:30]} - with `period_end` added: return one row per key "
            f"{info.get('key') or ''} and period_end. Group in one pass - no loops")
