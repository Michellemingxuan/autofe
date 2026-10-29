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

from agent.aggregates import PERIODS, linkage_rule, materialise_aggregate
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
    if period != "transaction" and not keys:
        return {"ok": False, "error": (f"by {period}, name the key the aggregate groups by "
                                       "with period_end - the source column(s) the linkage "
                                       "joins on, e.g. key='customer_id'")}
    if (problem := session.allowed(source)):
        return {"ok": False, "error": problem}
    linkage_path = ws.linkage_path(source)
    if not linkage_path.exists():
        return {"ok": False, "error": (f"{source!r} has no confirmed linkage - an aggregate "
                                       "joins through it; call propose_linkage first")}
    raw_path = session.build_paths().get(source)
    if raw_path is None:
        return {"ok": False, "error": f"{source!r} has no data to aggregate"}
    linkage = linkage_path.read_text()
    time_column = _time_column(linkage)
    if not time_column:
        return {"ok": False, "error": f"the linkage for {source!r} does not record its time column"}
    # The time column goes with it: a linkage recorded in a run has no header to read it from.
    spec = {"source": source, "period": period, "time_column": time_column,
            "rule": linkage_rule(linkage),
            "window_days": int(window_days) if window_days else None, "key": keys or None,
            "description": description.strip(), "code": code}

    def run(code: str, mode: str, **kwargs: Any):
        return session.run(code, mode, intent=f"aggregate:{name}", **kwargs)

    try:
        summary, linked = materialise_aggregate(
            run, name, spec, linkage, raw_path, engine=session.local_engine,
            linkage_engine=engine_of(linkage))
    except ValueError as error:
        return {"ok": False, "error": str(error)}
    if not summary.ok:
        return {**summary.for_agent(), "hint": _hint(session, source)}
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


def _time_column(linkage: str) -> str:
    from agent.aggregates import linkage_time_column

    return linkage_time_column(linkage) or ""


def _hint(session: Session, source: str) -> str:
    src = session.ws.sources().get(source)
    columns = list(src.columns) if src else []
    return (f"by transaction, aggregate(spark, rows) gets each id's {source!r} events, joined "
            f"by the linkage (id, as_of and {', '.join(columns[:20])}): return one row per id. "
            "By day, week or month it gets the raw rows with `period_end` added: return one "
            "row per key and period_end. Group in one pass - no loops")
