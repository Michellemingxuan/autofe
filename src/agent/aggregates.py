"""Aggregates: a source summarised before it is joined to the model ids.

A linked source holds a row per (id, event before its as-of date): a customer
under several as-of dates brings every event once per date, and a long source -
years of card spend - makes that table far larger than any feature needs. An
aggregate summarises it once, at one of two grains.

**By transaction** - one row per model id: its events, joined by the source's
linkage and cut to the window before its as-of date (``window_days``), are
summarised as they are joined, a chunk of ids at a time - so the ids x events
table is never held whole. The grain for a source that is a short history
before each transaction (the last 7 days of spend, 30 of payments).

**By day, week or month** - on the raw source, per customer and period, then
joined through the source's confirmed linkage. The aggregate names its `key`, so
a source too large to hold is split on disk by it and summarised a part at a time:

    aggregate(spark, raw)   one row per key and period_end   (raw: the source on disk)
    link(base_ids, table)   the buckets that end before each as-of date
    build(spark, sources, base)   sources["<aggregate>"]: id, as_of, period, values

The runner sets ``period_end`` - the last day of each event's period - so a
bucket reaches an as-of date only when all of it lies before that date; the
script may not move it (``agent._child``). The aggregate takes the source's time
column name, so the linkage written for the source joins it unchanged.

The same three steps run in discovery (on the source as discovery reads it),
at evaluation (on the whole source) and at inference (on the target cases'
history), from the aggregate's recorded code.
"""

from __future__ import annotations

import re
from typing import Any, Callable

__all__ = ["PERIODS", "linkage_time_column", "materialise_aggregate", "spec_key"]

PERIODS = ("transaction", "day", "week", "month")


def linkage_time_column(code: str) -> str | None:
    """The time column a confirmed linkage checks against as_of - from its header."""
    match = re.search(r"time_column=(\w+)", code)
    return match.group(1) if match else None


def linkage_rule(code: str) -> str:
    """strict (event < as_of) or inclusive (event <= as_of) - from its header."""
    match = re.search(r"rule=(\w+)", code)
    return match.group(1) if match else "strict"


def materialise_aggregate(run: Callable[..., tuple[str, Any]], name: str, spec: dict[str, Any],
                          linkage: str, raw_path: str, *, engine: str, linkage_engine: str,
                          where: str = "", partition: bool = False) -> tuple[Any, Any]:
    """Summarise one source with an aggregate's code, then join it with the source's
    linkage: the two runs' results. `run(code, mode, **kwargs)` runs a script the
    caller's way (a session, an evaluation, an inference); the second result is None
    when the first failed. `partition`: the joined rows go to disk by chunk of ids
    (evaluation, ``agent._child.Parts``)."""
    source = spec["source"]
    time_column = spec.get("time_column") or linkage_time_column(linkage)
    if time_column is None:
        raise ValueError(f"the linkage for {source!r} does not say its time column")
    suffix = f" {where}" if where else ""
    if spec["period"] == "transaction":
        if engine != linkage_engine:
            raise ValueError(f"a per-transaction aggregate runs the linkage of {source!r} inside "
                             f"it: write it for {linkage_engine}, the linkage's engine")
        window = spec.get("window_days")
        _, summary = run(spec["code"], "aggregate", engine=engine,
                         title=f"aggregate {name} (per transaction, {source}"
                               + (f", last {window} days" if window else "") + f"){suffix}",
                         raw={source: raw_path}, source=source,
                         options={"time_column": time_column, "period": "transaction",
                                  "rule": spec.get("rule") or linkage_rule(linkage),
                                  "window_days": window, "linkage": linkage})
        return summary, summary
    _, summary = run(spec["code"], "aggregate", engine=engine,
                     title=f"aggregate {name} ({spec['period']}ly {source}){suffix}",
                     raw={source: raw_path}, source=source,
                     options={"time_column": time_column, "period": spec["period"],
                              "key": spec.get("key")})
    if not summary.ok:
        return summary, None
    _, linked = run(linkage, "linkage", engine=linkage_engine,
                    title=f"linkage for {name}{suffix}",
                    raw={source: str(summary.out_path)}, source=source,
                    **({"options": {"partition": True}} if partition else {}))
    return summary, linked


def spec_key(spec: dict[str, Any], linkage: str) -> tuple[str, ...]:
    """What makes two aggregates the same work: source, grain, window, code, linkage, key."""
    return (spec["source"], spec["period"], str(spec.get("window_days")), spec["code"], linkage,
            ",".join(spec.get("key") or []))
