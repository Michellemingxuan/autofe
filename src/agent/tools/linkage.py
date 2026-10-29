"""Join a source to the model ids, point in time, with the analyst's confirmation.

A source reaches a feature only through its confirmed linkage. The proposal is
run on the screen ids and checked - no event on or after as_of, no ids that are
not model ids - before the analyst is asked; a leaky join never reaches them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from agent.extracts import cached_extract, linkage_key, save_extract
from agent.session import Session

__all__ = ["propose_linkage", "ensure_linked", "engine_of", "read_linked"]


def engine_of(code: str) -> str:
    """The engine a confirmed linkage was written for - recorded in its header. One
    from before engines were recorded was written for pandas."""
    match = re.search(r"^# engine: (\w+)$", code, flags=re.M)
    return match.group(1) if match else "pandas"


def linkage_engine(session: Session) -> str:
    return session.ws.cfg.agent.linkage_engine


# A cut is kept when it holds at most this share of the source's rows.
CUT_MAX_SHARE = 0.5


def _extract_option(session: Session, source: str, keys: list[str]) -> dict[str, Any]:
    """Run options that also cut discovery's sample of the source (agent.extracts) -
    none when there is no key, or the cut is already what discovery reads."""
    src = session.ws.sources()[source]
    if not keys or cached_extract(session.ws, source, str(src.discovery_path), keys):
        return {}
    folder = session.run_dir / "extracts"
    folder.mkdir(parents=True, exist_ok=True)
    return {"extract": {"key": keys, "path": str(folder / f"{source}.parquet")}}


def _options(engine: str, extract: dict[str, Any]) -> dict[str, Any]:
    """On pandas the screen ids' linked rows go to disk by chunk of ids, as at
    evaluation, so a feature is built a chunk at a time in discovery too."""
    if engine == "spark":
        return {}
    return {"options": {"partition": True, **extract}}


def read_linked(path: str) -> pd.DataFrame:
    """A linkage's rows - one file, or on disk by chunk of ids (agent._child.Parts)."""
    from agent._child import Parts, is_parts

    if not is_parts(path):
        return pd.read_parquet(path)
    parts = Parts.open(path)
    return pd.concat([parts.read(k) for k in range(parts.n_chunks)], ignore_index=True)


def _keep_extract(session: Session, source: str, keys: list[str], kwargs: dict[str, Any],
                  result: Any) -> None:
    made = kwargs.get("extract")
    stats = (result.result or {}).get("extract")
    # Kept only when it saves reading: a cut holding most of the source is no faster.
    if made and stats and Path(made["path"]).exists() \
            and stats["rows"] <= CUT_MAX_SHARE * max(stats["source_rows"], 1):
        src = session.ws.sources()[source]
        save_extract(session.ws, source, made["path"], str(src.discovery_path), keys, stats)
        session.emit("extract_saved", source=source, **stats)


def propose_linkage(session: Session, source: str, code: str, time_column: str,
                    rule: str = "strict", key: str | list[str] | None = None) -> dict[str, Any]:
    """Run link() on the screen ids, check point-in-time, then ask the analyst."""
    if (problem := session.allowed(source)):
        return {"ok": False, "error": problem}
    src = session.ws.sources().get(source)
    if src is None or not src.usable:
        return {"ok": False, "error": f"{source!r} is not a usable source; "
                                      f"usable: {sorted(session.raw_paths())}"}
    if rule not in ("strict", "inclusive"):
        return {"ok": False, "error": "rule must be strict (event < as_of) or "
                                      "inclusive (event <= as_of)"}
    keys = [k.strip() for k in (key.split(",") if isinstance(key, str) else key or [])
            if k.strip()] or linkage_key(code)
    engine = linkage_engine(session)
    extra = _extract_option(session, source, keys) if engine != "spark" else {}
    _, result = session.run(code, "linkage", intent=f"linkage:{source}",
                            title=f"linkage for {source}", raw=session.build_paths(),
                            source=source, engine=engine, **_options(engine, extra))
    if not result.ok:
        return result.for_agent()

    checks = _check(session, read_linked(result.out_path), time_column, rule)
    if checks.get("error"):
        return {"ok": False, **checks}

    decision = session.ask("linkage", {"source": source, "code": code, "engine": engine,
                                       "time_column": time_column, "rule": rule, **checks})
    if not decision.approved:
        return {"ok": False, "approved": False, "user_note": decision.note, "checks": checks}

    path = session.ws.linkage_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# linkage for {source}: time_column={time_column}, rule={rule}"
                    + (f", key={','.join(keys)}" if keys else "") + "\n"
                    f"# confirmed in run {session.run_id}\n# engine: {engine}\n{code}")
    session.linked[source] = result.out_path
    _keep_extract(session, source, keys, extra, result)
    return {"ok": True, "approved": True, "user_note": decision.note,
            "saved": str(path), "checks": checks,
            "next": f"features can now use sources[{source!r}]"}


def ensure_linked(session: Session, source: str) -> str | None:
    """Linked rows for a source with confirmed linkage; built once per run."""
    if source in session.linked:
        return None
    path = session.ws.linkage_path(source)
    if not path.exists():
        return (f"{source!r} has no confirmed linkage; call propose_linkage first "
                f"(confirmed: {session.ws.linked()})")
    code = path.read_text()
    keys = linkage_key(code)
    extra = _extract_option(session, source, keys) if engine_of(code) != "spark" else {}
    _, result = session.run(code, "linkage", intent=f"linkage:{source}",
                            title=f"reuse confirmed linkage for {source}",
                            raw=session.build_paths(), source=source, engine=engine_of(code),
                            **_options(engine_of(code), extra))
    if not result.ok:
        return f"the confirmed linkage for {source!r} failed: {result.error}"
    session.linked[source] = result.out_path
    _keep_extract(session, source, keys, extra, result)
    return None


def _at_full_size(session: Session, linked: pd.DataFrame) -> dict[str, Any]:
    """The linked rows at evaluation, scaled from the screen ids by the id ratio:
    a row per (id, event before its as-of date), so a customer under several as-of
    dates brings their events once per date - the join is many-to-many."""
    ws = session.ws
    full = ws.data_size()["full_rows"]
    if not full or not len(linked):
        return {}
    ratio = full / len(ws.screen)
    gb = float(linked.memory_usage(deep=True).sum()) * ratio / 1e9
    out: dict[str, Any] = {
        "events_per_id": round(len(linked) / max(linked[ws.id_col].nunique(), 1), 1),
        "rows_at_full_size": int(len(linked) * ratio),
        "gb_at_full_size": round(gb, 1)}
    try:
        machine = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return out
    if gb > 0.5 * machine:
        out["scale"] = (f"on the full data this linkage holds about {out['rows_at_full_size']:,} "
                        f"rows, {gb:.0f} GB - the machine has {machine:.0f} GB. Keep fewer "
                        "columns, and a shorter window of events before each as-of date.")
    return out


def _check(session: Session, linked: pd.DataFrame, time_column: str,
           rule: str) -> dict[str, Any]:
    id_col, screen = session.ws.id_col, session.ws.screen
    if time_column not in linked.columns:
        return {"error": f"link() output has no {time_column!r} column to check "
                         f"against as_of; columns: {list(linked.columns)}"}
    for col in (time_column, "as_of"):
        if pd.api.types.is_numeric_dtype(linked[col]):
            return {"error": f"{col!r} is numeric ({linked[col].dtype}); return it as "
                             "a date (pd.to_datetime / F.to_date) so it can be checked"}
    event = pd.to_datetime(linked[time_column], errors="coerce")
    as_of = pd.to_datetime(linked["as_of"], errors="coerce")
    late = (event > as_of) if rule == "inclusive" else (event >= as_of)
    unknown_ids = set(linked[id_col]) - set(screen[id_col])
    checks = {
        "rows": int(len(linked)),
        "match_rate": round(linked[id_col].nunique() / len(screen), 4),
        "point_in_time_violations": int(late.sum()),
        "unparsed_dates": int(event.isna().sum() + as_of.isna().sum()),
        "unknown_ids": len(unknown_ids),
        "head": linked.head(5).to_string(index=False, max_colwidth=25),
    }
    checks.update(_at_full_size(session, linked))
    if checks["point_in_time_violations"]:
        checks["error"] = (f"{checks['point_in_time_violations']} linked rows have "
                           f"{time_column} {'>' if rule == 'inclusive' else '>='} as_of; "
                           "filter them out in link()")
    elif checks["unknown_ids"]:
        checks["error"] = f"{len(unknown_ids)} ids in the output are not model ids"
    return checks
