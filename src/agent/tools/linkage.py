"""Join a source to the model ids, point in time, with the analyst's confirmation.

A source reaches a feature only through its confirmed linkage. The proposal is
run on the screen ids and checked - no event on or after as_of, no ids that are
not model ids - before the analyst is asked; a leaky join never reaches them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from agent.session import Session

__all__ = ["propose_linkage", "ensure_linked"]


def propose_linkage(session: Session, source: str, code: str, time_column: str,
                    rule: str = "strict") -> dict[str, Any]:
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
    _, result = session.run(code, "linkage", intent=f"linkage:{source}",
                            title=f"linkage for {source}", raw=session.raw_paths(),
                            source=source)
    if not result.ok:
        return result.for_agent()

    checks = _check(session, pd.read_parquet(result.out_path), time_column, rule)
    if checks.get("error"):
        return {"ok": False, **checks}

    decision = session.ask("linkage", {"source": source, "code": code,
                                       "time_column": time_column, "rule": rule, **checks})
    if not decision.approved:
        return {"ok": False, "approved": False, "user_note": decision.note, "checks": checks}

    path = session.ws.linkage_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# linkage for {source}: time_column={time_column}, rule={rule}\n"
                    f"# confirmed in run {session.run_id}\n{code}")
    session.linked[source] = result.out_path
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
    _, result = session.run(path.read_text(), "linkage", intent=f"linkage:{source}",
                            title=f"reuse confirmed linkage for {source}",
                            raw=session.raw_paths(), source=source)
    if not result.ok:
        return f"the confirmed linkage for {source!r} failed: {result.error}"
    session.linked[source] = result.out_path
    return None


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
    if checks["point_in_time_violations"]:
        checks["error"] = (f"{checks['point_in_time_violations']} linked rows have "
                           f"{time_column} {'>' if rule == 'inclusive' else '>='} as_of; "
                           "filter them out in link()")
    elif checks["unknown_ids"]:
        checks["error"] = f"{len(unknown_ids)} ids in the output are not model ids"
    return checks
