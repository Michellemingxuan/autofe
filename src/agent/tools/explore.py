"""Look before building: the catalog, a few rows, and exploratory code."""

from __future__ import annotations

import re
from typing import Any

from agent.memory import open_raw_columns, requested_columns
from agent.session import Session

__all__ = ["catalog", "scope", "sample_rows", "run_probe"]


def catalog(session: Session, query: str = "") -> dict[str, Any]:
    """Search model columns, the allowed sources and the scopes."""
    found = session.ws.catalog(query)
    allowed = set(session.params.sources)
    if "sources" in found:
        found["sources"] = [s for s in found["sources"] if s["name"] in allowed]
    if "matches" in found:
        found["matches"] = [m for m in found["matches"]
                            if not m["where"].startswith("source:")
                            or m["where"][7:] in allowed]
    return found


def scope(session: Session, query: str = "", status: str = "", table: str = "",
          scope_name: str = "") -> dict[str, Any]:
    """The scopes' variables - what each is, which scope and table hold it, whether the
    model uses it, and whether a data request already asks for it."""
    out = session.ws.scope_variables(query, status, table, scope_name=scope_name)
    taken = requested_columns(session)
    for v in out["variables"]:
        if (by := taken.get((str(v["table"]).lower(), str(v["variable"]).lower()))):
            v["requested"] = by
    if taken and status.strip() == "unused_raw" and not open_raw_columns(session, taken):
        out["note"] = ("every unused_raw variable is already requested - see `requested`. "
                       "Ask only for what those requests lack, or report that the scope is "
                       "used up for this direction.")
    return out


def sample_rows(session: Session, source: str = "model_database", n: int = 5,
                columns: list[str] | None = None) -> str:
    """A few labelled model rows, or the first rows of an allowed source."""
    tables = set(session.ws.scope()["table"].astype(str)) if len(session.ws.scope()) else set()
    if source in tables:
        return (f"{source!r} is a {session.ws.scope_of(source)} table: it has no rows in this "
                f"workspace. scope(table="
                f"{source!r}) lists its variables; an L3 request (screen_request) is how its "
                "data is obtained.")
    if source != "model_database" and (problem := session.allowed(source)):
        return problem
    try:
        return session.ws.sample_rows(source, min(int(n), 50), columns)
    except KeyError as error:
        return str(error)


def run_probe(session: Session, code: str, purpose: str = "") -> dict[str, Any]:
    """Run exploratory code over `base`, every allowed source (`raw`) and the
    linked sources (`sources`) - any confirmed one it names is joined first."""
    from agent.tools.linkage import ensure_linked

    for source in session.params.sources:
        if re.search(rf"sources\[\s*['\"]{re.escape(source)}['\"]", code) \
                and session.ws.linkage_path(source).exists():
            if problem := ensure_linked(session, source):
                return {"ok": False, "error": problem}
    _, result = session.run(code, "probe", intent="probe", title=purpose,
                            raw=session.raw_paths(), sources=dict(session.linked))
    session.failed_streak = 0                      # it looked: the next screen may go
    return result.for_agent()
