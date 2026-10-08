"""End the run with the agent's own account of it.

This is a tool rather than the model's last message on purpose: the agent
narrates between steps, and a reply that is only narration would otherwise end
the run early with no summary. The run stops here and nowhere else.
"""

from __future__ import annotations

from typing import Any

from agent.tools.data_pull import request_scope
from agent.session import Session

__all__ = ["report_findings", "validated_sql"]


def report_findings(session: Session, summary: str) -> dict[str, Any]:
    """End the run. A data-request run ends only when every proposal is challenged,
    and its summary carries the validated SQL of the kept ones, copied by the code."""
    if session.data_requests:
        undecided = [r["intent"] for r in session.data_requests if r.get("status") == "proposed"]
        if undecided:
            # The verdicts are the run's output; a report in prose is not one.
            return {"ok": False, "error": f"challenge {undecided} with challenge_request "
                                          "first, then report"}
    if session.gated and session.report_refusals < REPORT_REFUSALS:
        # A report short of the target ends the direction early - unless the agent
        # insists. Like ideas first, the runner asks for this; tools called directly
        # do not.
        left = _results_wanted(session)
        if left:
            session.report_refusals += 1
            return {"ok": False, "error": (
                f"{left} more result(s) wanted ({session.budget()}). Propose from your ideas "
                "before you report - screen_feature for an L1/L2 idea; for an L3 one, "
                "screen_request within a scope or propose_new_data beyond scope. If the "
                "direction is truly exhausted, call report_findings again and say why. "
                "Nothing was spent.")}
    if session.data_requests:
        summary = summary.strip() + "\n\n" + validated_sql(session)
    return session.end(summary)


# How often a report short of the target is sent back before it is accepted.
REPORT_REFUSALS = 2


def _results_wanted(session: Session) -> int:
    """How far a report would fall short of the target - while attempts remain.
    There is always room: a feature can be built from the data in hand, and a data
    request can look beyond scope when the scopes themselves are spent."""
    if session.attempts >= session.max_attempts:
        return 0
    return session.wanted()


def validated_sql(session: Session) -> str:
    """The kept requests' SQL - exactly as validated - and what was dropped."""
    kept = [r for r in session.data_requests if r.get("status") == "kept"]
    dropped = [r for r in session.data_requests if r.get("status") == "dropped"]
    lines = [f"Challenge: {len(kept)} kept, {len(dropped)} dropped (constructible from "
             "current data)."]
    in_scope = [r for r in kept if r.get("sql")]
    beyond = [r for r in kept if not r.get("sql")]
    if in_scope:
        lines += ["", "## Validated SQL - the kept requests within a scope"]
        for r in in_scope:
            lines += ["", f"### {r['intent']} {r['source_name']} ({request_scope(r)})",
                      "```sql", r["sql"].strip(), "```"]
    if beyond:
        lines += ["", "## Beyond scope - the kept ideas and the data they need"]
        for r in beyond:
            lines += ["", f"### {r['intent']} {r['source_name']}", str(r.get("data", "")).strip()]
    if dropped:
        lines += ["", "Dropped: " + ", ".join(f"{r['intent']} {r['source_name']}" for r in dropped)]
    return "\n".join(lines)
