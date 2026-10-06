"""End the run with the agent's own account of it.

This is a tool rather than the model's last message on purpose: the agent
narrates between steps, and a reply that is only narration would otherwise end
the run early with no summary. The run stops here and nowhere else.
"""

from __future__ import annotations

from typing import Any

from agent.session import Session

__all__ = ["report_findings", "validated_sql"]


def report_findings(session: Session, summary: str) -> dict[str, Any]:
    """End the run. A data-request run ends only when every proposal is challenged,
    and its summary carries the validated SQL of the kept ones, copied by the code."""
    if session.l3_only:
        undecided = [r["intent"] for r in session.data_requests if r.get("status") == "proposed"]
        if undecided:
            # The verdicts are the run's output; a report in prose is not one.
            return {"ok": False, "error": f"challenge {undecided} with challenge_request "
                                          "first, then report"}
    if session.ideas_required and session.report_refusals < REPORT_REFUSALS:
        # A report with intents unspent ends the direction short - unless the agent
        # insists. Like ideas first, the runner asks for this; tools called directly
        # do not.
        left = _intents_left(session)
        if left:
            session.report_refusals += 1
            return {"ok": False, "error": (
                f"{left} intent(s) left ({session.budget()}). Propose from your ideas "
                "before you report - screen_feature for an L1/L2 idea, screen_request for "
                "an L3 one. If the direction is truly exhausted, call report_findings "
                "again and say why. Nothing was spent.")}
    if session.l3_only:
        summary = summary.strip() + "\n\n" + validated_sql(session)
    return session.end(summary)


# How often a report with feature intents left is sent back before it is accepted.
REPORT_REFUSALS = 2


def _intents_left(session: Session) -> int:
    """The intents a report would leave unspent: every L1/L2 one - features can
    always be built from the data in hand - and L3 ones while a CAS column is left
    that no request asks for."""
    from agent.memory import open_raw_columns

    l3_room = bool(open_raw_columns(session))
    if session.quota:
        return sum(max(n - session.used_at(lv), 0) for lv, n in session.quota.items()
                   if lv != "L3" or l3_room)
    if set(session.params.levels) <= {"L3"} and not l3_room:
        return 0
    return max(session.K - session.intents_used, 0)


def validated_sql(session: Session) -> str:
    """The kept requests' SQL - exactly as validated - and what was dropped."""
    kept = [r for r in session.data_requests if r.get("status") == "kept"]
    dropped = [r for r in session.data_requests if r.get("status") == "dropped"]
    lines = [f"Challenge: {len(kept)} kept, {len(dropped)} dropped (constructible from "
             "current data)."]
    if kept:
        lines += ["", "## Validated SQL - the kept requests"]
        for r in kept:
            lines += ["", f"### {r['intent']} {r['source_name']}", "```sql", r["sql"].strip(), "```"]
    if dropped:
        lines += ["", "Dropped: " + ", ".join(f"{r['intent']} {r['source_name']}" for r in dropped)]
    return "\n".join(lines)
