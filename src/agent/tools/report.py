"""End the run with the agent's own account of it.

This is a tool rather than the model's last message on purpose: the agent
narrates between steps, and a reply that is only narration would otherwise end
the run early with no summary. The run stops here and nowhere else.
"""

from __future__ import annotations

from typing import Any

from agent.tools.data_pull import request_scope
from agent.composer import message
from agent.session import Session

__all__ = ["report_findings", "validated_sql"]


def report_findings(session: Session, summary: str) -> dict[str, Any]:
    """End the run. A data-request run ends only when every proposal is challenged,
    and its summary carries the validated SQL of the kept ones, copied by the code.
    Short of the target, a report is sent back; if the agent insists in a round that
    found something, the next round of ideas is asked for instead."""
    if session.data_requests:
        undecided = [r["intent"] for r in session.data_requests if r.get("status") == "proposed"]
        if undecided:
            # The verdicts are the run's output; a report in prose is not one.
            return {"ok": False, "error": f"challenge {undecided} with challenge_request "
                                          "first, then report"}
    left = _results_wanted(session) if session.gated else 0
    if left and session.report_refusals < REPORT_REFUSALS:
        # A report short of the target is sent back. Like ideas first, the runner
        # asks for this; tools called directly do not.
        session.report_refusals += 1
        return {"ok": False, "error": (
            f"{left} more result(s) wanted ({session.budget()}). Propose from your ideas "
            "before you report - screen_feature for an L1/L2 idea; for an L3 one, "
            "screen_request within a scope or propose_new_data beyond scope. If the "
            "direction is truly exhausted, call report_findings again and say why. "
            "Nothing was spent.")}
    if left and session.ideas_required and session.results() > session.round_results:
        # The agent insists, but this round still found something: the direction is
        # not exhausted, only these ideas are. A new round of ideas comes next; the
        # run ends early only on a round that finds nothing.
        session.round_spent = True
        return {"ok": False, "error": (
            f"Not yet: {left} more result(s) wanted ({session.budget()}), and this round "
            "still found some. This round is closed - a new round of ideas comes next, "
            "with what worked and what failed in front of you.")}
    if left and session.ideas_required and not session.rethought \
            and session.results() <= RETHINK_SHARE * session.K:
        # Exhausted, the agent says - with half the target or less. Once a run, it is
        # sent back to take stock: where the run stands and why the attempts failed,
        # then a new round of ideas built on that analysis.
        session.rethought = True
        session.round_spent = True
        session.rethink = rethink(session)
        session.emit("agent_nudged", attempt=0, reason="exhausted at half the target or less: "
                                                       "a round to rethink")
        return {"ok": False, "error": session.rethink}
    if session.data_requests:
        summary = summary.strip() + "\n\n" + validated_sql(session)
    return session.end(summary)


# A report of exhaustion at this share of the target or less is sent back - once a
# run - to rethink: a summary of the run, and a new round of ideas.
RETHINK_SHARE = 0.5

_FAILURE_KINDS = (
    ("did not beat base on the screen", r"is not above|gain"),
    ("redundant with what the model has", r"redundan|explained|model column|existing column|seen before|same as"),
    ("the script failed", r"script failed|build\(\) returned|timed out"),
)


def rethink(session: Session) -> str:
    """Where the run stands - what verified, why the rest failed - and the push to
    think again, for the agent that would stop at half the target or less."""
    import re
    from collections import Counter

    entries = [e for e in session.ledger if not e.get("deleted")]
    good = [f"`{e['name']}` ({e['level']}, {e['delta']:+.4f})" if isinstance(e.get("delta"), (int, float))
             else f"`{e['name']}` ({e['level']})" for e in entries if e.get("verified")]
    kept = [f"`{r['source_name']}` (L3)" for r in session.data_requests if r.get("status") == "kept"]
    failed = [e for e in entries if not e.get("verified")]
    kinds: Counter = Counter()
    for e in failed:
        reason = str(e.get("reason") or "")
        kinds[next((k for k, pat in _FAILURE_KINDS if re.search(pat, reason, re.I)), "other reasons")] += 1
    examples = [f"`{e['name']}` - {str(e.get('reason') or '').splitlines()[0][:120]}" for e in failed[-4:]]
    return message(
        "rethink", budget=session.budget(),
        worked=", ".join(good + kept) or "nothing yet",
        failed=("; ".join(f"{n} {k}" for k, n in kinds.most_common()) or "none"),
        examples=("\n".join(f"* {x}" for x in examples) or "* none"))


# How often a report short of the target is sent back, in a round, before the
# agent may insist.
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
