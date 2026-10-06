"""Challenge a data request: can what it asks for be built from the data we have?

A pull from the CAS is effort - a BigQuery job over very large tables, then a
source to link. It is worth it only for information the current data cannot
give. So in a data-request run every proposal goes through one loop:

    propose (screen_request validates the SQL) -> challenge -> kept / dropped

The challenge is the agent's own reflection on its proposal, with one question:

    Can this new information be constructed from the model database and the
    linked sources?

The judgement is the agent's; this tool is deterministic and only records and
checks it. The agent answers ``constructible``, ``partly`` or ``new``; a
"constructible" answer must come with the construction - a ``build()`` script
over the current data - and the tool runs it on the screen rows. The rules for
the outcome are fixed here, not left to the model:

* constructible, and the construction runs   -> dropped
* constructible, but it does not run         -> kept, the claim not shown
* partly / new                               -> kept

A dropped request gives its intent back: the current data already covers it,
so the agent must propose something else in its place. To keep that from
looping, the refunds stop once a run has made 2 x K proposals.

A construction that works is kept on record either way: it is a feature idea
for an L1/L2 run.
"""

from __future__ import annotations

import re
from typing import Any

from agent.session import Session
from agent.tools.linkage import ensure_linked

__all__ = ["challenge_request", "current_data_brief", "VERDICTS"]

VERDICTS = ("constructible", "partly", "new")


def challenge_request(session: Session, intent: str, verdict: str, reasoning: str,
                      columns: list[str] | None = None, code: str = "") -> dict[str, Any]:
    """Record the verdict on one proposal; run its construction; keep or drop it."""
    record = next((r for r in session.data_requests if r.get("intent") == intent), None)
    if record is None:
        return {"ok": False, "error": f"no proposal {intent!r}; proposals: "
                                      f"{[r.get('intent') for r in session.data_requests]}"}
    if record.get("status") != "proposed":
        return {"ok": False, "error": f"{intent} is already decided ({record['status']})"}
    verdict = verdict.strip().lower()
    if verdict not in VERDICTS:
        return {"ok": False, "error": f"verdict must be one of {VERDICTS}"}
    if verdict == "constructible" and not code.strip():
        return {"ok": False, "error": "a 'constructible' verdict needs the construction: a "
                                      "build(spark, sources, base) script - it will be run"}

    result: dict[str, Any] = {"verdict": verdict, "reasoning": reasoning.strip(),
                              "columns": list(columns or []), "code": code.strip(),
                              "code_ok": None, "code_error": None}
    if code.strip():
        result.update(_try_construction(session, record, code))
    if verdict == "constructible" and result["code_ok"]:
        status = "dropped"
    else:
        status = "kept"
        if verdict == "constructible":
            result["note"] = "said to be constructible, but the construction did not run"
    record["status"], record["challenge"] = status, result
    refunded = False
    if status == "dropped" and session.request_spent(record) \
            and len(session.data_requests) < 2 * session.K:
        record["spent"], record["refunded"] = False, True
        session.intents_used -= 1
        refunded = True
    session.save_requests()
    session.emit("request_challenged", intent=intent, source_name=record["source_name"],
                 status=status, refunded=refunded, **result)
    left = [r["intent"] for r in session.data_requests if r.get("status") == "proposed"]
    reply = {"intent": intent, "status": status, "code_ok": result["code_ok"],
             "remaining": left}
    if result["code_error"]:
        reply["code_error"] = result["code_error"]
    if status == "dropped":
        reply["next"] = (("dropped - the current data already covers it, so it does not count: "
                          "re-propose - ask for information the model database and the linked "
                          f"sources cannot give ({session.budget()}).") if refunded else
                         f"dropped - and no refund: this run has made {len(session.data_requests)} "
                         f"proposals, the limit of 2 x K ({session.budget()}).")
        return reply
    if not left:
        budget = f"{session.intents_used}/{session.K} intents used"
        reply["next"] = (f"every proposal so far is decided ({budget}); propose the next "
                         "request, or call report_findings")
    return reply


def _try_construction(session: Session, record: dict[str, Any], code: str) -> dict[str, Any]:
    """Run the construction on the screen rows - proof, or not."""
    ws = session.ws
    used = [s for s in ws.sources() if re.search(rf"['\"]{re.escape(s)}['\"]", code)]
    for source in used:
        if (problem := session.allowed(source)) or (problem := ensure_linked(session, source)):
            return {"code_ok": False, "code_error": problem}
    _, run = session.run(code, "feature", intent=record["intent"],
                         title=f"challenge: rebuild {record['source_name']} from current data",
                         sources={s: session.linked[s] for s in used})
    if not run.ok:
        return {"code_ok": False, "code_error": (run.error or "")[-600:]}
    return {"code_ok": True, "code_error": None}


