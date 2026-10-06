"""Build one feature on the screen rows and score it against base. One intent.

The feature's script runs out of process (``agent.execution``); its column is
then handed to the discovery screen (``discovery.screen.Screener``), whose
guards (``discovery.guards``: finite values, no spikes, not redundant with a
base column) run before the fit. The analyst's gates decide what is verified:
the Gini gain must clear ``min_gini_gain`` and, when set, the capture-rate gain
must clear ``min_capture_gain``.

A feature identical in values to one already screened - in this run or an
earlier direction (``agent.memory``) - is refused before it is scored, and the
intent is given back.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from agent.execution import check_code
from agent.memory import save_values, same_feature
from agent.session import FEATURE_LEVELS, Session
from agent.tools.ideas import ideas_needed
from agent.tools.linkage import ensure_linked

__all__ = ["screen_feature"]


def screen_feature(session: Session, name: str, description: str, level: str,
                   code: str) -> dict[str, Any]:
    ws, p = session.ws, session.params
    if session.finished:
        return {"ok": False, "error": "the run is finished"}
    if (missing := ideas_needed(session)):
        return missing
    if session.intents_used >= session.K:
        return {"ok": False, "error": f"all {session.K} intents are used; "
                                      "call report_findings"}
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
        return {"ok": False, "error": f"{name!r} is not a valid column name"}
    if name in ws.base_features or any(e["name"] == name for e in session.ledger):
        return {"ok": False, "error": f"{name!r} is already taken; choose a new name"}
    allowed_levels = [lv for lv in FEATURE_LEVELS if lv in p.levels]
    if level not in allowed_levels:
        return {"ok": False, "error": f"level must be one of {allowed_levels} in this run"}
    if (problem := session.level_full(level)):
        return {"ok": False, "error": problem}
    named = set(re.findall(r"sources\[\s*['\"]([^'\"]+)['\"]\s*\]", code))
    if (unknown := sorted(named - set(ws.sources()))):
        return {"ok": False, "error": (
            f"no source {unknown} - the sources are {sorted(session.params.sources)}. A data "
            "request is not data: its pull exists only once the analyst drops the result in "
            "the additional data folder, where it appears as a source. Nothing was spent.")}
    used = [s for s in ws.sources() if re.search(rf"['\"]{re.escape(s)}['\"]", code)]
    for source in used:
        if (problem := session.allowed(source)) or (problem := ensure_linked(session, source)):
            return {"ok": False, "error": problem}

    # The code guard reads the script without running it: a refusal costs nothing,
    # like every other check above. (run_code applies it again, for every mode.)
    if (problem := check_code(code)):
        return {"ok": False, "error": f"{problem}. Nothing was spent."}

    session.intents_used += 1
    intent = f"I{session.intents_used}"
    code_id, run = session.run(code, "feature", intent=intent, level=level, title=name,
                               sources={s: session.linked[s] for s in used})
    # `delta` is the Gini gain - the name the trace and the evaluation read.
    entry = {"intent": intent, "name": name, "description": description,
             "level": level, "code": code, "code_id": code_id, "sources": used,
             "delta": None, "base_score": None, "candidate_score": None,
             "capture_gain": None, "coverage": None, "verified": False, "reason": ""}

    if not run.ok:
        entry["reason"] = f"script failed: {run.error}"
    elif run.feature != name:
        entry["reason"] = f"build() returned column {run.feature!r}, expected {name!r}"
    else:
        built = pd.read_parquet(run.out_path).set_index(ws.id_col)[name]
        values = built.reindex(ws.screen[ws.id_col]).to_numpy()
        if (twin := same_feature(session, name, values)):
            session.intents_used -= 1                     # nothing new was proposed
            return {"ok": False, "error": f"{name!r} is {twin}. Nothing was spent - "
                                          "propose something that measures a different thing "
                                          "(another window, ratio or source is fine).",
                    "budget": session.budget()}
        save_values(session, name, values)
        entry["coverage"] = round(float(pd.notna(values).mean()), 4)
        screened = ws.screener.evaluate_values(name, values)
        entry.update(delta=screened.delta, base_score=screened.base_score,
                     candidate_score=screened.candidate_score,
                     capture_gain=screened.extras.get("capture_delta"))
        entry["reason"] = (screened.error or "screen failed") if not screened.ok \
            else _below_gates(entry, p)
        entry["verified"] = screened.ok and not entry["reason"]

    session.ledger.append(entry)
    session.save_ledger()
    session.emit("feature_screened", **{k: v for k, v in entry.items() if k != "code"},
                 intents_used=session.intents_used, K=session.K)
    if entry["verified"]:
        features = session.run_dir / "features"
        features.mkdir(exist_ok=True)
        (features / f"{name}.py").write_text(code)
        session.emit("feature_verified", name=name, description=description, level=level,
                     delta=entry["delta"], capture_gain=entry["capture_gain"], intent=intent)

    reply = {"intent": intent, "name": name, "verified": entry["verified"],
             "gini_gain": entry["delta"], "capture_gain": entry["capture_gain"],
             "coverage": entry["coverage"], "reason": entry["reason"],
             "gates": {"min_gini_gain": p.min_gini_gain, "min_capture_gain": p.min_capture_gain},
             "budget": session.budget()}
    if run.ok is False:
        reply["stdout"] = run.stdout
    if session.intents_used >= session.K:
        reply["next"] = "that was the last intent; call report_findings with a summary"
    return reply


def _below_gates(entry: dict[str, Any], p: Any) -> str:
    """The analyst's gates, in words, for the first one the feature misses."""
    if entry["delta"] <= p.min_gini_gain:
        return f"Gini gain {entry['delta']:+.4f} is not above {p.min_gini_gain:+.4f}"
    gain = entry["capture_gain"]
    if p.min_capture_gain is not None and (gain is None or gain <= p.min_capture_gain):
        shown = "n/a" if gain is None else f"{gain:+.4f}"
        return f"capture-rate gain {shown} is not above {p.min_capture_gain:+.4f}"
    return ""
