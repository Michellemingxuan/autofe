"""Build one feature on the screen rows and score it against base. One attempt.

The feature's script runs out of process (``agent.execution``); its column is
then handed to the discovery screen (``discovery.screen.Screener``), whose
guards (``discovery.guards``: finite values, no spikes, not redundant with a
base column) run before the fit. The analyst's gates decide what is verified:
the Gini gain must clear ``min_gini_gain`` and, when set, the capture-rate gain
must clear ``min_capture_gain``.

A feature identical in values to one already screened - in this run or an
earlier direction (``agent.memory``) - is refused before it is scored, and the
attempt is given back.
"""

from __future__ import annotations

import re
from collections import Counter
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
    if session.failed_streak >= FAILED_STREAK:
        return {"ok": False, "error": (
            f"{session.failed_streak} scripts in a row failed - look before the next attempt: "
            "run_probe to print the columns, dtypes and a few rows of the frames you use, "
            "then screen again. Nothing was spent.")}
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
        return {"ok": False, "error": f"{name!r} is not a valid column name"}
    idea = idea_of(name, session.ideas)
    failed = [e for e in session.ledger if not e.get("verified") and idea_of(e["name"], session.ideas) == idea]
    if len(failed) >= IDEA_FAILURES:
        return {"ok": False, "error": (
            f"the idea {idea!r} has failed {len(failed)} times "
            f"({'; '.join(str(e.get('reason') or '').splitlines()[0][:80] for e in failed)}). "
            "Move to a different idea - more fixes to the same one tend to repeat the "
            "mistake. Nothing was spent.")}
    if name in ws.base_features or any(e["name"] == name for e in session.ledger):
        return {"ok": False, "error": f"{name!r} is already taken; choose a new name"}
    allowed_levels = [lv for lv in FEATURE_LEVELS if lv in p.levels]
    if level not in allowed_levels:
        return {"ok": False, "error": f"level must be one of {allowed_levels} in this run"}
    if (problem := session.target_met(level)):
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

    session.take_attempt()
    intent = f"I{session.attempts}"
    code_id, run = session.run(code, "feature", intent=intent, level=level, title=name,
                               sources={s: session.linked[s] for s in used})
    # `delta` is the Gini gain - the name the trace and the evaluation read.
    entry = {"intent": intent, "name": name, "description": description,
             "level": level, "code": code, "code_id": code_id, "sources": used,
             "delta": None, "base_score": None, "candidate_score": None,
             "capture_gain": None, "coverage": None, "verified": False, "reason": ""}

    if not run.ok:
        entry["reason"] = f"script failed: {run.short_error}"
    elif run.feature != name:
        entry["reason"] = f"build() returned column {run.feature!r}, expected {name!r}"
    else:
        built = pd.read_parquet(run.out_path).set_index(ws.id_col)[name]
        values = built.reindex(ws.screen[ws.id_col]).to_numpy()
        if (twin := same_feature(session, name, values)):
            session.give_back_attempt()                   # nothing new was proposed
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
                 attempts=session.attempts, results=session.results(), K=session.K)
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
        # The usual cause is a column the frame does not have: show what it had.
        reply["frames"] = _frames(session, used)
    if run.elapsed_s and run.elapsed_s > SLOW_SCRIPT_S:
        reply["slow"] = (f"the script took {run.elapsed_s:.0f}s on {len(ws.screen):,} screen rows - "
                         "it loops over rows or ids. Vectorise: filter, then groupby().agg(); "
                         "no apply, iterrows or Python loops.")
    # One failure tends to breed the next - patching the same code on a wrong
    # belief. Two in a row: the next screen waits for a look at the data.
    session.failed_streak = 0 if run.ok else session.failed_streak + 1
    if session.failed_streak >= FAILED_STREAK:
        reply["next"] = (f"{session.failed_streak} scripts in a row failed. Stop patching: "
                         "run_probe to print the columns and dtypes of the frames you use "
                         "(see `frames`) - the next screen waits for it. Then fix the script, "
                         "or move to a different idea.")
    if (learned := lessons(session.ledger)):
        reply["lessons"] = learned
    if session.wanted() == 0:
        reply["next"] = f"the target of {session.K} is reached; call report_findings with a summary"
    elif session.attempts >= session.max_attempts:
        reply["next"] = "that was the last attempt; call report_findings with a summary"
    elif (over := session.round_over()):
        reply["next"] = over
    return reply


# Failed scripts in a row before the next screen waits for a probe.
FAILED_STREAK = 2
# Failed attempts one idea may have; the next is refused - move on.
IDEA_FAILURES = 2
# A script on the screen rows slower than this is told to vectorise.
SLOW_SCRIPT_S = 30


def idea_of(name: str, ideas: list[dict[str, Any]]) -> str:
    """The idea a feature belongs to: the idea it is named after, else its name
    without a variant's suffix - `x_v2`, `x_final`, `x_fixed` are all `x`."""
    named = [i["name"] for i in ideas if name == i["name"] or name.startswith(i["name"] + "_")]
    if named:
        return max(named, key=len)
    stem = name
    while (shorter := re.sub(r"_(v\d+|final\d*|fix(ed)?\d*|retry\d*|alt\d*|new\d*|\d+)$", "", stem)) != stem:
        stem = shorter
    return stem


def lessons(ledger: list[dict[str, Any]]) -> list[str]:
    """What the run's failures have in common, in words - a pattern repeated
    across attempts is easy to miss one reply at a time. Only patterns in what
    was measured: a failed script is fixed once (the probe gate sees to it), and
    repeating it as a lesson would keep a solved problem in view."""
    redundant = Counter(m.group(1) for e in ledger
                        if (m := re.search(r"against the existing column '([^']+)'", e.get("reason") or "")))
    below = sum(1 for e in ledger if "is not above" in str(e.get("reason", "")))
    sparse = sum(1 for e in ledger if e.get("coverage") is not None and e["coverage"] < 0.05)
    out = [f"{n} features were redundant with `{col}` - a ratio, difference or rescaling of it "
           "re-derives it; build on other columns" for col, n in redundant.items() if n >= 2]
    if below >= 3:
        out.append(f"{below} features did not beat base - change the signal, not the window")
    if sparse >= 2:
        out.append(f"{sparse} features were almost always missing - the event they need is rare; "
                   "measure something most customers have")
    return out


def _frames(session: Session, used: list[str]) -> dict[str, list[str]]:
    """The columns a feature script was given: `base`, and each source it read."""
    import pyarrow.dataset as ds

    ws = session.ws
    out = {"base": [ws.id_col, *ws.base_features[:60]]
                   + ([f"... {len(ws.base_features) - 60} more"] if len(ws.base_features) > 60 else [])}
    for source in used:
        path = session.linked.get(source)
        if path:
            out[f"sources[{source!r}]"] = ds.dataset(path).schema.names
    return out


def _below_gates(entry: dict[str, Any], p: Any) -> str:
    """The analyst's gates, in words, for the first one the feature misses."""
    if entry["delta"] <= p.min_gini_gain:
        return f"Gini gain {entry['delta']:+.4f} is not above {p.min_gini_gain:+.4f}"
    gain = entry["capture_gain"]
    if p.min_capture_gain is not None and (gain is None or gain <= p.min_capture_gain):
        shown = "n/a" if gain is None else f"{gain:+.4f}"
        return f"capture-rate gain {shown} is not above {p.min_capture_gain:+.4f}"
    return ""
