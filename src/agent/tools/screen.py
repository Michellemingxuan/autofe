"""Build one feature on the screen rows, check it, and score it against base. One attempt.

The feature's script runs out of process (``agent.execution``); its column is
then handed to the discovery screen (``discovery.screen.Screener``), whose
guards (``discovery.guards``: finite values, no spikes, not redundant with a
base column) run before the fit. The analyst's gates decide what is verified:
the Gini gain must clear ``min_gini_gain`` and, when set, the capture-rate gain
must clear ``min_capture_gain``.

Before the fit, the cheap checks (``agent.checks``), each in milliseconds:

* **C0 - seen before?** Before the script runs: the same code as an earlier
  feature (names and comments aside) is refused, and one that means much the
  same must say what differs. After it runs: identical values to an earlier
  feature (``agent.memory``) are refused. A refusal gives the attempt back.
* **C1 - redundancy with the model?** On the built values: a feature whose
  |Spearman rho| with a model column passes the limit fails without the fit. **C2 - new signal?** is measured too
  (its residual correlation) and recorded - information only, unless the config
  makes it a gate (``agent.checks.min_residual_rho``).

Every feature records what the checks measured (``checks``) and its idea, so
the board (``agent.board``) shows it. An idea whose features carry no signal is
not refused: each reply gives the round's ideas in a fresh Thompson-sampled
order (``work_next``), where it now draws low and comes later.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any

import pandas as pd

from agent.board import idea_for, idea_of, order_ideas, round_board, session_ideas  # noqa: F401 - idea_of, for callers
from agent.checks import code_fingerprint, model_checks, nearest_meaning
from agent.execution import check_code
from agent.memory import earlier_features, save_values, same_feature
from agent.model_history import history_of
from agent.session import FEATURE_LEVELS, Session
from agent.tools.ideas import ideas_needed
from agent.tools.linkage import ensure_linked

__all__ = ["screen_feature"]


def screen_feature(session: Session, name: str, description: str, level: str,
                   code: str, differs: str = "", idea: str = "") -> dict[str, Any]:
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
    idea, problem = idea_for(session, idea, name)
    if problem:
        return {"ok": False, "error": problem}
    # A name is taken by a base column or a feature that was built. A script that
    # failed built nothing: its name is free for the fixed script.
    if name in ws.base_features or any(e["name"] == name and _built(e) for e in session.ledger):
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

    # C0 - seen before? The code and the meaning, before anything runs.
    seen, seen_note = _seen_before(session, name, description, code, differs)
    if seen:
        return {"ok": False, "error": seen + " Nothing was spent."}

    session.take_attempt()
    intent = f"I{session.attempts}"
    code_id, run = session.run(code, "feature", intent=intent, level=level, title=name,
                               sources={s: session.linked[s] for s in used})
    # `delta` is the Gini gain - the name the trace and the evaluation read.
    record = next((i for i in session.ideas if i["name"] == idea), {})
    entry = {"intent": intent, "name": name, "description": description,
             "idea": idea, "engine": record.get("engine", ""),
             "differs": differs.strip(), "checks": {"passed": False, "seen_before": seen_note},
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
        # C1 and C2 - on the built values, before the fit.
        measured, failed = model_checks(ws.screener, values, ws.cfg.agent.checks)
        entry["checks"].update(measured)
        if failed:
            entry["reason"] = failed
        else:
            entry["checks"]["passed"] = True
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
             "checks": {k: v for k, v in entry["checks"].items() if v is not None},
             "gates": {"min_gini_gain": p.min_gini_gain, "min_capture_gain": p.min_capture_gain},
             "budget": session.budget()}
    if run.ok is False:
        reply["stdout"] = run.stdout
        # The usual cause is a column the frame does not have: show what it had.
        reply["frames"] = _frames(session, used)
    if run.elapsed_s and run.elapsed_s > ws.cfg.agent.timeouts.slow_script_s:
        reply["slow"] = (f"the script took {run.elapsed_s:.0f}s on {len(ws.screen):,} screen rows - "
                         "it loops over rows or ids. Vectorise: filter, then groupby().agg(); "
                         "no apply, iterrows or Python loops.")
    if run.ok and (note := scale_note(session, run)):
        reply["scale"] = note
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
    if session.round_ideas:
        # The board, as this round stands: each idea's standing after this feature.
        reply["ideas"] = round_board(session)
        # Which to work next: a fresh Thompson draw over the round's ideas - one whose
        # features failed draws low and comes later, never shut out.
        mine = {i["name"] for i in session.round_ideas}
        reply["work_next"] = order_ideas([i for i in session_ideas(session) if i["name"] in mine],
                                         seed=f"{session.run_id}:{intent}")
    if session.wanted() == 0:
        reply["next"] = f"the target of {session.K} is reached; call report_findings with a summary"
    elif session.attempts >= session.max_attempts:
        reply["next"] = "that was the last attempt; call report_findings with a summary"
    elif (over := session.round_over()):
        reply["next"] = over
    return reply


# Failed scripts in a row before the next screen waits for a probe.
FAILED_STREAK = 2
# A projection above this share of the evaluation time limit, or of the
# machine's memory, is sent back to the agent.
SCALE_WARN_SHARE = 0.5


def _machine_mb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**20
    except (ValueError, OSError, AttributeError):
        return None


def scale_note(session: Session, run: Any) -> str | None:
    """The screen is a sample; evaluation runs the same script on the full splits.
    Scaled by the row ratio - roughly, as joins and sorts grow faster than rows -
    a script that will not finish there, or not fit, is told so now."""
    ws = session.ws
    full = ws.data_size()["full_rows"]
    if not full or full <= len(ws.screen):
        return None
    ratio = full / len(ws.screen)
    limit_s = ws.cfg.agent.timeouts.eval_s
    problems = []
    projected_s = max((run.elapsed_s or 0) - (run.setup_s or 0), 0) * ratio
    if projected_s > SCALE_WARN_SHARE * limit_s:
        problems.append(f"about {projected_s / 60:.0f} min (the limit is {limit_s / 60:.0f})")
    machine = _machine_mb()
    # On spark the work is in the JVM, not this process: only the time says anything.
    if session.params.engine != "spark" and run.peak_mb and machine:
        projected_mb = run.peak_mb * ratio
        if projected_mb > SCALE_WARN_SHARE * machine:
            problems.append(f"about {projected_mb / 1024:.0f} GB of memory "
                            f"(the machine has {machine / 1024:.0f} GB)")
    if not problems:
        return None
    return (f"on the full data ({full:,} rows, {ratio:.0f}x the screen) this script would "
            f"take {' and '.join(problems)}, and fail at evaluation. Make the next one "
            "leaner: name only the columns you need, filter to the window before any "
            "join or groupby, aggregate to one row per id early, never merge two "
            "event-level frames.")


def _history_twin(name: str, description: str, past: list[dict[str, Any]],
                  session: Session) -> dict[str, Any] | None:
    """The model-history variable a feature reads like - the same name, or close in
    words - or None."""
    if not past:
        return None
    same = next((v for v in past if v["name"].lower() == name.lower()), None)
    if same:
        return same
    limit = session.ws.cfg.agent.checks.same_meaning
    near, overlap = nearest_meaning(f"{name} {description}", past)
    return near if near is not None and limit is not None and overlap >= limit else None


def _built(entry: dict[str, Any]) -> bool:
    """Whether a ledger entry built a feature - a failed script or a wrong shape did not."""
    return not re.match(r"script failed|build\(\) returned", str(entry.get("reason") or ""))


def _seen_before(session: Session, name: str, description: str, code: str,
                 differs: str) -> tuple[str | None, str]:
    """C0 before the script runs, against earlier features and the model history:
    the refusal (the same code; or the same meaning with no word on what differs) -
    or None - and, either way, what it found, for the board."""
    # A script that failed made no feature: there is nothing to have seen.
    earlier = [e for e in earlier_features(session)
               if not re.match(r"script failed|build\(\) returned", str(e.get("reason") or ""))]
    past = [{"name": v["variable"], "description": v["description"], **v}
            for v in history_of(session.ws)["variables"]]
    found = []
    if (twin := _history_twin(name, description, past, session)):
        status = twin.get("status") or "used"
        why = f" ({twin['why']})" if twin.get("why") else ""
        if not differs.strip():
            return (f"it reads like `{twin['name']}` of the model's {twin.get('generation') or 'history'} "
                    f"- {status}{why}. Say in `differs` what this one measures or fixes that it did "
                    "not, or propose something else."), ""
        found.append(f"like `{twin['name']}` of the model's {twin.get('generation') or 'history'} "
                     f"({status})")
    mine = code_fingerprint(code, name)
    if mine:
        same = next((e for e in earlier if code_fingerprint(e.get("code") or "", e["name"]) == mine), None)
        if same:
            where = "this run" if same["run_id"] == session.run_id else f"run {same['run_id']}"
            return (f"the same code as `{same['name']}` from {where} - only the names differ. "
                    "Propose something that computes another thing."), ""
    limit = session.ws.cfg.agent.checks.same_meaning
    near, overlap = nearest_meaning(f"{name} {description}", earlier) if earlier else (None, 0.0)
    if near is not None and limit is not None and overlap >= limit:
        where = "this run" if near["run_id"] == session.run_id else f"run {near['run_id']}"
        if not differs.strip():
            return (f"it reads much like `{near['name']}` from {where} ({near.get('description', '')[:120]!r}, "
                    f"overlap {overlap:.2f}). Say in `differs` what this one measures that it does "
                    "not - another window, ratio, source or population - or propose something else."), ""
        found.append(f"like `{near['name']}` ({where}, overlap {overlap:.2f})")
    if found:
        return None, "; ".join(found) + " - differs, as it says"
    return None, (f"new - closest `{near['name']}`, overlap {overlap:.2f}" if near is not None
                  else "new - nothing earlier")


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
