"""The logging board: every idea, the features and requests made from it, and
how each one ended - written from the run's own records, never from the agent's
account of them.

Two boards read the same records. This one is the proposal board: an idea, its
engine and its standing; each feature with its code, what the cheap checks
measured (agent.checks), its screen gain and its outcome. The evaluated board is
Results (agent.evaluate).

    Outcome      a feature: verified or failed (the reason says where)
                 a request: kept or dropped (dropped by its challenge, or by the analyst)
    Evaluation   a verified feature only: not evaluated, evaluated (with the
                 verdict), evaluated · removed (taken off by the analyst), removed

An idea's standing comes from its features (and, at L3, its requests):

* a feature **carries signal** when it was verified - the screen's fit cleared
  the gates; a request when the challenge kept it;
* it **carries none** when it was measured and failed - explained by the
  model (C1), below the gates, a screen guard, or a request the current data
  covers;
* a failed script or a wrong shape says nothing about the idea, and counts
  neither way.

From those, each idea keeps a Beta(1 + signal, 1 + none): ``p_useful`` is its
mean, and the status is ``explore`` (nothing measured yet), ``active`` (some
signal) or ``low`` (measured, no signal yet). Nothing is refused: the order the
ideas are worked in is a Thompson draw from each Beta (:func:`order_ideas`) -
over every round's ideas of the run, not only the latest round's - so
an idea whose features failed is drawn later and later - deprioritised, never
shut out, and back in front the moment one of its features carries signal. An
idea linked to an earlier run's (``same_as``) starts from that idea's record.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.session import Session
    from agent.workspace import Workspace

__all__ = ["idea_of", "idea_for", "feature_evidence", "idea_standing", "run_ideas", "idea_records",
           "order_ideas", "round_board", "board_rows", "prompt_board"]

_NO_EVIDENCE = (r"^script failed", r"^build\(\) returned")


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


def idea_for(session: "Session", idea: str, name: str) -> tuple[str, str | None]:
    """The idea a proposal tests - the one it names (``idea=``), else the idea its
    name starts with (older runs) - and an error when a run with ideas cannot tell."""
    names = [i["name"] for i in session.ideas]
    idea = str(idea or "").strip()
    if idea:
        if idea in names:
            return idea, None
        return "", (f"no idea named {idea!r}; this run's ideas are "
                    f"{', '.join(names[-10:]) or 'none yet'}. Nothing was spent.")
    guess = idea_of(name, session.ideas)
    if guess in names or not names:
        return (guess if guess in names else ""), None
    return "", ("say which idea this tests: idea=<one of " + ", ".join(names[-10:])
                + ">. Nothing was spent.")


def feature_evidence(entry: dict[str, Any]) -> str | None:
    """What one feature or request says about its idea: 'signal', 'none', or None."""
    if "source_name" in entry and "verified" not in entry:          # a data request
        status = entry.get("status") or ("approved" if entry.get("approved") else
                                          "rejected" if entry.get("approved") is False else "")
        return "signal" if status in ("kept", "approved") else \
            "none" if status in ("dropped", "rejected") else None
    if entry.get("verified"):
        return "signal"
    reason = str(entry.get("reason") or "")
    if not reason or any(re.search(p, reason) for p in _NO_EVIDENCE):
        return None
    return "none"


def idea_standing(evidence: list[str | None],
                  inherited: dict[str, int] | None = None) -> dict[str, Any]:
    """An idea's Beta and status from its features' evidence - and the record of the
    earlier idea it is linked to, if any."""
    signal = evidence.count("signal") + int((inherited or {}).get("signal", 0))
    none = evidence.count("none") + int((inherited or {}).get("none", 0))
    status = "active" if signal else "low" if none else "explore"
    return {"status": status, "signal": signal, "none": none,
            "alpha": 1 + signal, "beta": 1 + none,
            "p_useful": round((1 + signal) / (2 + signal + none), 3)}


def run_ideas(folder: Path) -> list[dict[str, Any]]:
    """The ideas a run recorded: its ideas.json, else (older runs) its events."""
    path = folder / "ideas.json"
    if path.exists():
        return json.loads(path.read_text())
    events = folder / "events.jsonl"
    out = []
    if events.exists():
        for line in open(events):
            if '"ideas_recorded"' not in line:
                continue
            event = json.loads(line)
            out += [{**i, "round": event.get("round"), "engine": i.get("engine")}
                    for i in event.get("ideas") or []]
    return out


def _entries(folder: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ledger, requests = folder / "ledger.json", folder / "data_requests.json"
    return (json.loads(ledger.read_text()) if ledger.exists() else [],
            json.loads(requests.read_text()) if requests.exists() else [])


def _of(entry: dict[str, Any], ideas: list[dict[str, Any]]) -> str:
    return entry.get("idea") or idea_of(entry.get("name") or entry.get("source_name") or "", ideas)


def _standings(ideas: list[dict[str, Any]], ledger: list[dict[str, Any]],
               requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ideas of one run, each with its standing, its features and requests."""
    out = []
    made = [e for e in [*ledger, *requests] if not e.get("deleted")]
    for idea in ideas:
        mine = [e for e in made if _of(e, ideas) == idea["name"]]
        evidence = [feature_evidence(e) for e in mine]
        standing = idea_standing(evidence, idea.get("inherited"))
        last = next((str(e.get("reason") or "") for e, ev in zip(reversed(mine), reversed(evidence))
                     if ev == "none"), "")
        out.append({**idea, **standing, "made": [e.get("name") or e.get("source_name") for e in mine],
                    "why": last.splitlines()[0][:160] if last else ""})
    return out


def session_ideas(session: "Session") -> list[dict[str, Any]]:
    """This run's ideas with their standing - from the session, as it is now."""
    return _standings(session.ideas, session.ledger, session.data_requests)


def idea_records(ws: "Workspace", skip_run: str | None = None) -> list[dict[str, Any]]:
    """Every run's ideas with their standing, oldest run first."""
    from agent.memory import _direction, _runs

    out = []
    for folder in _runs(ws):
        if folder.name == skip_run:
            continue
        ideas = run_ideas(folder)
        if not ideas:
            continue
        ledger, requests = _entries(folder)
        direction = _direction(folder)
        out += [{**i, "run_id": folder.name, "direction": direction}
                for i in _standings(ideas, ledger, requests)]
    return out


def order_ideas(standings: list[dict[str, Any]], seed: str) -> list[str]:
    """Thompson sampling over a round's ideas: one draw from each idea's Beta, highest
    draw first - which to work next. An idea whose features failed draws low, so it
    comes later; none is left out."""
    rng = random.Random(seed)
    draws = {i["name"]: rng.betavariate(i["alpha"], i["beta"]) for i in standings}
    return sorted(draws, key=lambda n: -draws[n])


def round_board(session: "Session") -> list[dict[str, Any]]:
    """This run's ideas, every round's, as the screen reply shows them: standing and
    order. An earlier round's idea that carried signal is still worth working."""
    return [{k: i.get(k) for k in ("name", "round", "engine", "status", "p_useful", "made")}
            for i in session_ideas(session)]


# --------------------------------------------------------------- the board
def _evaluations(ws: "Workspace") -> dict[str, dict[str, Any]]:
    """Per feature key, how its evaluation went: the verdict of the newest finished
    evaluation that measured it, and whether the analyst took it off the results.
    One the analyst kept anywhere counts as kept."""
    from agent.evaluate import list_evaluations, read_events, removed_variants

    out: dict[str, dict[str, Any]] = {}
    try:
        for summary in list_evaluations(ws):                     # newest first
            if summary["status"] != "done":
                continue
            events = read_events(Path(ws.cfg.agent.run_dir) / "evaluations" / summary["eval_id"]
                                 / "events.jsonl")
            done = next(e for e in reversed(events) if e["event"] == "eval_done")
            removed = removed_variants(events)
            verdicts = {v["feature"]: v for v in done.get("verdicts") or []}
            by_column = {f["column"]: f for f in summary["features"]}
            for row in done.get("comparison") or []:
                variant = row["variant"]
                if not variant.startswith("loi__"):
                    continue
                column = variant.split("__", 1)[1]
                key = by_column.get(column, {}).get("key")
                if not key or (key in out and not out[key]["removed"]):
                    continue
                out[key] = {"verdict": verdicts.get(column, {}).get("verdict") or "",
                            "removed": variant in removed}
    except Exception:  # noqa: BLE001 - a broken evaluation must not hide the board
        return out
    return out


def _evaluation(entry: dict[str, Any], key: str, evaluations: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """A feature's Evaluation column - and its verdict. Only a verified feature can be
    evaluated; a failed one has none."""
    if not entry.get("verified"):
        return "", ""
    done = evaluations.get(key)
    if done:
        removed = done["removed"] or entry.get("deleted")
        return ("evaluated · removed" if removed else "evaluated"), done["verdict"]
    return ("removed" if entry.get("deleted") else "not evaluated"), ""


def _request_outcome(r: dict[str, Any], finished: bool) -> tuple[str, str]:
    """A request's outcome - kept or dropped - and why a dropped one was dropped."""
    status = r.get("status") or ("approved" if r.get("approved") else
                                 "rejected" if r.get("approved") is False else "proposed")
    if r.get("deleted"):
        return "dropped", "removed by the analyst"
    if status == "dropped":
        reasoning = (r.get("challenge") or {}).get("reasoning") or ""
        return "dropped", "challenge: the current data already supplies it" + (
            f" - {reasoning}" if reasoning else "")
    if status == "rejected":
        return "dropped", "rejected by the analyst" + (f": {r['note']}" if r.get("note") else "")
    if status in ("kept", "approved"):
        return "kept", r.get("note") or (r.get("challenge") or {}).get("reasoning") or ""
    return ("dropped", "never challenged before the run ended") if finished \
        else ("pending", "awaiting its challenge")


def board_rows(ws: "Workspace") -> list[dict[str, Any]]:
    """The proposal board, one row per feature or request - and one per idea that
    made nothing - newest run first."""
    from agent.memory import _direction, _runs

    evaluations = _evaluations(ws)
    rows = []
    for folder in reversed(_runs(ws)):
        ideas = run_ideas(folder)
        ledger, requests = _entries(folder)
        if not ideas and not ledger and not requests:
            continue
        direction = _direction(folder)
        standings = {i["name"]: i for i in _standings(ideas, ledger, requests)}
        used = set()

        def idea_cols(name: str) -> dict[str, Any]:
            i = standings.get(name, {})
            return {"idea": name if i else "", "round": i.get("round"), "engine": i.get("engine") or "",
                "idea_measures": i.get("measures") or [],
                    "lens": i.get("lens") or "", "idea_status": i.get("status") or "",
                    "p_useful": i.get("p_useful"), "idea_description": i.get("description") or ""}

        for e in ledger:
            name = _of(e, ideas)
            used.add(name)
            checks = e.get("checks") or {}
            sent, verdict = _evaluation(e, f"{folder.name}:{e['name']}", evaluations)
            rows.append({"run_id": folder.name, "direction": direction, **idea_cols(name),
                         "kind": "feature", "name": e["name"], "level": e.get("level"),
                         "description": e.get("description") or "", "differs": e.get("differs") or "",
                         "code": e.get("code") or "", "attempt": e.get("intent"),
                         "seen_before": checks.get("seen_before") or "",
                         # C1: the largest |rho| with a model column - older runs kept it as
                         # the nearest column's rho.
                         "max_model_rho": checks.get("max_model_rho", abs(checks["nearest_rho"])
                                                     if checks.get("nearest_rho") is not None else None),
                         "nearest": checks.get("nearest") or "", "nearest_rho": checks.get("nearest_rho"),
                         "residual_rho": checks.get("residual_rho"), "note": checks.get("note") or "",
                         "gini_gain": e.get("delta"), "capture_gain": e.get("capture_gain"),
                         "status": ("verified" if e.get("verified") else "failed"),
                         "reason": e.get("reason") or "", "evidence": feature_evidence(e),
                         "sent": sent, "verdict": verdict})
        finished = _finished(folder)
        for r in requests:
            name = _of(r, ideas)
            used.add(name)
            outcome, why = _request_outcome(r, finished)
            rows.append({"run_id": folder.name, "direction": direction, **idea_cols(name),
                         "kind": "request", "name": r.get("source_name") or r.get("intent"),
                         "level": "L3", "description": r.get("gap") or "", "differs": "",
                         "code": r.get("sql") or r.get("data") or "", "attempt": r.get("intent"),
                         "seen_before": "", "max_model_rho": None, "nearest": "", "nearest_rho": None,
                         "residual_rho": None, "note": "",
                         "gini_gain": None, "capture_gain": None, "status": outcome, "reason": why,
                         "evidence": feature_evidence(r), "sent": "", "verdict": ""})
        for name, i in standings.items():
            if name not in used:
                rows.append({"run_id": folder.name, "direction": direction, **idea_cols(name),
                             "kind": "idea", "name": "", "level": i.get("level"), "description": "",
                             "differs": "", "code": "", "attempt": None, "seen_before": "",
                             "max_model_rho": None, "nearest": "", "nearest_rho": None,
                             "residual_rho": None, "note": "",
                             "gini_gain": None, "capture_gain": None, "status": "not tried",
                             "reason": "", "evidence": None, "sent": "", "verdict": ""})
    return rows


def _finished(folder: Path) -> bool:
    events = folder / "events.jsonl"
    return events.exists() and '"run_done"' in events.read_text()


def prompt_board(ws: "Workspace", skip_run: str | None = None, limit: int = 30) -> list[str]:
    """Earlier runs' ideas for the brief: engine, standing, and what their features
    showed - the low ones with why, so a new idea changes what made them fail."""
    ideas = [i for i in idea_records(ws, skip_run) if i["status"] != "explore"][-limit:]
    mark = {"active": "✓", "low": "·"}
    lines = []
    for i in ideas:
        engine = i.get("engine") or "?"
        lines.append(f"* {mark.get(i['status'], '·')} `{i['name']}` [{engine}, {i['status']}, "
                     f"P(useful) {i['p_useful']:.2f}] - {str(i.get('description') or '')[:140]}"
                     + (f" | made: {', '.join(f'`{m}`' for m in i['made'][:4])}" if i["made"] else "")
                     + (f" | why not: {i['why']}" if i["status"] == "low" and i["why"] else ""))
    return lines
