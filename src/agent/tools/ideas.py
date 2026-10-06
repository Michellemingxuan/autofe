"""Diverge before converging: a spread of ideas, through different lenses, first.

Left to itself, an agent briefed the same way each run reaches for the same
obvious move - the unused variable, aggregated the usual way. Two things push
it wider:

* **ideas** - the run opens with a stage of its own: the agent may look at the
  data, then answers with more ideas than it may spend on, as structured output
  (``agent.tools.Ideas``) - each named, with a level and a *lens*: a
  way of looking at behaviour (a trend, a recency, a concentration, an
  interaction ...). Code checks the spread - enough ideas, enough distinct
  lenses - and sends a short one back to fix; proposals come after.
* **focus lenses** - each run is given a few lenses to lean on, rotated across
  runs by a counter, so two runs on a similar direction start from different
  angles. At least ``MIN_FOCUS`` ideas must use them.

The earlier directions' features and requests are in the brief too
(``agent.memory``), and identical ones are refused - so the agent cannot simply
repeat itself, and is shown where it has not looked yet.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent.session import Session

__all__ = ["LENSES", "FOCUS_SIZE", "MIN_FOCUS", "MIN_LENSES", "focus_lenses", "lenses_for_turn",
           "next_turn", "record_ideas", "ideas_needed"]

# Ways of looking at a customer's behaviour - each a different kind of signal.
LENSES: dict[str, str] = {
    "level": "how much, right now",
    "trend": "the direction and speed of change over a window",
    "volatility": "how unstable or erratic it is",
    "recency": "how long since something last happened",
    "frequency": "how often something happens",
    "concentration": "how spread or concentrated - across merchants, categories, accounts",
    "own_history": "now against the customer's own past",
    "capacity": "against what the customer can bear - limit, income, balance",
    "interaction": "two signals together, where each alone says little",
    "sequence": "the order of events - what came before what",
    "absence": "what is missing or stopped - a payment, an activity",
    "calendar": "timing within the month, season, pay cycle",
    "peer": "against similar customers - segment, cohort",
    "consistency": "where two sources or two measures disagree",
}
FOCUS_SIZE = 3          # lenses a run is asked to lean on
MIN_FOCUS = 2           # ideas that must use them
MIN_LENSES = 4          # distinct lenses across the ideas


def _counter(session: Session) -> Path:
    return Path(session.ws.cfg.agent.run_dir) / "ideas" / "rotation.json"


def lenses_for_turn(turn: int) -> list[str]:
    """The focus lenses of the run that takes rotation turn `turn`."""
    names = list(LENSES)
    return [names[(turn * FOCUS_SIZE + i) % len(names)] for i in range(FOCUS_SIZE)]


def next_turn(run_dir: str | Path) -> int:
    """The rotation turn the next run will take - read, not advanced."""
    path = Path(run_dir) / "ideas" / "rotation.json"
    return json.loads(path.read_text())["next"] if path.exists() else 0


def focus_lenses(session: Session) -> list[str]:
    """This run's focus lenses: the next ones in the rotation, taken once per run."""
    if getattr(session, "focus", None):
        return session.focus
    path = _counter(session)
    turn = next_turn(session.ws.cfg.agent.run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"next": turn + 1}))
    session.focus = lenses_for_turn(turn)
    return session.focus


def _wanted(session: Session) -> int:
    return session.K + 2


def ideas_needed(session: Session) -> dict[str, Any] | None:
    """The refusal a proposing tool returns while there are no ideas yet."""
    if not getattr(session, "ideas_required", False) or session.ideas:
        return None
    return {"ok": False, "error": "no ideas recorded yet: the ideas come first. "
            "Nothing was spent."}


def _cas_variables(session: Session) -> list[str]:
    """The CAS variables an L3 idea may name - not the keys or the partition date."""
    scope = session.ws.scope()
    if not len(scope):
        return []
    keys = {c for t in set(scope["table"].astype(str))
            for k in ("partition", "identifiers") for c in session.ws.table_profile(t)[k]}
    return [v for v in scope["variable"].astype(str) if v not in keys and not v.endswith("pkey")]


def _names_cas(text: str, variables: list[str]) -> list[str]:
    return [v for v in variables if re.search(rf"\b{re.escape(v)}\b", text, flags=re.I)]


def record_ideas(session: Session, ideas: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the ideas - many, varied, each fitting its level - and record
    them. The error, when there is one, goes back to the model to fix."""
    levels = list(session.params.levels)
    mixed = len(session.quota) > 1
    cas = _cas_variables(session)
    clean, names = [], set()
    for i, idea in enumerate(ideas or [], 1):
        name = str(idea.get("name", "")).strip()
        lens = str(idea.get("lens", "")).strip().lower()
        description = str(idea.get("description") or idea.get("hypothesis") or "").strip()
        data = str(idea.get("data", "")).strip()
        level = str(idea.get("level", "")).strip().upper() or (levels[0] if len(levels) == 1 else "")
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,63}", name) or name in names:
            return {"ok": False, "error": f"idea {i}: give it a new snake_case name, got {name!r}"}
        if lens not in LENSES:
            return {"ok": False, "error": f"idea {i}: lens {lens!r} is not one of {list(LENSES)}"}
        if not description:
            return {"ok": False, "error": f"idea {i}: say what it measures and why it should carry risk"}
        if level not in levels:
            return {"ok": False, "error": f"idea {i}: its level must be one of {levels}"}
        named = _names_cas(f"{description} {data}", cas)
        if level == "L3" and not named:
            return {"ok": False, "error": (
                f"idea {i} ({name}): names no CAS variable - built only from the model database "
                "and the linked sources, this is an L1/L2 idea, not a data request. Rework it "
                "around CAS variables, written in its data exactly as named - e.g. "
                + ", ".join(f"`{v}`" for v in cas[:6]) + " (the brief lists them all) - or "
                "replace it.")}
        names.add(name)
        clean.append({"n": i, "name": name, "lens": lens, "level": level,
                      "description": description, "data": data, **({"cas": named} if named else {})})
    focus = focus_lenses(session)
    lenses = {c["lens"] for c in clean}
    problems = []
    if len(clean) < _wanted(session):
        problems.append(f"{len(clean)} ideas - give at least {_wanted(session)}")
    if len(lenses) < MIN_LENSES:
        problems.append(f"{len(lenses)} lenses ({sorted(lenses)}) - use at least {MIN_LENSES}")
    if sum(c["lens"] in focus for c in clean) < MIN_FOCUS:
        problems.append(f"fewer than {MIN_FOCUS} ideas use this run's focus {focus}")
    if mixed:
        bare = [lv for lv, n in session.quota.items() if n and not any(c["level"] == lv for c in clean)]
        if bare:
            problems.append(f"no ideas for {bare} - this run's intents include them "
                            f"({session.shares_left()})")
    if problems:
        return {"ok": False, "error": "not yet a spread: " + "; ".join(problems)}
    session.ideas = clean
    session.emit("ideas_recorded", ideas=clean, focus=focus)
    return {"ok": True, "ideas": len(clean), "lenses": sorted(lenses)}
