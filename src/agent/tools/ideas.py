"""Diverge before converging: a spread of ideas, from different engines, first.

Left to itself, an agent briefed the same way each run reaches for the same
obvious move - the unused variable, aggregated the usual way. Three things push
it wider:

* **ideas** - the run opens with a stage of its own: the agent may look at the
  data, then answers with more ideas than it may spend on, as structured output
  (``agent.tools.Ideas``) - each named, with a level, the *engine* that made it
  (``agent.engines``: mechanism, contrastive, analogy, morphological,
  others) and a *lens*, the kind of signal (a trend, a recency, a
  concentration ...). Code checks the spread - enough ideas, enough distinct
  engines and lenses - and sends a short one back to fix; proposals come after.
* **focus engines** - each round leans on a few engines, drawn by Thompson
  sampling over every engine's yield across runs (``agent.engines.draw_focus``):
  at least ``MIN_FOCUS`` ideas must come from them.
* **the board** - an idea that says what an earlier one said is checked on the
  way in (``agent.checks.similarity``): the same as another idea of this answer
  is sent back; the same as an idea of an earlier run is kept, linked to it
  (``same_as``), and starts from its record - so one whose features carried no
  signal is drawn late (Thompson sampling), not refused.

The earlier directions' ideas and features are in the brief (``agent.board``,
``agent.memory``), and identical features are refused - so the agent cannot
simply repeat itself, and is shown where it has not looked yet.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.board import idea_records
from agent.checks import similarity
from agent.engines import ENGINES, MIN_ENGINES, draw_focus
from agent.session import Session

__all__ = ["LENSES", "MIN_FOCUS", "MIN_LENSES", "MIN_ENGINES", "focus_engines",
           "record_ideas", "ideas_needed", "ideas_in_round", "save_ideas"]

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
MIN_FOCUS = 2           # ideas that must come from the round's focus engines
MIN_MEASURES, MAX_MEASURES = 2, 5   # ways an idea can be measured: it is not one feature
# An idea's name names a concept; a window or a version in it is a feature's name.
FEATURE_LIKE = re.compile(r"_(\d+(d|w|m|mo|y|yr)|v\d+|\d+)(_|$)")
MIN_LENSES = 4          # distinct lenses across the ideas


def focus_engines(session: Session) -> list[str]:
    """The upcoming round's focus engines: drawn once per round, by Thompson
    sampling over every engine's yield on the board, and recorded."""
    upcoming = session.round + 1
    drawn = getattr(session, "focus_drawn", None)
    if drawn is None:
        drawn = session.focus_drawn = {}
    if upcoming not in drawn:
        draw = draw_focus(idea_records(session.ws), seed=f"{session.run_id}:{upcoming}")
        drawn[upcoming] = draw["focus"]
        session.emit("engines_drawn", round=upcoming, focus=draw["focus"], draws=draw["draws"],
                     engine_yield=draw["yield"])
    session.focus = drawn[upcoming]
    return session.focus


def ideas_in_round(session: Session) -> int:
    """How many ideas this round asks for: a few more than the results still wanted,
    capped by the run's ideas per round - a large K is reached in several rounds."""
    return max(3, min(session.params.ideas_per_round, session.wanted() + 2))


def _floors(n: int) -> tuple[int, int, int]:
    """The spread a round of n ideas must show: distinct lenses, distinct engines,
    ideas from the focus engines."""
    return min(MIN_LENSES, n), min(MIN_ENGINES, n), min(MIN_FOCUS, n)


def ideas_needed(session: Session) -> dict[str, Any] | None:
    """The refusal a proposing tool returns while there are no ideas yet."""
    if not getattr(session, "ideas_required", False) or session.ideas:
        return None
    return {"ok": False, "error": "no ideas recorded yet: the ideas come first. "
            "Nothing was spent."}


def save_ideas(session: Session) -> None:
    """The run's ideas, as recorded - the board reads them."""
    (session.run_dir / "ideas.json").write_text(json.dumps(session.ideas, indent=2, default=str))


def _fixes(clean: list[dict[str, Any]], focus: list[str], used: set[str], lenses: set[str],
           min_engines: int, min_focus: int, on_focus: int, min_lenses: int,
           bare: list[str], n: int, most: int) -> list[str]:
    """What to change, concretely, for each rule an answer missed - one wrong move
    at a time is how a small round gets stuck."""
    out = []
    if len(clean) < n or len(clean) > most:
        out.append(f"give {n} to {most} ideas, no more")
    if len(used) < min_engines:
        spare = [e for e in ENGINES if e not in used]
        out.append(f"change the engine of {min_engines - len(used)} idea(s) to one not used yet "
                   f"- {', '.join(spare)}; keep {min_focus} on the focus engines {focus}")
    if on_focus < min_focus:
        out.append(f"make {min_focus - on_focus} more idea(s) come from the focus engines {focus}")
    if len(lenses) < min_lenses:
        spare = [k for k in LENSES if k not in lenses][:4]
        out.append(f"give {min_lenses - len(lenses)} idea(s) another lens - e.g. {', '.join(spare)}")
    if bare:
        out.append(f"make one idea level {' and one level '.join(bare)}")
    return out or ["keep what passes, change what is MISSING"]


def _same_as(session: Session, clean: list[dict[str, Any]],
             earlier: list[dict[str, Any]]) -> str | None:
    """Check the ideas against each other, this run's earlier rounds and the board
    (``earlier``, the earlier runs' ideas): an error, or None. The same as another
    idea of the answer, or of an earlier round - still in the order, to be worked
    by its name - is an error. A match with an idea of an earlier run is linked on
    the idea (``same_as``), with that idea's record (``inherited``)."""
    limit = session.ws.cfg.agent.checks.same_meaning
    if limit is None:
        return None
    text = lambda i: f"{i['name']} {i['description']}"           # noqa: E731
    problem = None
    for a in range(len(clean)):
        scores = similarity(text(clean[a]), [text(b) for b in clean[a + 1:]])
        for b, score in enumerate(scores, a + 1):
            if score >= limit and problem is None:
                problem = (f"ideas {clean[a]['name']!r} and {clean[b]['name']!r} say much the same "
                           f"thing (overlap {score:.2f}) - merge them, or make one measure "
                           "something else")
    for idea in clean if session.ideas else []:
        scores = similarity(text(idea), [text(e) for e in session.ideas])
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] >= limit and problem is None:
            twin = session.ideas[best]
            problem = (f"idea {idea['name']!r} reads like {twin['name']!r} of round "
                       f"{twin.get('round') or '?'} (overlap {scores[best]:.2f}) - that idea is "
                       f"still in the order: work it with idea=\"{twin['name']}\", or give an idea "
                       "that measures something else")
    for idea in clean if earlier else []:
        scores = similarity(text(idea), [text(e) for e in earlier])
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] < limit:
            continue
        twin = earlier[best]
        idea["same_as"] = f"{twin['run_id']}:{twin['name']}"
        idea["inherited"] = {"signal": twin["signal"], "none": twin["none"],
                             "why": twin.get("why") or ""}
    return problem


def _scope_variables(session: Session) -> list[str]:
    """The scope variables an L3 idea may name - not the keys or the partition date."""
    scope = session.ws.scope()
    if not len(scope):
        return []
    keys = {c for t in set(scope["table"].astype(str))
            for k in ("partition", "identifiers") for c in session.ws.table_profile(t)[k]}
    return [v for v in scope["variable"].astype(str) if v not in keys and not v.endswith("pkey")]


def _taken_variables(session: Session) -> dict[str, list[str]]:
    """Scope variables a request of this run or an earlier one asks for already -
    variable -> the requests."""
    from agent.memory import requested_columns

    out: dict[str, list[str]] = {}
    for (_table, column), by in requested_columns(session).items():
        out.setdefault(column, []).extend(by)
    return out


def _names_in(text: str, variables: list[str]) -> list[str]:
    return [v for v in variables if re.search(rf"\b{re.escape(v)}\b", text, flags=re.I)]


def record_ideas(session: Session, ideas: list[dict[str, Any]],
                 lenient: bool = False) -> dict[str, Any]:
    """Check the ideas - many, varied, each fitting its level - and record
    them. The error, when there is one, goes back to the model to fix.

    ``lenient`` is the last resort, when the answers kept missing the spread: the
    ideas that are each valid are recorded anyway, with the rules they missed, so
    the run - and the board - keep its ideas rather than going on without any."""
    if getattr(session, "explore", False) and session.round + 1 not in session.themes_drawn:
        return {"ok": False, "error": "this run explores without a direction: call draw_theme "
                                      "for this round's theme first, then give ideas about it"}
    levels = list(session.params.levels)
    mixed = len(session.quota) > 1
    in_scope = _scope_variables(session)
    scopes = ", ".join(session.ws.scopes) or "the scope"
    taken = _taken_variables(session) if "L3" in levels else {}
    earlier = {i["name"] for i in session.ideas}            # earlier rounds' ideas
    board = idea_records(session.ws, skip_run=session.run_id)   # earlier runs' ideas
    past = {e["name"]: e["run_id"] for e in board}
    first = len(session.ideas)                              # numbered across rounds
    clean, names, skipped = [], set(), []
    for i, idea in enumerate(ideas or [], 1):
        name = str(idea.get("name", "")).strip()
        lens = str(idea.get("lens", "")).strip().lower()
        engine = str(idea.get("engine", "")).strip().lower()
        description = str(idea.get("description") or idea.get("hypothesis") or "").strip()
        data = str(idea.get("data", "")).strip()
        level = str(idea.get("level", "")).strip().upper() or (levels[0] if len(levels) == 1 else "")
        measures = [str(m).strip() for m in (idea.get("measures") or []) if str(m).strip()]
        named = _names_in(f"{description} {data} {' '.join(measures)}", in_scope)
        # beyond_cas: the field's name before scopes had keywords.
        beyond = level == "L3" and bool(idea.get("beyond_scope") or idea.get("beyond_cas"))
        problem = None
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,63}", name) or name in names:
            problem = f"idea {i}: give it a new snake_case name, got {name!r}"
        elif name in earlier:
            problem = f"idea {i}: {name!r} was an idea of an earlier round - give a new one"
        elif name in past:
            problem = (f"idea {i}: {name!r} was an idea of run {past[name]} - give a new name; "
                       "if it is that idea again, say what is new about it")
        elif FEATURE_LIKE.search(name):
            problem = (f"idea {i}: {name!r} reads like a feature - a window or a version in "
                       "the name. Name the concept (liquidity_buffer_depletion, not "
                       "balance_std_6m); windows and statistics belong in its measures")
        elif engine not in ENGINES:
            problem = f"idea {i}: engine {engine!r} is not one of {list(ENGINES)}"
        elif lens not in LENSES:
            problem = f"idea {i}: lens {lens!r} is not one of {list(LENSES)}"
        elif not description:
            problem = f"idea {i}: say what it measures and why it should carry risk"
        elif level not in levels:
            problem = f"idea {i}: its level must be one of {levels}"
        elif not MIN_MEASURES <= len(measures) <= MAX_MEASURES and not (lenient and measures):
            problem = (f"idea {i} ({name}): give {MIN_MEASURES} to {MAX_MEASURES} measures - "
                       "different ways to measure it, each a candidate feature or data in "
                       "one line. An idea is a hypothesis several features can test")
        elif beyond and len(data.split()) < 4:
            problem = (f"idea {i} ({name}): beyond scope, say in its data what data it needs "
                       "and where it would come from - the system or team, the grain.")
        elif level == "L3" and not beyond and not named:
            problem = (
                f"idea {i} ({name}): names no scope variable. Within a scope ({scopes}), write "
                "the variables it needs in its data exactly as named - e.g. "
                + ", ".join(f"`{v}`" for v in in_scope[:6]) + " (the brief lists them). If the "
                "data lies outside every scope - external, strategy (RLA), calling - mark it "
                "beyond_scope. If the model database and the linked sources already carry it, "
                "it is an L1/L2 idea, not a data request.")
        elif level == "L3" and not beyond and all(v.lower() in taken for v in named):
            # Its request would ask for data already asked for, and be refused as such:
            # a new angle on the same columns is a feature, not a data request.
            by = sorted({b for v in named for b in taken[v.lower()]})
            problem = (
                f"idea {i} ({name}): its scope variables ({', '.join(named)}) are requested "
                f"already - by {'; '.join(by[:3])}. A request for them again is the same data. "
                "Name a scope variable not yet requested (with them or alone), or go "
                "beyond_scope.")
        if problem:
            if lenient:
                skipped.append(problem)
                continue
            return {"ok": False, "error": problem}
        names.add(name)
        clean.append({"n": first + i, "round": session.round + 1, "name": name,
                      "engine": engine, "lens": lens, "level": level,
                      "description": description, "measures": measures, "data": data,
                      **({"beyond_scope": True} if beyond else {}),
                      **({"variables": named} if named and not beyond else {})})
    focus = focus_engines(session)
    lenses = {c["lens"] for c in clean}
    used = {c["engine"] for c in clean}
    n = ideas_in_round(session)
    min_lenses, min_engines, min_focus = _floors(n)
    on_focus = sum(c["engine"] in focus for c in clean)
    # Every requirement, met or not: fixing only the one named breaks another.
    most = session.params.ideas_per_round
    checks = [(n <= len(clean) <= most, f"{n} to {most} ideas (you gave {len(clean)})"),
              (len(used) >= min_engines, f"at least {min_engines} different engines "
                                         f"(you used {len(used)}: {sorted(used)})"),
              (on_focus >= min_focus, f"at least {min_focus} from this round's focus engines "
                                      f"{focus} (you have {on_focus})"),
              (len(lenses) >= min_lenses, f"at least {min_lenses} different lenses "
                                          f"(you used {len(lenses)}: {sorted(lenses)})")]
    bare: list[str] = []
    if mixed:
        bare = [lv for lv in session.quota if session.wanted(lv) and
                not any(c["level"] == lv for c in clean)]
        checks.append((not bare, "an idea for every level still wanted "
                                 f"({session.targets_left()})" + (f" - none for {bare}" if bare else "")))
    missed = [text for ok, text in checks if not ok]
    if missed and not lenient:
        return {"ok": False, "error": "not yet a spread. An answer must meet ALL of these - keep "
                "what already passes: " + "; ".join(f"[{'ok' if ok else 'MISSING'}] {text}"
                                                    for ok, text in checks)
                + ". To fix: " + "; ".join(_fixes(clean, focus, used, lenses, min_engines,
                                                  min_focus, on_focus, min_lenses,
                                                  bare if mixed else [], n, most))}
    if lenient and not clean:
        return {"ok": False, "error": "no idea in the answer was valid: " + "; ".join(skipped)}
    if (problem := _same_as(session, clean, board)) and not lenient:
        return {"ok": False, "error": problem}
    session.ideas = session.ideas + clean
    save_ideas(session)
    session.round += 1
    session.round_ideas, session.round_attempts, session.round_spent = clean, 0, False
    session.round_results, session.report_refusals = session.results(), 0
    session.emit("ideas_recorded", ideas=clean, focus=focus, round=session.round,
                 **({"lenient": True, "missed": missed + skipped} if lenient else {}),
                 **({"theme": session.themes_drawn[session.round]}
                    if session.round in getattr(session, "themes_drawn", {}) else {}))
    linked = [{"idea": c["name"], "same_as": c["same_as"],
               "p_useful": round((1 + c["inherited"]["signal"])
                                 / (2 + c["inherited"]["signal"] + c["inherited"]["none"]), 3),
               **({"why_not": c["inherited"]["why"]} if c["inherited"]["why"] else {})}
              for c in clean if c.get("same_as")]
    return {"ok": True, "ideas": len(clean), "engines": sorted(used), "lenses": sorted(lenses),
            "round": session.round,
            **({"linked": linked, "note": "these read like earlier ideas and start from their "
                                          "record - a failed one is worked later"} if linked else {})}
