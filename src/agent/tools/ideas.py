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
* **the board** - two ideas of one answer alike in words are sent back. Whether
  an idea restates an earlier one is the agent's to say: it names that idea
  (``same_as``) - linked to it and starting from its record, so one whose
  features carried no signal is drawn late (Thompson sampling), not refused -
  or says how it differs (``differs``). Words only find the candidates
  (``agent.checks.similarity``): an idea close to an earlier one that says
  neither is shown the candidates and asked, once.

The earlier directions' ideas and features are in the brief (``agent.board``,
``agent.memory``), and identical features are refused - so the agent cannot
simply repeat itself, and is shown where it has not looked yet.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.board import carry_ideas, idea_records, idea_standing, one_per_family
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
        draw = draw_focus(one_per_family(idea_records(session.ws)), seed=f"{session.run_id}:{upcoming}")
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


# An earlier idea this close in words is shown to the idea's author, who says
# whether it is the same idea (``same_as``) or how it differs (``differs``) - the
# words find candidates, the agent decides. Restatements score from about 0.2.
SHORTLIST = 0.2
SHORTLIST_TOP = 3


def _same_as(session: Session, clean: list[dict[str, Any]],
             earlier: list[dict[str, Any]], lenient: bool = False) -> str | None:
    """Check each idea against the others of the answer and the ideas before it: an
    error, or None. Two ideas of the answer alike in words are sent back. An idea
    may name the earlier idea it restates (``same_as``): one of this run's - an
    earlier round's, or an earlier run's in its draw - is in the order already, so
    it is sent back to be worked by its name; one of an earlier run's is linked
    (``same_as``, the family's record in ``inherited``). An idea close in words to
    an earlier one that names none is asked, once, which it is - same as it, or
    different (``differs``); one this close (``same_meaning``) until it says."""
    from agent.board import one_per_family, session_ideas

    limit = session.ws.cfg.agent.checks.same_meaning
    text = lambda i: f"{i['name']} {i.get('description') or ''}"   # noqa: E731
    problems: list[str] = []
    for a in range(len(clean)) if limit is not None else []:
        scores = similarity(text(clean[a]), [text(b) for b in clean[a + 1:]])
        for b, score in enumerate(scores, a + 1):
            if score >= limit:
                problems.append(f"ideas {clean[a]['name']!r} and {clean[b]['name']!r} say much "
                                f"the same thing (overlap {score:.2f}) - merge them, or make one "
                                "measure something else")
    mine = {i["name"]: i for i in session.ideas}
    drawn = {i["name"]: i for i in session_ideas(session) if i.get("carried_from")}
    runs = {**{e["name"]: e for e in earlier}, **{e["key"]: e for e in earlier}}
    # The candidates: this run's ideas and one per family of the earlier runs'.
    pool = [{**i, "where": f"round {i.get('round') or '?'} of this run"} for i in session.ideas] + [
        {**e, "where": (f"run {e['run_id']}, in this run's draw" if e["name"] in drawn
                        else f"run {e['run_id']}")} for e in one_per_family(earlier)]
    questions = []
    for idea in clean:
        declared = str(idea.pop("_same_as", "") or "").strip()
        if declared:
            current = mine.get(declared) or drawn.get(declared)
            twin = runs.get(declared)
            if current is not None:
                if not lenient:
                    where = (f"of round {current.get('round') or '?'}" if declared in mine
                             else "of an earlier run, in this run's draw")
                    problems.append(f"idea {idea['name']!r} is {declared!r} {where} - that idea is "
                                    f"still in the order: work it with idea=\"{declared}\", and "
                                    "give a new idea in its place")
            elif twin is not None:
                idea["same_as"] = twin["key"]
                idea["inherited"] = {"signal": twin["signal"], "none": twin["none"],
                                     "why": twin.get("why") or ""}
            elif not lenient:
                problems.append(f"idea {idea['name']!r}: same_as names no earlier idea "
                                f"({declared!r}) - give an earlier idea's name, or leave it empty")
            continue
        if lenient or not pool:
            continue
        scores = similarity(text(idea), [text(c) for c in pool])
        near = sorted(((s, c) for s, c in zip(scores, pool) if s >= SHORTLIST),
                      key=lambda x: -x[0])[:SHORTLIST_TOP]
        if not near or idea.get("differs"):
            continue
        top, twin = near[0]
        if limit is not None and top >= limit and twin["name"] in mine:
            problems.append(f"idea {idea['name']!r} reads like {twin['name']!r} of round "
                            f"{twin.get('round') or '?'} (overlap {top:.2f}) - that idea is still in "
                            f"the order: work it with idea=\"{twin['name']}\", or give an idea "
                            "that measures something else (or say in `differs` how it does)")
            continue
        if idea["name"] in session.ideas_asked and (limit is None or top < limit):
            continue                        # asked once already: its silence means new
        session.ideas_asked.add(idea["name"])
        questions.append(f"`{idea['name']}` reads like " + "; ".join(
            f"`{c['name']}` ({c['where']}, P(useful) {c.get('p_useful', 0.5):.2f}, overlap {s:.2f}: "
            f"{str(c.get('description') or '')[:100]!r})" for s, c in near))
    if questions:
        problems.append(
            "say for each of these ideas whether it is one of the earlier ideas it reads like. "
            "The same idea - the same claim about who defaults and why, however worded: set "
            "`same_as` to that idea's name (it then keeps that idea's record; one already in "
            "this run's order is worked by its name instead). A different one: say in "
            "`differs` what it claims that the earlier one does not. "
            + " | ".join(questions))
    return "; ".join(problems) or None


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
                 lenient: bool = False, carry: list[str] | None = None) -> dict[str, Any]:
    """Check the ideas - many, varied, each fitting its level - and record
    them, with the earlier runs' ideas it carries into the draw (``carry``, by
    name). The error, when there is one, goes back to the model to fix.

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
        differs = str(idea.get("differs", "") or "").strip()
        clean.append({"n": first + i, "round": session.round + 1, "name": name,
                      "engine": engine, "lens": lens, "level": level,
                      "description": description, "measures": measures, "data": data,
                      "_same_as": str(idea.get("same_as", "") or "").strip(),
                      **({"differs": differs} if differs else {}),
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
    if (problem := _same_as(session, clean, board, lenient)) and not lenient:
        return {"ok": False, "error": problem}
    if (unknown := carry_ideas(session, carry or [])) and not lenient:
        return {"ok": False, "error": "; ".join(unknown) + " - name ideas from the board in "
                                      "your brief, or leave carry empty"}
    session.ideas = session.ideas + clean
    save_ideas(session)
    session.round += 1
    session.round_ideas, session.round_attempts, session.round_spent = clean, 0, False
    session.round_results, session.report_refusals = session.results(), 0
    session.emit("ideas_recorded", ideas=clean, focus=focus, round=session.round,
                 **({"carried": list(session.carried)} if session.carried else {}),
                 **({"lenient": True, "missed": missed + skipped} if lenient else {}),
                 **({"theme": session.themes_drawn[session.round]}
                    if session.round in getattr(session, "themes_drawn", {}) else {}))
    linked = [{"idea": c["name"], "same_as": c["same_as"],
               "p_useful": idea_standing([], c["inherited"])["p_useful"],
               **({"why_not": c["inherited"]["why"]} if c["inherited"]["why"] else {})}
              for c in clean if c.get("same_as")]
    return {"ok": True, "ideas": len(clean), "engines": sorted(used), "lenses": sorted(lenses),
            "round": session.round, **({"carried": len(session.carried)} if session.carried else {}),
            **({"linked": linked, "note": "these read like earlier ideas and start from their "
                                          "record - a failed one is worked later"} if linked else {})}
