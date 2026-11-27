"""The theme pool: directions to start from when the user gives none.

Once the task is described, one model call reads what the agent will read - the
task description and context, the column descriptions, the sources and the
scopes - and lists short themes worth a direction each: behaviours, risk
mechanisms, data that could carry them. The pool is kept in
``<agent.run_dir>/themes.json`` with a fingerprint of that context, and made
again only when the context changes.

The reviewed cases add to it: each pattern a case page of the wiki names under
**Patterns** (agent.wiki) is a theme, from the cases the model may read - its
home domain's. They are read from the wiki as it is, so a new case is in the
pool at once, and each keeps the case it came from.

A run started with no direction is an open exploration. The pool is the
agent's tool there: each round it draws a theme (``draw_theme``) - one the
fewest runs have explored, an unexplored one first; a case's pattern before a
made theme, then at random among equals - and builds that round's ideas around
it (a case's pattern with the case to read). A run walks across the pool round by
round, and successive runs walk on from where the earlier ones stopped.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent.workspace import Workspace

__all__ = ["OPEN", "POOL_SIZE", "case_themes", "context_of", "draw_theme", "generate", "pick",
           "pool", "save"]

POOL_SIZE = 10
OPEN = "Open exploration"                 # the direction of a run the user left empty


class Themes(BaseModel):
    themes: list[str] = Field(description=f"{POOL_SIZE} distinct themes, each 3 to 10 words")


def _path(ws: Workspace) -> Path:
    return Path(ws.cfg.agent.run_dir) / "themes.json"


def context_of(ws: Workspace, limit: int = 6000) -> str:
    """What the themes are drawn from - the task as the agent sees it."""
    d = ws.cfg.discovery
    columns = "; ".join(f"{c}: {ws.descriptions.get(c, '')}"[:90] for c in ws.base_features)
    parts = [f"Task: {d.task_description.strip()}",
             f"Context:\n{ws.task_context.strip()[:limit]}",
             f"Model columns ({len(ws.base_features)}): {columns[:limit]}",
             "Additional sources: " + (", ".join(
                 f"{s.name} ({', '.join(list(s.columns)[:12])})" for s in ws.sources().values())
                 or "none"),
             "Scopes data may be requested from: " + (", ".join(ws.scopes) or "none")]
    return "\n\n".join(parts)


def _fingerprint(ws: Workspace) -> str:
    return hashlib.sha256(context_of(ws).encode()).hexdigest()[:16]


def case_themes(ws: Workspace) -> list[dict[str, str]]:
    """The patterns of the wiki's cases this model may read: theme, pattern, case."""
    from agent import wiki

    try:
        found = wiki.pages(wiki.WIKI_DIR)
    except OSError:
        return []
    access = wiki.access_for(ws.cfg)
    out: dict[str, dict[str, str]] = {}
    for page in found.values():
        if page.kind == "case" and access.refusal(page) is None:
            for item in wiki.patterns(page):
                out.setdefault(item["theme"].lower(), item)
    return list(out.values())


def pool(ws: Workspace) -> dict[str, Any]:
    """The pool - the cases' patterns, then the made themes - each theme with how many
    runs explored it, and whether the task has changed since the themes were made
    (``stale``)."""
    path = _path(ws)
    saved = json.loads(path.read_text()) if path.exists() else {}
    runs = explored(ws)
    cases = case_themes(ws)
    taken = {c["theme"].lower() for c in cases}
    themes = ([{**c, "runs": runs.get(c["theme"].lower(), 0)} for c in cases]
              + [{"theme": t, "runs": runs.get(t.lower(), 0)} for t in saved.get("themes", [])
                 if t.lower() not in taken])
    ready = bool(ws.cfg.discovery.task_description.strip())
    return {"themes": themes, "made": saved.get("made"),
            "stale": ready and saved.get("fingerprint") != _fingerprint(ws),
            "ready": ready}


def explored(ws: Workspace) -> dict[str, int]:
    """Runs per theme: the explorations that drew it, and the runs whose direction
    the user wrote as it."""
    from agent.memory import _direction, _runs

    out: dict[str, int] = {}
    for folder in _runs(ws):
        seen = {_direction(folder).strip().lower()} - {"", OPEN.lower()}
        events = folder / "events.jsonl"
        if events.exists():
            seen |= {json.loads(line)["theme"].strip().lower()
                     for line in open(events) if '"theme_drawn"' in line}
        for theme in seen:
            out[theme] = out.get(theme, 0) + 1
    return out


def save(ws: Workspace, themes: list[str]) -> list[str]:
    clean = list(dict.fromkeys(t.strip().rstrip(".") for t in themes if t and t.strip()))
    path = _path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"themes": clean, "fingerprint": _fingerprint(ws),
                                "made": round(time.time(), 3)}, indent=2))
    return clean


def generate(ws: Workspace) -> list[str]:
    """Ask the model for the themes, and save them."""
    from agents import Agent, Runner

    from agent.agent import FirstAnswer
    from agent.llm import build_model

    agent = Agent(
        name="themes",
        instructions=(
            "You plan feature discovery for a credit-risk model. From the task below, list "
            f"{POOL_SIZE} distinct themes, each a starting direction for one discovery run: "
            "a customer behaviour, a risk mechanism, or a kind of data that could reveal "
            "default risk the model may miss. Each theme is 3 to 10 plain words, specific "
            "to this task - not a feature formula. Cover different behaviours, horizons "
            "and data: the model columns, the additional sources, the scopes data may be "
            "requested from, and data beyond them. No two themes may say the same thing."),
        model=build_model(ws.cfg.agent.llm, log_dir=Path(ws.cfg.agent.run_dir) / "llm_logs"),
        output_type=FirstAnswer(Themes))
    from agent.llm_trace import TraceHooks, store_for

    hooks = TraceHooks(store_for(ws.cfg), f"themes_{time.strftime('%Y%m%d_%H%M%S')}", "themes",
                       kind="themes")
    result = asyncio.run(Runner.run(agent, context_of(ws), max_turns=2, hooks=hooks))
    return save(ws, result.final_output.themes)


def pick(ws: Workspace, rng: random.Random | None = None,
         skip: set[str] | frozenset[str] = frozenset()) -> dict[str, Any] | None:
    """A theme the fewest runs explored - a case's pattern before a made theme, at
    random among equals - and not in `skip`; None when none is left."""
    themes = [t for t in pool(ws)["themes"] if t["theme"] not in skip]
    if not themes:
        return None
    least = min(t["runs"] for t in themes)
    first = [t for t in themes if t["runs"] == least]
    first = [t for t in first if t.get("case")] or first
    return (rng or random).choice(first)


def draw_theme(session: Any) -> dict[str, Any]:
    """An exploration's theme for its next round of ideas - the agent's tool."""
    if not session.explore:
        return {"ok": False, "error": "this run has a direction from the user: follow it"}
    upcoming = session.round + 1
    if upcoming in session.themes_drawn:
        return {"ok": True, "theme": session.themes_drawn[upcoming],
                "note": "already drawn for this round - give its ideas"}
    drawn = pick(session.ws, skip=set(session.themes_drawn.values()))
    if drawn is None:
        return {"ok": False, "error": "every theme in the pool is drawn in this run - report "
                                      "your findings"}
    theme, case = drawn["theme"], drawn.get("case")
    before = explored(session.ws).get(theme.lower(), 0)
    session.themes_drawn[upcoming] = theme
    session.emit("theme_drawn", theme=theme, round=upcoming, explored_before=before,
                 **({"case": case} if case else {}))
    if case:
        return {"ok": True, "theme": theme, "round": upcoming, "explored_before": before,
                "pattern": drawn["pattern"], "case": case,
                "next": f"a pattern a reviewed case showed: read the case ({case}, wiki_read) "
                        "for its findings, look at the data the pattern needs, then give this "
                        "round's ideas - each one about the pattern, through different lenses"}
    return {"ok": True, "theme": theme, "round": upcoming, "explored_before": before,
            "next": "look at the data this theme needs, then give this round's ideas - "
                    "each one about the theme, through different lenses"}
