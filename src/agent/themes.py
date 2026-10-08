"""The theme pool: directions to start from when the user gives none.

Once the task is described, one model call reads what the agent will read - the
task description and context, the column descriptions, the sources and the
scopes - and lists short themes worth a direction each: behaviours, risk
mechanisms, data that could carry them. The pool is kept in
``<agent.run_dir>/themes.json`` with a fingerprint of that context, and made
again only when the context changes.

A run started with no direction is an open exploration. The pool is the
agent's tool there: each round it draws a theme (``draw_theme``) - one the
fewest runs have explored, an unexplored one first, at random among equals -
and builds that round's ideas around it. A run walks across the pool round by
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

__all__ = ["OPEN", "POOL_SIZE", "context_of", "draw_theme", "generate", "pick", "pool", "save"]

POOL_SIZE = 30
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


def pool(ws: Workspace) -> dict[str, Any]:
    """The pool as saved, each theme with how many runs explored it, and whether the
    task has changed since it was made (``stale``)."""
    path = _path(ws)
    saved = json.loads(path.read_text()) if path.exists() else {}
    runs = explored(ws)
    themes = [{"theme": t, "runs": runs.get(t.lower(), 0)} for t in saved.get("themes", [])]
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
    result = asyncio.run(Runner.run(agent, context_of(ws), max_turns=2))
    return save(ws, result.final_output.themes)


def pick(ws: Workspace, rng: random.Random | None = None,
         skip: set[str] | frozenset[str] = frozenset()) -> str | None:
    """A theme the fewest runs explored, at random among equals, and not in `skip`
    - None when none is left."""
    themes = [t for t in pool(ws)["themes"] if t["theme"] not in skip]
    if not themes:
        return None
    least = min(t["runs"] for t in themes)
    return (rng or random).choice([t["theme"] for t in themes if t["runs"] == least])


def draw_theme(session: Any) -> dict[str, Any]:
    """An exploration's theme for its next round of ideas - the agent's tool."""
    if not session.explore:
        return {"ok": False, "error": "this run has a direction from the user: follow it"}
    upcoming = session.round + 1
    if upcoming in session.themes_drawn:
        return {"ok": True, "theme": session.themes_drawn[upcoming],
                "note": "already drawn for this round - give its ideas"}
    theme = pick(session.ws, skip=set(session.themes_drawn.values()))
    if theme is None:
        return {"ok": False, "error": "every theme in the pool is drawn in this run - report "
                                      "your findings"}
    before = explored(session.ws).get(theme.lower(), 0)
    session.themes_drawn[upcoming] = theme
    session.emit("theme_drawn", theme=theme, round=upcoming, explored_before=before)
    return {"ok": True, "theme": theme, "round": upcoming, "explored_before": before,
            "next": "look at the data this theme needs, then give this round's ideas - "
                    "each one about the theme, through different lenses"}
