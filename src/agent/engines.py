"""The idea engines: four ways to generate an idea, and room for any other.

An idea says which engine made it. The engines differ in where their knowledge
comes from and which way they reason, so a round drawn from several of them
spreads further than free brainstorming does:

    mechanism      forward from a theory of why people default (a theory library)
    contrastive    backward from cases the incumbent gets wrong
    analogy        sideways from a measure proven in another field
    morphological  from the empty cells of a seven-axis grid of what was tried
    others         any other route - practitioner knowledge, earlier model
                   generations, the product's rules, the market - named by the idea

Each guide is a markdown file in ``engines/`` - frontmatter (title, core
question, knowledge source, reasoning direction, cognitive basis), then the
method. The brief lists them in a table; the full guide is the ``engine`` tool's
answer, read when an idea is made with it.

**Which engines a round leans on** is drawn by Thompson sampling over their
yield: each engine keeps a Beta(1 + ideas that carried signal, 1 + ideas that
did not) from every run's board, one value is drawn per engine, and the top
:data:`FOCUS_ENGINES` are the round's focus. A strong engine is asked more, a
weak one still sometimes - and every draw is on record (``engines_drawn``).
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["ENGINES_DIR", "ENGINES", "FOCUS_ENGINES", "MIN_ENGINES", "engines", "engine_guide",
           "engine_yield", "draw_focus"]

ENGINES_DIR = Path(__file__).parent / "engines"
ORDER = ("mechanism", "contrastive", "analogy", "morphological", "others")
FOCUS_ENGINES = 2       # engines a round leans on
MIN_ENGINES = 3         # distinct engines across a round's ideas


@lru_cache(maxsize=1)
def engines() -> dict[str, dict[str, str]]:
    """Every engine's frontmatter and guide, in the canonical order."""
    out = {}
    for path in ENGINES_DIR.glob("*.md"):
        text = path.read_text()
        meta, body = {}, text
        if text.startswith("---"):
            head, body = text[3:].split("---", 1)
            for line in head.strip().splitlines():
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        out[meta.get("name", path.stem)] = {**meta, "body": body.strip()}
    return {k: out[k] for k in sorted(out, key=lambda k: (ORDER.index(k) if k in ORDER else 99, k))}


ENGINES = tuple(engines())


def engine_guide(name: str) -> dict[str, Any]:
    """An engine's full guide - the agent's tool."""
    key = str(name or "").strip().lower()
    if key not in engines():
        return {"ok": False, "error": f"no engine {name!r}; the engines are {list(ENGINES)}"}
    e = engines()[key]
    return {"ok": True, "engine": key, "title": e.get("title", key), "guide": e["body"]}


def engine_yield(ideas: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per engine, its ideas that carried signal and those that did not - from the
    board's idea records (agent.board.idea_records)."""
    out = {k: {"signal": 0, "none": 0, "untried": 0} for k in ENGINES}
    for idea in ideas:
        for engine in idea.get("engines") or [idea.get("engine")]:
            if engine not in out:
                continue
            status = idea.get("status")
            key = "signal" if status == "active" else "none" if status == "low" \
                else "untried"
            out[engine][key] += 1
    return out


def draw_focus(ideas: list[dict[str, Any]], seed: str) -> dict[str, Any]:
    """This round's focus engines: one Beta draw per engine from its yield, the
    top FOCUS_ENGINES taken. Seeded, so the draw is reproducible and on record."""
    rng = random.Random(seed)
    stats = engine_yield(ideas)
    draws = {k: round(rng.betavariate(1 + s["signal"], 1 + s["none"]), 4) for k, s in stats.items()}
    focus = sorted(draws, key=lambda k: -draws[k])[:FOCUS_ENGINES]
    return {"focus": focus, "draws": draws, "yield": stats}
