"""The prompt composer: it puts together what the LLM reads.

A run's prompt is its **brief** (the system instructions) and a few **messages**
(the user turns). Both are markdown templates in ``templates/``; the parts that
depend on the workspace and the run are built by :mod:`agent.composer.sections`.
The skills (``agent/skills/``) are appended to every brief in full.

    templates/feature_engineer.md   the brief of a direction (L1 / L2, and an L3 share)
    templates/data_scout.md         the brief of a data-request run (L3 only)
    templates/linkage_writer.md     the brief of a linkage job (Setup)
    templates/messages.md           the user turns: the task, ideas, propose, nudge
    sections.py                     the filled-in parts: memory, ideas, gates, current data ...

Two more things reach the model, written beside the code they describe: each
tool's description (the docstrings in ``agent/tools/__init__.py``), and each
tool's reply - its result, or why it refused and what to do next.

To read a run's brief without running anything:

    PYTHONPATH=src python -m agent.composer -c configs/synthetic_agent.yaml --levels L3

and every run keeps the brief it was given, as ``brief.md`` in its folder.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.composer import sections

__all__ = ["TEMPLATES_DIR", "template_for", "compose", "message"]

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _read(name: str) -> str:
    return (TEMPLATES_DIR / f"{name}.md").read_text()


def template_for(session: Any) -> str:
    """Which brief a run gets."""
    if session.kind == "linkage":
        return "linkage_writer"
    return "data_scout" if session.l3_only else "feature_engineer"


def _language(engine: str) -> str:
    return "PySpark" if engine == "spark" else "pandas"


def compose(session: Any) -> str:
    """The run's brief: its template, filled in."""
    ws, cfg, p = session.ws, session.ws.cfg, session.params
    name = template_for(session)
    fields: dict[str, Any] = {
        "id_col": ws.id_col, "id_format": cfg.data.id_format or "(not specified)",
        "skills": sections.skills(session), "data_size": sections.data_size(session),
    }
    if name == "linkage_writer":
        src = ws.sources()[session.source]
        fields.update(
            source=src.name, engine=session.local_engine,
            code_language=_language(session.local_engine),
            source_columns="\n".join(f"  * `{c}` - {d}  e.g. {src.samples.get(c, [])[:3]}"
                                     for c, d in src.columns.items()))
        return _read(name).format(**fields)

    fields.update(
        K=session.K, max_attempts=session.max_attempts, n_screen=f"{len(ws.screen):,}",
        task_description=cfg.discovery.task_description.strip(),
        task_context=ws.task_context.strip(), target=ws.target, n_base=len(ws.base_features),
        shot_list=sections.shot_list(ws), scope_notes=sections.scope_notes(ws),
        scopes=sections.scopes(ws),
        memory=sections.memory(session), ideas=sections.ideas(session),
        columns=sections.columns(session))
    if name == "data_scout":
        fields.update(
            linked=", ".join(ws.linked()) or "none",
            unlinked=", ".join(sorted(set(p.sources) - set(ws.linked()))) or "none",
            current_data=sections.current_data(session))
    else:
        feature_levels = [lv for lv in p.levels if lv != "L3"]
        fields.update(
            engine=p.engine, code_language=_language(p.engine), quota=sections.quota(session),
            linkage_language=_language(cfg.agent.linkage_engine),
            feature_levels=", ".join(feature_levels) or "none",
            l3_rule=("L3 is on: you may request data the direction needs and nobody has."
                     if "L3" in p.levels else "L3 is off: do not request data pulls."),
            sources=", ".join(p.sources) or "none - model columns only",
            gates=sections.gates(session))
    return _read(name).format(**fields)


@lru_cache(maxsize=1)
def _messages() -> dict[str, str]:
    text = _read("messages")
    parts = re.split(r"^## (\w+)\n", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def message(name: str, **fields: Any) -> str:
    """One user turn from messages.md, filled in."""
    return _messages()[name].format(**fields)
