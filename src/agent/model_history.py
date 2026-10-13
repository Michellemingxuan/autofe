"""The model's history: the variables earlier generations used, tried or dropped.

The analyst keeps it as a markdown document (``agent.model_history``) - the
current generation on a line of its own (``Current model: Gen9``), then one
table row per variable:

    | generation | variable | description | status | why |
    | Gen7       | pay_gap_avg | average days between payments | dropped | unstable |

It reaches the agent twice: in the brief, so ideas start from what earlier
generations learned; and in check C0 (agent.tools.screen), where a feature that
reads like a variable of the history must say what differs before it is built.
An empty table is fine - the document can grow as generations are written down.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = ["COLUMNS", "read", "history_of"]

COLUMNS = ("generation", "variable", "description", "status", "why")


def read(path: str | Path | None) -> dict[str, Any]:
    """The document's current generation and its variables, one dict each."""
    if not path or not Path(path).is_file():
        return {"current": "", "variables": []}
    text = Path(path).read_text(errors="replace")
    current = re.search(r"current model\s*:\s*(\S+)", text, flags=re.I)
    rows, header = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue                                        # the separator row
        if header is None:
            header = [c.lower() for c in cells]
            continue
        row = dict(zip(header, cells))
        if row.get("variable"):
            rows.append({k: row.get(k, "") for k in COLUMNS})
    return {"current": current.group(1) if current else "", "variables": rows}


def history_of(ws: Any) -> dict[str, Any]:
    """The workspace's model history, from the config's ``agent.model_history``."""
    return read(getattr(ws.cfg.agent, "model_history", None))
