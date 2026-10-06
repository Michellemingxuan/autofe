"""Agent-driven feature discovery: one agent, skills loaded on demand, tools.

See docs/superpowers/specs/2026-10-03-agentic-feature-discovery-design.md.

* :mod:`agent.workspace` - what the agent can see of a use case
* :mod:`agent.execution` - running its scripts in a guarded subprocess
* :mod:`agent.session`   - one direction run: tool logic, events, approvals
* :mod:`agent.agent`     - the openai-agents Agent and the streamed run
* :mod:`agent.synthetic` - a CDSS-shaped synthetic use case
"""

import os as _os
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[2]


def at_project_root(config_path: str) -> str:
    """Paths in a config are relative to the project root, so the entry points
    move there first - run from src/ or anywhere else, data/ is the one folder."""
    resolved = str(_Path(config_path).resolve())
    _os.chdir(PROJECT_ROOT)
    return resolved
