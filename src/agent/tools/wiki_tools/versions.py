"""The wiki's history in git: read it, put a page back, undo a commit - and finish,
with the message the raw file's commit is made with."""

from __future__ import annotations

import re
import subprocess
from typing import Any

from agent.tools.wiki_tools.keeper import Keeper, one_line
from agent.wiki import git

__all__ = ["wiki_history", "wiki_diff", "wiki_restore", "wiki_revert", "finish", "DIFF_CHARS"]

DIFF_CHARS = 8_000
_REV = re.compile(r"(?:[0-9a-f]{4,40}|HEAD)(?:[~^]\d*)*")


def _rev(rev: str) -> str | None:
    return None if _REV.fullmatch(str(rev).strip()) else f"{rev!r} is not a commit - wiki_history lists them"


def _paths(page: str) -> list[str]:
    return [f"{page.strip().strip('/').removesuffix('.md')}.md"] if page.strip() else []


def wiki_history(keeper: Keeper, page: str = "", n: int = 15) -> dict[str, Any]:
    out = git(keeper.root, "log", f"-n{max(1, min(int(n), 50))}", "--date=short",
              "--format=%h %ad %s", "--", *_paths(page), check=False)
    return {"ok": True, "commits": out.strip().splitlines()}


def wiki_diff(keeper: Keeper, rev: str, page: str = "") -> dict[str, Any]:
    if (problem := _rev(rev)):
        return {"ok": False, "error": problem}
    out = git(keeper.root, "show", "--format=%h %ad %s%n", "--date=short", rev.strip(), "--",
              *_paths(page), check=False)
    return {"ok": True, "diff": out[:DIFF_CHARS],
            **({"truncated": True} if len(out) > DIFF_CHARS else {})}


def wiki_restore(keeper: Keeper, page: str, rev: str) -> dict[str, Any]:
    pid, problem = keeper.editable(page)
    if problem or (problem := _rev(rev)):
        return {"ok": False, "error": problem}
    try:
        git(keeper.root, "checkout", rev.strip(), "--", f"{pid}.md")
    except RuntimeError as error:
        return {"ok": False, "error": f"{pid} as of {rev}: {error} - a page deleted in a "
                                      "commit is there as of the commit before it (<rev>^)"}
    keeper.record("restored", pid, f"as of {rev}")
    return {"ok": True, "page": pid, "restored": rev}


def wiki_revert(keeper: Keeper, rev: str) -> dict[str, Any]:
    """Undo a commit's page edits - the raw files and the records stay as they are."""
    if (problem := _rev(rev)):
        return {"ok": False, "error": problem}
    touched = git(keeper.root, "show", "--name-only", "--format=", rev.strip(), check=False).split()
    pages_touched = [p for p in touched if p.startswith("domains/")]
    if not pages_touched:
        return {"ok": False, "error": f"{rev} changed no page"}
    patch = git(keeper.root, "show", "--format=", rev.strip(), "--", "domains", check=False)
    done = subprocess.run(["git", "-C", str(keeper.root), "apply", "-R", "--whitespace=nowarn"],
                          input=patch, capture_output=True, text=True)
    if done.returncode:
        return {"ok": False, "error": f"{rev} cannot be undone cleanly - later edits touch "
                                      "the same lines; restore its pages one by one"}
    keeper.record("reverted", rev.strip())
    return {"ok": True, "reverted": rev.strip(), "pages": pages_touched}


def finish(keeper: Keeper, message: str) -> dict[str, Any]:
    keeper.message = one_line(message) or f"Ingest raw/{keeper.raw_name}"
    return {"ok": True, "done": True}
