"""Find pages and read them as files - frontmatter included, for edits to copy from."""

from __future__ import annotations

from typing import Any

from agent.tools.wiki_tools.keeper import Keeper
from agent.wiki import Access, _resolve, pages, search

__all__ = ["wiki_search", "wiki_read", "READ_CHARS"]

READ_CHARS = 16_000          # a page read at once, at most


def wiki_search(keeper: Keeper, query: str, domain: str = "", kind: str = "") -> dict[str, Any]:
    """Every page, cases of every domain included: the keeper sees the whole wiki."""
    return search(query, Access(cross_domain_cases="open"), domain, kind, root=keeper.root)


def wiki_read(keeper: Keeper, page: str) -> dict[str, Any]:
    all_pages = pages(keeper.root)
    pid = _resolve(page, all_pages)
    if pid is None:
        return {"ok": False, "error": f"no page {page!r} - wiki_search finds pages"}
    text = keeper.path(pid).read_text()
    return {"ok": True, "page": pid, "text": text[:READ_CHARS],
            **({"truncated": True} if len(text) > READ_CHARS else {}),
            "links": [{"link": link, "page": _resolve(link, all_pages, all_pages[pid].domain)}
                      for link in all_pages[pid].links]}
