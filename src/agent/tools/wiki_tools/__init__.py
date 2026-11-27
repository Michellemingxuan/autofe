"""The wiki keeper's tools: plain functions over a :class:`Keeper` - one raw file's
work on the wiki (agent.wiki_ingest runs the keeper, one commit per file).

    read      wiki_search · wiki_read                          find pages; read one as a file
    edit      create_page · add_to_page · change_in_page ·     add, change or delete a passage;
              delete_from_page · delete_page · move_page       create, delete or move a page
    versions  wiki_history · wiki_diff · wiki_restore ·        git log and show; a page as of a
              wiki_revert · finish                             commit; undo one; the message

What is not the keeper's is refused: the engines, the cases summaries, anything
outside ``domains/``; a passage not in the page exactly once; a page left without
its title or body; a case page not named by its case id, for a case another
domain has a page for, or without its Key points, Findings and Patterns.

:func:`for_keeper` wraps them for the agents SDK, bound to one Keeper, as
agent.tools.for_agent does the discovery agent's. Tests call the functions directly.
"""

from __future__ import annotations

import json
from typing import Any

from agent.tools.wiki_tools.edit import (add_to_page, change_in_page, create_page, delete_from_page,
                                         delete_page, move_page)
from agent.tools.wiki_tools.keeper import Keeper
from agent.tools.wiki_tools.read import wiki_read, wiki_search
from agent.tools.wiki_tools.versions import (finish, wiki_diff, wiki_history, wiki_restore,
                                             wiki_revert)

__all__ = ["Keeper", "TOOLS", "END_TOOL", "for_keeper", "wiki_search", "wiki_read", "create_page",
           "add_to_page", "change_in_page", "delete_from_page", "delete_page", "move_page",
           "wiki_history", "wiki_diff", "wiki_restore", "wiki_revert", "finish"]

TOOLS = ("wiki_search", "wiki_read", "create_page", "add_to_page", "change_in_page",
         "delete_from_page", "delete_page", "move_page", "wiki_history", "wiki_diff",
         "wiki_restore", "wiki_revert", "finish")
END_TOOL = "finish"


def _dump(value: Any) -> str:
    return json.dumps(value, default=str)


def for_keeper(keeper: Keeper) -> list[Any]:
    """The keeper's tools as SDK function tools, bound to one Keeper."""
    from agents import function_tool

    @function_tool(name_override="wiki_search")
    def _wiki_search(query: str, domain: str = "", kind: str = "") -> str:
        """Find pages by their words; returns ids, titles, summaries and matching lines.

        Args:
            query: words, e.g. "bust out cash advance".
            domain: a domain or a subdomain, e.g. fraud or customer_management/cdss_us_sbs;
                empty = all.
            kind: domain, subdomain, page, case, cases_summary, cognitive or experiment;
                empty = all.
        """
        return _dump(wiki_search(keeper, query, domain, kind))

    @function_tool(name_override="wiki_read")
    def _wiki_read(page: str) -> str:
        """A page's file in full - frontmatter included - and its links.

        Args:
            page: its id, e.g. domains/customer_management/cdss_us_sbs/bust_out.
        """
        return _dump(wiki_read(keeper, page))

    @function_tool(name_override="create_page")
    def _create_page(page: str, title: str, summary: str, body: str) -> str:
        """Create a page no page covers yet.

        Args:
            page: its id, snake_case - domains/<domain>/index or <page> (what holds
                across the domain), domains/<domain>/<subdomain>/index or <page> (one use
                case); the domains are underwriting, customer_management, fraud and
                marketing. A case is domains/<domain>/<subdomain>/cases/<case id>, named by
                the reviewed customer or account, e.g.
                domains/customer_management/cdss_us_sbs/cases/CR-20931.
            title: a plain title.
            summary: one sentence on what the page holds.
            body: its markdown from its # heading on. A case's has three sections, in
                this order: "## Key points" (the case in a few lines), "## Findings" (its
                facts, as reviewed) and "## Patterns" (what it shows of the domain, one
                bullet each: "* **<a short name>** - how it shows").
        """
        return _dump(create_page(keeper, page, title, summary, body))

    @function_tool(name_override="add_to_page")
    def _add_to_page(page: str, text: str, after: str = "") -> str:
        """Add text to a page: right after a passage of it, or at its end.

        Args:
            page: the page's id.
            text: the markdown to add.
            after: a passage of the page, copied exactly, unique in it. Empty = at the end.
        """
        return _dump(add_to_page(keeper, page, text, after))

    @function_tool(name_override="change_in_page")
    def _change_in_page(page: str, old: str, new: str) -> str:
        """Replace one passage of a page - frontmatter lines too (title, summary).

        Args:
            page: the page's id.
            old: the passage, copied exactly from wiki_read, unique in the page.
            new: what replaces it.
        """
        return _dump(change_in_page(keeper, page, old, new))

    @function_tool(name_override="delete_from_page")
    def _delete_from_page(page: str, old: str) -> str:
        """Remove one passage of a page.

        Args:
            page: the page's id.
            old: the passage, copied exactly from wiki_read, unique in the page.
        """
        return _dump(delete_from_page(keeper, page, old))

    @function_tool(name_override="delete_page")
    def _delete_page(page: str, reason: str) -> str:
        """Delete a page the material shows wrong or redundant.

        Args:
            page: the page's id.
            reason: one line on why.
        """
        return _dump(delete_page(keeper, page, reason))

    @function_tool(name_override="move_page")
    def _move_page(page: str, to: str) -> str:
        """Move a page to another id - a case to the domain it belongs to, say. The links
        to it are changed for you.

        Args:
            page: the page's id now.
            to: its new id.
        """
        return _dump(move_page(keeper, page, to))

    @function_tool(name_override="wiki_history")
    def _wiki_history(page: str = "", n: int = 15) -> str:
        """The wiki's commits, latest first - each ingest is one.

        Args:
            page: only the commits that touched this page; empty = all.
            n: how many, at most 50.
        """
        return _dump(wiki_history(keeper, page, n))

    @function_tool(name_override="wiki_diff")
    def _wiki_diff(rev: str, page: str = "") -> str:
        """What a commit changed.

        Args:
            rev: the commit, as wiki_history gives it.
            page: only this page's changes; empty = all.
        """
        return _dump(wiki_diff(keeper, rev, page))

    @function_tool(name_override="wiki_restore")
    def _wiki_restore(page: str, rev: str) -> str:
        """Put a page back as it was as of a commit - a page a commit deleted is there as
        of the commit before it (<rev>^).

        Args:
            page: the page's id.
            rev: the commit.
        """
        return _dump(wiki_restore(keeper, page, rev))

    @function_tool(name_override="wiki_revert")
    def _wiki_revert(rev: str) -> str:
        """Undo the page edits of one earlier commit.

        Args:
            rev: the commit, as wiki_history gives it.
        """
        return _dump(wiki_revert(keeper, rev))

    @function_tool(name_override="finish")
    def _finish(message: str) -> str:
        """The raw file is folded in: end with the commit message.

        Args:
            message: one line - what changed in the wiki, and why.
        """
        return _dump(finish(keeper, message))

    tools = {"wiki_search": _wiki_search, "wiki_read": _wiki_read, "create_page": _create_page,
             "add_to_page": _add_to_page, "change_in_page": _change_in_page,
             "delete_from_page": _delete_from_page, "delete_page": _delete_page,
             "move_page": _move_page, "wiki_history": _wiki_history, "wiki_diff": _wiki_diff,
             "wiki_restore": _wiki_restore, "wiki_revert": _wiki_revert, "finish": _finish}
    return [tools[name] for name in TOOLS]
