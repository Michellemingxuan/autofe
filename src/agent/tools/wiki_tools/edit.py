"""Edit pages as an editor does: add, change or delete a passage; create, delete or
move a page. A passage is found by its exact text, which must be in the page once."""

from __future__ import annotations

import re
from typing import Any

from agent.tools.wiki_tools.keeper import Keeper, one_line
from agent.wiki import _kind, _resolve, pages, within

__all__ = ["create_page", "add_to_page", "change_in_page", "delete_from_page", "delete_page",
           "move_page"]


def _once(text: str, part: str, pid: str) -> str | None:
    n = text.count(part) if part else 0
    if n == 1:
        return None
    return (f"the text to find is not in {pid} - copy it exactly from wiki_read" if n == 0
            else f"the text to find is {n} times in {pid} - give more of it, to be unique")


def _is_case(pid: str) -> bool:
    return _kind(tuple(pid.split("/")))[0] == "case"


def _case_taken(keeper: Keeper, pid: str, but: str = "") -> str | None:
    other = keeper.case_elsewhere(pid, but) if _is_case(pid) else None
    return other and (f"case {pid.rsplit('/', 1)[-1]} has its page already: {other} - edit "
                      "that page, or move it with move_page")


def create_page(keeper: Keeper, page: str, title: str, summary: str, body: str) -> dict[str, Any]:
    pid, problem = keeper.editable(page)
    if problem or (problem := _case_taken(keeper, pid)):
        return {"ok": False, "error": problem}
    if keeper.path(pid).exists():
        return {"ok": False, "error": f"{pid} exists - read it and edit it"}
    head = f"---\ntitle: {one_line(title)}\nsummary: {one_line(summary)}\n---\n"
    return keeper.save(pid, head + body.strip(), "created")


def add_to_page(keeper: Keeper, page: str, text: str, after: str = "") -> dict[str, Any]:
    pid, problem, current = keeper.existing(page)
    if problem or (after and (problem := _once(current, after, pid))):
        return {"ok": False, "error": problem}
    if not text.strip():
        return {"ok": False, "error": "nothing to add"}
    new = (current.replace(after, after.rstrip("\n") + "\n" + text.strip("\n"), 1) if after
           else current.rstrip() + "\n\n" + text.strip())
    return keeper.save(pid, new, "added")


def change_in_page(keeper: Keeper, page: str, old: str, new: str) -> dict[str, Any]:
    pid, problem, current = keeper.existing(page)
    if problem or (problem := _once(current, old, pid)):
        return {"ok": False, "error": problem}
    if not new.strip():
        return {"ok": False, "error": "to remove text, use delete_from_page"}
    return keeper.save(pid, current.replace(old, new, 1), "changed")


def delete_from_page(keeper: Keeper, page: str, old: str) -> dict[str, Any]:
    pid, problem, current = keeper.existing(page)
    if problem or (problem := _once(current, old, pid)):
        return {"ok": False, "error": problem}
    return keeper.save(pid, re.sub(r"\n{3,}", "\n\n", current.replace(old, "", 1)), "deleted")


def delete_page(keeper: Keeper, page: str, reason: str) -> dict[str, Any]:
    pid, problem, _ = keeper.existing(page)
    if problem:
        return {"ok": False, "error": problem}
    area = "/".join(pid.split("/")[1:-1])                  # the domain or subdomain it indexes
    before = pages(keeper.root)
    if pid.endswith("/index") and any(within(p, area) and p.id != pid and p.kind != "cases_summary"
                                      for p in before.values()):
        return {"ok": False, "error": f"{pid} indexes pages of {area} - delete them first"}
    linked = [p.id for p in before.values() if p.id != pid
              and any(_resolve(link, before, p.domain) == pid for link in p.links)]
    keeper.path(pid).unlink()
    keeper.record("removed", pid, reason)
    return {"ok": True, "page": pid, "removed": True,
            **({"linked_from": linked, "next": "remove or change the links to it"} if linked else {})}


def move_page(keeper: Keeper, page: str, to: str) -> dict[str, Any]:
    """A page to another id - another domain, say - with the links to it changed."""
    pid, problem, _ = keeper.existing(page)
    new, problem2 = keeper.editable(to)
    if problem or problem2 or (problem2 := _case_taken(keeper, new, but=pid)):
        return {"ok": False, "error": problem or problem2}
    if keeper.path(new).exists():
        return {"ok": False, "error": f"{new} exists - fold the page into it and delete this one"}
    if _is_case(pid) != _is_case(new):
        return {"ok": False, "error": "a case moves to cases/ of a domain, and only a case does"}
    before = pages(keeper.root)
    keeper.path(new).parent.mkdir(parents=True, exist_ok=True)
    keeper.path(pid).rename(keeper.path(new))
    keeper.record("moved", pid, f"to {new}")
    relinked = []
    for p in before.values():
        if p.id == pid:
            continue
        path = keeper.path(p.id)
        current = changed = path.read_text()
        for link in p.links:
            if _resolve(link, before, p.domain) == pid:
                changed = changed.replace(f"[[{link}]]", f"[[{new.removeprefix('domains/')}]]")
        if changed != current:
            path.write_text(changed)
            relinked.append(p.id)
    if relinked:
        keeper.record("relinked", ", ".join(relinked), f"to {new}")
    return {"ok": True, "page": new, "moved_from": pid, "relinked": relinked}
