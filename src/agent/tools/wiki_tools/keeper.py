"""The keeper's state for one raw file, and the checks every edit goes through."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent.wiki import CASE_SECTIONS, DOMAINS, _kind, frontmatter, pages, sections

__all__ = ["Keeper"]

_DOMAIN = re.compile(r"[a-z][a-z0-9_]{0,40}")
_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")     # a case page's name


def one_line(text: str) -> str:
    return " ".join(str(text).split())


def _case_problem(body: str) -> str | None:
    """Why a case's body lacks its sections - Key points, Findings, Patterns, in that
    order, each written, the patterns as bullets - or None."""
    have = [h for h in sections(body) if h in CASE_SECTIONS]
    if have != list(CASE_SECTIONS):
        return ("a case has three sections, in this order: "
                + ", ".join(f"## {h}" for h in CASE_SECTIONS) + f" (it has: {', '.join(have) or 'none'})")
    empty = [h for h in CASE_SECTIONS if not sections(body)[h]]
    if empty:
        return f"write its {', '.join(empty)}"
    if not re.search(r"^\s*[-*]\s+\S", sections(body)["Patterns"], re.M):
        return "write its Patterns as bullets: * **<a short name>** - how it shows in the domain"
    return None


class Keeper:
    """One raw file's work on the wiki: where the wiki is, what was changed, and the
    commit message once the keeper finishes."""

    def __init__(self, root: Path, raw_name: str):
        self.root, self.raw_name = Path(root), raw_name
        self.edits: list[dict[str, str]] = []
        self.message: str | None = None

    def path(self, pid: str) -> Path:
        return self.root / f"{pid}.md"

    def editable(self, page: str) -> tuple[str, str | None]:
        """The page's id, and why the keeper may not write it (None when it may)."""
        pid = str(page).strip().strip("/").removesuffix(".md")
        parts = tuple(pid.split("/"))
        kind = _kind(parts) if parts and parts[0] == "domains" else None
        if kind is None:
            return pid, (f"{pid!r} is not a domain page: domains/<domain>/index or <page>, "
                         "domains/<domain>/<subdomain>/index or <page>, or "
                         "domains/<domain>/<subdomain>/cases/<case id>")
        if parts[1] not in DOMAINS:
            return pid, (f"{pid!r}: the domains are {', '.join(DOMAINS)} - file it in a "
                         "subdomain of the one it belongs to")
        if kind[0] == "cases_summary":
            return pid, f"{pid} is made from the subdomain's cases - edit the cases instead"
        if len(parts) > 3 and not _DOMAIN.fullmatch(parts[2]):
            return pid, f"{pid!r}: use a snake_case name for the subdomain"
        if kind[0] == "case":
            if not _CASE_ID.fullmatch(parts[-1]):
                return pid, (f"{pid!r}: a case page is named by its case id - the reviewed customer "
                             "or account, e.g. domains/customer_management/cdss_us_sbs/cases/CR-20931")
        elif not _NAME.fullmatch(parts[-1]):
            return pid, f"{pid!r}: use snake_case names"
        return pid, None

    def case_elsewhere(self, pid: str, but: str = "") -> str | None:
        """The page another domain has for this case id, if any - a case has one page."""
        name = pid.rsplit("/", 1)[-1].lower()
        return next((p.id for p in pages(self.root).values() if p.kind == "case"
                     and p.id not in (pid, but) and p.case_id.lower() == name), None)

    def existing(self, page: str) -> tuple[str, str | None, str]:
        """An editable page that exists: its id, why not, and its text."""
        pid, problem = self.editable(page)
        if not problem and not self.path(pid).exists():
            problem = f"no page {pid} - create_page makes one"
        return pid, problem, ("" if problem else self.path(pid).read_text())

    def record(self, op: str, pid: str, note: str = "") -> None:
        self.edits.append({"op": op, "page": pid, **({"note": note} if note else {})})

    def save(self, pid: str, text: str, op: str, note: str = "") -> dict[str, Any]:
        """Write a page, if it keeps its title and body - and a case, its three sections."""
        meta, body = frontmatter(text)
        if not meta.get("title") or not body.strip():
            return {"ok": False, "error": f"{pid} would lose its title or its body - not saved"}
        if _kind(tuple(pid.split("/")))[0] == "case" and (problem := _case_problem(body)):
            return {"ok": False, "error": f"{pid}: {problem} - not saved"}
        path = self.path(pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n")
        self.record(op, pid, note)
        return {"ok": True, "page": pid, op: True}
