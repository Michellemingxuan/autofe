"""The idea engines' wiki: what the agent knows beyond the data, as markdown pages.

    wiki/                          its own git repository, kept by the agent
      raw/                         the user's material - the only folder users touch
      domains/<domain>/            underwriting, customer_management, fraud, marketing
        index.md                   the domain: overview, links to its subdomains
        <page>.md                  what holds across its subdomains
        <subdomain>/               one use case - customer_management/cdss_us_sbs
          index.md                 the subdomain: overview, links to its pages and cases
          <page>.md                a subpage - bust_out, collections ...
          cases/
            summary.md             what the cases teach together: no ids or numbers
            <case id>.md           one reviewed case, named by its id
      cognitive/<engine>.md        the idea engines - copied from the code (wiki_seed/)
      experiments/<model>/<page>   reflections on a model's results
      index.json                   the catalog: id, kind, domain, title, summary, links

A page's id is its path without ``.md`` (``domains/customer_management/cdss_us_sbs/
bust_out``); its ``domain`` is where it is - ``customer_management``, or
``customer_management/cdss_us_sbs`` for a subdomain's page. Its
frontmatter gives a ``title`` and a one-line ``summary``. A case page is named by
its case id - the reviewed customer or account (``cases/CR-20931``) - so each case
has one page and ``[[CR-20931]]`` links it. Its body has three sections: **Key
points**, **Findings** (the facts of the case) and **Patterns** (what it shows of the
domain, one named bullet each - the themes open explorations draw). ``[[id]]`` in the body
links another page - by id, by a domain's or subdomain's name (its index) or by a
page name that is unique in the wiki or in the linking page's subdomain or domain.

**Access** is per model (``agent.wiki``): its home - a subdomain - is open; the
rest is open but for the single cases of other subdomains - their
``cases/summary.md`` says what the cases teach - unless ``cross_domain_cases`` is
``open``. Cognitive pages are
open to every model, experiment pages to their own.

**Versions.** The wiki is a git repository of its own, outside autofe's: every
ingest is one commit (agent.wiki_ingest), and the agent that keeps it reads the
history and restores from it. :func:`ensure_wiki` makes it on first use from
``src/agent/wiki_seed`` - the engines and a README - and keeps the engines in
step with the code.

The agent reads it as Claude Code reads a codebase - no embeddings: the page
index is in its brief, :func:`search` ranks pages by their words (BM25), and
:func:`read` gives a page with its links to follow.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["WIKI_DIR", "SEED_DIR", "Page", "Access", "pages", "index", "search", "read",
           "access_for", "brief_index", "frontmatter", "ensure_wiki", "git", "CASE_SECTIONS",
           "sections", "patterns", "DOMAINS"]

WIKI_DIR = Path(__file__).resolve().parents[2] / "wiki"
SEED_DIR = Path(__file__).parent / "wiki_seed"
READ_CHARS = 12_000          # a page read at once, at most
INDEX_LISTED = 60            # pages listed in the brief, at most - search finds the rest
CASE_SECTIONS = ("Key points", "Findings", "Patterns")    # a case page's, in this order
DOMAINS = ("underwriting", "customer_management", "fraud", "marketing")   # each has subdomains

_LINK = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from has in is it its of on or that the "
                  "their they this to was what when which who with".split())


def frontmatter(text: str) -> tuple[dict[str, str], str]:
    """A page's ``key: value`` frontmatter and its body."""
    if not text.startswith("---"):
        return {}, text
    head, _, body = text[3:].partition("\n---")
    meta = {}
    for line in head.strip().splitlines():
        key, _, value = line.partition(":")
        if key.strip():
            meta[key.strip()] = value.strip()
    return meta, body.strip()


@dataclass
class Page:
    id: str
    kind: str                    # domain | subdomain | page | case | cases_summary | cognitive | experiment
    domain: str | None           # <domain> or <domain>/<subdomain>; an experiment's model
    title: str
    summary: str
    body: str
    links: list[str] = field(default_factory=list)   # as written; resolved by read()
    case_id: str | None = None   # a case's: the reviewed customer or account it is named by

    def entry(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "domain": self.domain, "title": self.title,
                "summary": self.summary, "links": self.links,
                **({"case_id": self.case_id} if self.case_id else {})}


def git(root: Path, *args: str, check: bool = True) -> str:
    """Run git in the wiki's repository; its output."""
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and done.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {(done.stderr or done.stdout).strip()}")
    return done.stdout


def ensure_wiki(root: Path = WIKI_DIR) -> Path:
    """The wiki as a git repository: made from the seed on first use - pages already
    there are kept - and its engines copied from the code when they changed."""
    root = Path(root)
    fresh = not (root / ".git").exists()
    added = []                                  # seed files the wiki did not have
    for src in SEED_DIR.rglob("*"):
        dst = root / src.relative_to(SEED_DIR)
        if src.is_file() and (not dst.exists() or (dst.parts[len(root.parts)] == "cognitive"
                                                   and dst.read_bytes() != src.read_bytes())):
            if not dst.exists():
                added.append(str(src.relative_to(SEED_DIR)))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    (root / "raw").mkdir(exist_ok=True)
    if fresh:
        git(root, "init", "-q")
        git(root, "config", "user.name", "autofe wiki keeper")
        git(root, "config", "user.email", "wiki@autofe.local")
    if fresh:
        index(root)
    paths = ["."] if fresh else ["cognitive", *added]
    if git(root, "status", "--porcelain", "--", *paths).strip():
        git(root, "add", "-A", "--", *paths)
        git(root, "commit", "-q", "-m", "Seed the wiki" if fresh else "Update from the seed in the code",
            "--", *paths)                       # those alone, not what else is staged
    return root


def sections(body: str) -> dict[str, str]:
    """A page's ``## `` sections, by heading, in order."""
    found = re.split(r"^##\s+(.+?)\s*$", body, flags=re.M)
    return {head: text.strip() for head, text in zip(found[1::2], found[2::2])}


def patterns(page: Page) -> list[dict[str, str]]:
    """A case's patterns: each bullet of its Patterns section, by its bold name
    (``* **Minimum-pay slide** - ...``) or, with none, its first words."""
    out = []
    items = re.split(r"^\s*[-*]\s+", sections(page.body).get("Patterns", ""), flags=re.M)
    for item in items[1:]:                                # [0]: any text before the bullets
        item = " ".join(item.split())
        if not item:
            continue
        named = re.match(r"\*\*(.+?)\*\*", item)
        name = named.group(1) if named else " ".join(item.split()[:8])
        out.append({"theme": name.strip(" .:-"), "pattern": item, "case": page.id})
    return out


def _kind(parts: tuple[str, ...]) -> tuple[str, str | None] | None:
    """A page's kind and domain from its path, or None for a file outside the layout."""
    if parts[0] == "domains" and len(parts) == 3:                 # domains/<domain>/<page>
        return ("domain" if parts[2] == "index" else "page"), parts[1]
    if parts[0] == "domains" and len(parts) == 4 and "cases" not in parts[2:]:
        return ("subdomain" if parts[3] == "index" else "page"), f"{parts[1]}/{parts[2]}"
    if parts[0] == "domains" and len(parts) == 5 and parts[3] == "cases" and parts[2] != "cases":
        return ("cases_summary" if parts[4] == "summary" else "case"), f"{parts[1]}/{parts[2]}"
    if parts[0] == "cognitive" and len(parts) == 2:
        return "cognitive", None
    if parts[0] == "experiments" and len(parts) == 3:
        return "experiment", parts[1]
    return None


def _summary(meta: dict[str, str], body: str) -> str:
    if meta.get("summary") or meta.get("question"):
        return meta.get("summary") or meta["question"]
    for para in re.split(r"\n\s*\n", body):
        para = " ".join(para.split())
        if para and not para.startswith(("#", "|", "-", "*", ">")):
            return re.split(r"(?<=[.!?])\s", para)[0][:200]
    return ""


def pages(root: Path = WIKI_DIR) -> dict[str, Page]:
    """Every page of the wiki, by id. The raw folder is not part of it."""
    out = {}
    for path in sorted(Path(root).rglob("*.md")):
        parts = path.relative_to(root).with_suffix("").parts
        kind = _kind(parts)
        if kind is None:
            continue
        meta, body = frontmatter(path.read_text())
        heading = re.search(r"^#\s+(.+)$", body, re.M)
        title = meta.get("title") or (heading.group(1).strip() if heading else parts[-1])
        pid = "/".join(parts)
        out[pid] = Page(pid, kind[0], kind[1], title, _summary(meta, body), body,
                        list(dict.fromkeys(m.strip() for m in _LINK.findall(body))),
                        parts[-1] if kind[0] == "case" else None)
    return out


def index(root: Path = WIKI_DIR) -> list[dict[str, Any]]:
    """The catalog of every page, written to ``index.json`` when it changed."""
    entries = [p.entry() for p in pages(root).values()]
    path = Path(root) / "index.json"
    text = json.dumps(entries, indent=2) + "\n"
    if path.parent.exists() and (not path.exists() or path.read_text() != text):
        path.write_text(text)
    return entries


def _resolve(link: str, all_pages: dict[str, Page], near: str | None = None) -> str | None:
    """The page a ``[[link]]`` names: an id, an id without ``domains/``, a domain (its
    index), or a page name -
    unique in the wiki, or in the linking page's domain (``near``)."""
    link = link.strip().strip("/").removesuffix(".md")
    if link in all_pages:
        return link
    if f"domains/{link}" in all_pages:                # <domain>/<page>, <domain>/cases/<case>
        return f"domains/{link}"
    if f"domains/{link}/index" in all_pages:
        return f"domains/{link}/index"
    named = ([pid for pid in all_pages if pid.rsplit("/", 1)[-1] == link]
             or [pid for pid in all_pages if pid.endswith(f"/{link}/index")])  # a subdomain's name
    for area in ([near, near.split("/")[0]] if near else []):       # its subdomain, its domain
        if len(named) > 1:
            named = [pid for pid in named if pid.startswith(f"domains/{area}/")] or named
    return named[0] if len(named) == 1 else None


def within(page: Page, area: str | None) -> bool:
    """Whether a page is in a domain or subdomain - a domain holds its subdomains'."""
    return bool(area) and (page.domain == area or (page.domain or "").startswith(f"{area}/"))


@dataclass
class Access:
    """What one model may read: its home domain, and how much of the others."""
    home: str | None = None
    model: str | None = None
    cross_domain_cases: str = "summary"       # summary | open

    def refusal(self, page: Page) -> str | None:
        """Why the model may not read a page, or None when it may."""
        if page.kind == "experiment" and page.domain != self.model:
            return f"{page.id} reflects on the model {page.domain!r}, not this one"
        if page.kind == "case" and not within(page, self.home) and self.cross_domain_cases != "open":
            return (f"{page.id} is a case of {page.domain!r}, a cross-domain for this model "
                    f"(home: {self.home or 'none set'}) - its cases are closed. Read "
                    f"domains/{page.domain}/cases/summary for what they teach together")
        return None


def access_for(cfg: Any) -> Access:
    """A model's access, from its config's ``agent.wiki``."""
    wiki = cfg.agent.wiki
    return Access(home=wiki.domain, model=cfg.run.name, cross_domain_cases=wiki.cross_domain_cases)


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def search(query: str, access: Access, domain: str = "", kind: str = "", limit: int = 8,
           root: Path = WIKI_DIR) -> dict[str, Any]:
    """The pages the model may read that best match a query, by BM25 over their
    title, summary and body (the title and summary count three and two times)."""
    terms = list(dict.fromkeys(_words(query)))
    if not terms:
        return {"ok": False, "error": "give some words to search for, e.g. \"bust out spend\""}
    found = [p for p in pages(root).values() if access.refusal(p) is None
             and (not domain or within(p, domain.strip("/"))) and (not kind or p.kind == kind)]
    docs = [Counter(_words(f"{p.title} " * 3 + f"{p.summary} " * 2 + (p.case_id or "")
                           + " " + p.body)) for p in found]
    if not docs:
        return {"ok": True, "query": query, "results": [], "note": "no readable page matches the filters"}
    avg = sum(sum(d.values()) for d in docs) / len(docs)
    df = {t: sum(t in d for d in docs) for t in terms}
    scored = []
    for page, doc in zip(found, docs):
        size = sum(doc.values())
        score = sum(math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
                    * doc[t] * 2.2 / (doc[t] + 1.2 * (0.25 + 0.75 * size / avg))
                    for t in terms if doc[t])
        if score > 0:
            scored.append((score, page))
    scored.sort(key=lambda s: -s[0])
    return {"ok": True, "query": query, "results": [
        {"page": p.id, "title": p.title, "kind": p.kind, "domain": p.domain, "summary": p.summary,
         "score": round(s, 2), "lines": _lines(p.body, terms)} for s, p in scored[:limit]],
        "total_matches": len(scored)}


def _lines(body: str, terms: list[str], n: int = 2) -> list[str]:
    """The body's lines with the most query words - where the page says it."""
    ranked = sorted(((sum(t in _words(line) for t in terms), i, line.strip())
                     for i, line in enumerate(body.splitlines()) if line.strip()),
                    key=lambda r: (-r[0], r[1]))
    return [line[:240] for hits, _, line in ranked[:n] if hits]


def read(page: str, access: Access, root: Path = WIKI_DIR) -> dict[str, Any]:
    """One page in full, with the pages it links - to follow, or a refusal."""
    all_pages = pages(root)
    pid = _resolve(page, all_pages)
    if pid is None:
        return {"ok": False, "error": f"no page {page!r} - wiki_search finds pages by their words"}
    found = all_pages[pid]
    if (why := access.refusal(found)):
        return {"ok": False, "error": why}
    links = []
    for link in found.links:
        target = _resolve(link, all_pages, found.domain)
        if target is None:
            links.append({"link": link, "page": None, "note": "not written yet"})
        else:
            closed = access.refusal(all_pages[target])
            links.append({"page": target, "title": all_pages[target].title,
                          **({"closed": True} if closed else {})})
    return {"ok": True, "page": pid, "title": found.title, "kind": found.kind,
            "domain": found.domain, **({"case_id": found.case_id} if found.case_id else {}),
            "text": found.body[:READ_CHARS],
            **({"truncated": True} if len(found.body) > READ_CHARS else {}), "links": links}


def brief_index(access: Access, root: Path = WIKI_DIR) -> str:
    """The pages the model may read, one line each - home domain first, then the
    other domains, then cognitive and experiment pages."""
    readable = [p for p in pages(root).values()
                if access.refusal(p) is None and p.kind != "cognitive"]
    order = {"domain": 0, "subdomain": 1, "page": 2, "cases_summary": 3, "case": 4, "experiment": 5}
    home = access.home.split("/")[0] if access.home else None
    readable.sort(key=lambda p: (p.kind == "experiment", not within(p, access.home),
                                 (p.domain or "").split("/")[0] != home,
                                 p.domain or "", order[p.kind], p.id))
    lines = [f"* `{p.id}` - {p.title}" + (f": {p.summary}" if p.summary else "")
             for p in readable[:INDEX_LISTED]]
    if len(readable) > INDEX_LISTED:
        lines.append(f"* ... and {len(readable) - INDEX_LISTED} more - wiki_search finds them")
    return "\n".join(lines)
