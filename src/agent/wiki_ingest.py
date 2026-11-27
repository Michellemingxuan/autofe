"""Turn what the user drops in ``wiki/raw/`` into wiki pages - by an agent that keeps them.

Users touch only ``raw/``; the rest of the wiki is the keeper's. For each raw file
not ingested in its current version (by its hash), the **keeper** - a model with
tools, like the discovery agent - is given the file, the domains and the pages
most like it (BM25, agent.wiki), and works on the pages as an editor does, with
the tools in agent.tools.wiki_tools: search and read; add, change or delete a
passage; create, delete or move a page; read the history, restore a page, undo
a commit; finish with the commit message.

Deduplication is the keeper's call: it reads the pages most like the material and
adds to them, or skips, instead of writing the same thing twice. A case page is
named by its case id - the reviewed customer or account - so a case has one page
and is never written twice.

The harness then makes each subdomain's ``cases/summary.md`` again where its cases
changed - one model call that updates the page it was (keeps what still holds, adds
what the cases now show, drops what they contradict), with no ids, names or
numbers, since other subdomains read it in their place - rewrites ``index.json``, logs the edits in ``log.md`` and commits it
all as one commit of the wiki's own repository. A file that fails leaves no edits
(the pages go back to the last commit), is recorded with its error and tried
again once it changes. Edits made by hand outside ``raw/`` are committed first, as
such, so the history keeps them.

The server runs it in the background when it starts and when a run starts:

    PYTHONPATH=src python -m agent.wiki_ingest -c configs/cdss_us_sbs.yaml
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from agent.tools.wiki_tools import END_TOOL, Keeper, for_keeper
from agent.tools.wiki_tools.keeper import one_line
from agent.wiki import (WIKI_DIR, Access, _resolve, ensure_wiki, frontmatter, git, index, pages,
                        search, sections)

__all__ = ["CasesSummary", "pending", "ingest"]

RAW_CHARS = 40_000           # a raw file given at once, at most
SHORTLIST = 8
MAX_TURNS = 60
_NOT_HAND = (":(exclude)raw", ":(exclude)index.json")     # not edits by hand


class CasesSummary(BaseModel):
    summary: str = Field(description="one sentence: what the cases teach together")
    body: str = Field(description="the page's markdown from its # heading on: the patterns "
                                  "across the cases. No ids, names, amounts, dates or other "
                                  "numbers")


KEEP = """You keep a wiki of domain knowledge that an agent reads when it invents
features for default models. Users add material in raw/; you fold it into the
pages. Nobody else edits them, and every change you make is versioned.

Pages, by id (the path without .md). There are four domains - underwriting,
customer_management, fraud and marketing - and each has subdomains: one use case
each, a product, portfolio or model (customer_management/cdss_us_sbs).
* domains/<domain>/index - a domain: an overview, links to its subdomains
* domains/<domain>/<page> - what holds across the domain's subdomains
* domains/<domain>/<subdomain>/index - a subdomain: an overview, its routes and
  concepts, links to its pages and cases
* domains/<domain>/<subdomain>/<page> - a subpage: one concept, mechanism or set of
  signals
* domains/<domain>/<subdomain>/cases/<case id> - one reviewed case, named by its id
  as the material gives it (domains/customer_management/cdss_us_sbs/cases/CR-20931),
  in three sections:
    ## Key points - the case in a few lines: what happened, what the model saw
    ## Findings - the facts, as reviewed: the behaviour, its timing, the signals
    ## Patterns - what it shows of the domain, beyond this one customer: one
       bullet each, "* **<a short name, 3 to 8 words>** - how it shows and what
       would reveal it". These patterns are the themes the discovery agent
       explores: name each as a customer behaviour or a risk mechanism that a
       feature could measure ("Refinancing under strain"), not as this case and
       not as a remark on the model - what the model missed goes in Key points
The cases/summary pages are made from the cases - do not edit them. Cognitive
pages (the idea engines) are not yours to edit either.

How to work:
1. Read the pages most like the material (listed below) and search for more.
2. Fold the material in with small edits: add what is new to the page it belongs
   on, change what it corrects, delete what it shows wrong or what another page
   already says. Create a page only for what no page covers; delete a page that
   has become redundant. Never write the same thing twice.
3. File it where it belongs: the domain - underwriting, customer_management, fraud
   or marketing; there are no others - and the subdomain of the use case it is
   about. A new subdomain gets an index page, linked from its domain's index.
   What holds across a domain's subdomains goes on a page of the domain itself.
   Material about another field (a fraud team's notes, a marketing study) goes in
   that field's domain and is linked from the subdomains it bears on.
4. Each specific case the material describes - a customer or an account that was
   reviewed - is a case page of its own, named by its case id. A case has one page
   only: new material on a known case edits its page, and a case found to belong to
   another subdomain is moved there (move_page), keeping its id. Link it as [[<case id>]].
   A pattern the material gives without a case id goes on a subpage, not in cases/.
   Keep names, addresses and card numbers out of every page; the ids are enough.
5. When the material says an earlier edit was wrong, look at the history
   (wiki_history, wiki_diff) and restore or revert rather than patching blindly.
6. Link pages with [[id]] - a domain's or subdomain's name alone links its index -
   and link a new subpage from its subdomain's index.
7. Write plainly and briefly, for an analyst: what the material teaches, not a copy.
8. finish with a commit message: one line on what changed and why.
"""

SUMMARISE = """You keep the page that says what the cases of `{domain}` teach
together, in a wiki an agent reads when it invents features for default models: the
patterns across the cases, where models were late, what would have shown it - in
plain short points. Models of other subdomains read it in place of the cases, so it
holds no ids, names, merchants, amounts, dates or other numbers.

The cases have changed. Update the page as it is now - do not write it afresh:
keep each point that still holds, with its wording and headings (some come from
case reviews whose cases have no page of their own); add what the cases now show;
change or drop what they now contradict. Leave out how many cases there are - it
is added for you.

The page now:
{current}

The cases - their key points and patterns:
{cases}
"""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _state_path(root: Path) -> Path:
    return root / "raw" / ".ingested.json"


def _state(root: Path) -> dict[str, Any]:
    path = _state_path(root)
    return json.loads(path.read_text()) if path.exists() else {}


def pending(root: Path = WIKI_DIR) -> list[Path]:
    """The raw files not ingested in their current version."""
    raw, done = Path(root) / "raw", _state(Path(root))
    if not raw.exists():
        return []
    return [p for p in sorted(raw.rglob("*")) if p.is_file() and not p.name.startswith(".")
            and done.get(p.relative_to(raw).as_posix(), {}).get("sha")
            != _sha(p.read_text(errors="replace"))]


def _run_keeper(cfg: Any, keeper: Keeper, prompt: str) -> None:
    """The keeper as an SDK agent: its tools bound to the Keeper, until it finishes."""
    from agents import Agent, Runner, StopAtTools

    from agent.llm import build_model
    from agent.llm_trace import TraceHooks, store_for

    agent = Agent(name="wiki_keeper", instructions=KEEP,
                  model=build_model(cfg.agent.llm, log_dir=Path(cfg.agent.run_dir) / "llm_logs"),
                  tools=for_keeper(keeper),
                  tool_use_behavior=StopAtTools(stop_at_tool_names=[END_TOOL]))
    hooks = TraceHooks(store_for(cfg), f"wiki_keeper_{time.strftime('%Y%m%d_%H%M%S')}",
                       "wiki_keeper", kind="wiki_keeper")
    asyncio.run(Runner.run(agent, prompt, max_turns=MAX_TURNS, hooks=hooks))


def _ask(cfg: Any, name: str, prompt: str, output: type[BaseModel]) -> Any:
    """One structured model call, traced like the themes'."""
    from agents import Agent, Runner

    from agent.agent import FirstAnswer
    from agent.llm import build_model
    from agent.llm_trace import TraceHooks, store_for

    agent = Agent(name=name, instructions="Answer with the structured output asked for.",
                  model=build_model(cfg.agent.llm, log_dir=Path(cfg.agent.run_dir) / "llm_logs"),
                  output_type=FirstAnswer(output))
    hooks = TraceHooks(store_for(cfg), f"{name}_{time.strftime('%Y%m%d_%H%M%S')}", name, kind=name)
    return asyncio.run(Runner.run(agent, prompt, max_turns=2, hooks=hooks)).final_output


def _prompt(root: Path, name: str, text: str) -> str:
    """The keeper's task for one raw file: the subdomains, the pages most like it, the file."""
    all_pages = pages(root)
    near = search(text[:4000], Access(cross_domain_cases="open"), limit=SHORTLIST,
                  root=root).get("results", [])
    domains = sorted({p.domain for p in all_pages.values() if p.kind == "subdomain"})
    unwritten = sorted({link for p in all_pages.values() for link in p.links
                        if _resolve(link, all_pages, p.domain) is None})
    listed = "\n".join(f"* `{r['page']}` - {r['title']}: {r['summary']}" for r in near) or "(none)"
    return (f"Subdomains: {', '.join(domains) or 'none yet'}. Linked but not written yet: "
            f"{', '.join(unwritten) or 'none'}.\n\nThe pages most like the material:\n{listed}"
            f"\n\nThe raw material, raw/{name}:\n\n{text[:RAW_CHARS]}")


def _summarise_cases(root: Path, domain: str,
                     summarise: Callable[[str, str, str], CasesSummary]) -> bool:
    """A subdomain's cases/summary.md, updated from its cases - removed with the last case."""
    cases = [p for p in pages(root).values() if p.kind == "case" and p.domain == domain]
    path = root / "domains" / domain / "cases" / "summary.md"
    if not cases:
        path.unlink(missing_ok=True)
        return False
    meta, body = frontmatter(path.read_text()) if path.exists() else ({}, "")
    current = re.sub(r"\n+(?:Cases: |Drawn from ).*\Z", "", body.strip(), flags=re.S) or "(none yet)"
    # Their key points and patterns: the findings hold the facts and numbers, the id the case.
    made = summarise(domain, current, "\n\n".join(
        f"### {p.title}\n{p.summary}\n\n" + "\n\n".join(
            f"## {h}\n{sections(p.body).get(h, '')}" for h in ("Key points", "Patterns"))
        for p in cases))
    title = meta.get("title") or f"What the {domain.rsplit('/', 1)[-1].replace('_', ' ')} cases teach"
    # A count, not links: case pages are named by their ids, which this page leaves out.
    drawn = f"Drawn from {len(cases)} case{'s' if len(cases) > 1 else ''} of [[{domain}]]."
    path.write_text(f"---\ntitle: {title}\nsummary: {one_line(made.summary)}\n---\n"
                    f"{made.body.strip()}\n\n{drawn}\n")
    return True


def _changed(root: Path) -> list[str]:
    """The paths changed since the last commit - deleted ones too, so a case moved out
    of a domain counts as a change to its cases."""
    return [line[3:].strip().strip('"').split(" -> ")[-1]
            for line in git(root, "status", "--porcelain", "-uall").splitlines()]


def _log(root: Path, lines: list[str]) -> None:
    with (root / "log.md").open("a") as out:
        out.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')}\n\n" + "\n".join(lines) + "\n")


def ingest(cfg: Any, root: Path = WIKI_DIR,
           keep: Callable[[Keeper, str], None] | None = None,
           summarise: Callable[[str, str, str], CasesSummary] | None = None) -> list[dict[str, Any]]:
    """Ingest every pending raw file, one commit each; what each became. ``keep`` and
    ``summarise`` stand in for the model calls (tests)."""
    root = ensure_wiki(Path(root))
    keep = keep or (lambda keeper, prompt: _run_keeper(cfg, keeper, prompt))
    summarise = summarise or (lambda domain, current, cases: _ask(
        cfg, "wiki_cases", SUMMARISE.format(domain=domain, current=current, cases=cases), CasesSummary))
    if git(root, "status", "--porcelain", "--", ".", *_NOT_HAND).strip():
        _log(root, ["* edits by hand outside raw/, committed as they were made - the wiki is "
                    "kept by the agent; add material in raw/ instead"])
        git(root, "add", "-A", "--", ".", *_NOT_HAND)
        git(root, "commit", "-q", "-m", "Edits by hand outside raw/ - kept as they were made")
    state, done = _state(root), []
    shas = {v.get("sha"): k for k, v in state.items() if not v.get("error")}
    for path in pending(root):
        name = path.relative_to(root / "raw").as_posix()
        text = path.read_text(errors="replace")
        record: dict[str, Any] = {"sha": _sha(text), "at": time.strftime("%Y-%m-%d %H:%M")}
        if record["sha"] in shas:            # the same file again, under another name
            record["same_as"] = shas[record["sha"]]
            message, lines = f"raw/{name} is raw/{shas[record['sha']]} again - skipped", []
        else:
            keeper = Keeper(root, name)
            try:
                keep(keeper, _prompt(root, name, text))
                cases = sorted({m.group(1) for p in _changed(root) if (m := re.fullmatch(
                    r"domains/([^/]+/[^/]+)/cases/(?!summary\.md$)[^/]+\.md", p))})
                for domain in cases:
                    _summarise_cases(root, domain, summarise)
                record["edits"] = keeper.edits
                message = keeper.message or f"Ingest raw/{name}"
                lines = ([f"  * {e['op']} `{e['page']}`" + (f" - {e['note']}" if e.get("note") else "")
                          for e in keeper.edits]
                         + [f"  * made `domains/{d}/cases/summary` again" for d in cases])
                shas[record["sha"]] = name
            except Exception as error:  # noqa: BLE001 - no edits kept; tried again once it changes
                git(root, "checkout", "HEAD", "--", "domains", check=False)
                git(root, "clean", "-fdq", "--", "domains", check=False)
                record["error"] = f"{type(error).__name__}: {error}"
                message, lines = f"Could not ingest raw/{name}", [f"  * {record['error']}"]
        state[name] = record
        _state_path(root).write_text(json.dumps(state, indent=2) + "\n")
        _log(root, [f"* raw/{name}: {message}", *lines])
        index(root)
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", message[:200], "-m", f"raw/{name}")
        record["commit"] = git(root, "rev-parse", "--short", "HEAD").strip()
        done.append({"file": name, **record})
    return done


def main(argv: list[str] | None = None) -> None:
    import argparse

    from agent import at_project_root
    from validation.config import load_config

    parser = argparse.ArgumentParser(description="Ingest wiki/raw into the wiki's pages.")
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args(argv)
    for item in ingest(load_config(at_project_root(args.config))):
        print(json.dumps(item))


if __name__ == "__main__":
    main()
