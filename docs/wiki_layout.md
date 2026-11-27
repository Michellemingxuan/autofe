# Idea-engine wiki: layout

```
wiki/                                    its own git repository - kept by the agent, outside autofe's
├── raw/                                 the only folder users touch: drop material here
│   ├── <anything>.md                    hashed, so each version is ingested once
│   └── .ingested.json                   each file's hash, what it became, its commit
│
├── domains/                             four domains, fixed; each has subdomains - one use case each
│   ├── customer_management/
│   │   ├── index.md                     the domain: overview, links to its subdomains
│   │   ├── <page>.md                    what holds across its subdomains
│   │   └── cdss_us_sbs/                 a subdomain: one use case - a product, a portfolio, a model
│   │       ├── index.md                 the subdomain: overview, links to its pages and cases
│   │       ├── bust_out.md              CBO (credit bust-out); links to fraud
│   │       ├── risk_signals.md          the signals of deterioration, recurring and rare
│   │       ├── reading_signals.md       how to read them: trajectory, timing, convergence
│   │       ├── payment_behaviour.md     how paying changes on the way to default
│   │       └── cases/
│   │           ├── summary.md           what the cases teach together: no ids, names or numbers
│   │           └── <case id>.md         one reviewed case, named by its id: CR-20931.md
│   │
│   ├── underwriting/                    index.md, and subdomains as material arrives
│   ├── fraud/
│   └── marketing/
│
├── cognitive/                           the idea engines - copied from src/agent/wiki_seed/
│   ├── mechanism.md                     (the code owns them; the wiki follows the code)
│   ├── contrastive.md
│   ├── analogy.md
│   ├── morphological.md
│   └── others.md
│
├── experiments/                         reflections on evaluation results (trigger to be decided)
│   └── <model>/
│       └── <page>.md                    what helped, what did not, and why
│
├── index.json                           catalog: id, kind, domain (or domain/subdomain), title, summary, links, case_id
└── log.md                               what each ingest changed, and why
```

On first use the wiki is made from `src/agent/wiki_seed/` (the engines, the four
domains' index pages and a README) and committed as "Seed the wiki"; pages
already in the folder are kept, and a seed file the wiki lacks is added later.

The domains are fixed: **underwriting**, **customer_management**, **fraud** and
**marketing**. The keeper adds subdomains - one per use case, e.g.
`customer_management/cdss_us_sbs` - each with its index linked from its domain's.
A page directly under a domain holds what is true across its subdomains; cases
are always in a subdomain.
autofe's own repository ignores `wiki/`, so the wiki's pages - which may hold
material from private use cases - never reach it.

A page is markdown with frontmatter: `title` and `summary`. A case page is named
by its case id - the reviewed customer or account, as the material gives it - so
a case has one page in the whole wiki, and `[[CR-20931]]` links it. A pattern
with no case id behind it is a subpage, not a case. Links are `[[id]]`: the path
without `.md`, with or without `domains/`, a domain's or subdomain's name (its
index), or a page name (the linking page's subdomain first, then its domain).

A case page has three sections, in this order:

```markdown
# <title>

## Key points        the case in a few lines: what happened, what the model saw
## Findings          the facts, as reviewed: the behaviour, its timing, the signals
## Patterns          what it shows of the domain, one bullet each:
* **<a short name>** - how it shows, and what would reveal it
```

The patterns are themes: an open exploration (a run with no direction) draws
the patterns of the cases its model may read - its home subdomain's - before the
themes made from the task description, the least explored first, and reads the
case for its findings. The cases summary is made from the key points and the
patterns only, not the findings.

## Access, per model

Each model has a home - a subdomain; every other subdomain is a cross-domain. Set
in the model's config:

```yaml
agent:
  wiki:
    domain: customer_management/cdss_us_sbs   # the home: <domain>/<subdomain>
    cross_domain_cases: summary    # summary | open
```

| Page                          | Home        | Cross-domain        |
|-------------------------------|-------------|---------------------|
| domain and subdomain pages    | open        | open                |
| `cases/summary.md`            | open        | open                |
| `cases/<case id>.md`          | open        | closed (an option)  |
| `cognitive/`                  | open        | -                   |
| `experiments/<model>/`        | own model   | -                   |

## Discovery agent's tools

- **Index in the brief:** the titles and one-line summaries of the pages the model may read.
- **`wiki_search(query, domain?, kind?)`:** (domain: a domain or a subdomain) ranked word matching over title, summary, case id and body, filtered by access; returns the top pages with snippets.
- **`wiki_read(page)`:** the full page, its case id and its `[[links]]`, capped in size; a closed page returns a short refusal.

## Ingest: the keeper

`src/agent/wiki_ingest.py` runs it; its tools are in `src/agent/tools/wiki_tools/`
(`read.py`, `edit.py`, `versions.py`, bound for the SDK by `for_keeper`). It runs in the background when the server starts and
when a run starts, or by hand: `PYTHONPATH=src python -m agent.wiki_ingest -c <config>`.

Edits made by hand outside `raw/` are committed first, as such, so the history
keeps them. Then, for each raw file not ingested in its current version, the
**keeper** - a model with tools - is given the file, the subdomains, the links not
written yet and the eight pages most like it, and edits the wiki as an editor:

| Tool | What it does |
|---|---|
| `wiki_search`, `wiki_read` | find pages; read one as a file, frontmatter included |
| `create_page` | a page no page covers; a case is named by its case id |
| `add_to_page` | add text after a passage of a page, or at its end |
| `change_in_page` | replace one passage - frontmatter lines too |
| `delete_from_page` | remove one passage |
| `delete_page` | a page shown wrong or redundant; says which pages link to it |
| `move_page` | a page to another id - a case to its subdomain - with the links to it changed |
| `wiki_history` | `git log`, of the wiki or of one page |
| `wiki_diff` | `git show` of a commit, or of one page in it |
| `wiki_restore` | a page as of a commit (a deleted page: as of `<rev>^`) |
| `wiki_revert` | undo the page edits of one commit (raw files and records stay) |
| `finish` | the commit message |

The tools refuse what is not the keeper's: the engines, the cases summaries,
anything outside `domains/`; a passage that is not in the page exactly once; a
page left without its title or body; a case page not named by a case id, or for
a case another subdomain has a page for.

Then the harness updates `cases/summary.md` of each subdomain whose cases changed -
one model call that starts from the page as it is (keeps what holds, adds what the
cases now show, drops what they contradict), with no ids, names or numbers -
rewrites `index.json`, appends the edits to `log.md` and commits it all: **one
commit per raw file**, its message the keeper's. A file that fails leaves no
edits - the pages go back to the last commit - and is recorded with its error,
then tried again once it changes. The same text under another name is skipped.
