# Idea-engine wiki: layout

```
wiki/
├── raw/                                 drop .md files here; ingest reads them automatically
│   └── <anything>.md                    hashed, so each file is ingested once
│
├── domains/
│   ├── credit_risk/                     seeded first
│   │   ├── index.md                     the domain: overview, links to subpages and cases
│   │   ├── bust_out.md                  CBO (credit bust-out); links to fraud/
│   │   ├── risk_signals.md              the signals of deterioration, recurring and rare
│   │   ├── reading_signals.md           how to read them: trajectory, timing, convergence
│   │   ├── payment_behaviour.md         how paying changes on the way to default
│   │   └── cases/
│   │       ├── summary.md               what the cases teach together: patterns, no ids or numbers
│   │       ├── minimum_payer_slide.md   one typical case and its recorded pattern
│   │       └── bust_out_after_line_increase.md
│   │
│   ├── fraud/                           later
│   │   ├── index.md
│   │   └── cases/
│   │       ├── summary.md
│   │       └── <case>.md
│   ├── marketing/                       later
│   └── <outside payments>/              e.g. insurance, telecom churn
│
├── cognitive/                           the idea engines, moved from src/agent/engines/
│   ├── mechanism.md
│   ├── contrastive.md
│   ├── analogy.md
│   ├── morphological.md
│   └── others.md
│
├── experiments/                         reflections on evaluation results (trigger to be decided)
│   └── <model>/
│       └── <page>.md                    what helped, what did not, and why
│
├── index.json                           catalog: id, kind, domain, title, one-line summary, links
└── log.md                               what each ingest changed, and why
```

## Access, per model

Each model has a home domain; every other domain is a cross-domain. Set in the
model's config:

```yaml
agent:
  wiki:
    domain: credit_risk            # the home domain
    cross_domain_cases: summary    # summary | open
```

| Page                          | Home domain | Cross-domain        |
|-------------------------------|-------------|---------------------|
| `index.md`, subpages          | open        | open                |
| `cases/summary.md`            | open        | open                |
| `cases/<case>.md`             | open        | closed (an option)  |
| `cognitive/`                  | open        | -                   |
| `experiments/<model>/`        | own model   | -                   |

## Agent tools

- **Index in the brief:** the titles and one-line summaries of the pages the model may read, from `index.json`.
- **`wiki_search(query, domain?, kind?)`:** ranked word matching over title, summary and body, filtered by access; returns the top pages with snippets.
- **`wiki_read(page)`:** the full page and its `[[links]]`, capped in size; a closed page returns a short refusal.

Ingest is a harness step, not an agent tool: one LLM call per new raw file, given the most similar existing pages, decides whether to create a page, merge into one or skip it as a duplicate, then updates `index.json`, the domain's `cases/summary.md` and `log.md`.
