# Agentic Feature Discovery Design

## Goal

Turn feature discovery from a fixed propose → screen loop into one agent that,
given a **direction** written in plain text, reads the model database and any
additional data sources, writes feature code at increasing levels of reach,
screens it, and - when the data it needs is missing - writes the SQL to pull it.
The user watches every step live, approves the steps that touch data, and
chooses which verified features are evaluated together.

## Scope (v1)

* One agent with several skills; one direction per run; directions run one at a
  time (state is kept per direction so parallel runs can come later).
* L1 and L2 run autonomously. Linkage and L3 data pulls wait for the user.
* L3 SQL is run by the user in BigQuery; the result is dropped into
  `data/<use case>/additional_data/` and picked up from there. A BigQuery client
  tool is a later version.
* Development and tests run on synthetic data generated from the schemas.
* Code lives in this repo: `src/agent/` (backend) and `web/` (frontend).

## The run

```
direction (text)
   │
   ▼
agent ── load_skill(data | feature | evaluate) ──┐
   │                                             │
   ├─ L1a  model DB only            ─┐           │  every script is shown
   ├─ L1b  one source + linkage     ─┼─ run_code ─┼─ to the user as it runs
   ├─ L2c  source + model DB        ─┘   │        │
   │                                     ▼        │
   │                          screen_feature (screen_train → screen_val)
   │                                     │
   │                          pass → verified ledger    fail → reason back to agent
   │
   └─ L3   gap → SQL within CAS scope → ⏸ user approves → user runs in BigQuery
                → file lands in additional_data/ → detected → new cached source
   
stop: K intents used, or the agent calls finish
   │
   ▼
evaluation pipeline (deterministic): leave_one_in vs base on valid/test
   │
   ▼
user ticks combinations (e.g. f1 + f3) → base + f1 + f3 variant → report on
test, the out-of-time hold-out
```

An **intent** is one feature idea the agent pursues to a screen result. K is
purely a cap on intents per direction; the run does not stop early on a count
of verified features.

## Materials

| Material | Source | Shown to the agent as |
| --- | --- | --- |
| Model database | train/valid/test + screen samples | `sample_rows`, `catalog` |
| Column descriptions | `column_descriptions.json` | `catalog` |
| Few-shot rows | `few_shot.csv` | `sample_rows` (a tool, not pasted into the prompt) |
| Task description | `task_context.md` | system prompt |
| Direction | user text, per run | first user message |
| Scope | CAS / CAS_analytics files | `catalog`, each variable flagged `in_model` or `unused_raw` |
| Cached sources | `additional_data/<name>.parquet` + `<name>.json` (schema + description) | `catalog`, `sample_rows` |
| Linkage | `linkage/<source>.py`, generated once, confirmed by the user | used implicitly by `run_code` |

### The model database key

Each row's id is `<customer_id>_<dt>_<marker>`. `dt` is the row's as-of date.
Linkage parses the id into `customer_id`, `as_of = dt`, `marker`, and joins a
source on `customer_id` with **`event_dt < as_of`**, which is the point-in-time
rule every additional-data feature must respect. (Whether events dated exactly
`as_of` are allowed is decided once, at linkage confirmation.)

### Source detection

A watcher scans `data/<use case>/additional_data/` for a data file with a
sibling `<name>.json`. A new pair emits `source_detected`; the frontend shows it
with its schema, and the agent sees it in `catalog` from the next step on. A
source with no confirmed linkage can be read in `probe` mode but not used in a
feature script.

## Skills

Markdown with YAML frontmatter under `src/agent/skills/`, in the format of
`AgenticSys_v2/skills/`. Unlike there, skills load **on demand** through
`load_skill(name)` so the prompt only carries what the current step needs. The
agent may use them together within one intent.

| Skill | Covers |
| --- | --- |
| `data` | Reading the catalog and samples; text-to-PySpark/pandas/SQL; writing linkage and the point-in-time rule; L3 gap analysis → a text description of what is missing + SQL within the CAS scope |
| `feature` | L1a / L1b / L2c recipes; the feature-script contract; what makes a feature redundant with the base set; how to vary ideas across intents |
| `evaluate` | Reading screen results and guard failures; when to refine an idea, when to abandon it, when to escalate to L3 |

## Tools

All `@function_tool`, registered on the one agent.

| Tool | Does | Gate |
| --- | --- | --- |
| `load_skill(name)` | Returns the skill body | - |
| `catalog(query)` | Searches model columns, CAS scope (`in_model` / `unused_raw`) and cached sources with descriptions | - |
| `sample_rows(source, n, columns)` | Few rows from the model DB or a source | - |
| `run_code(code, mode)` | Runs PySpark/pandas in a subprocess with a timeout; returns stdout, error, output schema and head. `mode=probe` explores; `mode=feature` enforces the contract below | - |
| `propose_linkage(source, code)` | Runs the join on the screen sample, reports match rate and a point-in-time check, then waits | ⏸ user confirms; saved to `linkage/<source>.py` |
| `screen_feature(feature_id)` | Existing `discovery.screen.Screener` + guards: fit on screen_train, score on screen_val, delta vs base | - |
| `screen_request(description, sql)` | Records an L3 request and waits | ⏸ user approves; user runs it in BigQuery |
| `finish(summary)` | Ends the direction | - |

### Feature-script contract

```python
def build(spark, sources: dict, base) -> "DataFrame[id, <feature_name>]":
    ...
```

* `base` is the model database; `sources` holds the cached sources already
  joined through their confirmed linkage, restricted to the ids being scored.
* One new column per script, keyed on `id`. L1b and L2c are each one script.
* During discovery the script only sees the screen_train and screen_val ids,
  so iterating on a 20 GB source stays cheap. Only verified features are
  materialised on the full splits.
* A guard rejects scripts that read a source directly instead of through
  `sources`, which is what keeps the point-in-time rule enforced.

PySpark runs as a local session, used like pandas but able to hold the full
histories.

## Verification and evaluation

* **Verified** = passed `screen_feature`: model fit on screen_train, scored on
  screen_val, through the existing guards. Valid and test are never shown to
  the agent.
* After the run, the validation pipeline reports `leave_one_in` against base on
  valid and test for every verified feature.
* The user ticks a combination; the pipeline trains `base + chosen` as a named
  variant. This needs one addition to `validation.stages.modeling`:
  `model.variants` accepting `{name: combo_a, add: [f1, f3]}` beside the
  existing kinds.
* Test is the out-of-time hold-out and the only evidence; the agent never sees
  it. A second hold-out will be added later for a final check, at which point
  test's role in the report moves to it.

## Backend

Reused from `AgenticSys_v2`:

* `llm/factory.py` + `safechain_client.py`: `openai` | `safechain` behind one
  `AsyncOpenAI`-shaped client, so the openai-agents SDK runs unchanged.
* `Runner.run_streamed` over one `Agent`; stream items mapped to SSE events as in
  `runner/turn/sse.py::map_run_item`.
* Flask + SSE with a replay buffer and heartbeat (`server.py`), and JSONL event
  logging (`EventLogger`).

New: the approval gate. A gated tool emits `approval_required`, then blocks on a
per-request `threading.Event` until `POST /api/runs/<id>/approvals/<req_id>`
resolves it (approve / reject with a note, which goes back to the agent).

### Events

| Event | Payload |
| --- | --- |
| `run_started` | run_id, direction, K |
| `skill_loaded` | name |
| `tool_started` / `tool_completed` | call_id, tool, args, result, duration_ms |
| `code_status` | code_id, intent, level, code, state (`running` / `ok` / `error`), stdout, error |
| `feature_screened` | feature_id, code_id, delta, passed, reason |
| `feature_verified` | feature_id, name, description, level |
| `approval_required` / `approval_resolved` | req_id, kind (`linkage` / `data_pull`), code or SQL, decision, note |
| `source_detected` | name, path, schema |
| `eval_started` / `eval_done` | variants, comparison table |
| `run_done` | stopped_because, intents_used, verified |

Every script the agent writes - probes, linkage, L1, L2 - goes out as
`code_status`, so all code is visible to the user.

## Frontend

`web/`, copied from CaseReviewChat's journey shell: React + Vite + Zustand,
CSS Modules, the Amex tokens, `useSSE` with reconnect and replay.

| Pane | Shows |
| --- | --- |
| Rail | Directions (past and current runs); detected sources |
| Direction | The direction text, K, start / stop; agent messages |
| Trace | Every step in order; code cards with state and logs; approval cards with approve / reject + note |
| Ledger | Each intent: level, code, screen delta, verdict; checkboxes on verified features → "Evaluate combination" |
| FlowStrip | Live graph: Direction → Skill → run_code → Screen → Verified, with the L3 branch → Approval → BigQuery → Source; nodes lit by state, intent counter `n / K` |

## Synthetic data

A generator builds, from the schemas: a model database with ids in the
`<customer_id>_<dt>_<marker>` form, and spend / payment / balance histories
keyed by `customer_id` and `event_dt`. It plants:

* one aggregate with real signal (e.g. 90-day payment-to-spend ratio) that an
  L1b feature should verify;
* one leak - target-correlated events dated after `as_of` - that the
  point-in-time rule must stop.

A correct end-to-end run verifies the first and never uses the second.

## Layout

```
src/agent/
  agent.py          builds the Agent: instructions, tools, model
  runner.py         one direction run; maps stream items to events
  server.py         Flask routes, SSE, approvals
  tools/            catalog, sample_rows, run_code, linkage, screen, data_pull
  skills/           data.md, feature.md, evaluate.md
  llm/              factory + safechain client (from AgenticSys_v2)
  synthetic.py      the generator above
web/                frontend
data/<use case>/
  additional_data/  cached sources: <name>.parquet + <name>.json
  linkage/          confirmed linkage per source
```

## Build order

1. Synthetic data + `run_code` with the contract, tested without an LLM.
2. Agent with `catalog`, `sample_rows`, `run_code`, `screen_feature`; L1a end
   to end, CLI only, events to JSONL.
3. Linkage gate and source detection; L1b / L2c.
4. Flask + SSE, then the frontend panes and FlowStrip.
5. Evaluation after the run and combination variants.
6. L3 requests.

## Out of scope for v1

Parallel directions, BigQuery client execution, long-term memory across
directions beyond the existing history file.
