# autofe

A modular pipeline for answering one question: **do the newly proposed features
actually add anything on top of the incumbent feature set?**

```
data ─▶ 1. data quality ─▶ 2. feature selection ─▶ 3. model builds ─▶ 4. analysis ─▶ 5. verdict
        (placeholder)        spearman + MI          XGBoost 1.7.6      SHAP, Gini     four gates
```

Every stage is an independent module with a single `run_*` entry point, driven by
one YAML config, and fans its work out across processes.

## Environment

XGBoost is pinned to **1.7.6**; nothing else is. The full suite is green on both
numpy generations, so the pipeline itself does not force the choice:

| | numpy | pandas | scipy | scikit-learn | shap |
| --- | --- | --- | --- | --- | --- |
| tested | 1.26.4 | 3.0.5 | 1.17.1 | 1.9.0 | 0.49.1 |
| tested | 2.0.0 | 2.3.2 | 1.16.2 | 1.7.2 | 0.49.1 |

### "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.0"

This is an ABI mismatch in the *environment*, not in autofe — some package with a
compiled extension was built against numpy 1.x while numpy 2.x is installed. The
traceback names the culprit a few lines below the warning. In this stack the
candidates are `shap`, `scikit-learn`, `scipy`, `pandas` and `numba`; `xgboost` is
not one, because it reaches its native library through `ctypes` rather than the
numpy C API.

To identify it:

```bash
for m in numpy pandas scipy sklearn shap numba xgboost; do
    python -c "import $m; print('$m', $m.__version__)" 2>&1 | tail -2
done
```

Then either upgrade that package to a build that supports numpy 2, or pin
`numpy<2` — both work, since autofe uses no numpy-2-only API.

## Install

```bash
pip install -e ".[dev]"        # or: pip install -r requirements.txt
```

XGBoost is pinned to **1.7.6**. On an existing environment where numpy/scipy are
already set, `pip install --no-deps xgboost==1.7.6` avoids disturbing them.

The preparation notebooks download from `archive.ics.uci.edu`. In a private
environment with no egress, prepare local train/valid/test tables and point
`data.paths` at them instead; validation itself does not reach the network.

## Run

```bash
# Run data/bankruptcy/prepare.ipynb once to build the fixed split files.
autofe -c configs/bankruptcy.yaml                      # or: python -m validation.cli -c ...
autofe -c configs/bankruptcy.yaml --set run.n_jobs=8   # dotted overrides, repeatable
autofe -c configs/bankruptcy.yaml --plan               # show the stages; load nothing
autofe -c configs/bankruptcy.yaml --check              # validate wiring/data; train nothing
```

During a run, the terminal prints a compact pipeline board. The same transitions
are persisted in `run.log`. Each stage also records its duration, checks,
warning/failure reason, and summary in `pipeline_status.json`; a failed run
therefore still says exactly where it stopped.

For a new use case, generate the small config instead of copying a demo's
domain-specific discovery settings:

```bash
autofe init configs/churn.yaml \
  --train data/churn/train.csv --valid data/churn/valid.csv --test data/churn/test.csv \
  --target churned --task binary --id customer_id --new-prefix cand_
autofe -c configs/churn.yaml --check
autofe -c configs/churn.yaml
```

The generated config uses safe defaults, infers incumbent numeric columns, and
includes the model variants needed by the per-candidate verdict. Add advanced
settings only when the use case needs them.

Three demos ship with the repo:

| Config | Data | Purpose |
| --- | --- | --- |
| `configs/bankruptcy.yaml` | UCI Taiwanese Bankruptcy Prediction (real, binary, 3.2% event rate) | Realistic end-to-end run; rare events, all-continuous ratios |
| `configs/malware.yaml` | UCI NATICUSdroid Android permissions (29,332 apps, 50% positive) | Balanced target, and every feature a 0/1 flag - so a useful feature has to combine flags, not transform one |
| `configs/myocardial.yaml` | UCI Myocardial Infarction Complications (1,700 patients, 23% positive) | Small, 7.6% of cells missing, 98 of 110 features coded categories - the hard case |

From Python:

```python
from validation import run_pipeline

result = run_pipeline("configs/bankruptcy.yaml")
print(result.analysis.comparison)          # gini gain per variant
print(result.feature_selection.selected)   # new features that survived screening
```

Candidates engineered in a session never need to round-trip through a file:

```python
from validation import Pipeline, load_config

frames = {name: part.assign(cand_x=part["a"] / part["b"])
          for name, part in frames.items()}
cfg = load_config("configs/bankruptcy.yaml")
cfg.features.new = ["cand_x"]
result = Pipeline(cfg).run(frames=frames)   # or run(dataset=...) for a prebuilt one
```

**Start here:** [`notebooks/usage.ipynb`](notebooks/usage.ipynb)
is a fully executed walkthrough — the one-line run, how to read each artifact,
the stage-by-stage API, and a worked discover -> verify -> keep-or-shift-out loop.

## Agent-driven discovery

One agent works on one **direction** - a plain-text idea for improving the
model - with up to K **intents** (features screened). It reads the model
database and the extra sources, writes each feature as a script (L1: model
columns or an aggregate of one source; L2: a source aggregate combined with
model columns; L3: new data, requested as SQL), and screens it: fit on screen_train, score on
screen_valid. A feature is verified when it clears the analyst's gates: its
Gini gain above `agent.min_gini_gain` and, when set, its capture-rate gain
above `agent.min_capture_gain` (both settable per run). Valid and test never
reach the agent. The agent is briefed with three skills (`src/agent/skills/`:
`data_sourcing`, `feature`, `evaluate`) and acts through the tools in
`src/agent/tools/`; the screen tool reuses `discovery.screen` and
`discovery.guards`. Overview: [`docs/modeling_agent_architecture.svg`](docs/modeling_agent_architecture.svg). Design:
[`docs/superpowers/specs/2026-10-03-agentic-feature-discovery-design.md`](docs/superpowers/specs/2026-10-03-agentic-feature-discovery-design.md).

The entry points resolve paths from the project root wherever they are run
from, so `data/` is the one data folder.

```bash
PYTHONPATH=src python -m agent.synthetic                           # CDSS-shaped synthetic data -> data/synthetic_agent
PYTHONPATH=src python -m agent.server -c configs/synthetic_agent.yaml   # backend, :49010
cd web && npm install && npm run build                              # served by the backend at /
# or: cd web && npm run dev                                         # :5173, proxies /api
```

From the terminal instead (approvals asked at the prompt; `--yes` approves all):

```bash
PYTHONPATH=src python -m agent.cli -c configs/synthetic_agent.yaml -d "payment behaviour vs spend" -k 5
PYTHONPATH=src python -m agent.evaluate -c configs/synthetic_agent.yaml \
    --feature <run_id>:pay_to_spend_90d --feature <other_run_id>:balance_6m_volatility \
    --combo pay_vol=<run_id>:pay_to_spend_90d,<other_run_id>:balance_6m_volatility
```

The UI follows the journey in three steps.

1. **Setup** - point at the data, in five blocks, each prefilled from the
   config file and applied on its own (saved as overrides in
   `<agent.run_dir>/workspace_overrides.json` only if the workspace loads):
   * **Model database** - set the folder once, then just the split file
     names; each path shows whether it exists and its size.
   * **Context for the agent** - task description, and the small files
     (task context, column descriptions): a path, or an upload kept under
     `<agent.run_dir>/uploads`.
   * **Shots** - labelled examples the agent reads by category with its
     `shots` tool: the **clustering shots** - the prepare step's file
     (`discovery.few_shot_path`), or generated on the page from the screen's
     fit rows (KMeans per class, N rows × B batches) - then your own categories, appended in
     order (`agent.shot_spec_paths`), one markdown file each:

     ```markdown
     # Early cures
     ## Context
     Customers who went 30 days past due and cured within two cycles.
     ## IDs
     - 105131_20240301_B
     - 104444_20240701_A, 104990_20240501_C
     ## Same examples each discovery?
     No - rotate, 4 per batch
     ```

     Ids are looked up in the train split only. "Yes" shows every id in
     every run; "no" rotates through batches of the given size, one batch
     per discovery run (the counter carries across runs, as does the
     clustering shots' rotation). The page lists the categories - rows
     found, ids missing, how they rotate - and any can be deleted.
   * **Additional data** - a folder of `<name>.parquet|csv` +
     `<name>_data_sample.json`, or a source added by path where it lies
     (`sources.json` - big data is never uploaded or copied; its sample JSON
     can be uploaded). For each source with data the agent **proposes its
     linkage** - the point-in-time join to the model ids - and you approve
     it after seeing the code, the match rate and the point-in-time check.
   * **Scope** - the CAS variables, flagged by whether the model uses them:
     by default the `*_flagged.csv` files in the additional data folder
     (`agent.scope_glob`), plus any listed in `agent.scope_paths`. **Scope
     notes** (`agent.scope_notes_paths`; .md, .txt, .docx, .pdf - uploaded or
     by path) are your guidance on using it, given to the agent verbatim.
2. **Discover** - a direction plus its **parameters**, prefilled from the
   config's `agent` section: K, the model, the engine, the verification
   threshold, which levels (L1, L2, L3) and which sources. The run
   shows as a trace grouped into steps - explore, linkage, each intent, data
   requests, summary - beside a timeline. A direction, or one intent of it,
   can be deleted.
3. **Evaluate** - every verified feature from every direction in one pool;
   click one to read its code and the linkage it used. Pick any, group some
   into combinations, run: `leave_one_in` for each and `combo__<name>` for
   each combination on the full splits. On real data this is the slow part,
   so it streams - each feature script, each pipeline stage with its time,
   and the pipeline's log. Results, all on test (the out-of-time hold-out):
   Gini gain, capture-rate gain at the top 10%, 5% and 1%, the SHAP rank of
   each new feature in its model, and the verdict. Evaluations can be deleted.

**A data-request run** - tick only L3. Nothing is built or screened. One
agent reads the CAS scope, your scope notes and the shots, and takes each data
pull it wants (up to K) through one loop:

1. *Propose* - a rationale, the features it would enable, and BigQuery SQL.
2. *Validate* - the tool reads the SQL against the CAS column lists: a table
   outside the scope, an invented column, no identifier selected or no
   partition-date filter sends it back before it costs anything.
3. *Challenge* - the agent reflects on its own proposal: can this information
   be built from the model database and the linked sources? A pull is effort,
   so it is worth it only if not. A "constructible" verdict must come with the
   construction, and the construction is run on the screen rows.
4. *Kept or dropped* - only a construction that runs drops the proposal.

The run ends only when every proposal is challenged, and its summary carries
the validated SQL of the kept requests. Nothing waits for approval; the kept
and dropped requests are reviewed when the run ends - on the run page, or
downloaded together as `data_requests.md` (also in the run folder, with one
`.sql` file per kept request).

As in AgenticSys_v2, the agents reason and the tools are deterministic Python:
a tool runs code, checks it, records it - it never calls a model.

What waits for you, and nothing else does: confirming a linkage (in Setup, or
mid-run if a direction needs a source that has none), and an **L3 data pull**
- the agent writes the gap and BigQuery SQL over the CAS scope; approve it,
run it, and drop the result in as a source.

The API the UI calls is listed in `src/agent/server.py`;
`tests/test_agent_server.py` drives every route without an LLM and fails if
the frontend calls a path the server lacks.

Sources are `<name>.parquet|csv` beside `<name>_data_sample.json`
(`{column: [description, [samples]]}`); CAS scope files are the
`*_flagged.csv` exports in the same folder. Scripts run in a subprocess with a
guard against file access, on pandas or Spark (`agent.engine`). The LLM is
`agent.llm`: `openai`, or `safechain` through the client in `src/agent/llm/`
(copied from AgenticSys_v2: a call stalled at 40s is re-issued, capped at 180s -
`SAFECHAIN_STALL_RETRY_S`, `SAFECHAIN_CALL_TIMEOUT_S`; events in
`outputs/agent/llm_logs/`).

## The demo run

[UCI dataset 572](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction):
6,819 Taiwanese companies (1999-2009), 95 financial ratios, 3.2% bankruptcy rate.
The scenario is that an incumbent model already uses the profitability / leverage
/ growth ratios and a team proposes adding the **cash-flow family** (11 ratios).
`data/bankruptcy/prepare.ipynb` also plants two controls among the candidates so the
screens have something known to catch: `cand_dup_roa_c` (a near-copy of an
incumbent) and `cand_noise` (pure noise).

What the run found (18s, 8 cores):

* **Data quality** dropped `net_income_flag` - constant across all 6,819 rows.
* **Selection** kept 5 of 13 candidates. Both planted controls were caught
  (`cand_dup_roa_c` at |rho|=0.993 vs. its anchor; `cand_noise` at |rho|=0.007 vs.
  the outcome). Six real cash-flow ratios were dropped as redundant *with each
  other* - `cash_flow_to_sales`, `_to_equity`, `_to_total_assets` and
  `_to_liability` are ~0.96-0.97 correlated, so the family contributes its best
  member rather than four copies of the same signal.
* **Gini gain: essentially zero.** `base_plus_new` scored 0.8902 test Gini against
  the baseline's 0.8905 (-0.0003), and no `leave_one_in` variant moved it beyond
  +/-0.007. The cash-flow family carries 7.5% of SHAP attribution but that
  attribution is redistribution, not new information - the incumbent ratios
  already span it.

Two caveats that the demo makes concrete, and that apply to any run:

* Test has only ~44 bankruptcies, so Gini differences of +/-0.01 are inside the
  noise band. A point estimate of gini gain is not on its own evidence of lift;
  bootstrap the metric or repeat over seeds before calling a feature a win.
* Train Gini is 0.998 against 0.89 on test. That gap is the model overfitting
  95 ratios to 4k rows, not a pipeline defect - but it is why the comparison is
  made on `test`, never on `train`.

## Layout

| Path | What it does |
| --- | --- |
| `src/validation/config.py` | Typed config dataclasses; unknown keys are errors, not silent no-ops |
| `src/validation/data.py` | Load fixed splits, resolve base/new feature lists, clean sentinels |
| `src/validation/preflight.py` | Fast checks for data, leakage, model, and verdict contracts |
| `src/validation/status.py` | Terminal graph plus live HTML/JSON run state |
| `src/validation/parallel.py` | The only place that talks to joblib; also budgets XGBoost threads |
| `src/validation/metrics.py` | Adjusted Gini, decile accuracy, capture rate |
| `src/validation/stages/data_quality.py` | **Placeholder** — port AIME_DataStability here |
| `src/validation/stages/feature_selection.py` | Spearman + mutual information screens |
| `src/validation/stages/modeling.py` | Variant construction and XGBoost training |
| `src/validation/stages/analysis.py` | Metrics, Gini gain vs. baseline, SHAP ranking |
| `src/validation/stages/verdict.py` | The four-gate cascade and the batch decision |
| `src/validation/pipeline.py` | Sequences the stages, writes every artifact |
| `src/agent/` | Agent-driven discovery: setup, workspace, guarded code runner, session (tools, events, approvals), agent, server, evaluation |
| `src/agent/composer/` | **The prompt composer**: the brief and message templates (`templates/feature_engineer.md`, `data_scout.md`, `linkage_writer.md`, `messages.md`) and the filled-in sections - see its README; print a run's prompt with `python -m agent.composer` |
| `src/agent/skills/` | The skills - `data_sourcing`, `feature`, `evaluate` - appended to every brief in full |
| `src/agent/tools/` | What the agent can do - one module per kind of tool; each checks, runs and records |
| `web/` | The frontend: Setup (data, sources, linkage), Discover (parameters, step trace, timeline), Evaluate (pool, combinations, streamed results) |
| `notebooks/usage.ipynb` | Executed walkthrough, including the discover/verify loop |
| `data/<use case>/` | A use case: its build script, raw input, and shaped table |
| `data/` | Every table the pipeline reads or writes (tables git-ignored, build scripts tracked) |
| `configs/` | Run configs |

## Project layout

```
configs/            run configs - one file fully describes a run
data/               one folder per use case - the script and its output together
  bankruptcy/
    prepare.ipynb   downloads, shapes, checks, and splits the UCI table
    raw/            the downloaded archive, before shaping  (ignored)
    train.csv valid.csv test.csv                            (ignored)
    column_mapping.csv
    column_descriptions.json   what each column means, for a discovery run
  malware/          same shape: prepare.ipynb + mapping + descriptions
  myocardial/
src/validation/     the pipeline: data, quality, screening, models, verdict
src/discovery/      proposing candidate features and screening them cheaply
src/preprocessing/  what every data/<use case>/prepare.ipynb shares
notebooks/          executed walkthrough
outputs/            one timestamped directory per run
tests/
```

**One folder per use case under `data/`.** A use case that is split upstream keeps
its parts side by side, which is exactly the shape `data.paths` expects:

```
data/<use case>/
    train.csv valid.csv test.csv        # fixed split; parquet works too
```

Each preparation notebook reads its use case's config for the target, id column,
candidate list and output path, so those are declared once in YAML rather than
repeated on a command line. The scripts own only what a config cannot express —
how the raw source becomes a table — and then verify that what they built matches
what the config declares.

The tables themselves are git-ignored — train/valid/test CSV or parquet files,
screen samples, few-shot rows, and `raw/` — so nothing large or confidential is
committed. What is tracked is everything needed to rebuild them: each use case's
`prepare.ipynb`, its `column_mapping.csv`, and its `column_descriptions.json`.
Those are code and documentation, so they survive a fresh clone.

So a fresh clone executes `data/<use case>/prepare.ipynb` once; it downloads the
source, rebuilds the fixed splits, and checks them against the config.

Paths in a config are relative to where you run from, so run from the project
root (the notebook does `os.chdir(ROOT)` in its first cell for the same reason).

## Inputs: two shapes, same Dataset

| Your data | Config | Call |
| --- | --- | --- |
| **Already split, separate files** | `data.paths: {train:…, valid:…, test:…}` | `run()` |
| **Already split, in memory** | — | `run(frames={"train": df, …})` |

Splitting is deliberately an upstream preparation decision: the frames are used
exactly as given, so out-of-time or sampling logic is stable across repeated runs.
Features are resolved against `train`, and a
column present in train but missing from another split is an error, not a silent
NaN column.

## Stage 1 — data quality & stability (placeholder)

`stages/data_quality.py` has the full plumbing (config, parallel fan-out, report
shape, downstream contract) with two `TODO` functions to fill in from
[`AIME_DataStability`](https://github.aexp.com/sssheno/AIME_DataStability):

* `_profile_chunk` — per-feature quality metrics (currently missing rate and
  cardinality only).
* `_stability_chunk` — PSI/CSI across `data_quality.by_col` periods; returns an
  empty frame until ported.

### Distribution consistency (implemented)

The one stability check that is real: for every feature, PSI between the reference
split and each other split, so you can see whether a variable is shaped the same
way in valid and test as it is in train.

```yaml
data_quality:
  distribution_check: true
  distribution_reference: train   # train | valid | test
  distribution_bins: 10
  max_psi: 0.25
```

Bin edges are the reference's quantiles, and **NaN gets its own bucket** — a feature
that is 2% null in train and 40% null in test has genuinely shifted, and binning
missingness away would hide exactly that. Read `psi_max` (worst across the compared
splits) and `psi_worst_split`: below 0.10 is no meaningful shift, 0.25 and above
means the variable is not the same thing in the two samples.

On a random split nothing shifts, which is the point of running it — the check earns
its keep on an out-of-time split. Verified against a deliberately shifted split: the
variable used to order the split scored PSI 7.05 and 33 of 97 features exceeded 0.25.

Return one row per feature plus a boolean `passed`.

Two switches control it, and they are independent:

```yaml
data_quality:
  enabled: false        # false = skip the stage entirely (the default)
  drop_failed: false    # true = failing features are removed before selection;
                        # false = they are only reported
```

## Stage 2 — feature selection

Both screens run on a sample of the training split.

The stage's primary output is a verdict per proposed feature —
`feature_selection_verdicts.csv`, `result.verdicts`, and the lead table in
`report.md`:

| feature | verdict | decided_by | reason |
| --- | --- | --- | --- |
| cash_current_liability | IN | – | |
| cand_dup_roa_c | OUT | redundancy screen | redundant with roa_c… (\|rho\|=0.993) |
| cand_noise | OUT | signal screen | weak spearman vs target (-0.0066) |

It spans stages on purpose: a candidate cut by data quality never reaches the
screens, so it appears here as `OUT / data quality` instead of silently vanishing.

**One asymmetry worth stating.** Incumbent features are never screened, so a
candidate can be dropped for duplicating an incumbent but never the reverse — even
when the candidate is the stronger of the pair. That is deliberate (the incumbent
set is the status quo being challenged), but it means the process leans toward
rejection, and a narrow loss is not evidence the feature is worthless.

**Signal** — Spearman rho and mutual information between each new feature and the
outcome. Below `spearman.target_min_abs` or `mutual_info.target_min`, the feature
is dropped.

**Redundancy** — Spearman rho and normalized MI between each new feature and
(a) the incumbent features and (b) the new features already kept. Candidates are
considered strongest-first; one that duplicates something already in the set is
dropped, so a cluster of correlated candidates contributes its best member rather
than all of them.

Both directions of mutual information are computed, and both are reported per
candidate so the trade-off is one sort rather than two thresholds:

| Column | Meaning | Want |
| --- | --- | --- |
| `mi_target` | raw MI with the outcome, in nats | higher |
| `nmi_target` | the same, normalized to 0–1 | higher |
| `mi_redundancy_base` | MI with the incumbent set | lower |
| `mrmr_score` | `nmi_target − mi_redundancy_base` | higher |
| `spearman_redundancy_base` | max \|rho\| with the incumbent set | lower |

`feature_selection.ranking` picks the order candidates are considered in, which
matters because the greedy screen keeps whichever member of a correlated cluster
it reaches first:

* `spearman` (default) — strongest \|rho\| with the outcome first.
* `mi` — highest mutual information with the outcome first.
* `mrmr` — re-scored after every pick: relevance minus redundancy with the
  incumbents *and* the candidates kept so far.

Two things to know before switching to `mrmr`. The score subtracts *normalized*
relevance from normalized redundancy — raw nats would be swamped, since for a 3%
event `H(y)` is only ~0.14 nats while redundancy lives on 0–1. And set
`mutual_info.redundancy_stat: mean` when the incumbent set is large; a max over
80+ features is near 1 for almost everything and flattens the ranking. The drop
*gate* always uses max regardless — being a near-copy of one incumbent is what
makes a candidate redundant, and a mean would hide it.

Implementation notes:

* Spearman is one BLAS call per chunk — rank once, then pairwise-complete Pearson
  expressed as matrix products (matches `pandas.corr(method="spearman")` to ~1e-3;
  the difference is global vs. per-pair ranking under NaN).
* MI discretizes each column into quantile buckets once, **giving NaN its own
  bucket** so "missing" carries information, then computes joint histograms with
  `np.bincount`. Normalized MI (`MI / min(H(a), H(b))`) is what the redundancy
  threshold compares against, so it is comparable across cardinalities.
* Set `mutual_info.target_method: sklearn` to score signal with sklearn's kNN
  estimator instead of the histogram one.

## Stage 3 — model builds

A *variant* is a named feature list. Configure any mix of:

| Variant | Features |
| --- | --- |
| `base` | incumbent only — the champion |
| `base_plus_new` | incumbent + all selected new features |
| `new_only` | new features alone |
| `leave_one_in` | base + one new feature — the marginal value of each candidate |
| `leave_one_out` | base + all new except one — what is lost by removing it |

`leave_one_in` / `leave_one_out` expand to one model per new feature, so a run
with 20 candidates trains 20+ models — that is what the parallelism is for.

## Stage 4 — outcome analysis

* Adjusted Gini and capture rate per variant per split, both with a gain column
  against the baseline (`gini_gain_*`, `capture_gain_*`). Every percent in
  `analysis.capture_rate_percents` lands in `metrics_by_variant_split.csv`;
  `analysis.comparison_capture_percents` (default `[0.05]`) picks which reach the
  comparison table, since each adds a column per split.
* `calc_accuracy` (a decile *calibration* measure, not classification accuracy) is
  computed into `metrics_by_variant_split.csv` but kept out of the comparison table —
  set `analysis.include_accuracy: true` to show it. On small splits it is mostly
  noise: with ~44 events a perfectly calibrated model scores ~0.75 ± 0.11.

Capture rate is quantised by the event count: with 44 bankruptcies in test it can
only move in steps of 1/44 = 0.023, so treat a one-step difference between variants
as a tie, not a win.
* **Gini gain**: each variant's adjusted Gini minus `analysis.baseline_variant`'s,
  absolute and relative.
* **SHAP** (`TreeExplainer`, falling back to XGBoost's exact `pred_contribs`):
  mean |SHAP| ranking per variant, plus the share of total attribution landing on
  the new features. XGBoost's own `total_gain` importance sits next to it in
  `feature_ranking.csv`.

## Stage 5 — the verdict

The four stages above produce evidence; this one turns it into a decision. Four
gates, applied in order — a feature that fails one is never measured at the next,
because it genuinely cannot be:

| Gate | Fails when | Threshold |
| --- | --- | --- |
| data quality | the column itself is unusable | `data_quality.*` |
| feature selection | no signal, or signal the incumbents already carry | `feature_selection.*` |
| gini gain | it entered a model and the model got no better | `verdict.min_gini_gain` |
| shap rank | the model kept it but leans on it barely | `verdict.max_shap_rank_pct` |

Then one call on the batch: **if nothing clears all four gates, the verdict is
`TRY A NEW BATCH`** — propose a different family rather than lowering a threshold.

```python
result.batch.verdict     # 'KEEP' | 'TRY A NEW BATCH'
result.batch.failed_at   # {'feature selection': 8, 'gini gain': 4, 'shap rank': 1}
result.verdicts          # one row per candidate, every gate's outcome
```

`failed_at` is the part worth acting on, because *where* a batch dies says what to
do next. Falling at feature selection means the family duplicates what you already
have — look elsewhere. Falling at gini gain means the features were genuinely new
but did not move the model.

Two caveats. The per-feature gini gate needs `leave_one_in` variants to attribute
gain to a single feature; without them that gate reports `not evaluable` rather
than guessing, and a feature is never failed for missing infrastructure. And the
thresholds are blunt — with few positives in test, `min_gini_gain: 0.005` sits
inside the noise band, so a lone `KEEP` is not yet evidence.

## Parallelism

`run.n_jobs` and `run.backend` control every stage. Fan-out points: feature-selection
chunks (`feature_selection.chunk_size` new features per task), one task per model
variant, and one task per variant for metrics and SHAP.

Two things worth knowing:

* XGBoost brings its own thread pool. `parallel.threads_per_worker` divides the
  machine's cores by the number of process workers so the two layers don't
  oversubscribe; override with `model.threads_per_model`.
* Feature matrices are handed to workers as raw numpy arrays, which lets joblib
  memory-map them instead of pickling a copy per worker.

Set `run.backend: sequential` (or `run.n_jobs: 1`) to debug — results are
identical, which `tests/test_pipeline.py` asserts.

## Output

Each run writes `outputs/<run name>/<timestamp>/`:

```
report.md                              human-readable summary
summary.json                           machine-readable summary + library versions
pipeline_status.json                   stage state, timings, checks and errors
preflight.json                         resolved data/config contract checks
config.resolved.yaml                   the fully resolved config
run.log
data_quality_report.csv
feature_selection_target_stats.csv     spearman/MI vs outcome per candidate
feature_selection_redundancy.csv       closest existing feature per candidate
feature_selection_spearman_matrix.csv  new x all correlation matrix
feature_selection_mi_matrix.csv        new x all normalized MI matrix
candidate_verdicts.csv                 the four-gate cascade, one row per candidate
batch_verdict.json                     KEEP or TRY A NEW BATCH, and where the batch died
feature_selection_verdicts.csv         IN/OUT per candidate at the selection stage
feature_selection_decisions.json       kept / dropped with reasons
metrics_by_variant_split.csv
variant_comparison.csv                 gini + gini gain per variant
shap_ranking.csv, xgb_importance.csv, feature_ranking.csv
models/<variant>.json
```

## Using it as a discovery loop

The pipeline is built to be called repeatedly: propose candidates, verify them,
then either keep them or shift out and propose something else. `Pipeline.run(frames=...)`
takes in-memory split tables, so a round is one function call — see section 4 of the
notebook for a working `verify()` and a two-round ledger.

What the demo loop turned up, which is worth knowing before trusting a round:

* Round 1 (the cash-flow family) and round 2 (30 engineered interactions between
  the top SHAP drivers) both shifted out. The interactions took ~18% of SHAP
  attribution and still gained nothing on held-out data.
* **SHAP share is not evidence of value.** Attribution measures what the model
  *uses*; gini gain measures what it *gained*. A boosted ensemble already composes
  interactions from raw features, so pre-multiplying them moves credit around
  without improving the ranking.
* A fixed gain threshold is a blunt decision rule. With few positives in test,
  bootstrap the gain or repeat across seeds before believing a `KEEP`.
* Nothing here detects leakage. A feature built from the outcome will pass both
  screens and post a large gain; that check stays with whoever proposes it.

## Tests

```bash
pytest -q
```

Metric tests are pure-pandas; the end-to-end tests skip if XGBoost is missing.
