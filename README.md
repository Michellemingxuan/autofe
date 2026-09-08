# mllite

A modular pipeline for answering one question: **do the newly proposed features
actually add anything on top of the incumbent feature set?**

```
data ─▶ 1. data quality ─▶ 2. feature selection ─▶ 3. model builds ─▶ 4. analysis ─▶ 5. verdict
        (placeholder)        spearman + MI          XGBoost 1.7.6      SHAP, Gini     four gates
```

Every stage is an independent module with a single `run_*` entry point, driven by
one YAML config, and fans its work out across processes.

## Install

```bash
pip install -e ".[dev]"        # or: pip install -r requirements.txt
```

XGBoost is pinned to **1.7.6**. On an existing environment where numpy/scipy are
already set, `pip install --no-deps xgboost==1.7.6` avoids disturbing them.

The demo downloads from `archive.ics.uci.edu`. In a private environment with no
egress, point `data.path` at a local table instead - nothing else in the pipeline
reaches the network.

## Run

```bash
python data/bankruptcy/prepare.py             # downloads the demo dataset
mllite -c configs/bankruptcy.yaml                      # or: python -m mllite.cli -c ...
mllite -c configs/bankruptcy.yaml --set run.n_jobs=8   # dotted overrides, repeatable
```

Two demos ship with the repo:

| Config | Data | Purpose |
| --- | --- | --- |
| `configs/bankruptcy.yaml` | UCI Taiwanese Bankruptcy Prediction (real, binary, 3.2% event rate) | Realistic end-to-end run |
| `configs/example_synthetic.yaml` | `data/synthetic/make.py` (regression, known ground truth) | Fast sanity check that the screens find what is really there |

From Python:

```python
from mllite import run_pipeline

result = run_pipeline("configs/bankruptcy.yaml")
print(result.analysis.comparison)          # gini gain per variant
print(result.feature_selection.selected)   # new features that survived screening
```

Candidates engineered in a session never need to round-trip through a file:

```python
from mllite import Pipeline, load_config

frame["cand_x"] = frame["a"] / frame["b"]
cfg = load_config("configs/bankruptcy.yaml")
cfg.features.new = ["cand_x"]
result = Pipeline(cfg).run(frame=frame)     # or run(dataset=...) for a prebuilt one
```

**Start here:** [`notebooks/usage_walkthrough.ipynb`](notebooks/usage_walkthrough.ipynb)
is a fully executed walkthrough — the one-line run, how to read each artifact,
the stage-by-stage API, and a worked discover -> verify -> keep-or-shift-out loop.

## The demo run

[UCI dataset 572](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction):
6,819 Taiwanese companies (1999-2009), 95 financial ratios, 3.2% bankruptcy rate.
The scenario is that an incumbent model already uses the profitability / leverage
/ growth ratios and a team proposes adding the **cash-flow family** (11 ratios).
`data/bankruptcy/prepare.py` also plants two controls among the candidates so the
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
| `src/mllite/config.py` | Typed config dataclasses; unknown keys are errors, not silent no-ops |
| `src/mllite/data.py` | Load, resolve base/new feature lists, clean sentinels, split |
| `src/mllite/parallel.py` | The only place that talks to joblib; also budgets XGBoost threads |
| `src/mllite/metrics.py` | Adjusted Gini, decile accuracy, capture rate |
| `src/mllite/stages/data_quality.py` | **Placeholder** — port AIME_DataStability here |
| `src/mllite/stages/feature_selection.py` | Spearman + mutual information screens |
| `src/mllite/stages/modeling.py` | Variant construction and XGBoost training |
| `src/mllite/stages/analysis.py` | Metrics, Gini gain vs. baseline, SHAP ranking |
| `src/mllite/stages/verdict.py` | The four-gate cascade and the batch decision |
| `src/mllite/pipeline.py` | Sequences the stages, writes every artifact |
| `notebooks/usage_walkthrough.ipynb` | Executed walkthrough, including the discover/verify loop |
| `data/<use case>/` | A use case: its build script, raw input, and shaped table |
| `data/` | Every table the pipeline reads or writes (git-ignored) |
| `configs/` | Run configs |

## Project layout

```
configs/            run configs - one file fully describes a run
data/               one folder per use case - the script and its output together
  bankruptcy/
    prepare.py      downloads + shapes the UCI table
    raw/            the downloaded archive, before shaping
    modeling.parquet
    column_mapping.csv
  synthetic/
    make.py         generates the ground-truth table
    modeling.parquet
src/mllite/         the pipeline
notebooks/          executed walkthrough
outputs/            one timestamped directory per run
tests/
```

**One folder per use case under `data/`.** A use case that is split upstream keeps
its parts side by side, which is exactly the shape `data.paths` expects:

```
data/<use case>/
    modeling.parquet                    # one table, pipeline splits it
    train.parquet valid.parquet test.parquet   # or already split
```

Both prep scripts read their use case's config for the target, id column,
candidate list and output path, so those are declared once in YAML rather than
repeated on a command line. The scripts own only what a config cannot express —
how the raw source becomes a table — and then verify that what they built matches
what the config declares.

`data/` is git-ignored, so nothing large or confidential is ever committed — which
is also why the build scripts live under `scripts/<use case>/` rather than beside
their output. They are code: they need to survive a fresh clone, and
`data/synthetic/make.py` is imported by the test suite.

Paths in a config are relative to where you run from, so run from the project
root (the notebook does `os.chdir(ROOT)` in its first cell for the same reason).

## Inputs: four shapes, same Dataset

| Your data | Config | Call |
| --- | --- | --- |
| One table, pipeline splits it | `data.path` + `data.split.mode: random \| time` | `run()` |
| One table with a train/valid/test column | `data.path` + `split.mode: column` | `run()` |
| **Already split, separate files** | `data.paths: {train:…, valid:…, test:…}` | `run()` |
| **Already split, in memory** | — | `run(frames={"train": df, …})` |
| One frame in memory | — | `run(frame=df)` |

When `data.paths` is set it wins over `data.path` and the whole `split` block is
ignored — the frames are used exactly as given, so out-of-time or sampling logic
you applied upstream is preserved. Features are resolved against `train`, and a
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
then either keep them or shift out and propose something else. `Pipeline.run(frame=...)`
takes an in-memory table, so a round is one function call — see section 4 of the
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
