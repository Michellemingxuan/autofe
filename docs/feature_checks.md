# Feature checks C0–C3

Every feature the agent proposes (`screen_feature`, `src/agent/tools/screen.py`)
passes four checks, cheapest first. A check that refuses before anything is built
costs nothing; a feature that is built and fails costs one attempt.

| Check | Question | Where | Cost | Outcome when it fails |
|---|---|---|---|---|
| **C0** | Seen before? | `tools/screen._seen_before`, `memory.same_feature`, `checks.code_fingerprint`, `checks.similarity` | ms | refused – nothing spent (values: attempt refunded) |
| **C1** | Does the model already have it? | `checks.Sketch.explained`, `checks.model_checks` | ms | failed: `explained by the model: …` |
| **C2** | How well does it alone rank the outcome (variable Gini)? | `checks.Sketch.variable_gini` | ms | information only (a gate if configured) |
| **C3** | Does it move the model on held-out rows? | `discovery.screen.Screener.evaluate_values`, `tools/screen._below_gates` | seconds | failed: a guard's message, or `Gini gain … is not above …` |

The board (`LoggingBoard`) shows one column per check: **C0 (seen before?)**,
**C1 (redundancy)**, **C2 (variable Gini)**, **C3 (screen gain)**. Thresholds are
under `agent.checks` and `agent.*` in the config. The values quoted here come from
`configs/cdss_us_sbs.yaml`.

---

## The order of one attempt

```
pre-flight   name valid and free · idea named · level allowed · sources exist and linked · code guard
C0 (before)  model history · code fingerprint · meaning              ── refuse: nothing spent
             ── the attempt is taken, the script runs on the screen rows ──
             script failed / wrong column                            ── failed: "script failed: …"
C0 (after)   identical values                                        ── refuse: attempt given back
             values saved (values/<name>.parquet), coverage recorded
C1, C2       on the built values                                     ── C1 fails: no fit
C3           guards → fit base + feature → Gini gain, capture gain → gates
verified     = the screen ran ok and no gate was missed
```

Everything measured is stored on the ledger entry (`entry["checks"]`, `delta`,
`capture_gain`, `reason`). The board reads that entry, and the screen reply returns
the same fields to the agent.

The pre-flight checks the name, then the sources, then the code. A name is free
unless a base column has it or a feature **built in this run or any earlier run**
has it. A failed script built nothing, so its name stays free for the fixed script.

---

## C0 – seen before?

C0 compares the new feature with **every undeleted feature built before it, in this
run and in all earlier runs** (`memory.earlier_features`), and with the **model
history** (`model_history.history_of`). It checks four things.

### C0.1 Model history (before the run)

The model's past variables come from `agent.model_history`, e.g.
`data/cdss_us_sbs/model_history.md`. A feature *reads like* one of them if either
of these holds:

- it has the same name (case-insensitive), or
- its word overlap (C0.3's measure) with a variable's name and description is
  ≥ `same_meaning`.

Without `differs` it is refused: *"it reads like `X` of the model's Gen9 – used
(why). Say in `differs` what this one measures or fixes that it did not."* With
`differs`, it is noted on the board and goes on.

### C0.2 Same code (before the run)

`checks.code_fingerprint(code, feature_name)` turns the script into a fingerprint
of what it computes:

1. **Parse** the script to its AST (abstract syntax tree), the structure Python
   builds from source. Comments and layout disappear at this step.
2. **Normalise** the tree:
   - Local names are renamed in order of first use: `v0`, `v1`, …
   - API names are kept: `build`, `base`, `sources`, `spark`, `pd`, `np`, `F`,
     `Window`, `link`, `raw`.
   - Docstrings and annotations are removed.
   - The feature's own column name becomes `'<feature>'`.
3. **Hash** the dumped tree:

```
fingerprint = sha1( ast.dump(normalised tree) )[:16]      # 16 hex characters, 64 bits
twin        = an earlier built feature with the same fingerprint  → refused
```

Two scripts that differ only in names, comments, docstrings or layout get the same
fingerprint. A different constant (a 30-day instead of a 90-day window), a
different input column, or a reordered expression gives a different fingerprint,
which C0.4 then checks.

| Script | Fingerprint |
|---|---|
| `out["pay_ratio_90d"] = base["pay_90d"] / base["spend_90d"]  # comment` | `0a4dcd8cfbed6ef0` |
| same, but the variable is `result`, the name is `repaid_share`, and it has a docstring | `0a4dcd8cfbed6ef0` → refused |
| same with `30d` windows | `5b2f8b9c1e9e1be3` → passes C0.2 |

### C0.3 Same meaning (before the run)

`checks.similarity` scores how close two texts are in words: a tf-idf cosine with
no embedding and no model call. The text compared is `name + " " + description`.

**Words.** Underscores become spaces and everything is lower case. Tokens are
`[a-z0-9]+`. Words shorter than 3 letters are dropped, and so are stop words: plain
English plus *customer, account, feature, measure* and their plurals. A trailing
*s* is cut from a word longer than 4 letters that doesn't end in *ss*.

**Score.** The new text t is compared with o₁ … oₘ, and the corpus is those n = m+1
texts:

```
weight_d(w)   = count_d(w) · ( ln((1+n) / (1+df(w))) + 1 )      df(w): texts containing w
overlap(t, o) = Σ_w weight_t(w)·weight_o(w) / (‖weight_t‖·‖weight_o‖)      in [0, 1]
nearest       = the o with the largest overlap
```

If overlap ≥ `same_meaning` (0.6) and `differs` is empty, the feature is refused:
*"it reads much like `X` from run R (…, overlap 0.64). Say in `differs` what this
one measures that it does not."* With `differs`, it goes on. Either way, the board
records the nearest earlier feature and the overlap.

Rare words shared by both texts weigh the most. Because idf is recomputed over each
comparison set, the same pair scores differently against a large board than
against a few texts. For example, one real pair of ideas scored 0.40 against the
board and 0.31 as a pair alone. Long descriptions push scores down.

### C0.4 Identical values (after the build)

Each screened feature's values on the screen rows are kept in
`values/<name>.parquet`. The new values x are compared with each stored y, from
every run, by Spearman's rank correlation:

```
rows   = ids where x and y are both present          (inner join, missing dropped)
skip   if |rows| < 50  or  x has < 2 distinct values  or  y has < 2
rho_s  = Pearson( rank(x), rank(y) )                  (ties get the average rank)
twin   if |rho_s| ≥ same_values (0.99)               → refused, attempt given back
```

The test uses |ρ|, so a negation (−x) or any monotone transform (log x, x·c, x+c)
is caught. A real variation is not: a 30-day and a 90-day sum of the same events
correlate well below 0.99, so it is screened. This level catches what C0.2 misses,
such as the same quantity computed by different code (`a/b` and `1/(b/a)`).

---

## C1 – redundancy with the model

C1 asks whether the feature is a near-copy of a column the model already has.

**Precomputed once per screener (`Sketch`).** Every model column is ranked on the
screen rows (train and held-out together), with these rules:

```
r(v)  = rank(v) − mean(rank(v)),   a missing value → 0   (the centre: it carries no order)
z_j   = r(column_j) / std(r(column_j))                   (constant columns dropped)
```

**Per feature f.** One matrix product gives f's rank correlation with every column:

```
z_f    = r(f) / std(r(f))
rho_j  = (1/n) · Σ_rows z_j · z_f            for every model column j
C1     = max_j |rho_j|,   nearest = argmax_j      (board: max_model_rho, nearest, nearest_rho)
```

**Decision.**

```
fails  if C1 > max_model_rho (0.9)
  unless |rho_r| ≥ keep_if_residual_rho (0.05)  and  C1 < REBUILT (0.99)   → noted, fitted anyway
         (rho_r: rank correlation with the base model's residual, below)
```

A constant feature has no C1 and passes on to C3. C1 is pairwise on purpose: it
names one column in a number anyone can read. A feature rebuilt from two or three
columns together can pass C1, and the fit in C3 judges that one.

C1 isn't exactly Spearman when values are missing. Setting a missing value to the
centre (instead of dropping the row) pulls the correlation of a sparse feature
toward 0.

---

## C2 – variable Gini

C2 asks how well the feature alone ranks the outcome: its **variable Gini**, the
Gini of the feature used by itself as a score for y, on every screen row
(`checks.Sketch.variable_gini`):

```
f      = the feature, a missing value set to its median (it ranks in the middle)
r_i    = rank of f_i among all screen rows (ties share the average rank)
AUC    = ( Σ_{y_i = 1} r_i − n1·(n1 + 1)/2 ) / (n1 · n0)      Mann-Whitney
Gini   = | 2·AUC − 1 |                                       board: variable_gini
```

The absolute value makes the direction irrelevant: a feature that ranks defaulters
low is as informative as one that ranks them high. A constant feature, or a screen
with one class, has none.

C2 is **information only** by default (`min_variable_gini: null`) and shown on the
board. With `min_variable_gini` set to a number, a feature fails when its Gini is
below it: *"no signal: the feature alone ranks the outcome with a Gini of 0.012,
below 0.05"*.

C2 says nothing about what the model already has. A strong variable Gini can
belong to a near-copy of a model column (C1 catches that), and a weak one can
still add to the model through a non-monotone shape or with other columns (C3
judges that).

**C1's exception** is separate from C2. It still uses the rank correlation with
the base model's residual, `rho_r` (board: `residual_rho`):

```
e      = y − ŷ_base                                   on the screen's held-out rows
rho_r  = r(f)·r(e) / sqrt( (r(f)·r(f)) · (r(e)·r(e)) )     centred ranks, as in C1
noise  ≈ 1 / sqrt(held-out rows)                      4,000 rows → about ±0.016
```

A feature's own Gini can't say whether it tracks what the model misses; the
residual can.

---

## C3 – screen gain

C3 asks whether adding the feature raises the model's score on rows the fit did not
see (`Screener.evaluate_values` → `_score_extended`, then `_below_gates`).

### Guards (`discovery/guards.py`), before the fit

| Guard | Rule | Message starts |
|---|---|---|
| finite | no ±inf and no non-numeric values (NaN is allowed: the booster reads it as missing) | `… has N infinite or non-numeric values` |
| scale | the largest \|value\| must be ≤ `spike_factor` (1000) × the 99th percentile of \|value\|. If the 99th percentile is 0 (a sparse column), the 10th percentile of the nonzero values is used instead. Columns with ≤ 2 distinct values are skipped. | `… spikes on the sample data` |
| redundancy | the largest \|Spearman ρ\| with any base column (pandas `corrwith` on ranks, missing values dropped pairwise) must be ≤ the limit. The inherited limit is `discovery.redundancy_max_abs` if set, else `feature_selection.spearman.redundancy_max_abs` (0.80). For the agent it is raised to at least `REBUILT` (**0.99**) whenever C1 is on, so C1 makes the redundancy call and this guard only stops a ranked copy. | `… is redundant: \|rho\|=… against the existing column` |
| matrix | the whole base + feature matrix is numeric and finite | `The sample matrix has infinite …` |

A guard that fails ends the screen: the feature fails with that message, and the
attempt is spent.

### The fit

- **Estimator.** The same estimator and parameters as the final verdict
  (`validation.model.fit_booster`, `model.params`), so a screen gain and a verdict
  gain are the same kind of quantity.
- **Rounds.** `discovery.screen_boost_rounds` (150).
- **Rows.** It fits on the screen's train rows and scores on its held-out rows. It
  never scores on rows it fit, because in-sample a booster rebuilds a ratio from
  its inputs and the gain becomes noise. The held-out split is named `eval`, so
  early stopping never watches it.
- **Column subsampling is off** (`colsample_* = 1`). Otherwise the number of
  columns alone moves the score: a constant column measured −0.0055.
- **Baseline.** Fit once per screener and reused, so every candidate is compared
  with the same reference.

### The scores

Both are computed on the held-out rows, from predictions p and outcomes y.

**Adjusted (normalised) Gini** (`validation.metrics.calc_adj_gini`): the model's
Gini divided by the Gini of a perfect ordering:

```
sort rows by p (descending);  c_k = cumulative share of all positives in the top k rows
G_model   = (2·Σ_k c_k − c_N) / N − 1                  (= 2·AUC − 1, from the capture curve)
G_perfect = the same, with rows sorted by y
adj_gini  = G_model / G_perfect
```

**Capture rate** (`validation.metrics.capture_rate`): the share of all positives
that falls in the top `discovery.capture_percent` (5%) of rows ranked by p.

**Gains** are base + feature minus base, from the same fits:

```
delta        = adj_gini(base + f) − adj_gini(base)          → board: C3 (screen gain), gini_gain
capture_gain = capture(base + f) − capture(base)
```

### Gates (`_below_gates`)

```
fails if delta ≤ min_gini_gain                       (agent.min_gini_gain: 0.0 → strictly above)
fails if min_capture_gain is set and capture_gain ≤ min_capture_gain   (null → not gated)
verified = the screen ran ok and neither gate failed
```

Each run can set its own gates (`RunParams.min_gini_gain`, `min_capture_gain`). A
verified feature's script is saved to `features/<name>.py`.

C3 is a screen, not the decision. The verdict comes later from the evaluation
(`leave_one_in` over the full splits, with tuning, SHAP and the verdict gates,
e.g. `min_gini_gain: 0.005` there).

---

## Thresholds

| Key | Value | Check | Meaning |
|---|---|---|---|
| `agent.checks.same_meaning` | 0.6 | C0.1, C0.3 (and ideas) | word overlap at which two texts read the same; `null` turns it off |
| `agent.checks.same_values` | 0.99 | C0.4 | \|Spearman\| at which two features are one |
| `memory.same_feature` (code) | 50 rows, 2 values | C0.4 | minimum needed to compare at all |
| `agent.checks.max_model_rho` | 0.9 | C1 | \|ρ\| with one model column above which the feature is redundant; `null` turns it off |
| `agent.checks.keep_if_residual_rho` | 0.05 | C1 | the \|residual ρ\| that saves a C1 failure |
| `checks.REBUILT` (code) | 0.99 | C1 | above this, the exception cannot save it |
| `agent.checks.min_variable_gini` | null | C2 | a number makes C2 a gate |
| `discovery.spike_factor` | 1000 | C3 guard | max \|value\| over its 99th percentile |
| C3 redundancy guard (agent) | max(0.80, 0.99) = 0.99 | C3 guard | \|Spearman\| with a base column: the inherited limit (`discovery.redundancy_max_abs`, else `feature_selection.spearman.redundancy_max_abs`), raised to `REBUILT` while C1 is on. The discovery stage keeps 0.80 |
| `discovery.screen_boost_rounds` | 150 | C3 | boosting rounds of the screen fit |
| `discovery.capture_percent` | 0.05 | C3 | top share of rows for the capture rate |
| `agent.min_gini_gain` | 0.0 | C3 gate | the Gini gain must be strictly above this |
| `agent.min_capture_gain` | null | C3 gate | the capture gain must be strictly above this, if set |

---

## Points to know

- **C1 makes the redundancy call.** C1 (0.9, with its residual exception) is the agent's
  redundancy rule. The C3 guard inherits 0.80 from feature selection, but for the
  agent's screener it is raised to 0.99 (`agent/workspace.py`). Without that, the
  guard would reject every feature the exception saves from C1, and a feature between 0.80
  and 0.90 would pass C1 only to fail in C3 after its attempt was spent. The
  fixed discovery stage (`discovery/stage.py`) still screens at 0.80. The
  evaluation's pipeline also still applies feature selection
  (`validation/stages/feature_selection.py`: |Spearman| > 0.80, or normalised MI
  > 0.80, against the columns it compares with). A feature the exception saved from C1 is
  therefore flagged there as redundant. Whether it is dropped or only measured
  depends on `run.gates` (`enforce` or `open`).
- **C0.3 rarely fires.** Real paraphrases score about 0.3–0.4 against long
  descriptions, so C0.2 and C0.4 do most of the deduplication.
- **C2 is not a gate by default.** A feature's own Gini ignores what the model
  already has, so C1 and C3 decide; set `min_variable_gini` to drop features
  that alone barely rank the outcome.
- **C3's gain is one fit on one split.** Small gains near 0 are within noise. The
  evaluation decides.
