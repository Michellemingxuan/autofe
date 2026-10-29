---
name: feature
description: Write one feature as a script at level L1 or L2, and screen it with screen_feature.
---
# Feature skill

## Levels

* **L1** - the model database alone: ratios, differences, interactions,
  thresholds, missingness patterns of model columns. Cheap, but the incumbent
  model can already approximate many of these; prefer ones that encode domain
  meaning.
* **L2** - anything that reads additional data: a source through its confirmed
  linkage, or an aggregate of one (below) - on its own (a count, a trend, a
  recency, a share by category) or combined with model columns (90-day spend
  divided by the credit limit).
* **L3** - data nobody has yet: not a feature level here, but a request
  (see the data_sourcing skill) that makes later L1/L2 features possible.

The level follows the code: a script that reads `sources[...]` is L2, one that
does not is L1. A wrong level is sent back, at no cost.

## Long sources: aggregate first, then build

A linked source is long: a row per id and event before its as-of date, so a
customer under several as-of dates brings every event once per date. Years of
card spend linked this way is far larger than a feature needs, and every
feature re-reads it. Summarise the source first, then build on the summary:

1. **`define_aggregate`** - a novel summary of the source, made once:
   temporary features you will build on. Two grains:
   * **`period="transaction"`** - one row per model id: the summary of its own
     events before its as-of date. You write `aggregate(spark, raw)`; `raw` is
     a chunk of ids' events joined by the linkage (`<id>`, `as_of` and the
     source's columns), cut to the last `window_days` days when you give it.
     Group by `<id>`. The runner joins and summarises a chunk of ids at a time,
     so the ids x events table is never held whole. The grain when the source
     is a short history before each transaction (the last 7 days of spend, 30
     days of payments) - or when the window is fixed.
   * **`period="day" | "week" | "month"`** - one row per customer and period,
     on the raw source before any join. `raw` is the source with `period_end`
     added (the last day of each event's period); group by the key the linkage
     joins on and `period_end`, and name that key (`key="<key>"`): a source too
     large for memory is split by it and summarised a part at a time. The runner then joins it through
     the source's linkage - only periods that end before each as-of date. The
     grain when features need several windows over a long history (3 vs 12
     months, a trend, a volatility).
   Both are checked point in time. No attempt is spent.
2. **`screen_feature`** - read it as `sources["<aggregate>"]` and combine with
   model columns: per transaction, map its columns onto `base` by the id; per
   period, roll the periods up into a window first. One aggregate serves many
   features: spend against the limit, the share at risky merchants, the trend,
   the months since the last spike.

What makes an aggregate worth defining:
* Several columns in one pass - the totals, counts, maxima and category shares
  the direction needs (`spend`, `n_txn`, `max_txn`, `travel_spend`,
  `n_merchants`), not one number.
* Shapes that combine later: sums, counts, maxima, minima, sums of squares
  (for a variance), counts by category, distinct counts, the first and last
  event day.
* The grain the idea needs: `transaction` for one window before each
  transaction; `month` for trends and volatility over months; `week` for
  payday or month-end rhythm; `day` when windows vary but must end exactly at
  the as-of date (a month that ends after it is left out whole).

pandas example - a per-transaction aggregate of the last 7 days
(`define_aggregate(name="<source>_7d", source="<source>", period="transaction",
window_days=7, ...)`):

```python
def aggregate(spark, raw):
    r = raw.assign(risky=raw["<amount>"].where(raw["<risk_score>"] < 0, 0.0))
    return (r.groupby("<id>", as_index=False)
             .agg(spend_7d=("<amount>", "sum"), n_txn_7d=("<amount>", "size"),
                  max_txn_7d=("<amount>", "max"), risky_spend_7d=("risky", "sum"),
                  n_merchants_7d=("<merchant>", "nunique")))
```

Read a linked source directly only for what needs single events (the exact
day of the last payment, the gap between two events), and filter it to the
window first.


## Before you propose

The brief's "Earlier directions" section lists every feature already proposed
and not deleted, with its result. Read it before you give your ideas, and plan
around it:
* Do not propose one of them again, under a new name or with the same
  computation. A feature with the same values is refused.
* A variation is fine when it measures something else - another window,
  another ratio, another source, another population. In `description`, say
  what is new compared with the nearest earlier feature.
* A ✗ (not verified) feature is a lead only with a real change: say what the
  change is and why it should now carry signal.
* Give each attempt to a different idea. Two features of one idea compete for
  the same signal.

## The checks before the fit

`screen_feature` checks a feature cheaply before it is fitted; the reply's
`checks` says what each measured.

* **C0 - seen before?** The same code as an earlier feature (only the names
  changed) is refused. A name and description that read much like an earlier
  feature's are refused until `differs` says what this one measures that it
  does not. Identical values (rank correlation 0.99 or more) are refused after
  the build. A refusal costs nothing.
* **C1 - redundancy with the model?** `max_model_rho` is the largest |Spearman
  rho| between the feature and any one model column; `nearest` is that column.
  Above the limit it is a near-copy the model already has: build on other data,
  not another window of the same column.
* **C2 - signal?** `variable_gini` is the feature's own Gini: how well it alone
  ranks the outcome, |2·AUC - 1| on the screen rows - information, not a gate:
  every feature that passes C0 and C1 is fitted. It says nothing about what the
  model already has: a strong one may be a copy (C1), a weak one may still add
  to the model (a non-monotone shape, or with other columns), so judge by the
  screen's gain.

## The contract

`screen_feature(name, description, level, code, idea, differs)` runs your
script on the screen rows, checks it and scores it. `idea` names the idea the
feature tests; `differs` says what it measures that a look-alike earlier
feature does not (only when the reply asks). The script defines:

```python
def build(spark, sources, base):
    # base:    model rows - id column + base features (no target), each id once
    # sources: {name: linked rows} for confirmed sources (id, as_of, events),
    #          and for this run's aggregates (id, as_of, one row per period)
    # spark:   the SparkSession, or None on the pandas engine
    # return:  one row per id: the id column and exactly one column named `name`
```

Rules:
* Read only the columns the brief lists under "The columns you can use" - for
  `base` and for each source. A name not listed there does not exist.
* Return exactly two columns: the id column and `name`. One row per id. `base`
  has each id once - on the full data too, where the model rows can repeat an
  id - so start from `base[["<id>"]]` and map onto it; no de-duplication needed.
* Ids with no events may be missing from the result - they become NaN. Fill
  them yourself when "no events" has a meaning (a count of 0, say).
* Read sources and aggregates only through `sources["<name>"]`; mention each by
  its quoted name, so the runner joins it in.
* Windows count back from `as_of`: `<event_date> >= as_of - 90 days`. Linkage has
  already removed events at or after as_of.
* Guard divisions: `np.where(den > 0, num / den, np.nan)` (pandas) or
  `F.when(den > 0, num / den)` (spark). Infinite values are rejected.
* No file reads, writes, or os/sys imports.

pandas, the slips that cost attempts:
* Dates: keep them as pandas Series - `(s["as_of"] - s["<event_date>"]).dt.days`.
  `.values` turns them into numpy `datetime64`, which has no `.days`, and
  `np.timedelta64(1, "M")` (months) is not supported - count days.
* `base` has no `as_of`: every `sources[...]` row carries it.
* Speed: filter rows first, then `groupby("<id>").agg(...)` - no loops over rows,
  ids or groups (see "No loops" below: the runner refuses them).
* A ratio's denominator can be 0 or missing: guard it (see above), and fill
  "no events" with what it means (a count of 0), not with 0 for everything.

## No loops - whole-frame operations only

A feature, an aggregate or a linkage may not loop in Python over rows, ids or
groups: `iterrows`, `itertuples`, `apply(..., axis=1)`, `for ... in
df.groupby(...)`, `for x in df["<col>"]`, `for i in range(len(...))`. On the
screen it takes seconds; on the full data it is millions of passes, each
scanning whole frames - hours, past the time limit. The runner refuses such a
script before it runs (no attempt spent). Every per-id question has a
whole-frame answer:

| You want, per id | Write |
|---|---|
| a value in a window before as_of | filter `s[(s["<event_date>"] < s["as_of"]) & (s["<event_date>"] >= s["as_of"] - pd.Timedelta(days=N))]`, then `groupby("<id>").agg(...)` |
| the last (or first) event before as_of | `s.sort_values("<event_date>").groupby("<id>").tail(1)` (`.head(1)` for the first) |
| the latest value vs the one before | sort, then `g = s.groupby("<id>")["<amount>"]`; `s["prev"] = g.shift(1)`; keep the last row per id |
| the n-th latest period | sort descending, then `groupby("<id>").nth(n)` |
| gaps between consecutive events | sort, then `s.groupby("<id>")["<event_date>"].diff().dt.days`, then aggregate the gaps |
| a flag from several columns of one row | `np.where(cond_a & cond_b, 1, 0)` on whole columns - never `apply(axis=1)` |
| a count or sum under a condition | a 0/1 or masked column first (`s["<amount>"].where(cond, 0)`), then `groupby(...).sum()` |
| a trend (slope) over periods | `t` = days before as_of; per id the slope is `cov(t, y) / var(t)`: aggregate `t`, `y`, `t*y`, `t*t` with sums and counts, then combine |
| a value from another frame per id | `merge` (many-to-one, `validate="many_to_one"`) or `map` onto the ids - never filter a frame inside a loop |
| the id's parts (customer, date) | `parts = base["<id>"].str.split("_", expand=True)` - one call for every id |

Example - last full month's payment minus spend, from two monthly aggregates:

```python
def build(spark, sources, base):
    p = sources["<payments>_by_month"][["<id>", "as_of", "<event_date>", "total_payment"]]
    s = sources["<spends>_by_month"][["<id>", "<event_date>", "monthly_spend"]]
    m = p.merge(s, on=["<id>", "<event_date>"], how="inner")       # one row per id and month
    last = m.sort_values("<event_date>").groupby("<id>").tail(1)    # the latest month before as_of
    gap = (last["total_payment"] - last["monthly_spend"]).set_axis(last["<id>"])
    out = base[["<id>"]].copy()
    out["<feature>"] = out["<id>"].map(gap)
    return out
```

A loop over a handful of fixed things - your own list of windows, of
categories, of columns - is fine: `for days in (30, 90, 180):` builds three
filters, not millions.

## Scale - the screen is a sample

Your script is tried on the screen rows, but a verified feature is computed
again at evaluation on the full model data, against the whole of each source -
your brief's "Data size" gives both. A linked source there holds every event of
every id: often hundreds of millions of rows. Write every script for that:
* At evaluation `build()` runs on a chunk of ids at a time, each with only its
  ids' rows: compute each id's value from its own rows and base columns - no
  ranks, quantiles or means across ids.
* Name each column you read in quotes - `s["<amount>"]`, `s[["<id>", "as_of",
  "<event_date>"]]`, `base["<column>"]`. On pandas, only the columns your code
  names (plus the id and `as_of`) are loaded - of the sources and of `base`; a
  column built from a variable is not.
* Filter first: cut the events to your window (and to the categories you need)
  before any groupby, merge or sort.
* A long source is summarised once with `define_aggregate`, not re-read by
  every feature: a month per row instead of every event (see "Long sources").
* **Many-to-many joins are the main danger.** A merge returns, for each key,
  the left rows with that key times the right rows with it. Two event-level
  frames joined on the id - a source with itself, two sources, events with
  events - give every id (its events x its other events) rows: 300 x 300 is
  90,000 rows for one id, and the full data has millions of ids. Before every
  merge, ask: is the key unique on at least one side?
  * Aggregate each side to one row per id (`groupby("<id>").agg(...)`), then
    merge the aggregates - always one-to-one.
  * Merging events with a per-id table (`base`, an aggregate) is many-to-one:
    fine. Pass `validate="many_to_one"` so pandas checks it.
  * Need two sources together per event ("payments within 30 days of a spend")?
    Aggregate both to (id, day) or (id, month) first, then join on those keys.
  * The runner refuses a many-to-many merge in a feature, with the numbers -
    rewrite the join; do not patch around it.
* No pivot or `unstack` over a column with many values, no cross joins, no
  `sort_values` over all events when a `groupby(...).max()` / `idxmax` gives the
  answer, no `.astype(str)` on a large column.
* Prefer one `groupby(...).agg(...)` with several outputs to several passes.
* Each screen reply says how the script scales: a `scale` note means it would
  not finish, or not fit, on the full data. Treat it like a failure - the next
  script must be leaner.

The examples show the shape of the code only. Names in `<angle brackets>` stand
for real ones from your brief: `<id>` the id column, `<source>` a source,
`<event_date>` and `<amount>` its columns, `<feature>` your feature's name.

pandas example (L2) - a monthly aggregate, then features on it. First the summary,
once (`define_aggregate(name="<source>_by_month", source="<source>",
period="month", key="<key>", ...)`):

```python
def aggregate(spark, raw):
    r = raw[["<key>", "period_end", "<amount>", "<category>"]]
    r = r.assign(travel=r["<amount>"].where(r["<category>"] == "travel", 0.0))
    return (r.groupby(["<key>", "period_end"], as_index=False)
             .agg(spend=("<amount>", "sum"), n_txn=("<amount>", "size"),
                  max_txn=("<amount>", "max"), travel=("travel", "sum")))
```

Then a feature on it - the monthly spend of the last 3 months against that of
the 9 before (`<event_date>` is the period's last day). Dividing by a model
column such as the credit limit is the same one more step:

```python
def build(spark, sources, base):
    m = sources["<source>_by_month"]
    age = (m["as_of"] - m["<event_date>"]).dt.days
    out = base[["<id>"]].set_index("<id>")
    out["recent"] = m[age < 92].groupby("<id>")["spend"].sum() / 3
    out["before"] = m[(age >= 92) & (age < 365)].groupby("<id>")["spend"].sum() / 9
    out["<feature>"] = np.where(out["before"] > 0,
                                out["recent"].fillna(0) / out["before"], np.nan)
    return out[["<feature>"]].reset_index()
```

pandas example (L2) - straight from a linked source, a sum over the last 90 days:

```python
def build(spark, sources, base):
    s = sources["<source>"]
    recent = s[s["<event_date>"] >= s["as_of"] - pd.Timedelta(days=90)]
    out = recent.groupby("<id>")["<amount>"].sum().rename("<feature>").reset_index()
    out = base[["<id>"]].merge(out, on="<id>", how="left").fillna({"<feature>": 0.0})
    return out
```

PySpark example (same feature) - only when your brief's engine is PySpark:

```python
def build(spark, sources, base):
    s = sources["<source>"]
    recent = s.where(F.col("<event_date>") >= F.date_sub(F.col("as_of"), 90))
    agg = recent.groupBy("<id>").agg(F.sum("<amount>").alias("<feature>"))
    return base.select("<id>").join(agg, "<id>", "left").fillna({"<feature>": 0.0})
```

## Name and describe

`name`: snake_case, new, says what it is and over what window
(`<measure>_<window>`, e.g. `<amount>_sum_90d`).
`description`: one sentence a risk analyst would understand - what it measures
and why it should carry default risk.

Probe first when unsure of a column's type or range; a failed screen still
uses an attempt.
