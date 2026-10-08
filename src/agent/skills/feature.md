---
name: feature
description: Write one feature as a script at level L1 or L2, and screen it with screen_feature.
---
# Feature skill

## Levels

* **L1** - from one kind of data:
  * model columns only: ratios, differences, interactions, thresholds,
    missingness patterns. Cheap, but the incumbent model can already
    approximate many of these; prefer ones that encode domain meaning.
  * or an aggregation of one extra source, through its confirmed linkage:
    counts, sums, ratios over windows before as_of, recency, trends,
    volatility, shares by category.
* **L2** - an extra-source aggregate combined with model columns, e.g. 90-day
  spend divided by the credit limit.
* **L3** - data nobody has yet: not a feature level here, but a request
  (see the data_sourcing skill) that makes later L1/L2 features possible.

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

## The contract

`screen_feature(name, description, level, code)` runs your script on the
screen rows and scores it. The script defines:

```python
def build(spark, sources, base):
    # base:    model rows - id column + base features (no target)
    # sources: {name: linked rows} for confirmed sources (id, as_of, events)
    # spark:   the SparkSession, or None on the pandas engine
    # return:  one row per id: the id column and exactly one column named `name`
```

Rules:
* Read only the columns the brief lists under "The columns you can use" - for
  `base` and for each source. A name not listed there does not exist.
* Return exactly two columns: the id column and `name`. One row per id.
* Ids with no events may be missing from the result - they become NaN. Fill
  them yourself when "no events" has a meaning (a count of 0, say).
* Read sources only through `sources["<name>"]`; mention each source you use by
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
* Speed: filter rows first, then `groupby("<id>").agg(...)` - never `apply` over
  rows, `iterrows`, or a Python loop over ids. On the screen rows (your brief
  gives their number) vectorised code takes seconds; longer means it is looping.
* A ratio's denominator can be 0 or missing: guard it (see above), and fill
  "no events" with what it means (a count of 0), not with 0 for everything.

## Scale - the screen is a sample

Your script is tried on the screen rows, but a verified feature is computed
again at evaluation on the full model data, against the whole of each source -
your brief's "Data size" gives both. A linked source there holds every event of
every id: often hundreds of millions of rows. Write every script for that:
* Name each column you read in quotes - `s["<amount>"]`, `s[["<id>", "as_of",
  "<event_date>"]]`. On pandas, only the columns your code names (plus the id
  and `as_of`) are loaded; a column built from a variable is not.
* Filter first: cut the events to your window (and to the categories you need)
  before any groupby, merge or sort.
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

pandas example (L1) - a sum of an amount over the last 90 days:

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
