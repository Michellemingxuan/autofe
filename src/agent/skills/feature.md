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
* Spend each intent on a different idea. Two features of one idea compete for
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
* Windows count back from `as_of`: `event_dt >= as_of - 90 days`. Linkage has
  already removed events at or after as_of.
* Guard divisions: `np.where(den > 0, num / den, np.nan)` (pandas) or
  `F.when(den > 0, num / den)` (spark). Infinite values are rejected.
* No file reads, writes, or os/sys imports.

pandas example (L1):

```python
def build(spark, sources, base):
    s = sources["spends"]
    recent = s[s["event_dt"] >= s["as_of"] - pd.Timedelta(days=90)]
    out = recent.groupby("id")["amount"].sum().rename("spend_90d").reset_index()
    out = base[["id"]].merge(out, on="id", how="left").fillna({"spend_90d": 0.0})
    return out
```

Spark example (same feature):

```python
def build(spark, sources, base):
    s = sources["spends"]
    recent = s.where(F.col("event_dt") >= F.date_sub(F.col("as_of"), 90))
    agg = recent.groupBy("id").agg(F.sum("amount").alias("spend_90d"))
    return base.select("id").join(agg, "id", "left").fillna({"spend_90d": 0.0})
```

## Name and describe

`name`: snake_case, new, says what it is (`pay_to_spend_90d`).
`description`: one sentence a risk analyst would understand - what it measures
and why it should carry default risk.

Probe first when unsure of a column's type or range; a failed screen still
spends an intent.
