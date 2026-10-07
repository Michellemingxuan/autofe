---
name: data_sourcing
description: Source the data a feature needs - explore the model rows and the extra sources, write the point-in-time linkage for a source, and (L3) write SQL for data that is missing.
---
# Data sourcing skill

## What you can read

* `catalog(query)` - search model columns, sources, and the CAS scope; each
  model or source column comes with example values, so one search shows what a
  column holds. Several keywords in one query are searched together. Empty
  query = overview. Each CAS variable has a status:
  * `in_model` - already used by the incumbent model (via a matched variable)
  * `in_model_unused` - in the model's inputs but carries no importance
  * `unused_raw` - a raw variable the model does not use: the scope for L3
* `sample_rows(source, n, columns)` - a few rows. `model_database` gives
  labelled screen rows; a source name gives its first rows (or its JSON
  samples when only the schema exists).
* `shots(category)` - curated labelled examples, by category. Empty
  category = the list: the clustering shots (representative rows of every
  region and class) and the user's own categories, each with a context saying
  what its rows are about. Read the ones the direction touches - they are the
  cases the user cares about.
* `run_probe(code, purpose)` - run code to look at data. Available names:
  * `base` - the model rows: the id column and the base features. No target.
  * `raw[name]` - a usable source as it is on disk; in a probe, its first
    200,000 rows - enough to see keys, types and formats, fast on any size.
  * `sources[name]` - a source already joined through its confirmed linkage.
  * `pd`, `np`; with the spark engine also `spark` and `F`
    (`pyspark.sql.functions`), and every frame is a Spark DataFrame.
  Print what you need, or set `result = <frame>` to see its head and schema.
  Keep output small: aggregate, count, describe - do not print whole frames.

Scripts may not open or read files, import os/sys/subprocess, or write
anything; the frames above are all they get.

## Linkage - once per source, confirmed by the user

A source can be used in a feature only after its linkage is confirmed. Write it
with `propose_linkage(source, code, time_column, rule)`. The code defines:

```python
def link(base_ids, source):
    # base_ids: the model ids (one column, the id column)
    # source:   the raw source
    # return:   source rows joined to the ids that may see them, with columns
    #           <id column>, as_of, the source's columns (keep time_column)
```

Steps:
1. Parse the model id into its parts using the id format given in your brief
   (for a format like `<key>_<date>_<marker>`: split on `_`; the date part is the
   as-of date). Name the date column `as_of`.
2. Find the source's key and time columns from its description; cast key types
   to match (ids are often strings in one and numbers in the other).
3. Join on the key; keep only events before the as-of date:
   `strict` = `event < as_of` (the default - same-day events may not be known
   yet), `inclusive` = `event <= as_of`.
4. Do not aggregate here. Linkage returns event rows; features aggregate.
   You may drop rows far older than any feature could want (e.g. > 2 years).

Write `link()` for the linkage engine your brief names - pandas, unless it says
PySpark. A link reads the whole source, so keep only the columns you need before
the join, and join on the key alone. On PySpark, `source` is a Spark DataFrame
and `F` is `pyspark.sql.functions`. The example shows the shape only: `<id>` is
the id column, `<key>` the source's join key, `<event_date>` its event date,
`<column>` a column to keep - take the real names, and how the id splits, from
your brief:

```python
def link(base_ids, source):
    ids = (base_ids
           .withColumn("<key>", F.split("<id>", "_").getItem(0))
           .withColumn("as_of", F.to_date(F.split("<id>", "_").getItem(1), "yyyyMMdd")))
    events = (source.select("<key>", "<event_date>", "<column>")
              .withColumn("<key>", F.col("<key>").cast("string"))
              .withColumn("<event_date>", F.to_date("<event_date>")))
    joined = ids.join(events, "<key>")
    return joined.where(F.col("<event_date>") < F.col("as_of"))     # strict
```

The tool reports rows, match rate (share of model ids with at least one event),
and point-in-time violations, then waits for the user. A violation fails the
proposal - fix the filter. If the user rejects, read their note and revise.

Probe the join first (`run_probe`) so the proposal is right the first time.

## L3 - when the data you need is not there

When the direction needs information no model column or source carries, it is
one of two kinds of request. Both are challenged: can the data that exists now
already supply it?

**Within the CAS scope** - the tables the analyst listed, with SQL:
1. Check `scope(status="unused_raw")` and `catalog` for the CAS variables that
   carry it. A variable marked `requested` is already asked for by an earlier
   request (the brief lists them). Build on the others. Use a requested column
   only beside new columns that add information.
2. Its idea (the run's first stage) writes the CAS variables it needs in `data`,
   spelled as `scope()` lists them.
3. Call `screen_request(gap, sql, source_name)`:
   * `gap` - one paragraph: what is missing, why the direction needs it, and
     which features it would enable.
   * `sql` - BigQuery SQL over the CAS tables, selecting only the needed
     columns, the join key, and the event date, for the customers and the date
     range of the model sample. Filter early; these tables are very large.
   * `source_name` - a short snake_case name for the new source.
   The SQL is screened against the CAS columns; a refusal costs nothing.

**Beyond the CAS scope** - data the bank or the market holds elsewhere:
external information (bureau triggers, macro, merchant or industry data), the
strategies applied to an account (RLA, line actions, collections treatment),
calling and contact history, servicing and complaints, and more. Nobody here can
describe all of it; the idea is what counts.
1. Its idea is marked `beyond_cas`, and its `data` says what it needs and where
   it would come from.
2. Call `propose_new_data(gap, source_name, data, features)` - no SQL. Be
   specific: the behaviour it shows, the grain (per account, per call, per
   month), how far back, and why the model cannot see it now.

**The challenge - every request, either kind.** Right after proposing, challenge
it yourself with `challenge_request(intent, verdict, reasoning, columns, code)` -
be sceptical, and ask one question: can the information it asks for be built
from the data that exists now?
* `constructible` - the model database and the linked sources already give it,
  with the same meaning. Write the construction: a pandas `build(spark, sources,
  base)` returning the id column and ONE column named `proxy`. It is run; if it
  runs, the request is dropped.
* `partly` - a close proxy exists, but something real is missing. Give the
  proxy's construction if you can.
* `new` - the current data does not carry it.
Judge the information, not the name. A kept request is a result; a dropped one is
not - propose something else.

The user reviews the kept requests when the run ends (data_requests.md); data
that arrives lands in the additional data folder as a new source. Do not wait for
it - continue with what you can do now.
