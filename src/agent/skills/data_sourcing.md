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
  * `raw[name]` - a usable source as it is on disk.
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
   (e.g. `<customer_id>_<dt>_<marker>`: split on `_`; `dt` is the as-of date,
   YYYYMMDD). Name the date column `as_of`.
2. Find the source's key and time columns from its description; cast key types
   to match (ids are often strings in one and numbers in the other).
3. Join on the key; keep only events before the as-of date:
   `strict` = `event < as_of` (the default - same-day events may not be known
   yet), `inclusive` = `event <= as_of`.
4. Do not aggregate here. Linkage returns event rows; features aggregate.
   You may drop rows far older than any feature could want (e.g. > 2 years).

The tool reports rows, match rate (share of model ids with at least one event),
and point-in-time violations, then waits for the user. A violation fails the
proposal - fix the filter. If the user rejects, read their note and revise.

Probe the join first (`run_probe`) so the proposal is right the first time.

## L3 - when the data you need is not there

When the direction needs information no model column or source carries:
1. Check `scope(status="unused_raw")` and `catalog` for the CAS variables that
   carry it. A variable marked `requested` is already asked for by an earlier
   request (the brief lists them). Build on the others. Use a requested column
   only beside new columns that add information. If the direction needs only
   requested columns, say so in your report - do not ask for them again.
2. The ideas come first - the run's first stage. Each L3 idea writes the CAS
   variables it needs in `data`, spelled as `scope()` lists them. An idea with
   no CAS variable is an L1/L2 idea and is sent back.
3. Call `screen_request(gap, sql, source_name)`:
   * `gap` - one paragraph: what is missing, why the direction needs it, and
     which features it would enable.
   * `sql` - BigQuery SQL over the CAS tables, selecting only the needed
     columns, the join key, and the event date, for the customers and the date
     range of the model sample. Filter early; these tables are very large.
   * `source_name` - a short snake_case name for the new source.
4. The user approves, runs the SQL, and drops the result in the additional data
   folder. Do not wait for it - continue with what you can build now.
