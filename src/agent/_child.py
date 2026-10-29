"""Runs one agent script in its own process: ``python -m agent._child spec.json``.

The parent writes a spec naming the inputs and the mode, and reads back a JSON
result. Everything the script may touch is handed to it already loaded:

    base      the model rows - id + base columns, no target
    sources   each source joined through its confirmed linkage (feature mode)
    raw       each source as it is on disk, loaded on first access (probe, linkage) -
              in a probe, its first PROBE_ROWS rows: a probe looks, it does not compute
    spark, F  a local SparkSession and pyspark.sql.functions (engine=spark)
    pd, np

On pandas, a feature reads only the source columns its code names in quotes
(plus the id and as_of): a linked source on the full splits is far larger than
memory needs to hold for one feature. The peak memory of the run is reported, so
the parent can project the script to the full data.

On pandas a source is never loaded whole by a linkage or an aggregate: it is read
BATCH_ROWS rows at a time (``stream_link``). At evaluation the joined rows go to
disk by chunk of ids (``Parts``) and a feature is built a chunk at a time.

Every pandas merge is sized before it runs (``guard_merges``): the rows out of a
join are, per key, the left rows times the right rows - with repeated keys on
both sides (many-to-many) that multiplies. A feature may not do it at all; any
script is stopped before a join that would not fit in memory.

Modes:

    probe      run the code; report what it printed, and `result` if it set one
    linkage    call link(base_ids, source) -> rows with id, as_of and the source columns
    aggregate  call aggregate(spark, raw) on one source:
               * by day, week or month - on the raw source, before any join: the runner
                 adds `period_end` - the last day of each event's period - and the script
                 groups by the source's key and period_end. Out comes one row per key and
                 period, period_end standing for the source's time column, so the source's
                 own linkage can join it point in time
               * by transaction - on each id's events, joined by the source's linkage and
                 cut to the window before its as-of date, CHUNK_IDS ids at a time: the
                 ids x events table is never held whole. Out comes one row per id
    feature    call build(spark, sources, base) -> one row per id, one new column
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import resource
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

_MAX_TEXT = 6000
# A probe reads this many rows of each raw source: enough to see keys, types and
# formats, and fast on a source of any size. Linkage and features read every row.
PROBE_ROWS = 200_000
PERIOD_END = "period_end"
# Ids joined to their events at a time, in a per-transaction aggregate on pandas,
# and the ids a feature is built on at a time at evaluation.
CHUNK_IDS = int(os.environ.get("AUTOFE_CHUNK_IDS", "50000"))
# Source rows read at a time on pandas: a linkage or an aggregate streams a source
# in batches of this many rows - it never loads it whole.
BATCH_ROWS = int(os.environ.get("AUTOFE_BATCH_ROWS", "4000000"))
# A day/week/month aggregate on pandas: the source on disk per partition by its key
# (a CSV's bytes; parquet is compressed, so a quarter of it).
PART_BYTES = int(os.environ.get("AUTOFE_PART_BYTES", str(1024 * 2**20)))


def add_period_end(frame: Any, time_column: str, period: str, engine: str) -> Any:
    """`period_end`: the last day of the day, week (to Sunday) or month each event falls
    in, at midnight. A bucket is joined to an as-of date only when that day is before
    it, so every event in it is before it too."""
    if engine == "spark":
        from pyspark.sql import functions as F

        day = F.to_date(F.col(time_column))
        end = {"day": day, "week": F.next_day(F.date_sub(day, 1), "Sun"),
               "month": F.last_day(day)}[period]
        return frame.withColumn(PERIOD_END, F.to_timestamp(end))
    t = pd.to_datetime(frame[time_column], errors="coerce")
    end = {"day": lambda: t.dt.normalize(),
           "week": lambda: t.dt.to_period("W-SUN").dt.end_time.dt.normalize(),
           "month": lambda: t.dt.to_period("M").dt.end_time.dt.normalize()}[period]()
    return frame.assign(**{PERIOD_END: end})


class _Lazy(dict):
    """A dict of paths that loads each frame the first time it is read."""

    def __init__(self, paths: dict[str, str], load):
        super().__init__()
        self._paths, self._load = paths, load

    def __getitem__(self, key):
        if not dict.__contains__(self, key):
            if key not in self._paths:
                raise KeyError(f"no source {key!r}; available: {sorted(self._paths)}")
            dict.__setitem__(self, key, self._load(self._paths[key]))
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        return key in self._paths

    def keys(self):
        return self._paths.keys()

    def __iter__(self):
        return iter(self._paths)

    def __len__(self):
        return len(self._paths)


def _rss_mb() -> float:
    """Peak resident memory so far: kilobytes on Linux, bytes on macOS."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def named_columns(code: str, columns: list[str], keep: set[str]) -> list[str] | None:
    """The columns a script names as quoted strings, plus `keep`; None when it names
    none of them - then it may select by position or dtype, and gets every column."""
    named = [c for c in columns if re.search(r"""['"]""" + re.escape(c) + r"""['"]""", code)]
    if not [c for c in named if c not in keep]:
        return None
    return [c for c in columns if c in named or c in keep]


# A join may take this share of the machine's memory, at most.
MERGE_MEMORY_SHARE = 0.5
# Bytes a cell of a joined frame takes, roughly (numbers 8, short strings more).
_CELL_BYTES = 16


def _machine_bytes() -> float | None:
    import os

    try:
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _keys(frame: pd.DataFrame, cols: list[str], index: bool) -> pd.Series | None:
    """Rows per key value on one side of a join."""
    if index:
        return frame.index.value_counts(dropna=False)
    if not cols or any(c not in frame.columns for c in cols):
        return None                                  # pandas raises its own error
    keys = frame[cols[0]] if len(cols) == 1 else frame[cols]
    return keys.value_counts(dropna=False)


def size_merge(left: pd.DataFrame, right: pd.DataFrame, *, how: str = "inner", on=None,
               left_on=None, right_on=None, left_index: bool = False,
               right_index: bool = False, **_: Any) -> dict[str, Any] | None:
    """Rows a merge will produce, from the rows per key on each side - without
    making it. None when the keys cannot be read (pandas reports that itself)."""
    if how == "cross":
        return {"rows": len(left) * len(right), "many_to_many": len(left) > 1 and len(right) > 1,
                "left_per_key": len(right), "right_per_key": len(left)}
    as_list = lambda v: [] if v is None else ([v] if isinstance(v, str) else list(v))  # noqa: E731
    lcols, rcols = as_list(left_on if left_on is not None else on), \
        as_list(right_on if right_on is not None else on)
    if not lcols and not rcols and not (left_index or right_index):
        lcols = rcols = [c for c in left.columns if c in right.columns]
    lc, rc = _keys(left, lcols, left_index), _keys(right, rcols, right_index)
    if lc is None or rc is None or lc.empty or rc.empty:
        return None
    try:
        lc, rc = lc.align(rc, join="inner")
    except Exception:  # noqa: BLE001 - mismatched key types: pandas reports it
        return None
    matched = int((lc * rc).sum())
    rows = matched + (len(left) if how in ("left", "outer") else 0) \
        + (len(right) if how in ("right", "outer") else 0)
    return {"rows": rows, "many_to_many": bool(len(lc) and lc.max() > 1 and rc.max() > 1),
            "left_per_key": float(rc.max()) if len(rc) else 0.0,
            "right_per_key": float(lc.max()) if len(lc) else 0.0}


def guard_merges(mode: str) -> None:
    """Size every pandas merge before it runs. In a feature, a many-to-many merge
    is refused: aggregate a side to one row per key first. Anywhere, a merge that
    would not fit in memory is stopped with its numbers, not killed by the OS."""
    original = pd.merge
    machine = _machine_bytes()

    def merge(left, right, *args, **kwargs):
        if not args and isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
            size = size_merge(left, right, **kwargs)
            if size:
                bigger = max(len(left), len(right))
                if mode == "feature" and size["many_to_many"] and size["rows"] > bigger:
                    raise ValueError(
                        f"many-to-many merge: keys repeat on both sides ({len(left):,} x "
                        f"{len(right):,} rows -> {size['rows']:,} rows). On the full data it "
                        "multiplies further. Aggregate one side to one row per key first "
                        "(groupby(...).agg(...)), then merge.")
                need = size["rows"] * (len(left.columns) + len(right.columns)) * _CELL_BYTES
                if machine and need > MERGE_MEMORY_SHARE * machine:
                    raise MemoryError(
                        f"this merge would make {size['rows']:,} rows (about {need / 1e9:.0f} GB; "
                        f"the machine has {machine / 1e9:.0f} GB): "
                        + ("keys repeat on both sides, so the rows multiply. "
                           if size["many_to_many"] else "")
                        + "Cut both sides first - only the columns needed, only the keys "
                          "and dates in range - or aggregate a side to one row per key.")
        return original(left, right, *args, **kwargs)

    def frame_merge(self, right, *args, **kwargs):
        return merge(self, right, *args, **kwargs)

    pd.merge = merge
    pd.DataFrame.merge = frame_merge


def _schema(path: str) -> list[str]:
    if is_parts(path):
        return json.loads(open(os.path.join(path, "_meta.json")).read())["columns"]
    if path.endswith(".csv"):
        return list(pd.read_csv(path, nrows=0).columns)
    import pyarrow.dataset as ds

    return list(ds.dataset(path, format="parquet").schema.names)


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "toPandas"):
        return frame.toPandas()
    if isinstance(frame, pd.Series):
        return frame.to_frame()
    raise TypeError(f"expected a DataFrame, got {type(frame).__name__}")


def iter_source(path: str, columns: list[str] | None = None):
    """A source on disk, BATCH_ROWS rows at a time."""
    if path.endswith(".csv"):
        yield from pd.read_csv(path, usecols=columns, chunksize=BATCH_ROWS, low_memory=False)
        return
    import pyarrow.dataset as ds

    for batch in ds.dataset(path, format="parquet").to_batches(columns=columns,
                                                               batch_size=BATCH_ROWS):
        if batch.num_rows:
            yield batch.to_pandas()


def stream_link(link: Any, ids: pd.DataFrame, path: str, columns: list[str] | None = None):
    """(source batch, linked rows) for each batch of the source. A link joins and
    filters row by row, so its rows on the batches are its rows on the whole; the
    first two batches check that - a link that aggregates or dedupes across events
    would give other rows."""
    first = None
    for i, batch in enumerate(iter_source(path, columns)):
        linked = _to_pandas(link(ids, batch))
        if i == 0:
            first = (batch, len(linked))
        elif i == 1:
            both = len(_to_pandas(link(ids, pd.concat([first[0], batch], ignore_index=True))))
            if both != first[1] + len(linked):
                raise ValueError(
                    f"link() is not row by row: on two batches of the source it gives "
                    f"{both:,} rows together but {first[1] + len(linked):,} apart. A long "
                    "source is read in batches, so link() may only join the events to the "
                    "ids and filter them - never aggregate, deduplicate or rank across events")
            first = None
        yield batch, linked


def is_parts(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "_meta.json"))


class Parts:
    """Rows written to disk by chunk of ids - the base ids in their order, CHUNK_IDS
    to a chunk: <folder>/part-<k>/<batch>.parquet, with _empty.parquet (the
    columns, no rows) and _meta.json. A feature then reads one chunk at a time."""

    def __init__(self, folder: str, ids: pd.Series, chunk: int = 0):
        import math

        self.folder, self.chunk = folder, chunk or CHUNK_IDS
        os.makedirs(folder, exist_ok=True)
        position = pd.Series(np.arange(len(ids)) // self.chunk, index=ids.to_numpy())
        self.chunk_of = position[~position.index.duplicated()]
        self.n_chunks = max(math.ceil(len(ids) / self.chunk), 1)
        self.rows, self.batches, self.columns, self.head = 0, 0, None, None

    def write(self, frame: pd.DataFrame, id_col: str) -> None:
        if self.columns is None and len(frame.columns):
            self.columns = list(frame.columns)
            frame.head(0).to_parquet(os.path.join(self.folder, "_empty.parquet"), index=False)
        if frame.empty:
            return
        if self.head is None:
            self.head = frame.head(8)
        for k, rows in frame.groupby(frame[id_col].map(self.chunk_of), sort=False):
            folder = os.path.join(self.folder, f"part-{int(k):05d}")
            os.makedirs(folder, exist_ok=True)
            rows.to_parquet(os.path.join(folder, f"{self.batches:05d}.parquet"), index=False)
        self.batches += 1
        self.rows += len(frame)

    def read(self, k: int, columns: list[str] | None = None) -> pd.DataFrame:
        folder = os.path.join(self.folder, f"part-{k:05d}")
        files = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
        if not files:
            return pd.read_parquet(os.path.join(self.folder, "_empty.parquet"), columns=columns)
        return pd.concat([pd.read_parquet(os.path.join(folder, f), columns=columns)
                          for f in files], ignore_index=True)

    def close(self) -> dict[str, Any]:
        with open(os.path.join(self.folder, "_meta.json"), "w") as fh:
            json.dump({"chunk_ids": self.chunk, "n_chunks": self.n_chunks,
                       "rows": self.rows, "columns": self.columns or []}, fh)
        head = self.head if self.head is not None else pd.DataFrame(columns=self.columns or [])
        return {**_describe(head), "n_rows": int(self.rows), "chunks": self.n_chunks}

    @classmethod
    def open(cls, folder: str) -> "Parts":
        meta = json.loads(open(os.path.join(folder, "_meta.json")).read())
        parts = cls.__new__(cls)
        parts.folder, parts.chunk, parts.n_chunks = folder, meta["chunk_ids"], meta["n_chunks"]
        return parts


class _Done(Exception):
    """A mode that finished early, with its result set."""


def _per_transaction(spec: dict[str, Any], options: dict[str, Any], code: str, aggregate: Any,
                     base: Any, spark: Any, F: Any, load: Any, engine: str) -> dict[str, Any]:
    """aggregate(spark, rows) on each id's events: the source joined to the ids by its
    linkage, cut to the window before each as-of date, then summarised to one row
    per id. On pandas the ids go CHUNK_IDS at a time."""
    id_col, time_column = spec["id_col"], options["time_column"]
    strict = options.get("rule", "strict") != "inclusive"
    window = options.get("window_days")
    scope: dict[str, Any] = {"pd": pd, "np": np, "spark": spark, "F": F}
    exec(compile(options["linkage"], "linkage.py", "exec"), scope)  # noqa: S102 - the confirmed linkage
    link = scope["link"]
    path = spec["raw"][spec["source"]]

    if engine == "spark":
        events = load(path)
        rows = link(base.select(id_col), events)
        t, as_of = F.col(time_column).cast("timestamp"), F.col("as_of").cast("timestamp")
        late = rows.where(t >= as_of if strict else t > as_of).count()
        if late:
            raise ValueError(f"{late:,} joined rows are on or after their as-of date")
        if window:
            rows = rows.where(t >= as_of - F.expr(f"INTERVAL {int(window)} DAYS"))
        n_linked = rows.count()
        table = aggregate(spark, rows)
        if id_col not in table.columns:
            raise ValueError(f"aggregate() must return {id_col!r}: group by it")
        n_out = table.count()
        if table.select(id_col).distinct().count() != n_out:
            raise ValueError(f"aggregate() must return one row per {id_col!r}")
        table.write.mode("overwrite").parquet(spec["out_path"])
        shown = spark.read.parquet(spec["out_path"]).limit(200000).toPandas()
        return {**_describe(shown), "n_rows": int(n_out), "source_rows": int(events.count()),
                "linked_rows": int(n_linked)}

    # On pandas: the source a batch at a time, joined to every id and cut to the
    # window, onto disk by chunk of ids; then each chunk summarised.
    columns = named_columns(options["linkage"] + "\n" + code, _schema(path), {time_column})
    ids = base[[id_col]]
    parts = Parts(spec["out_path"] + ".linked", ids[id_col])
    n_source, n_linked = 0, 0
    try:
        for batch, rows in stream_link(link, ids, path, columns):
            n_source += len(batch)
            t = pd.to_datetime(rows[time_column], errors="coerce")
            as_of = pd.to_datetime(rows["as_of"], errors="coerce")
            late = int(((t >= as_of) if strict else (t > as_of)).sum())
            if late:
                raise ValueError(f"{late:,} joined rows are on or after their as-of date")
            if window:
                rows = rows[t >= as_of - pd.Timedelta(days=int(window))]
            n_linked += len(rows)
            parts.write(rows, id_col)
        parts.close()
        out = []
        for k in range(parts.n_chunks):
            rows = parts.read(k)
            if rows.empty:
                continue
            part = _to_pandas(aggregate(spark, rows))
            if id_col not in part.columns:
                raise ValueError(f"aggregate() must return {id_col!r}: group by it")
            out.append(part)
    finally:
        import shutil

        shutil.rmtree(parts.folder, ignore_errors=True)
    table = pd.concat(out, ignore_index=True) if out else pd.DataFrame({id_col: []})
    if table[id_col].duplicated().any():
        raise ValueError(f"aggregate() must return one row per {id_col!r}; "
                         f"{int(table[id_col].duplicated().sum()):,} repeat")
    table.to_parquet(spec["out_path"], index=False)
    return {**_describe(table), "n_rows": int(len(table)), "source_rows": int(n_source),
            "linked_rows": int(n_linked)}


def _by_period(spec: dict[str, Any], options: dict[str, Any], code: str, aggregate: Any,
               spark: Any, load: Any) -> tuple[pd.DataFrame, int, int]:
    """A day/week/month aggregate on pandas: (its rows, the source rows, the rows
    with a moved period_end). A source larger than PART_BYTES is first split on
    disk by its key - every row of a key in one partition - and each partition
    summarised in turn."""
    import math
    import shutil

    path, time_column, period = spec["raw"][spec["source"]], options["time_column"], options["period"]
    keys = list(options.get("key") or [])
    schema = _schema(path)
    unknown = [k for k in keys if k not in schema]
    if unknown:
        raise ValueError(f"the key {unknown} is not a column of {spec['source']!r}")
    keep = named_columns(code, schema, {time_column, *keys})

    def summarise(events: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        events = add_period_end(events, time_column, period, "pandas")
        table = _to_pandas(aggregate(spark, events))
        if PERIOD_END not in table.columns:
            raise ValueError(f"aggregate() must keep {PERIOD_END!r} - group by the "
                             f"source's key and {PERIOD_END!r}")
        if time_column in table.columns:
            raise ValueError(f"aggregate() must not return {time_column!r}: "
                             f"{PERIOD_END!r} stands for it")
        shifted = int((~table[PERIOD_END].isin(set(events[PERIOD_END].dropna()))).sum())
        return table, shifted

    size = (os.path.getsize(path) if os.path.isfile(path) else
            sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs))
    n_parts = math.ceil(size * (1 if path.endswith(".csv") else 4) / PART_BYTES)
    if not keys or n_parts <= 1:
        events = load(path, keep)
        table, shifted = summarise(events)
        return table, len(events), shifted

    folder = spec["out_path"] + ".parts"
    n_in, shifted, out = 0, 0, []
    try:
        for b, batch in enumerate(iter_source(path, keep)):
            n_in += len(batch)
            # On text, so a key read as a number in one batch and text in another
            # still lands in one partition.
            h = pd.util.hash_pandas_object(batch[keys].astype(str), index=False) % n_parts
            for p, rows in batch.groupby(h.to_numpy(), sort=False):
                os.makedirs(os.path.join(folder, str(p)), exist_ok=True)
                rows.to_parquet(os.path.join(folder, str(p), f"{b:05d}.parquet"), index=False)
        for p in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            files = sorted(os.listdir(os.path.join(folder, p)))
            events = pd.concat([pd.read_parquet(os.path.join(folder, p, f)) for f in files],
                               ignore_index=True)
            table, moved = summarise(events)
            shifted += moved
            out.append(table)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return pd.concat(out, ignore_index=True), n_in, shifted


def base_columns(code: str, columns: list[str], id_col: str, mode: str) -> list[str] | None:
    """The model columns a script loads: the id alone for a linkage or an aggregate;
    for a feature, the id and the columns its code names in quotes - all of them
    when it reaches them another way (by position, by dtype, by listing them)."""
    if mode in ("linkage", "aggregate"):
        return [id_col]
    if re.search(r"base\.(columns|iloc|select_dtypes|filter|drop)\b", code):
        return None
    return [c for c in columns if c == id_col
            or re.search(r"""['"]""" + re.escape(c) + r"""['"]""", code)]


def _require_ids(frame: pd.DataFrame, id_col: str) -> None:
    missing = {id_col, "as_of"} - set(frame.columns)
    if missing:
        raise ValueError(f"link() must return columns {sorted(missing)} as well")


def _link_in_batches(spec: dict[str, Any], code: str, link: Any, ids: pd.DataFrame,
                     options: dict[str, Any]) -> dict[str, Any]:
    """link() over the source a batch at a time. With `partition` the rows go to disk
    by chunk of ids (``Parts``) - a feature is then built a chunk at a time - else to
    one file. `keep` (the features' code) names the only columns to hold; `extract`
    also keeps discovery's cut (agent.extracts): the rows of every customer the ids
    link to."""
    id_col, path = spec["id_col"], spec["raw"][spec["source"]]
    if options.get("aggregate"):
        link = _filled(link, options["aggregate"])
    columns = None
    if options.get("keep"):
        columns = named_columns(code + "\n" + options["keep"], _schema(path), set())
    extract = options.get("extract") or {}
    keys = list(extract.get("key") or [])
    if keys and (missing := [k for k in keys if k not in _schema(path)]):
        raise ValueError(f"the key {missing} is not a column of {spec['source']!r}")
    parts = Parts(spec["out_path"], ids[id_col]) if options.get("partition") else None
    pieces, cut, n_source = [], [], 0
    for batch, linked in stream_link(link, ids, path, columns):
        _require_ids(linked, id_col)
        n_source += len(batch)
        if parts is not None:
            parts.write(linked, id_col)
        else:
            pieces.append(linked)
        if keys:
            if missing := [k for k in keys if k not in linked.columns]:
                raise ValueError(f"link() must keep the key {missing}")
            cut.append(batch[_key_text(batch, keys).isin(set(_key_text(linked, keys))).to_numpy()])
    if n_source == 0:                                # an empty source: its columns still
        empty = pd.read_csv(path, usecols=columns, nrows=0) if path.endswith(".csv") \
            else pd.read_parquet(path, columns=columns)
        linked = _to_pandas(link(ids, empty))
        _require_ids(linked, id_col)
        if parts is not None:
            parts.write(linked, id_col)
        else:
            pieces.append(linked)
    if parts is not None:
        described = parts.close()
    else:
        linked = pd.concat(pieces, ignore_index=True)
        linked.to_parquet(spec["out_path"], index=False)
        described = _describe(linked)
    if options.get("aggregate") and n_source and not described.get("n_rows"):
        raise ValueError(
            f"the linkage kept none of the aggregate's {n_source:,} rows. An aggregate has "
            "only its key, the time column and its values; the source's other columns reach "
            "the linkage empty - so a filter on one of them (a status, a second date) drops "
            "every row. Apply that filter in the aggregate's code instead")
    if keys:
        _write_cut(cut, extract["path"])
        described["extract"] = {"rows": int(sum(len(c) for c in cut)), "source_rows": int(n_source)}
    return described


def _source_columns(path: str) -> pd.DataFrame:
    """A source's columns and types, no rows."""
    if path.endswith(".csv"):
        return pd.read_csv(path, nrows=1000, low_memory=False).head(0)
    import pyarrow.dataset as ds

    return ds.dataset(path, format="parquet").schema.empty_table().to_pandas()


def _filled(link: Any, aggregate: dict[str, Any]) -> Any:
    """link() for an aggregate: the linkage was written for the raw source and may
    select its columns by name. The source columns the aggregate lacks are added
    empty, in the source's types, and taken out of what link() returns; the
    aggregate's own columns, which such a selection drops, are put back by the
    key and the time column - one aggregate row each."""
    empty = _source_columns(aggregate["source"])
    keys, time = list(aggregate["key"] or []), aggregate["time_column"]

    def by_row(frame: pd.DataFrame) -> pd.Series:
        return _key_text(frame, keys) + "\x1f" + \
            pd.to_datetime(frame[time].reset_index(drop=True)).astype(str)

    def link_aggregate(ids: pd.DataFrame, table: Any) -> Any:
        table = _to_pandas(table)
        added = [c for c in empty.columns if c not in table.columns]
        filler = {}
        for c in added:
            dtype = empty[c].dtype
            # A missing value has no integer or boolean form.
            dtype = "float64" if pd.api.types.is_integer_dtype(dtype) else \
                "object" if pd.api.types.is_bool_dtype(dtype) else dtype
            filler[c] = pd.Series(index=table.index, dtype=dtype)
        out = _to_pandas(link(ids, pd.concat([table, pd.DataFrame(filler)], axis=1)))
        out = out.drop(columns=[c for c in added if c in out.columns])
        lost = [c for c in table.columns if c not in out.columns]
        if lost and len(out):
            if not keys:
                raise ValueError(f"the linkage drops the aggregate's columns {lost}, and the "
                                 "aggregate names no key to join them back by - define it "
                                 "again with key=")
            if missing := [c for c in [*keys, time] if c not in out.columns]:
                raise ValueError(f"link() must keep {missing}: the aggregate's columns are "
                                 "joined back by its key and time column")
            values = table[lost].set_axis(by_row(table))
            out = out.reset_index(drop=True)
            out[lost] = values.reindex(by_row(out)).to_numpy()
        elif lost:
            out = out.assign(**{c: table[c].iloc[:0] for c in lost})
        return out

    return link_aggregate


def _key_text(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    """A key as text - the linkage may cast it (a number in the source, text in the id)."""
    text = frame[keys[0]].astype(str)
    for k in keys[1:]:
        text = text + "\x1f" + frame[k].astype(str)
    return text.reset_index(drop=True)


def _write_cut(cut: list[pd.DataFrame], path: str) -> None:
    """One file of the kept rows. A CSV read in batches can type a column one way in
    one batch and another in the next: such a column is kept as text."""
    frame = pd.concat(cut, ignore_index=True) if cut else pd.DataFrame()
    for column in frame.columns[frame.dtypes == object]:
        kinds = frame[column].dropna().map(type).unique()
        if len(kinds) > 1:
            frame[column] = frame[column].where(frame[column].isna(), frame[column].astype(str))
    frame.to_parquet(path, index=False)


def _built(frame: Any, id_col: str) -> pd.DataFrame:
    frame = _to_pandas(frame)
    if id_col not in frame.columns:
        raise ValueError(f"build() must return the {id_col!r} column")
    new = [c for c in frame.columns if c != id_col]
    if len(new) != 1:
        raise ValueError(f"build() must return {id_col!r} and exactly one feature "
                         f"column, got {new}")
    return frame


def _build_in_chunks(build: Any, spec: dict[str, Any], code: str, base: pd.DataFrame,
                     chunk: int, load: Any) -> pd.DataFrame:
    """build() on `chunk` ids at a time, each with only its ids' linked rows: at
    evaluation a linked source is on disk by chunk of ids (``Parts``), never whole.
    A source of one row per id (a per-transaction aggregate) is read once and cut
    to each chunk's ids."""
    id_col, out = spec["id_col"], []
    whole: dict[str, pd.DataFrame] = {}
    for k in range(max(-(-len(base) // chunk), 1)):
        ids = base.iloc[k * chunk:(k + 1) * chunk]

        def read(path: str, ids=ids, k=k) -> pd.DataFrame:
            columns = named_columns(code, _schema(path), {id_col, "as_of"})
            if is_parts(path):
                return Parts.open(path).read(k, columns)
            if path not in whole:
                whole[path] = load(path, columns)
            frame = whole[path]
            return frame[frame[id_col].isin(ids[id_col])].reset_index(drop=True)

        out.append(_built(build(None, _Lazy(spec.get("sources", {}), read),
                                ids.reset_index(drop=True)), id_col))
    return pd.concat(out, ignore_index=True)


def _describe(frame: pd.DataFrame) -> dict[str, Any]:
    return {"n_rows": int(len(frame)),
            "columns": {c: str(t) for c, t in frame.dtypes.items()},
            "head": frame.head(8).to_string(index=False, max_colwidth=30)[:_MAX_TEXT]}


def main(spec_path: str) -> None:
    spec = json.loads(open(spec_path).read())
    engine, mode = spec["engine"], spec["mode"]
    started = time.perf_counter()
    start_mb = _rss_mb()
    out: dict[str, Any] = {"ok": False, "mode": mode}
    printed = io.StringIO()

    try:
        if engine == "spark":
            from pyspark.sql import SparkSession, functions as F

            builder = (SparkSession.builder.master(spec.get("spark_master", "local[*]"))
                       .appName("autofe-agent"))
            for key, value in spec.get("spark_conf", {}).items():   # set before the JVM starts
                builder = builder.config(key, value)
            spark = builder.getOrCreate()

            def load(path: str):
                return spark.read.csv(path, header=True, inferSchema=True) \
                    if path.endswith(".csv") else spark.read.parquet(path)
        else:
            spark, F = None, None

            def load(path: str, columns: list[str] | None = None):
                if is_parts(path):                   # linked rows on disk by chunk of ids
                    parts = Parts.open(path)
                    return pd.concat([parts.read(k, columns) for k in range(parts.n_chunks)],
                                     ignore_index=True)
                # low_memory=False reads each column's type from the whole file, not
                # chunk by chunk - no mixed-type columns from a CSV export.
                return (pd.read_csv(path, low_memory=False, usecols=columns)
                        if path.endswith(".csv") else pd.read_parquet(path, columns=columns))

        def sample(path: str):
            if engine == "spark":
                return load(path).limit(rows)
            if path.endswith(".csv"):
                return pd.read_csv(path, nrows=rows, low_memory=False)
            import pyarrow.dataset as ds

            return ds.dataset(path).head(rows).to_pandas()

        code = open(spec["code_path"]).read()
        if engine != "spark":
            guard_merges(mode)
        # Starting Spark's JVM takes the same seconds on any data: not the script's cost.
        out["setup_s"] = round(time.perf_counter() - started, 2)

        def load_feature_source(path: str):
            # Spark reads lazily and prunes columns itself.
            if engine == "spark":
                return load(path)
            return load(path, named_columns(code, _schema(path), {spec["id_col"], "as_of"}))

        base = load(spec["base_path"]) if engine == "spark" or mode == "probe" \
            else load(spec["base_path"], base_columns(code, _schema(spec["base_path"]),
                                                      spec["id_col"], mode))
        sources = _Lazy(spec.get("sources", {}),
                        load_feature_source if mode == "feature" else load)
        raw = _Lazy(spec.get("raw", {}), sample if mode == "probe" else load)
        rows = int(spec.get("probe_rows", PROBE_ROWS))
        namespace: dict[str, Any] = {"pd": pd, "np": np, "spark": spark, "F": F,
                                     "base": base, "sources": sources, "raw": raw}

        with contextlib.redirect_stdout(printed):
            exec(compile(code, spec["code_path"], "exec"), namespace)  # noqa: S102
            id_col = spec["id_col"]

            if mode == "probe":
                if "result" in namespace:
                    out["result"] = _describe(_to_pandas(namespace["result"]))

            elif mode == "linkage":
                if "link" not in namespace:
                    raise ValueError("the linkage script must define link(base_ids, source)")
                if engine == "spark":
                    linked = namespace["link"](base.select(id_col), raw[spec["source"]])
                    linked.write.mode("overwrite").parquet(spec["out_path"])
                    linked = spark.read.parquet(spec["out_path"]).limit(200000).toPandas()
                    _require_ids(linked, id_col)
                    out["result"] = _describe(linked)
                else:
                    out["result"] = _link_in_batches(spec, code, namespace["link"], base[[id_col]],
                                                     spec.get("options") or {})

            elif mode == "aggregate":
                if "aggregate" not in namespace:
                    raise ValueError("the aggregate script must define aggregate(spark, raw)")
                options = spec.get("options") or {}
                time_column, period = options["time_column"], options["period"]
                path = spec["raw"][spec["source"]]
                if period == "transaction":
                    out["result"] = _per_transaction(spec, options, code, namespace["aggregate"],
                                                     base, spark, F, load, engine)
                    raise _Done
                if engine == "spark":
                    events = add_period_end(load(path), time_column, period, engine)
                    table = namespace["aggregate"](spark, events)
                    columns = list(table.columns)
                    if PERIOD_END not in columns:
                        raise ValueError(f"aggregate() must keep {PERIOD_END!r} - group by the "
                                         f"source's key and {PERIOD_END!r}")
                    if time_column in columns:
                        raise ValueError(f"aggregate() must not return {time_column!r}: "
                                         f"{PERIOD_END!r} stands for it")
                    ends = events.select(PERIOD_END).distinct()
                    shifted = table.join(ends, PERIOD_END, "left_anti").count()
                    n_in, n_out = events.count(), table.count()
                    table = table.withColumnRenamed(PERIOD_END, time_column)
                    table.write.mode("overwrite").parquet(spec["out_path"])
                    shown = spark.read.parquet(spec["out_path"]).limit(200000).toPandas()
                else:
                    table, n_in, shifted = _by_period(spec, options, code, namespace["aggregate"],
                                                      spark, load)
                    keys = list(options.get("key") or [])
                    missing = [k for k in keys if k not in table.columns]
                    if missing:
                        raise ValueError(f"aggregate() must return its key {missing} - "
                                         f"group by it and {PERIOD_END!r}")
                    repeats = int(table.duplicated(keys + [PERIOD_END]).sum()) if keys else 0
                    if repeats:
                        raise ValueError(f"aggregate() must return one row per {keys} and "
                                         f"{PERIOD_END!r}; {repeats:,} repeat")
                    n_out = len(table)
                    table = table.rename(columns={PERIOD_END: time_column})
                    table.to_parquet(spec["out_path"], index=False)
                    shown = table
                if shifted:
                    raise ValueError(f"{shifted:,} rows of aggregate() have a {PERIOD_END!r} no "
                                     "event has - keep it as the runner set it; a shifted "
                                     "period could reach past the as-of date")
                out["result"] = {**_describe(shown), "n_rows": int(n_out), "source_rows": int(n_in)}

            elif mode == "feature":
                if "build" not in namespace:
                    raise ValueError("the feature script must define build(spark, sources, base)")
                parts = [Parts.open(p) for p in spec.get("sources", {}).values()
                         if engine != "spark" and is_parts(p)]
                if parts:
                    frame = _build_in_chunks(namespace["build"], spec, code, base,
                                             parts[0].chunk, load)
                else:
                    frame = _built(namespace["build"](spark, sources, base), id_col)
                new = [c for c in frame.columns if c != id_col]
                if frame[id_col].duplicated().any():
                    raise ValueError(f"build() returned {int(frame[id_col].duplicated().sum())} "
                                     "duplicate ids; aggregate to one row per id")
                frame.to_parquet(spec["out_path"], index=False)
                out["feature"] = new[0]
                out["result"] = _describe(frame)
            else:
                raise ValueError(f"unknown mode {mode!r}")
        out["ok"] = True
    except _Done:
        out["ok"] = True
    except Exception:  # noqa: BLE001 - every failure goes back to the agent as text
        tb = traceback.format_exc()
        out["error"] = tb[-_MAX_TEXT:]
    out["stdout"] = printed.getvalue()[-_MAX_TEXT:]
    out["elapsed_s"] = round(time.perf_counter() - started, 2)
    # What the script's data took, above the interpreter and its imports.
    out["peak_mb"] = round(max(_rss_mb() - start_mb, 0.0), 1)
    with open(spec["result_path"], "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main(sys.argv[1])
