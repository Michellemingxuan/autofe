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

Every pandas merge is sized before it runs (``guard_merges``): the rows out of a
join are, per key, the left rows times the right rows - with repeated keys on
both sides (many-to-many) that multiplies. A feature may not do it at all; any
script is stopped before a join that would not fit in memory.

Modes:

    probe    run the code; report what it printed, and `result` if it set one
    linkage  call link(base_ids, source) -> rows with id, as_of and the source columns
    feature  call build(spark, sources, base) -> one row per id, one new column
"""

from __future__ import annotations

import contextlib
import io
import json
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

        base = load(spec["base_path"])
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
                base_ids = base.select(id_col) if engine == "spark" else base[[id_col]]
                linked = namespace["link"](base_ids, raw[spec["source"]])
                if engine == "spark":
                    linked.write.mode("overwrite").parquet(spec["out_path"])
                    linked = spark.read.parquet(spec["out_path"]).limit(200000).toPandas()
                else:
                    linked = _to_pandas(linked)
                    linked.to_parquet(spec["out_path"], index=False)
                missing = {id_col, "as_of"} - set(linked.columns)
                if missing:
                    raise ValueError(f"link() must return columns {sorted(missing)} as well")
                out["result"] = _describe(linked)

            elif mode == "feature":
                if "build" not in namespace:
                    raise ValueError("the feature script must define build(spark, sources, base)")
                frame = _to_pandas(namespace["build"](spark, sources, base))
                if id_col not in frame.columns:
                    raise ValueError(f"build() must return the {id_col!r} column")
                new = [c for c in frame.columns if c != id_col]
                if len(new) != 1:
                    raise ValueError(f"build() must return {id_col!r} and exactly one feature "
                                     f"column, got {new}")
                if frame[id_col].duplicated().any():
                    raise ValueError(f"build() returned {int(frame[id_col].duplicated().sum())} "
                                     "duplicate ids; aggregate to one row per id")
                frame.to_parquet(spec["out_path"], index=False)
                out["feature"] = new[0]
                out["result"] = _describe(frame)
            else:
                raise ValueError(f"unknown mode {mode!r}")
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
