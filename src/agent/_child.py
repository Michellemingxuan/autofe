"""Runs one agent script in its own process: ``python -m agent._child spec.json``.

The parent writes a spec naming the inputs and the mode, and reads back a JSON
result. Everything the script may touch is handed to it already loaded:

    base      the model rows - id + base columns, no target
    sources   each source joined through its confirmed linkage (feature mode)
    raw       each source as it is on disk, loaded on first access (probe, linkage) -
              in a probe, its first PROBE_ROWS rows: a probe looks, it does not compute
    spark, F  a local SparkSession and pyspark.sql.functions (engine=spark)
    pd, np

Modes:

    probe    run the code; report what it printed, and `result` if it set one
    linkage  call link(base_ids, source) -> rows with id, as_of and the source columns
    feature  call build(spark, sources, base) -> one row per id, one new column
"""

from __future__ import annotations

import contextlib
import io
import json
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

            def load(path: str):
                # low_memory=False reads each column's type from the whole file, not
                # chunk by chunk - no mixed-type columns from a CSV export.
                return (pd.read_csv(path, low_memory=False) if path.endswith(".csv")
                        else pd.read_parquet(path))

        def sample(path: str):
            if engine == "spark":
                return load(path).limit(rows)
            if path.endswith(".csv"):
                return pd.read_csv(path, nrows=rows, low_memory=False)
            import pyarrow.dataset as ds

            return ds.dataset(path).head(rows).to_pandas()

        base = load(spec["base_path"])
        sources = _Lazy(spec.get("sources", {}), load)
        raw = _Lazy(spec.get("raw", {}), sample if mode == "probe" else load)
        rows = int(spec.get("probe_rows", PROBE_ROWS))
        namespace: dict[str, Any] = {"pd": pd, "np": np, "spark": spark, "F": F,
                                     "base": base, "sources": sources, "raw": raw}
        code = open(spec["code_path"]).read()

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
    with open(spec["result_path"], "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main(sys.argv[1])
