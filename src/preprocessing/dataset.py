"""What every ``data/<dataset>/prepare.py`` script does the same way.

A prepare script has exactly one job a config cannot express: turn a published
archive into the modeling table the config already describes. Everything around
that job - fetching the archive, making column names usable, checking that the
table really contains what the config declares, writing the three artifacts - is
identical across datasets, so it lives here and each script keeps only its own
recipe.

The division of labour is worth stating, because it is what keeps the configs
honest:

    configs/<name>.yaml      where the table goes, the target, the id column,
                             and which features are the candidates
    data/<name>/prepare.py   how the archive becomes that table
    this module              the parts that do not vary, including the checks

The checks matter more than they look. A config can name a candidate feature
that the script does not build, and nothing downstream notices: the feature is
simply absent from every variant, the run completes, and the verdict is about a
set that was never evaluated. :func:`write_modeling_table` refuses that case
instead.

Three artifacts come out of every build:

    modeling.parquet            the table the pipeline reads
    column_mapping.csv          sanitized name -> the archive's original
    column_descriptions.json    what each column means, for a discovery run

The last one is not decoration. A proposer that knows a column only as
``cash_flow_to_liability`` cannot bring any real-world knowledge to it, so the
original human-written header travels with the table.
"""

from __future__ import annotations

import io
import json
import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "sanitize",
    "sanitize_columns",
    "fetch_archive",
    "declared_candidates",
    "resolve_path",
    "write_modeling_table",
]


def sanitize(name: str) -> str:
    """
    A column name that reads well and survives every downstream consumer.

    Published tables use headers written for humans - ``Bankrupt?``, ``ROA(C)
    before interest and depreciation before interest``,
    ``android.permission.GET_ACCOUNTS``. Lowercased snake_case keeps them
    readable while removing the punctuation that makes a name awkward to type,
    to reference in a config, or to index in generated code. The original is
    never lost: it goes to ``column_mapping.csv`` and, more importantly, becomes
    the column's description.
    """
    name = str(name).strip().lower()
    name = name.replace("%", " pct ").replace("&", " and ")
    name = re.sub(r"[^\w]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def sanitize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rename every column, returning the frame and the original -> new mapping."""
    mapping = pd.DataFrame({
        "original": list(frame.columns),
        "column": [sanitize(c) for c in frame.columns],
    })
    duplicated = mapping["column"][mapping["column"].duplicated()].tolist()
    if duplicated:
        # Two different headers collapsing to one name would silently drop a
        # column on rename, so it is an error rather than a warning.
        raise ValueError(
            f"sanitizing produced duplicate column name(s): {sorted(set(duplicated))}. "
            "Disambiguate them in the prepare script before sanitizing."
        )
    renamed = frame.copy()
    renamed.columns = mapping["column"].tolist()
    return renamed, mapping


def fetch_archive(url: str, dest: Path, member: str, timeout: int = 180) -> Path:
    """
    Download ``url`` into ``dest`` once and return the path to ``member``.

    Cached on the extracted file, so re-running a prepare script does no network
    I/O. Handles both a zip archive and a bare file, because UCI serves each.
    """
    target = dest / member
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read()

    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if member not in names:
                raise KeyError(f"{url} does not contain {member!r}; it holds {names}")
            archive.extract(member, dest)
    else:
        target.write_bytes(payload)

    return target


def declared_candidates(cfg: Any, frame: pd.DataFrame) -> list[str]:
    """The candidate features the config asks the pipeline to evaluate."""
    declared = list(cfg.features.new)
    if cfg.features.new_prefix:
        declared += [c for c in frame.columns
                     if c.startswith(cfg.features.new_prefix) and c not in declared]
    return declared


def resolve_path(path: str | Path, root: Path) -> Path:
    """Config paths are relative to the repo root, not the working directory."""
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


@dataclass
class BuildReport:
    """What was written, for the script to print and for tests to assert on."""

    rows: int
    columns: int
    positive_rate: float
    positives: int
    base_features: int
    candidates: list[str]
    described: int
    paths: dict[str, Path] = field(default_factory=dict)


def write_modeling_table(
    cfg: Any,
    frame: pd.DataFrame,
    *,
    root: Path,
    id_col: str,
    mapping: pd.DataFrame | None = None,
    descriptions: Mapping[str, str] | None = None,
    extra_descriptions: Mapping[str, str] | None = None,
) -> BuildReport:
    """
    Check the built table against the config, then write the three artifacts.

    Raises rather than writing whenever the table and the config disagree: a
    table that does not contain what the config declares produces a run whose
    verdict is about the wrong feature set, and that failure is invisible in the
    output.
    """
    out = resolve_path(cfg.data.path, root)
    out.parent.mkdir(parents=True, exist_ok=True)

    if cfg.data.target not in frame.columns:
        raise KeyError(f"target {cfg.data.target!r} is not in the built table")

    declared = declared_candidates(cfg, frame)
    absent = [c for c in declared if c not in frame.columns]
    if absent:
        raise KeyError(
            f"features.new names column(s) this script does not build: {absent}"
        )
    if not declared and not cfg.discovery.enabled:
        raise ValueError(
            f"{cfg.data.target!r} has no candidate features to evaluate: "
            "features.new is empty and discovery is disabled, so the run would "
            "have nothing to judge. Declare candidates or enable discovery."
        )

    target = frame[cfg.data.target]
    if set(target.dropna().unique()) - {0, 1}:
        raise ValueError(
            f"target {cfg.data.target!r} is not binary; it holds "
            f"{sorted(target.dropna().unique())[:8]}"
        )

    # Descriptions default to the archive's own headers, which is what a human
    # wrote and the most useful thing a proposer can be told about a column.
    if descriptions is None:
        if mapping is None:
            raise ValueError("pass either `descriptions` or `mapping`")
        descriptions = dict(zip(mapping["column"], mapping["original"]))
    descriptions = {str(k): str(v).strip() for k, v in descriptions.items()}
    if extra_descriptions:
        descriptions.update({str(k): str(v).strip()
                             for k, v in extra_descriptions.items()})
    reserved = {cfg.data.target, id_col, *cfg.data.id_cols}
    descriptions = {k: v for k, v in descriptions.items()
                    if k in frame.columns and k not in reserved}

    paths = {"table": out,
             "descriptions": out.parent / "column_descriptions.json"}
    frame.to_parquet(out, index=False)
    paths["descriptions"].write_text(
        json.dumps(descriptions, indent=2, ensure_ascii=False), encoding="utf-8")
    if mapping is not None:
        paths["mapping"] = out.parent / "column_mapping.csv"
        mapping.to_csv(paths["mapping"], index=False)

    base = [c for c in frame.columns if c not in set(declared) | reserved]
    return BuildReport(
        rows=len(frame),
        columns=frame.shape[1],
        positive_rate=float(target.mean()),
        positives=int(target.sum()),
        base_features=len(base),
        candidates=declared,
        described=len(descriptions),
        paths=paths,
    )


def print_report(report: BuildReport, cfg: Any, config_path: str | Path,
                 root: Path, id_col: str) -> None:
    """The summary a prepare script prints, identical for every dataset."""
    table = report.paths["table"]
    print(f"wrote {report.rows:,} rows x {report.columns} cols -> {table}")
    print(f"  target {cfg.data.target!r}: {report.positive_rate:.2%} positive "
          f"({report.positives:,} of {report.rows:,})")
    print(f"  id column: {id_col!r}")
    print(f"  incumbent features: {report.base_features}")
    print(f"  candidate features: {len(report.candidates)} "
          f"(declared in {Path(config_path).name})")
    print(f"  column descriptions: {report.paths['descriptions'].name} "
          f"({report.described} columns, for a discovery run)")
    print(f"\nready:  mllite -c {config_path}")
    print(f"   with discovery:  mllite -c {config_path} --set discovery.enabled=true")


def load_config_from(root: Path, config_path: str | Path) -> Any:
    """Import ``validation`` off the repo's src tree and load the config.

    Prepare scripts run as plain files (``python data/x/prepare.py``), not as
    part of an installed package, so they have to put ``src`` on the path
    themselves.
    """
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from validation import load_config

    return load_config(str(config_path))
