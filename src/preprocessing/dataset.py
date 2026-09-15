"""The small pieces every dataset's prepare step uses, and the checks on its output.

A prepare step - ``data/<name>/prepare.ipynb`` for the UCI demos, a script for a
private extract - is specific to its dataset. Only what never varies lives here:

    sanitize, sanitize_columns   readable snake_case names, originals kept
    fetch_archive                download once, then read from the local cache
    stratified_split             the one train / valid / test split
    write_splits                 check the splits against the config, then write

Every prepare step writes, at and beside the config's ``data.paths``:

    train.csv valid.csv test.csv   the tables the pipeline reads
    column_mapping.csv             sanitized name -> the original
    column_descriptions.json       what each column means, for discovery

plus, when discovery is configured, the few-shot example rows (see
:mod:`preprocessing.shots`). The split is made here, once, and the pipeline
reads it exactly as written - it never re-splits.

The descriptions are not decoration. A proposer that knows a column only as
``cash_flow_to_liability`` cannot bring any real-world knowledge to it, so the
original human-written header travels with the table.

The checks matter more than they look. A config can name a candidate feature
that the prepare step does not build, and nothing downstream notices: the feature
is simply absent from every variant, the run completes, and the verdict is about
a set that was never evaluated. :func:`write_splits` refuses that case instead,
and refuses an id that appears in two splits, which leaks the outcome.
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "SPLITS",
    "sanitize",
    "sanitize_columns",
    "fetch_archive",
    "stratified_split",
    "build_sample",
    "declared_candidates",
    "resolve_path",
    "write_splits",
]

SPLITS = ("train", "valid", "test")


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
            "Disambiguate them before sanitizing."
        )
    renamed = frame.copy()
    renamed.columns = mapping["column"].tolist()
    return renamed, mapping


def fetch_archive(url: str, dest: Path, member: str, timeout: int = 180) -> Path:
    """
    Download ``url`` into ``dest`` once and return the path to ``member``.

    Cached on the extracted file, so re-running a prepare step does no network
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


def stratified_split(
    frame: pd.DataFrame,
    target: str,
    *,
    valid_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Split once into train / valid / test, keeping the target rate in every part.

    Stratified because the outcome is often rare: an unstratified draw can hand
    valid or test a materially different base rate, which shows up as noise in
    the Gini comparison the pipeline exists to make. Each class is shuffled with
    the seed and cut by the two sizes, so the same seed always gives the same
    split, and rows keep their original order within each part. ``frame`` needs
    a unique index.
    """
    rng = np.random.default_rng(seed)
    labels = pd.Series("train", index=frame.index, dtype=object)
    for _, index in frame.groupby(frame[target], sort=False).groups.items():
        index = np.asarray(index)
        shuffled = index[rng.permutation(len(index))]
        n_test = int(round(len(index) * test_size))
        n_valid = int(round(len(index) * valid_size))
        labels.loc[shuffled[:n_test]] = "test"
        labels.loc[shuffled[n_test:n_test + n_valid]] = "valid"
    return {name: frame.loc[labels == name].reset_index(drop=True) for name in SPLITS}


def build_sample(
    frame: pd.DataFrame,
    target: str,
    size: int,
    *,
    balance: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Draw ``size`` rows - class-balanced by default - for discovery's screen.

    Balanced means each class contributes up to ``size / n_classes`` rows, so a
    rare class is kept whole instead of the handful a uniform draw would give.
    On an imbalanced target a uniform draw of a few thousand rows holds only a
    few positives, which makes the screen's score mostly noise; the screen
    compares two models on the same rows, so it does not need the base rate.
    The same seed always gives the same rows.
    """
    if size >= len(frame) and not balance:
        return frame.copy()

    rng = np.random.default_rng(seed)
    if not balance:
        take = rng.choice(len(frame), size=min(size, len(frame)), replace=False)
        return frame.iloc[np.sort(take)].reset_index(drop=True)

    groups = [group for _, group in frame.groupby(target, sort=True)]
    per_class = max(1, size // max(1, len(groups)))
    parts = []
    for group in groups:
        n = min(per_class, len(group))
        take = rng.choice(len(group), size=n, replace=False)
        parts.append(group.iloc[np.sort(take)])
    sample = pd.concat(parts, axis=0)
    # Shuffle so class order cannot leak into row order.
    return sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)


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


def write_splits(
    cfg: Any,
    frames: Mapping[str, pd.DataFrame],
    *,
    root: Path,
    mapping: pd.DataFrame | None = None,
    descriptions: Mapping[str, str] | None = None,
    extra_descriptions: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """
    Check the three splits against the config, then write them and their sidecars.

    Raises rather than writing whenever the tables and the config disagree: a
    table that does not contain what the config declares produces a run whose
    verdict is about the wrong feature set, and that failure is invisible in the
    output. Returns the path of every file written.
    """
    paths = dict(cfg.data.paths or {})
    missing = [s for s in SPLITS if s not in paths]
    if missing:
        raise ValueError(f"data.paths has no entry for {missing}; the config must "
                         "say where each split is written")
    if sorted(frames) != sorted(SPLITS):
        raise ValueError(f"expected frames for {list(SPLITS)}, got {sorted(frames)}")

    columns = list(frames["train"].columns)
    for name in ("valid", "test"):
        absent = [c for c in columns if c not in frames[name].columns]
        extra = [c for c in frames[name].columns if c not in columns]
        if absent or extra:
            raise ValueError(f"{name!r} does not have train's columns: "
                             f"missing {absent[:5]}, extra {extra[:5]}")
    frames = {name: frames[name][columns] for name in SPLITS}

    target = cfg.data.target
    if target not in columns:
        raise KeyError(f"target {target!r} is not in the built tables")
    values = pd.concat([frame[target] for frame in frames.values()]).dropna().unique()
    if set(values) - {0, 1}:
        raise ValueError(f"target {target!r} is not binary; it holds {sorted(values)[:8]}")

    declared = declared_candidates(cfg, frames["train"])
    absent = [c for c in declared if c not in columns]
    if absent:
        raise KeyError(f"features.new names column(s) that were not built: {absent}")
    if not declared and not cfg.discovery.enabled:
        raise ValueError(
            f"{target!r} has no candidate features to evaluate: features.new is "
            "empty and discovery is disabled, so the run would have nothing to "
            "judge. Declare candidates or enable discovery."
        )

    id_cols = list(cfg.data.id_cols)
    absent = [c for c in id_cols if c not in columns]
    if absent:
        raise KeyError(f"data.id_cols names column(s) that were not built: {absent}")
    if id_cols:
        # The same unit in two splits leaks the outcome, and nothing downstream
        # can tell a leaked split from a clean one.
        keys = {name: set(map(tuple, frame[id_cols].astype(str).to_numpy()))
                for name, frame in frames.items()}
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            shared = keys[left] & keys[right]
            if shared:
                raise ValueError(
                    f"{len(shared):,} id(s) appear in both {left!r} and {right!r} "
                    f"(e.g. {sorted(shared)[:3]}). That is leakage - fix the split.")

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
    reserved = {target, *id_cols}
    descriptions = {k: v for k, v in descriptions.items()
                    if k in columns and k not in reserved}

    written: dict[str, Path] = {}
    for name in SPLITS:
        out = resolve_path(paths[name], root)
        out.parent.mkdir(parents=True, exist_ok=True)
        frames[name].to_csv(out, index=False)
        written[name] = out

    folder = written["train"].parent
    written["descriptions"] = folder / "column_descriptions.json"
    written["descriptions"].write_text(
        json.dumps(descriptions, indent=2, ensure_ascii=False), encoding="utf-8")
    if mapping is not None:
        written["mapping"] = folder / "column_mapping.csv"
        mapping.to_csv(written["mapping"], index=False)
    return written
