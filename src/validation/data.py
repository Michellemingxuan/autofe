"""Stage 0: turn the train / valid / test tables into a ``Dataset``.

Input
    the three tables named in the config's ``data.paths``, or the same frames
    passed in memory. The split was made once, by the dataset's prepare step,
    and is used exactly as given - nothing here re-splits.

Output
    a ``Dataset``: one cleaned frame per split, plus the resolved incumbent
    (``base``) and candidate (``new``) feature lists. Every later stage reads it.

The steps, in order:

    read        read_frame               parquet / csv / pickle
    features    resolve_features         explicit lists and prefixes, against train
    clean       clean_missing            sentinel codes and +/-inf -> NaN
    indicators  add_missing_indicators   optional 0/1 "was missing" columns

What each split is for
    train   fits every model, and is where feature selection and the data
            quality reference look.
    valid   a tool, never evidence. It tunes hyperparameters, stops training,
            and gives the discovery screen the feedback it reports back to the
            proposer. Because proposals are *selected* on it, a gain here is
            not independent confirmation of anything.
    test    the hold-out, and the only evidence about a proposed feature.
            Nothing fits on it, tunes on it, stops on it, or screens on it.
            Two things read it: the analysis and verdict gates, which is the
            evidence; and the data quality stability check, which compares each
            column's distribution across all three splits. The latter is not a
            leak - PSI never looks at the target, so it measures whether a
            column is stable, not whether it predicts.

    Keeping test out of every loop is what makes its number mean something. The
    cost of breaking that rule is invisible - the run still completes, the
    numbers still look reasonable, and they are simply no longer about
    unseen data.

Relative paths resolve against the working directory, so run from the repo root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from validation.config import Config, DataConfig, FeatureConfig
from validation.logging_utils import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "valid", "test")


@dataclass
class Dataset:
    frames: Dict[str, pd.DataFrame]
    target: str
    base_features: List[str]
    new_features: List[str]
    weight_col: Optional[str] = None
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def all_features(self) -> List[str]:
        return list(self.base_features) + list(self.new_features)

    def split(self, name: str) -> pd.DataFrame:
        if name not in self.frames:
            raise KeyError(f"split {name!r} not available; have {sorted(self.frames)}")
        return self.frames[name]

    def available_splits(self) -> List[str]:
        return [s for s in SPLITS if s in self.frames and len(self.frames[s])]

    def describe(self) -> Dict[str, object]:
        return {
            "rows_per_split": {k: int(len(v)) for k, v in self.frames.items()},
            "n_base_features": len(self.base_features),
            "n_new_features": len(self.new_features),
            "target": self.target,
        }

    def with_features(self, base: List[str], new: List[str]) -> "Dataset":
        return Dataset(self.frames, self.target, base, new, self.weight_col, dict(self.meta))


# --------------------------------------------------------------------------- #
# Entry points - from files or from frames, both ending in _assemble
# --------------------------------------------------------------------------- #
def build_dataset(cfg: Config) -> Dataset:
    """Read the train / valid / test tables named in ``data.paths``."""
    if not cfg.data.paths:
        raise ValueError("data.paths is empty; point it at the train / valid / test "
                         "tables the dataset's prepare step wrote")
    frames = {name: read_frame(cfg.data, path) for name, path in cfg.data.paths.items()}
    return prepare_dataset_from_frames(frames, cfg)


def prepare_dataset_from_frames(frames: Dict[str, pd.DataFrame], cfg: Config) -> Dataset:
    """Build a ``Dataset`` from train / valid / test frames already in memory.

    This is the entry point for a discover -> verify loop, where candidate
    features are engineered in the session and never round-trip through a file.
    The frames are taken as-is, so whatever split produced them is preserved.
    """
    unknown = sorted(set(frames) - set(SPLITS))
    if unknown:
        raise ValueError(f"frame names must be train/valid/test, got extra: {unknown}")
    if "train" not in frames or not len(frames["train"]):
        raise ValueError("a non-empty 'train' frame is required")
    for name, frame in frames.items():
        if cfg.data.target not in frame.columns:
            raise KeyError(f"target column {cfg.data.target!r} not in the {name!r} frame")
    ordered = {name: frames[name] for name in SPLITS if name in frames and len(frames[name])}
    return _assemble(ordered, cfg)


def _assemble(frames: Dict[str, pd.DataFrame], cfg: Config) -> Dataset:
    """Resolve features against train, then subset and clean every split alike."""
    reference = frames["train"]
    base, new = resolve_features(reference, cfg.features, cfg.data)
    keep = list(dict.fromkeys(
        base + new + [cfg.data.target]
        + list(cfg.data.id_cols)
        + ([cfg.data.weight_col] if cfg.data.weight_col in reference.columns else [])
    ))

    prepared = {}
    for name, frame in frames.items():
        absent = [c for c in keep if c not in frame.columns]
        if absent:
            raise KeyError(f"the {name!r} split is missing column(s) present in train: {absent}")
        prepared[name] = clean_missing(frame[keep], base + new, cfg.data.missing_values)

    # After cleaning, so indicators capture sentinels and infinities too.
    prepared, base, new = add_missing_indicators(prepared, base, new, cfg)

    dataset = Dataset(
        frames=prepared,
        target=cfg.data.target,
        base_features=base,
        new_features=new,
        weight_col=cfg.data.weight_col,
    )
    logger.info("dataset ready: %s", dataset.describe())
    return dataset


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_frame(cfg: DataConfig, path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"data file does not exist: {path}. Relative paths resolve against the "
            f"working directory ({Path.cwd()}), so run from the repo root - and build "
            "the tables first with the dataset's prepare step.")
    fmt = cfg.format
    if fmt == "auto":
        suffix = path.suffix.lower()
        fmt = {
            ".parquet": "parquet", ".pq": "parquet",
            ".csv": "csv", ".gz": "csv", ".txt": "csv",
            ".pkl": "pickle", ".pickle": "pickle",
        }.get(suffix, "csv")
    logger.info("reading %s as %s", path, fmt)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "pickle":
        return pd.read_pickle(path)
    # round_trip: a float comes back bit-for-bit as it was written, so a model
    # fitted on the CSV is the model that would have been fitted on the frame.
    return pd.read_csv(path, float_precision="round_trip")


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def resolve_features(df: pd.DataFrame, cfg: FeatureConfig, data_cfg: DataConfig) -> tuple[List[str], List[str]]:
    """Turn explicit lists and/or prefixes into concrete, de-duplicated columns."""
    reserved = set(cfg.exclude) | set(data_cfg.id_cols) | {data_cfg.target}
    if data_cfg.weight_col:
        reserved.add(data_cfg.weight_col)

    def _resolve(names: List[str], prefix: Optional[str]) -> List[str]:
        out = [c for c in names if c not in reserved]
        if prefix:
            out += [c for c in df.columns if c.startswith(prefix) and c not in reserved and c not in out]
        return out

    new = _resolve(cfg.new, cfg.new_prefix)
    base = [c for c in _resolve(cfg.base, cfg.base_prefix) if c not in set(new)]

    if not base and not cfg.base and not cfg.base_prefix:
        # Default: everything numeric that is neither reserved nor a new feature.
        base = [
            c for c in df.columns
            if c not in reserved and c not in set(new) and pd.api.types.is_numeric_dtype(df[c])
        ]
        logger.info("features.base not given; inferred %d incumbent column(s)", len(base))

    missing = [c for c in base + new if c not in df.columns]
    if missing:
        raise KeyError(f"feature column(s) missing from data: {missing}")
    return base, new


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_missing(df: pd.DataFrame, columns: List[str], sentinels: List[float]) -> pd.DataFrame:
    """Replace sentinel codes and infinities with NaN, so they read as missing.

    Sentinels are configured; infinities are not, because there is no case where
    +/-inf is a meaningful model input. They arrive from ratio features with a zero
    denominator, and left alone they corrupt quantile binning, the PSI reference
    edges, and the correlation kernel - where squaring a near-DBL_MAX value
    overflows and turns the result into NaN.
    """
    if not columns:
        return df
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return df

    df = df.copy()
    if sentinels:
        df[numeric] = df[numeric].replace(list(sentinels), np.nan)

    block = df[numeric].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    non_finite = ~np.isfinite(block)
    n_non_finite = int(non_finite.sum()) - int(df[numeric].isna().to_numpy().sum())
    if n_non_finite > 0:
        affected = [numeric[i] for i in np.where(non_finite.any(axis=0))[0]
                    if not df[numeric[i]].isna().all()]
        block[non_finite] = np.nan
        df[numeric] = block
        logger.warning("converted %d infinite value(s) to NaN across %d column(s): %s",
                       n_non_finite, len(affected),
                       affected[:5] + (["..."] if len(affected) > 5 else []))
    return df


def add_missing_indicators(
    frames: Dict[str, pd.DataFrame],
    base: List[str],
    new: List[str],
    cfg: Config,
) -> tuple[Dict[str, pd.DataFrame], List[str], List[str]]:
    """Append a 0/1 column per eligible feature, recording where it was missing.

    Eligibility is decided on **train** and the same columns are then added to every
    split, so the feature set cannot differ between them. A column that is never
    missing, or always missing, is skipped: its indicator would be constant and
    carry nothing.
    """
    indicators = cfg.data.missing_indicators
    if not indicators.enabled:
        return frames, base, new

    train = frames["train"]
    source = list(new) if indicators.scope == "candidates" else list(base) + list(new)

    made: List[tuple] = []
    for column in source:
        rate = float(train[column].isna().mean())
        if not 0.0 < rate < 1.0 or rate < indicators.min_missing_rate:
            continue
        name = f"{column}{indicators.suffix}"
        if name in train.columns:
            logger.warning("missing indicator %r already exists as a column; skipping", name)
            continue
        made.append((column, name, rate))

    if not made:
        logger.info("missing indicators enabled, but no column qualified "
                    "(need %.1f%% <= missing rate < 100%%)", indicators.min_missing_rate * 100)
        return frames, base, new

    prepared = {
        split: frame.assign(**{name: frame[column].isna().astype("int8")
                               for column, name, _ in made})
        for split, frame in frames.items()
    }

    if indicators.treat_as == "new":
        new = list(new) + [name for _, name, _ in made]
    else:
        in_base = set(base)
        base = list(base) + [n for c, n, _ in made if c in in_base]
        new = list(new) + [n for c, n, _ in made if c not in in_base]

    logger.info("added %d missing indicator(s) as %s feature(s): %s",
                len(made), indicators.treat_as,
                ", ".join(f"{n} ({r:.1%})" for _, n, r in made[:5])
                + (", ..." if len(made) > 5 else ""))
    return prepared, base, new
