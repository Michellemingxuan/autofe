"""Loading, split assignment and feature-list resolution.

A ``Dataset`` is the single object every downstream stage consumes: one frame per
split plus the resolved base/new feature lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from mllite.config import Config, DataConfig, FeatureConfig
from mllite.logging_utils import get_logger

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
# Loading
# --------------------------------------------------------------------------- #
def read_frame(cfg: DataConfig, path: Optional[str] = None) -> pd.DataFrame:
    path = Path(path if path is not None else cfg.path)
    if not path.exists():
        raise FileNotFoundError(f"data.path does not exist: {path}")
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
    return pd.read_csv(path, nrows=cfg.nrows)


def resolve_features(df: pd.DataFrame, cfg: FeatureConfig, data_cfg: DataConfig) -> tuple[List[str], List[str]]:
    """Turn explicit lists and/or prefixes into concrete, de-duplicated columns."""
    reserved = set(cfg.exclude) | set(data_cfg.id_cols) | {data_cfg.target}
    if data_cfg.weight_col:
        reserved.add(data_cfg.weight_col)
    if data_cfg.split.column:
        reserved.add(data_cfg.split.column)
    if data_cfg.split.time_col:
        reserved.add(data_cfg.split.time_col)

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


def clean_missing(df: pd.DataFrame, columns: List[str], sentinels: List[float]) -> pd.DataFrame:
    """Replace sentinel codes with NaN so XGBoost treats them as missing."""
    if not sentinels or not columns:
        return df
    df = df.copy()
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric:
        df[numeric] = df[numeric].replace(list(sentinels), np.nan)
    return df


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def assign_splits(df: pd.DataFrame, cfg: DataConfig, seed: int) -> Dict[str, pd.DataFrame]:
    split_cfg = cfg.split
    if split_cfg.mode == "column":
        col = split_cfg.column
        labels = df[col].astype(str).str.lower()
        frames = {name: df.loc[labels == name].copy() for name in SPLITS}
        frames = {k: v for k, v in frames.items() if len(v)}
        unknown = sorted(set(labels.unique()) - set(SPLITS))
        if unknown:
            logger.warning("split column %r has unused label(s): %s", col, unknown)
        if "train" not in frames:
            raise ValueError(f"split column {col!r} produced no train rows")
        return frames

    if split_cfg.mode == "time":
        ordered = df.sort_values(split_cfg.time_col)
        n = len(ordered)
        n_test = int(round(n * split_cfg.test_size))
        n_valid = int(round(n * split_cfg.valid_size))
        n_train = n - n_test - n_valid
        if n_train <= 0:
            raise ValueError("time split leaves no training rows; lower valid_size/test_size")
        frames = {"train": ordered.iloc[:n_train].copy()}
        if n_valid:
            frames["valid"] = ordered.iloc[n_train:n_train + n_valid].copy()
        if n_test:
            frames["test"] = ordered.iloc[n_train + n_valid:].copy()
        return frames

    # random
    rng = np.random.default_rng(seed)
    if split_cfg.stratify:
        return _stratified_split(df, cfg, rng)
    draw = rng.random(len(df))
    test_cut = split_cfg.test_size
    valid_cut = test_cut + split_cfg.valid_size
    frames = {"test": df.loc[draw < test_cut].copy()}
    frames["valid"] = df.loc[(draw >= test_cut) & (draw < valid_cut)].copy()
    frames["train"] = df.loc[draw >= valid_cut].copy()
    frames = {k: v for k, v in frames.items() if len(v)}
    if "train" not in frames:
        raise ValueError("random split left no training rows")
    return frames


def _stratified_split(df: pd.DataFrame, cfg: DataConfig, rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    """Random split that preserves the target rate in every split.

    Matters whenever the outcome is rare: an unstratified draw can hand valid or
    test a materially different base rate, which shows up as noise in the Gini
    comparison the whole pipeline exists to make.
    """
    split_cfg = cfg.split
    labels = pd.Series("train", index=df.index, dtype=object)
    for _, index in df.groupby(df[cfg.target], sort=False).groups.items():
        index = np.asarray(index)
        shuffled = index[rng.permutation(len(index))]
        n_test = int(round(len(index) * split_cfg.test_size))
        n_valid = int(round(len(index) * split_cfg.valid_size))
        labels.loc[shuffled[:n_test]] = "test"
        labels.loc[shuffled[n_test:n_test + n_valid]] = "valid"
    frames = {name: df.loc[labels == name].copy() for name in SPLITS}
    frames = {k: v for k, v in frames.items() if len(v)}
    if "train" not in frames:
        raise ValueError("stratified split left no training rows")
    return frames


def build_dataset(cfg: Config) -> Dataset:
    """Read whatever the config points at and produce a cleaned ``Dataset``.

    ``data.paths`` (already-split inputs) takes precedence over ``data.path``.
    """
    if cfg.data.paths:
        logger.info("reading %d pre-split input(s); data.split is ignored", len(cfg.data.paths))
        frames = {name: read_frame(cfg.data, path) for name, path in cfg.data.paths.items()}
        return prepare_dataset_from_frames(frames, cfg)
    return prepare_dataset(read_frame(cfg.data), cfg)


def prepare_dataset(df: pd.DataFrame, cfg: Config) -> Dataset:
    """Build a ``Dataset`` from one in-memory table, splitting it per ``data.split``.

    This is the entry point for a discover -> verify loop, where candidate
    features are engineered in the session and never round-trip through a file.
    """
    if cfg.data.nrows and len(df) > cfg.data.nrows:
        df = df.head(cfg.data.nrows)
    if cfg.data.target not in df.columns:
        raise KeyError(f"target column {cfg.data.target!r} not in data")
    return _assemble(assign_splits(df, cfg.data, cfg.run.seed), cfg)


def prepare_dataset_from_frames(frames: Dict[str, pd.DataFrame], cfg: Config) -> Dataset:
    """Build a ``Dataset`` from inputs that are already split.

    Use this when train/valid/test are prepared upstream - separate files, or
    separate frames in memory. No splitting happens and ``data.split`` is unused;
    the frames are taken as-is, so any out-of-time or sampling logic upstream is
    preserved exactly.
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
        + [c for c in (cfg.data.weight_col, cfg.data.split.column, cfg.data.split.time_col)
           if c and c in reference.columns]
    ))

    prepared = {}
    for name, frame in frames.items():
        absent = [c for c in keep if c not in frame.columns]
        if absent:
            raise KeyError(f"the {name!r} split is missing column(s) present in train: {absent}")
        prepared[name] = clean_missing(frame[keep], base + new, cfg.data.missing_values)

    dataset = Dataset(
        frames=prepared,
        target=cfg.data.target,
        base_features=base,
        new_features=new,
        weight_col=cfg.data.weight_col,
    )
    logger.info("dataset ready: %s", dataset.describe())
    return dataset
