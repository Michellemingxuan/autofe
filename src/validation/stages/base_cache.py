"""The base model, trained once and kept: an evaluation reuses it.

Every evaluation compares challengers against ``base`` - the incumbent features
only. Retraining it each time - and retuning it, and re-explaining it with
SHAP - costs the most of any variant and changes nothing: the same data, the
same features and the same settings give the same model. So when
``model.base_cache_dir`` is set, the base variant is kept there under a
fingerprint of everything that decides it, and a later run that would build
the same base loads it instead:

* the data - per split, the base columns' values, the target and the weights,
  hashed column by column;
* the base feature list, the model parameters and the task;
* the boosting rounds and early stopping;
* the tuning settings, when tuning is on - and the winning parameters are kept
  with the model, so a shared tuning that would have tuned base hands every
  other variant the same parameters it did before;
* the seed and the xgboost version.

Any change to one of them is a new fingerprint, so a stale base is never used.
Its SHAP ranking is kept beside it, keyed by the SHAP settings.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from validation.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["fingerprint", "load", "save", "shap_key", "load_shap", "save_shap"]


def fingerprint(features: list, matrices: Dict[str, np.ndarray], targets: Dict[str, np.ndarray],
                weights: Dict[str, Optional[np.ndarray]], column_index: Dict[str, int],
                params: Dict[str, Any], cfg: Any) -> str:
    """A key for the base model: equal exactly when it would come out the same."""
    import xgboost

    h = hashlib.sha256()
    tuning = cfg.model.tuning
    settings = {
        "features": list(features), "params": params, "task": cfg.model.task,
        "num_boost_round": cfg.model.num_boost_round,
        "early_stopping_rounds": cfg.model.early_stopping_rounds,
        "tuning": asdict(tuning) if tuning.enabled else None,
        "seed": cfg.run.seed, "xgboost": xgboost.__version__,
    }
    h.update(json.dumps(settings, sort_keys=True, default=str).encode())
    for split in sorted(matrices):
        h.update(f"|{split}|{matrices[split].shape[0]}|".encode())
        for name in features:              # a column at a time: no copy of the whole matrix
            h.update(np.ascontiguousarray(matrices[split][:, column_index[name]]).tobytes())
        h.update(np.ascontiguousarray(targets[split]).tobytes())
        if weights.get(split) is not None:
            h.update(np.ascontiguousarray(weights[split]).tobytes())
    return h.hexdigest()[:20]


def _entry(cache_dir: str, key: str) -> Path:
    return Path(cache_dir) / key


def load(cache_dir: str, key: str, model_dir: Optional[str]) -> Optional[tuple[Any, Dict[str, Any]]]:
    """The kept base result and the parameters it was trained with, or None. Its
    saved model is copied beside the run's own models, where SHAP reads it."""
    entry = _entry(cache_dir, key)
    path = entry / "result.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            result, trained_with = pickle.load(fh)
    except Exception as error:  # noqa: BLE001 - a broken entry is retrained, not fatal
        logger.warning("base cache %s unreadable (%s); training base again", key, error)
        return None
    model = entry / "base.json"
    if model.exists():
        if model_dir:
            target = Path(model_dir) / "base.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model, target)
            result.model_path = str(target)
        else:
            result.model_path = str(model)
    result.cache_entry = str(entry)
    return result, trained_with


def save(cache_dir: str, key: str, result: Any, trained_with: Dict[str, Any]) -> None:
    """Keep a freshly trained base, with the parameters it was trained with."""
    entry = _entry(cache_dir, key)
    entry.mkdir(parents=True, exist_ok=True)
    if result.model_path and Path(result.model_path).exists():
        shutil.copyfile(result.model_path, entry / "base.json")
    with open(entry / "result.pkl", "wb") as fh:
        pickle.dump((result, trained_with), fh)
    result.cache_entry = str(entry)


def shap_key(split: str, rows: int, cfg: Any) -> str:
    shap = cfg.analysis.shap
    return hashlib.sha256(json.dumps(
        {"split": split, "rows": rows, "sample_size": shap.sample_size, "seed": cfg.run.seed},
        sort_keys=True).encode()).hexdigest()[:12]


def load_shap(entry: str, key: str) -> Optional[pd.DataFrame]:
    path = Path(entry) / f"shap_{key}.pkl"
    return pd.read_pickle(path) if path.exists() else None


def save_shap(entry: str, key: str, frame: pd.DataFrame) -> None:
    frame.to_pickle(Path(entry) / f"shap_{key}.pkl")
