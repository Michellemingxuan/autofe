"""Stage 3 - XGBoost model builds (pinned to 1.7.6).

The unit of work is a *variant*: a named feature list. Verifying new features is
then just training several variants and comparing them:

    base            incumbent features only (the champion)
    base_plus_new   incumbent + every selected new feature
    new_only        the new features on their own
    leave_one_in    base + exactly one new feature, one variant per new feature
    leave_one_out   base + all new features except one, one variant per feature

Variants are independent, so they are trained in parallel; each worker's XGBoost
thread budget is divided out of the machine's cores to avoid oversubscription.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from validation.config import Config
from validation.data import Dataset
from validation.logging_utils import get_logger, timed
from validation.metrics import calc_adj_gini
from validation.model import fit_booster
from validation.parallel import parallel_map, resolve_n_jobs, threads_per_worker

logger = get_logger(__name__)

DEFAULT_PARAMS = {
    "regression": {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
    },
    "binary": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
    },
}


DEFAULT_SEARCH_SPACE: Dict[str, Any] = {
    "eta": [0.02, 0.05, 0.1],
    "max_depth": {"low": 3, "high": 8, "int": True},
    "min_child_weight": {"low": 1, "high": 30, "int": True},
    "subsample": {"low": 0.6, "high": 1.0},
    "colsample_bytree": {"low": 0.5, "high": 1.0},
    "reg_lambda": {"low": 0.5, "high": 10.0, "log": True},
}


@dataclass
class VariantSpec:
    name: str
    features: List[str]
    note: str = ""


@dataclass
class ModelResult:
    name: str
    features: List[str]
    note: str = ""
    model_path: Optional[str] = None
    best_iteration: Optional[int] = None
    best_score: Optional[float] = None
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: Dict[str, Any] = field(default_factory=dict)
    tuned: bool = False
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Variant construction
# --------------------------------------------------------------------------- #
def build_variants(base: List[str], new: List[str], requested: List[str]) -> List[VariantSpec]:
    variants: List[VariantSpec] = []
    seen = set()

    def add(spec: VariantSpec) -> None:
        if not spec.features:
            logger.warning("variant %r has no features; skipping", spec.name)
            return
        if spec.name in seen:
            return
        seen.add(spec.name)
        variants.append(spec)

    for kind in requested:
        if kind == "base":
            add(VariantSpec("base", list(base), "incumbent features only"))
        elif kind == "base_plus_new":
            add(VariantSpec("base_plus_new", list(base) + list(new), "incumbent + all selected new"))
        elif kind == "new_only":
            add(VariantSpec("new_only", list(new), "new features only"))
        elif kind == "leave_one_in":
            for feature in new:
                add(VariantSpec(f"loi__{feature}", list(base) + [feature], f"base + {feature}"))
        elif kind == "leave_one_out":
            for feature in new:
                rest = [f for f in new if f != feature]
                add(VariantSpec(f"loo__{feature}", list(base) + rest, f"base + all new except {feature}"))
    return variants


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _train_variant(
    spec: VariantSpec,
    matrices: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    weights: Dict[str, Optional[np.ndarray]],
    column_index: Dict[str, int],
    params_by_variant: Dict[str, Dict[str, object]],
    num_boost_round: int,
    early_stopping_rounds: Optional[int],
    verbose_eval: int,
    nthread: int,
    model_dir: Optional[str],
    tuned: bool = False,
) -> ModelResult:
    result = ModelResult(name=spec.name, features=list(spec.features), note=spec.note)
    try:
        cols = [column_index[f] for f in spec.features]
        fitted = fit_booster(
            {split: matrix[:, cols] for split, matrix in matrices.items()},
            targets,
            spec.features,
            params_by_variant[spec.name],
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            weights=weights,
            nthread=nthread,
            verbose_eval=verbose_eval,
        )
        booster = fitted.booster
        result.params = fitted.params
        result.tuned = tuned
        result.best_iteration = fitted.best_iteration
        result.best_score = fitted.best_score
        result.predictions = dict(fitted.predictions)

        result.importance = _importance_frame(booster, spec.features)

        if model_dir:
            path = Path(model_dir) / f"{spec.name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(path))
            result.model_path = str(path)
    except Exception as exc:  # a single bad variant should not sink the run
        result.error = f"{type(exc).__name__}: {exc}"
        logger.exception("variant %s failed", spec.name)
    return result


def _importance_frame(booster, features: List[str]) -> pd.DataFrame:
    """Split-based importances, including total gain (the tree 'Gini gain')."""
    scores = {kind: booster.get_score(importance_type=kind)
              for kind in ("gain", "total_gain", "weight", "cover", "total_cover")}
    rows = []
    for feature in features:
        row = {"feature": feature}
        for kind, mapping in scores.items():
            row[f"imp_{kind}"] = float(mapping.get(feature, 0.0))
        rows.append(row)
    frame = pd.DataFrame(rows)
    total = frame["imp_total_gain"].sum()
    frame["imp_total_gain_pct"] = frame["imp_total_gain"] / total if total > 0 else np.nan
    return frame.sort_values("imp_total_gain", ascending=False, ignore_index=True)




# --------------------------------------------------------------------------- #
# Hyperparameter tuning
# --------------------------------------------------------------------------- #
def sample_params(space: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """Draw one random configuration.

    A list is a categorical choice; a dict is a range - ``{"low", "high"}`` plus
    optional ``"int"`` and ``"log"``. Anything else is a fixed value.
    """
    drawn: Dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, (list, tuple)):
            drawn[name] = spec[int(rng.integers(len(spec)))]
        elif isinstance(spec, dict) and "low" in spec and "high" in spec:
            low, high = float(spec["low"]), float(spec["high"])
            if spec.get("log"):
                value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                value = float(rng.uniform(low, high))
            drawn[name] = int(round(value)) if spec.get("int") else value
        else:
            drawn[name] = spec
    return drawn


def _score_trial(
    task: tuple,
    matrices: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    weights: Dict[str, Optional[np.ndarray]],
    column_index: Dict[str, int],
    base_params: Dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: Optional[int],
    nthread: int,
) -> Dict[str, Any]:
    """Train one candidate configuration and score it on the valid split.

    Scored with the same adjusted Gini the pipeline reports, so tuning optimises
    the quantity the verdict is later read from. Only train and valid are touched
    - test never enters tuning.
    """
    import xgboost as xgb

    spec, trial_params = task
    out: Dict[str, Any] = {"variant": spec.name, "params": trial_params,
                           "score": float("nan"), "best_iteration": None, "error": None}
    try:
        cols = [column_index[f] for f in spec.features]
        run_params = {**base_params, **trial_params, "nthread": nthread}
        dtrain = xgb.DMatrix(np.asarray(matrices["train"][:, cols], dtype=np.float32),
                             label=targets["train"], weight=weights.get("train"),
                             missing=np.nan, nthread=nthread)
        dvalid = xgb.DMatrix(np.asarray(matrices["valid"][:, cols], dtype=np.float32),
                             label=targets["valid"], weight=weights.get("valid"),
                             missing=np.nan, nthread=nthread)
        booster = xgb.train(run_params, dtrain, num_boost_round=num_boost_round,
                            evals=[(dvalid, "valid")],
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=False)
        best = getattr(booster, "best_iteration", None)
        kwargs = {"iteration_range": (0, best + 1)} if best is not None else {}
        preds = booster.predict(dvalid, **kwargs)
        frame = pd.DataFrame({"actual": targets["valid"], "pred": preds})
        out["score"] = float(calc_adj_gini(frame, "actual", "pred"))
        out["best_iteration"] = int(best) if best is not None else None
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("tuning trial failed for %s: %s", spec.name, out["error"])
    return out


def run_tuning(
    variants: List[VariantSpec],
    cfg: Config,
    matrices: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    weights: Dict[str, Optional[np.ndarray]],
    column_index: Dict[str, int],
    base_params: Dict[str, Any],
) -> tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Return the winning params per variant, plus the full trial log."""
    tuning = cfg.model.tuning
    space = tuning.search_space or DEFAULT_SEARCH_SPACE
    rng = np.random.default_rng(tuning.seed if tuning.seed is not None else cfg.run.seed)

    if "valid" not in matrices:
        raise ValueError("model.tuning needs a valid split to score against; "
                         "give data.paths a valid table")

    if tuning.mode == "shared":
        target = next((v for v in variants if v.name == tuning.tune_on), None)
        if target is None:
            target = variants[0]
            logger.warning("tuning.tune_on=%r not built; tuning on %r instead",
                           tuning.tune_on, target.name)
        to_tune = [target]
    else:
        to_tune = list(variants)

    tasks = [(spec, sample_params(space, rng))
             for spec in to_tune for _ in range(tuning.n_trials)]
    logger.info("tuning: %s mode, %d trial(s) across %d variant(s) = %d fit(s)",
                tuning.mode, tuning.n_trials, len(to_tune), len(tasks))

    nthread = cfg.model.threads_per_model or threads_per_worker(cfg.run.n_jobs, len(tasks))
    with timed(logger, "hyperparameter search"):
        trials = parallel_map(
            _score_trial, tasks,
            n_jobs=cfg.run.n_jobs, backend=cfg.run.backend, desc="trial",
            matrices=matrices, targets=targets, weights=weights, column_index=column_index,
            base_params=base_params, num_boost_round=cfg.model.num_boost_round,
            early_stopping_rounds=cfg.model.early_stopping_rounds, nthread=nthread,
        )

    log = pd.DataFrame([{**t, "params": json.dumps(t["params"], sort_keys=True)} for t in trials])
    best_by_variant: Dict[str, Dict[str, Any]] = {}
    for spec in to_tune:
        scored = [t for t in trials if t["variant"] == spec.name and not np.isnan(t["score"])]
        if not scored:
            logger.error("every tuning trial failed for %s; keeping the configured params", spec.name)
            continue
        best = max(scored, key=lambda t: t["score"])
        best_by_variant[spec.name] = best["params"]
        logger.info("tuned %s: valid %s=%.4f with %s",
                    spec.name, tuning.metric, best["score"], best["params"])

    params_by_variant: Dict[str, Dict[str, Any]] = {}
    for spec in variants:
        if tuning.mode == "shared":
            winner = best_by_variant.get(to_tune[0].name, {})
        else:
            winner = best_by_variant.get(spec.name, {})
        params_by_variant[spec.name] = {**base_params, **winner}
    return params_by_variant, log


def run_modeling(
    dataset: Dataset,
    cfg: Config,
    model_dir: Optional[Path] = None,
) -> tuple[List[ModelResult], pd.DataFrame]:
    variants = build_variants(dataset.base_features, dataset.new_features, cfg.model.variants)
    if not variants:
        raise ValueError("no model variants to train; check model.variants and the feature lists")

    features = dataset.all_features
    column_index = {name: i for i, name in enumerate(features)}
    splits = dataset.available_splits()

    matrices = {s: dataset.split(s)[features].to_numpy(dtype=np.float32) for s in splits}
    targets = {s: dataset.split(s)[dataset.target].to_numpy(dtype=np.float32) for s in splits}
    weights = {
        s: (dataset.split(s)[dataset.weight_col].to_numpy(dtype=np.float32) if dataset.weight_col else None)
        for s in splits
    }

    params = dict(DEFAULT_PARAMS[cfg.model.task])
    params.update(cfg.model.params or {})
    params.setdefault("seed", cfg.run.seed)

    tuning_log = pd.DataFrame()
    if cfg.model.tuning.enabled:
        params_by_variant, tuning_log = run_tuning(
            variants, cfg, matrices, targets, weights, column_index, params)
    else:
        params_by_variant = {spec.name: dict(params) for spec in variants}

    nthread = cfg.model.threads_per_model or threads_per_worker(cfg.run.n_jobs, len(variants))
    logger.info(
        "training %d variant(s) on %d worker(s), %d xgboost thread(s) each",
        len(variants), min(resolve_n_jobs(cfg.run.n_jobs), len(variants)), nthread,
    )

    with timed(logger, "model training"):
        results = parallel_map(
            _train_variant, variants,
            n_jobs=cfg.run.n_jobs, backend=cfg.run.backend, desc="variant",
            matrices=matrices, targets=targets, weights=weights, column_index=column_index,
            params_by_variant=params_by_variant, num_boost_round=cfg.model.num_boost_round,
            early_stopping_rounds=cfg.model.early_stopping_rounds,
            verbose_eval=cfg.model.verbose_eval, nthread=nthread,
            model_dir=str(model_dir) if (model_dir and cfg.model.save_models) else None,
            tuned=cfg.model.tuning.enabled,
        )

    failed = [r.name for r in results if r.error]
    if failed:
        logger.error("%d variant(s) failed: %s", len(failed), failed)
    return results, tuning_log
