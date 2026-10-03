"""Stage 4 - outcome analysis.

Performance-oriented version for large datasets.

Main changes from the original implementation:

* Metrics targets are stored as float32 instead of float64.
* SHAP is computed on a bounded sample and in row chunks so the full
  SHAP matrix is never materialised.
* SHAP only constructs the feature matrix needed by the current variant.
* SHAP variants are processed sequentially by default on memory-constrained
  machines; an optional ``analysis.shap.n_jobs`` can override this.
* The SHAP reduction keeps only sum(|SHAP|) per feature, which is all that is
  needed for mean absolute SHAP and share/ranking.

The public result structures and output columns are kept compatible with the
original analysis stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from validation.config import Config
from validation.data import Dataset
from validation.logging_utils import get_logger, timed
from validation.metrics import evaluate_predictions, gini_gain
from validation.parallel import parallel_map
from validation.stages.modeling import ModelResult

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    shap_ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    new_feature_share: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        if self.comparison.empty:
            return {}
        cols = [c for c in self.comparison.columns if c.startswith("gini_gain_")]
        best = (
            self.comparison.sort_values(cols[0], ascending=False)
            if cols
            else self.comparison
        )
        return {
            "variants": int(len(self.comparison)),
            "best_variant": str(best.iloc[0]["variant"]),
            "new_feature_shap_share": self.new_feature_share,
        }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _capture_percents(cfg: Config) -> List[float]:
    """Everything asked for, so a headline percent is never silently missing."""
    return sorted(
        set(cfg.analysis.capture_rate_percents)
        | set(cfg.analysis.comparison_capture_percents)
    )


def _pct_label(percent: float) -> str:
    return f"top{percent * 100:g}"


def _metrics_for_variant(
    result: ModelResult,
    targets: Dict[str, np.ndarray],
    splits: List[str],
    capture_percents: List[float],
    accuracy_bins: int,
) -> pd.DataFrame:
    rows = []
    for split in splits:
        if split not in result.predictions or split not in targets:
            continue
        frame = pd.DataFrame(
            {
                "actual": targets[split],
                "pred": result.predictions[split],
            }
        )
        scores = evaluate_predictions(
            frame,
            "actual",
            "pred",
            capture_percents=capture_percents,
            accuracy_bins=accuracy_bins,
        )
        rows.append(
            {
                "variant": result.name,
                "split": split,
                "n_features": len(result.features),
                **scores,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def _shap_for_variant(
    result: ModelResult,
    frame: pd.DataFrame,
    nthread: int,
    chunk_size: int,
) -> pd.DataFrame:
    """Mean |SHAP| per feature for one variant, computed in row chunks.

    Only the current variant's feature matrix is materialised. SHAP values are
    reduced immediately to per-feature sums, so an entire rows x features SHAP
    matrix is never kept in memory.
    """
    if result.error or not result.model_path:
        return pd.DataFrame()

    if frame.empty or not result.features:
        return pd.DataFrame()

    import xgboost as xgb

    sample = frame[result.features].to_numpy(dtype=np.float32, copy=False)

    booster = xgb.Booster()
    booster.load_model(result.model_path)
    booster.set_param("nthread", nthread)

    try:
        import shap

        explainer = shap.TreeExplainer(booster)
        use_shap_package = True
    except Exception as exc:
        logger.warning(
            "shap unavailable for %s (%s); using pred_contribs",
            result.name,
            exc,
        )
        explainer = None
        use_shap_package = False

    chunk_size = max(1, int(chunk_size))
    sum_abs = np.zeros(len(result.features), dtype=np.float64)
    n_rows = 0

    try:
        for start in range(0, sample.shape[0], chunk_size):
            stop = min(start + chunk_size, sample.shape[0])
            chunk = np.ascontiguousarray(sample[start:stop], dtype=np.float32)

            dmatrix = xgb.DMatrix(
                chunk,
                feature_names=list(result.features),
                missing=np.nan,
                nthread=nthread,
            )

            if use_shap_package:
                values = explainer.shap_values(
                    dmatrix,
                    check_additivity=False,
                )

                # Multiclass: combine classes by summing absolute attribution
                # before the feature-wise reduction.
                if isinstance(values, list):
                    chunk_sum = np.zeros(
                        len(result.features),
                        dtype=np.float64,
                    )
                    for value in values:
                        chunk_sum += np.abs(value).sum(axis=0)
                else:
                    chunk_sum = np.abs(values).sum(axis=0)
            else:
                # XGBoost's pred_contribs includes one final bias column.
                values = booster.predict(dmatrix, pred_contribs=True)
                if values.ndim != 2 or values.shape[1] < len(result.features) + 1:
                    raise ValueError(
                        "pred_contribs returned an unexpected shape "
                        f"{values.shape} for {len(result.features)} features"
                    )
                chunk_sum = np.abs(values[:, : len(result.features)]).sum(axis=0)

            sum_abs += np.asarray(chunk_sum, dtype=np.float64)
            n_rows += stop - start

            del dmatrix, values, chunk, chunk_sum

    except Exception as exc:
        logger.warning(
            "SHAP failed for %s (%s); falling back to XGBoost pred_contribs",
            result.name,
            exc,
        )

        # Retry from scratch with XGBoost's native contribution path.
        sum_abs.fill(0.0)
        n_rows = 0

        for start in range(0, sample.shape[0], chunk_size):
            stop = min(start + chunk_size, sample.shape[0])
            chunk = np.ascontiguousarray(sample[start:stop], dtype=np.float32)

            dmatrix = xgb.DMatrix(
                chunk,
                feature_names=list(result.features),
                missing=np.nan,
                nthread=nthread,
            )

            values = booster.predict(dmatrix, pred_contribs=True)
            expected_cols = len(result.features) + 1
            if values.ndim != 2 or values.shape[1] < expected_cols:
                raise ValueError(
                    "pred_contribs returned an unexpected shape "
                    f"{values.shape} for {len(result.features)} features"
                )

            sum_abs += np.abs(
                values[:, : len(result.features)]
            ).sum(axis=0)
            n_rows += stop - start

            del dmatrix, values, chunk

    if n_rows <= 0:
        return pd.DataFrame()

    mean_abs = sum_abs / float(n_rows)

    out = pd.DataFrame(
        {
            "variant": result.name,
            "feature": list(result.features),
            "mean_abs_shap": mean_abs,
        }
    )

    total = float(out["mean_abs_shap"].sum())
    out["shap_share"] = (
        out["mean_abs_shap"] / total if total > 0 else np.nan
    )
    out = out.sort_values(
        "mean_abs_shap",
        ascending=False,
        ignore_index=True,
    )
    out["shap_rank"] = np.arange(1, len(out) + 1)
    return out


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def run_analysis(
    dataset: Dataset,
    cfg: Config,
    results: List[ModelResult],
) -> AnalysisResult:
    ok = [r for r in results if not r.error]
    if not ok:
        logger.error("no successful model variants to analyse")
        return AnalysisResult()

    splits = [
        s
        for s in cfg.analysis.metrics_on
        if s in dataset.available_splits()
    ]

    # Predictions are already one-dimensional; float32 is sufficient for
    # metric evaluation and cuts target memory in half versus float64.
    targets = {
        s: dataset.split(s)[dataset.target].to_numpy(dtype=np.float32)
        for s in splits
    }

    with timed(logger, "metric evaluation"):
        metric_frames = parallel_map(
            _metrics_for_variant,
            ok,
            n_jobs=cfg.run.n_jobs,
            backend=cfg.run.backend,
            desc="metrics",
            targets=targets,
            splits=splits,
            capture_percents=_capture_percents(cfg),
            accuracy_bins=cfg.analysis.accuracy_bins,
        )

    metrics = (
        pd.concat(
            [f for f in metric_frames if len(f)],
            ignore_index=True,
        )
        if metric_frames
        else pd.DataFrame()
    )

    comparison = _build_comparison(metrics, ok, cfg, splits)

    shap_ranking = pd.DataFrame()
    if cfg.analysis.shap.enabled:
        shap_ranking = _run_shap(dataset, cfg, ok)

    importance = (
        pd.concat(
            [r.importance.assign(variant=r.name) for r in ok if not r.importance.empty],
            ignore_index=True,
        )
        if any(not r.importance.empty for r in ok)
        else pd.DataFrame()
    )

    feature_ranking = _merge_rankings(shap_ranking, importance)
    share = _new_feature_share(shap_ranking, dataset.new_features)

    return AnalysisResult(
        metrics=metrics,
        comparison=comparison,
        shap_ranking=shap_ranking,
        importance=importance,
        feature_ranking=feature_ranking,
        new_feature_share=share,
    )


def _run_shap(
    dataset: Dataset,
    cfg: Config,
    results: List[ModelResult],
) -> pd.DataFrame:
    shap_cfg = cfg.analysis.shap
    split = (
        shap_cfg.on_split
        if shap_cfg.on_split in dataset.available_splits()
        else "train"
    )

    frame = dataset.split(split)

    # Bound the SHAP workload. This setting already exists in the original
    # configuration and is therefore preserved.
    if shap_cfg.sample_size and len(frame) > shap_cfg.sample_size:
        frame = frame.sample(
            n=shap_cfg.sample_size,
            random_state=cfg.run.seed,
        )

    explainable = [r for r in results if r.model_path]
    if len(explainable) < len(results):
        logger.warning(
            "shap skipped for %d variant(s) without a saved model "
            "(set model.save_models: true)",
            len(results) - len(explainable),
        )
    if not explainable:
        return pd.DataFrame()

    # Optional settings. Defaults are intentionally conservative for a
    # memory-constrained laptop. They can be added to config later without
    # breaking older configs.
    chunk_size = int(getattr(shap_cfg, "chunk_size", 2000))
    requested_jobs = getattr(shap_cfg, "n_jobs", None)

    if requested_jobs is None:
        # SHAP is memory-heavy; one variant at a time is the safest default.
        shap_jobs = 1
    else:
        shap_jobs = max(1, int(requested_jobs))

    from validation.parallel import threads_per_worker

    nthread = threads_per_worker(
        shap_jobs,
        len(explainable),
    )

    # Do not allocate the all-feature matrix here. Each worker reads only
    # result.features, which is generally much smaller than dataset.all_features.
    with timed(
        logger,
        f"shap on {len(frame)} {split} row(s) in {chunk_size}-row chunks",
    ):
        frames = parallel_map(
            _shap_for_variant,
            explainable,
            n_jobs=shap_jobs,
            backend=cfg.run.backend,
            desc="shap",
            frame=frame,
            nthread=nthread,
            chunk_size=chunk_size,
        )

    frames = [f for f in frames if len(f)]
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )


def _build_comparison(
    metrics: pd.DataFrame,
    results: List[ModelResult],
    cfg: Config,
    splits: List[str],
) -> pd.DataFrame:
    """One row per variant: Gini per split plus gain over the baseline variant."""
    if metrics.empty:
        return pd.DataFrame()

    wide = metrics.pivot(
        index="variant",
        columns="split",
        values="adj_gini",
    )
    wide.columns = [f"adj_gini_{c}" for c in wide.columns]

    if cfg.analysis.include_accuracy:
        extras = metrics.pivot(
            index="variant",
            columns="split",
            values="accuracy",
        )
        extras.columns = [f"accuracy_{c}" for c in extras.columns]
        wide = wide.join(extras)

    for percent in cfg.analysis.comparison_capture_percents:
        source = f"capture_rate_{percent:g}"
        if source not in metrics.columns:
            continue
        captured = metrics.pivot(
            index="variant",
            columns="split",
            values=source,
        )
        captured.columns = [
            f"capture_{_pct_label(percent)}_{c}"
            for c in captured.columns
        ]
        wide = wide.join(captured)

    notes = {r.name: r for r in results}
    wide["n_features"] = [len(notes[v].features) for v in wide.index]
    wide["note"] = [notes[v].note for v in wide.index]
    wide["best_iteration"] = [notes[v].best_iteration for v in wide.index]

    baseline = cfg.analysis.baseline_variant
    if baseline in wide.index:
        for split in splits:
            col = f"adj_gini_{split}"
            if col not in wide.columns:
                continue
            champion = float(wide.loc[baseline, col])
            gains = [
                gini_gain(float(v), champion)
                for v in wide[col]
            ]
            wide[f"gini_gain_{split}"] = [
                g["gini_gain"] for g in gains
            ]
            wide[f"gini_gain_pct_{split}"] = [
                g["gini_gain_pct"] for g in gains
            ]

        # capture rate is the metric a reviewer actually feels, so give it a gain too
        for percent in cfg.analysis.comparison_capture_percents:
            for split in splits:
                col = f"capture_{_pct_label(percent)}_{split}"
                if col not in wide.columns:
                    continue
                champion = float(wide.loc[baseline, col])
                wide[
                    f"capture_gain_{_pct_label(percent)}_{split}"
                ] = [
                    float(v) - champion
                    for v in wide[col]
                ]
    else:
        logger.warning(
            "baseline variant %r not among trained variants; no gini gain computed",
            baseline,
        )

    return wide.reset_index()


def _merge_rankings(
    shap_ranking: pd.DataFrame,
    importance: pd.DataFrame,
) -> pd.DataFrame:
    if shap_ranking.empty and importance.empty:
        return pd.DataFrame()
    if shap_ranking.empty:
        return importance
    if importance.empty:
        return shap_ranking
    merged = shap_ranking.merge(
        importance,
        on=["variant", "feature"],
        how="outer",
    )
    return merged.sort_values(
        ["variant", "mean_abs_shap"],
        ascending=[True, False],
        ignore_index=True,
    )


def _new_feature_share(
    shap_ranking: pd.DataFrame,
    new_features: List[str],
) -> Dict[str, float]:
    """Fraction of each variant's total mean |SHAP| that lands on new features."""
    if shap_ranking.empty or not new_features:
        return {}
    new = set(new_features)
    share: Dict[str, float] = {}
    for variant, group in shap_ranking.groupby("variant"):
        total = group["mean_abs_shap"].sum()
        if total > 0:
            share[str(variant)] = float(
                group.loc[
                    group["feature"].isin(new),
                    "mean_abs_shap",
                ].sum()
                / total
            )
    return share
