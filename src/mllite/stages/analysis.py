"""Stage 4 - outcome analysis.

Three things, all fanned out across model variants:

* **Metrics**  - adjusted Gini, accuracy and capture rates per split.
* **Gini gain** - each variant's adjusted Gini minus the baseline variant's,
                  which is the headline answer to "did the new features help?".
* **SHAP**     - TreeExplainer feature ranking (mean |SHAP|), alongside the
                  model's own total-gain importance, plus the share of total
                  attribution captured by the new features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from mllite.config import Config
from mllite.data import Dataset
from mllite.logging_utils import get_logger, timed
from mllite.metrics import evaluate_predictions, gini_gain
from mllite.parallel import parallel_map
from mllite.stages.modeling import ModelResult

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
        best = self.comparison.sort_values(cols[0], ascending=False) if cols else self.comparison
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
    return sorted(set(cfg.analysis.capture_rate_percents)
                  | set(cfg.analysis.comparison_capture_percents))


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
        frame = pd.DataFrame({"actual": targets[split], "pred": result.predictions[split]})
        scores = evaluate_predictions(
            frame, "actual", "pred",
            capture_percents=capture_percents, accuracy_bins=accuracy_bins,
        )
        rows.append({"variant": result.name, "split": split, "n_features": len(result.features), **scores})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def _shap_for_variant(
    result: ModelResult,
    matrix: np.ndarray,
    column_index: Dict[str, int],
    nthread: int,
) -> pd.DataFrame:
    """Mean |SHAP| per feature for one variant, computed in its own worker."""
    if result.error:
        return pd.DataFrame()
    import xgboost as xgb

    cols = [column_index[f] for f in result.features]
    sample = np.asarray(matrix[:, cols], dtype=np.float32)
    dmatrix = xgb.DMatrix(sample, feature_names=list(result.features), missing=np.nan, nthread=nthread)

    booster = xgb.Booster()
    booster.load_model(result.model_path)
    booster.set_param("nthread", nthread)

    try:
        import shap

        explainer = shap.TreeExplainer(booster)
        values = explainer.shap_values(dmatrix, check_additivity=False)
        if isinstance(values, list):           # multiclass -> stack
            values = np.mean([np.abs(v) for v in values], axis=0)
    except Exception as exc:  # fall back to XGBoost's own exact tree SHAP
        logger.warning("shap unavailable for %s (%s); using pred_contribs", result.name, exc)
        values = booster.predict(dmatrix, pred_contribs=True)[:, :-1]  # drop bias column

    mean_abs = np.abs(values).mean(axis=0)
    frame = pd.DataFrame({
        "variant": result.name,
        "feature": list(result.features),
        "mean_abs_shap": mean_abs,
    })
    total = frame["mean_abs_shap"].sum()
    frame["shap_share"] = frame["mean_abs_shap"] / total if total > 0 else np.nan
    frame = frame.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    frame["shap_rank"] = np.arange(1, len(frame) + 1)
    return frame


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

    splits = [s for s in cfg.analysis.metrics_on if s in dataset.available_splits()]
    targets = {s: dataset.split(s)[dataset.target].to_numpy(dtype=np.float64) for s in splits}

    with timed(logger, "metric evaluation"):
        metric_frames = parallel_map(
            _metrics_for_variant, ok,
            n_jobs=cfg.run.n_jobs, backend=cfg.run.backend, desc="metrics",
            targets=targets, splits=splits,
            capture_percents=_capture_percents(cfg),
            accuracy_bins=cfg.analysis.accuracy_bins,
        )
    metrics = pd.concat([f for f in metric_frames if len(f)], ignore_index=True) if metric_frames else pd.DataFrame()

    comparison = _build_comparison(metrics, ok, cfg, splits)

    shap_ranking = pd.DataFrame()
    if cfg.analysis.shap.enabled:
        shap_ranking = _run_shap(dataset, cfg, ok)

    importance = pd.concat(
        [r.importance.assign(variant=r.name) for r in ok if not r.importance.empty],
        ignore_index=True,
    ) if any(not r.importance.empty for r in ok) else pd.DataFrame()

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


def _run_shap(dataset: Dataset, cfg: Config, results: List[ModelResult]) -> pd.DataFrame:
    shap_cfg = cfg.analysis.shap
    split = shap_cfg.on_split if shap_cfg.on_split in dataset.available_splits() else "train"
    frame = dataset.split(split)
    if shap_cfg.sample_size and len(frame) > shap_cfg.sample_size:
        frame = frame.sample(n=shap_cfg.sample_size, random_state=cfg.run.seed)

    features = dataset.all_features
    column_index = {name: i for i, name in enumerate(features)}
    matrix = frame[features].to_numpy(dtype=np.float32)

    explainable = [r for r in results if r.model_path]
    if len(explainable) < len(results):
        logger.warning("shap skipped for %d variant(s) without a saved model "
                       "(set model.save_models: true)", len(results) - len(explainable))
    if not explainable:
        return pd.DataFrame()

    from mllite.parallel import threads_per_worker
    with timed(logger, f"shap on {len(frame)} {split} row(s)"):
        frames = parallel_map(
            _shap_for_variant, explainable,
            n_jobs=cfg.run.n_jobs, backend=cfg.run.backend, desc="shap",
            matrix=matrix, column_index=column_index,
            nthread=threads_per_worker(cfg.run.n_jobs, len(explainable)),
        )
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_comparison(metrics: pd.DataFrame, results: List[ModelResult], cfg: Config, splits: List[str]) -> pd.DataFrame:
    """One row per variant: Gini per split plus gain over the baseline variant."""
    if metrics.empty:
        return pd.DataFrame()

    wide = metrics.pivot(index="variant", columns="split", values="adj_gini")
    wide.columns = [f"adj_gini_{c}" for c in wide.columns]
    if cfg.analysis.include_accuracy:
        extras = metrics.pivot(index="variant", columns="split", values="accuracy")
        extras.columns = [f"accuracy_{c}" for c in extras.columns]
        wide = wide.join(extras)

    for percent in cfg.analysis.comparison_capture_percents:
        source = f"capture_rate_{percent:g}"
        if source not in metrics.columns:
            continue
        captured = metrics.pivot(index="variant", columns="split", values=source)
        captured.columns = [f"capture_{_pct_label(percent)}_{c}" for c in captured.columns]
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
            gains = [gini_gain(float(v), champion) for v in wide[col]]
            wide[f"gini_gain_{split}"] = [g["gini_gain"] for g in gains]
            wide[f"gini_gain_pct_{split}"] = [g["gini_gain_pct"] for g in gains]

        # capture rate is the metric a reviewer actually feels, so give it a gain too
        for percent in cfg.analysis.comparison_capture_percents:
            for split in splits:
                col = f"capture_{_pct_label(percent)}_{split}"
                if col not in wide.columns:
                    continue
                champion = float(wide.loc[baseline, col])
                wide[f"capture_gain_{_pct_label(percent)}_{split}"] = [
                    float(v) - champion for v in wide[col]
                ]
    else:
        logger.warning("baseline variant %r not among trained variants; no gini gain computed", baseline)

    return wide.reset_index()


def _merge_rankings(shap_ranking: pd.DataFrame, importance: pd.DataFrame) -> pd.DataFrame:
    if shap_ranking.empty and importance.empty:
        return pd.DataFrame()
    if shap_ranking.empty:
        return importance
    if importance.empty:
        return shap_ranking
    merged = shap_ranking.merge(importance, on=["variant", "feature"], how="outer")
    return merged.sort_values(["variant", "mean_abs_shap"], ascending=[True, False], ignore_index=True)


def _new_feature_share(shap_ranking: pd.DataFrame, new_features: List[str]) -> Dict[str, float]:
    """Fraction of each variant's total mean |SHAP| that lands on new features."""
    if shap_ranking.empty or not new_features:
        return {}
    new = set(new_features)
    share: Dict[str, float] = {}
    for variant, group in shap_ranking.groupby("variant"):
        total = group["mean_abs_shap"].sum()
        if total > 0:
            share[str(variant)] = float(group.loc[group["feature"].isin(new), "mean_abs_shap"].sum() / total)
    return share
