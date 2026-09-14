"""Scoring metrics.

``cal_acc_worker``, ``calc_accuracy``, ``calc_adj_gini`` and ``capture_rate`` keep
the exact math supplied for the project; the only additions are guards that make
degenerate inputs (empty frame, zero actual total) return NaN instead of raising.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
import pandas as pd

MISSING_SENTINEL = -9999


# --------------------------------------------------------------------------- #
# Accuracy and Gini
# --------------------------------------------------------------------------- #
def cal_acc_worker(dsn: pd.DataFrame, y_pred: str, y_act: str) -> float:
    return ((dsn[y_pred] - dsn[y_act]).abs().sum() / dsn[y_act].abs().sum())


def calc_accuracy(df: pd.DataFrame, true_var: str, pred_var: str, k: int = 10) -> float:
    """
    This funciton returns the accuracy of model predicitons, accuracy is defined as
         1 - sum of abs(true - prediction) across bins / k
    df: dataframe with the true values and predicted values
    true_var: col name of the true values
    pred_var: col name of the predicted values
    k: number of bins used for accuracy calculation, 10 by default
    """
    df = df.loc[~((df[true_var].isnull()) | (df[true_var] == MISSING_SENTINEL))]
    n = len(df)
    if n == 0:
        return float("nan")
    y = df.sort_values(pred_var, ascending=False)
    y['rank'] = list(range(n))
    y['grp'] = y['rank'].apply(lambda x: math.floor(k * x / n))
    z = y.groupby(by=['grp'])[[pred_var, true_var]].mean()
    if z[true_var].abs().sum() == 0:
        return float("nan")
    return (1 - cal_acc_worker(z, pred_var, true_var))


# Adjusted GINI --> we take every sample, rather than taking decile.
def calc_adj_gini(df: pd.DataFrame, true_var: str, pred_var: str) -> float:
    """
    This funciton returns the Enhanced Gini of model predicitons
    df: dataframe with the true values and predicted values
    true_var: col name of the true values
    pred_var: col name of the predicted values
    """
    df = df.dropna(subset=[true_var])
    if df.empty:
        return float("nan")
    total = df[true_var].sum()
    if total == 0:
        return float("nan")
    y1 = df[[true_var, pred_var]].sort_values(pred_var, ascending=False)[true_var].cumsum().values / total
    y2 = df[[true_var, pred_var]].sort_values(true_var, ascending=False)[true_var].cumsum().values / total
    x1 = ((2 * y1.sum() - y1[-1]) / df.shape[0]) - 1  # 2 x AUC minus 1
    x2 = ((2 * y2.sum() - y2[-1]) / df.shape[0]) - 1
    if x2 == 0:
        return float("nan")
    return x1 / x2


def capture_rate(df: pd.DataFrame, actual: str, pred: str, percent: float = 0.05) -> float:
    df = df[~df[actual].isnull()]
    if df.empty or df[actual].sum() == 0:
        return float("nan")
    df = df.sort_values(by=[pred], ascending=[False])
    num_rows = int(len(df) * (percent))
    return df.head(num_rows)[actual].sum() / df[actual].sum()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def evaluate_predictions(
    df: pd.DataFrame,
    true_var: str,
    pred_var: str,
    capture_percents: Sequence[float] = (0.01, 0.05, 0.10),
    accuracy_bins: int = 10,
) -> Dict[str, float]:
    """Full metric bundle for one split of one model."""
    scored = df[[true_var, pred_var]].copy()
    out: Dict[str, float] = {
        "n_rows": float(len(scored)),
        "adj_gini": calc_adj_gini(scored, true_var, pred_var),
        "accuracy": calc_accuracy(scored, true_var, pred_var, k=accuracy_bins),
        "actual_mean": float(scored[true_var].mean()) if len(scored) else float("nan"),
        "pred_mean": float(scored[pred_var].mean()) if len(scored) else float("nan"),
    }
    for percent in capture_percents:
        out[f"capture_rate_{percent:g}"] = capture_rate(scored, true_var, pred_var, percent)
    return out


def gini_gain(challenger: float, champion: float) -> Dict[str, float]:
    """Absolute and relative lift of a challenger Gini over the champion's."""
    if champion is None or np.isnan(champion) or np.isnan(challenger):
        return {"gini_gain": float("nan"), "gini_gain_pct": float("nan")}
    gain = challenger - champion
    pct = gain / abs(champion) if champion != 0 else float("nan")
    return {"gini_gain": float(gain), "gini_gain_pct": float(pct)}
