"""Reject generated features whose *values* would break a model.

The sandbox decides whether a block is safe to run. These checks decide whether
what it produced is safe to fit on, which is a different question and the one
that actually bites.

Both guards exist because of failures observed in practice:

* Non-finite values. A generated ratio produces NaN or inf on some rows, a
  linear model raises at fit time, and the round dies with a stack trace instead
  of feedback the model can act on.

* Finite but enormous values. This is the subtle one. A model asked to guard a
  division writes ``a / (b + 1e-6)``, which does not prevent a blow-up - it
  converts one zero denominator into a value of order 1e6. That is finite, so it
  passes every NaN/inf check, and then an unscaled linear fit cannot converge,
  every coefficient collapses toward zero, and the model predicts a constant.
  Worse, the damage is invisible if the offending row sits in a split used for
  refitting rather than the one used for scoring: validation looks healthy while
  the deployed model is rubbish.

Both are reported with guidance aimed at the model, because the message is fed
back into the next round's prompt.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from discovery.sandbox import CandidateError

__all__ = [
    "check_finite",
    "check_scale",
    "check_matrix_finite",
    "check_redundancy",
    "DEFAULT_SPIKE_FACTOR",
]

# A lone value this many times its own 99th percentile is a spike, not a tail.
# Ordinary heavy-tailed features run 10-100x; the epsilon-guard failure above
# runs into the hundreds of thousands.
DEFAULT_SPIKE_FACTOR = 1000.0


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def check_finite(frame: pd.DataFrame, column: str, split_name: str) -> None:
    """Require the generated column to be numeric and finite on this split."""
    if column not in frame.columns:
        raise CandidateError(
            f"Generated feature '{column}' is missing from the {split_name} data."
        )

    values = _numeric(frame, column)
    bad = ~np.isfinite(values)
    if not bad.any():
        return

    positions = np.flatnonzero(bad)
    examples = [
        {"row": int(pos), "value": repr(frame.iloc[pos][column])}
        for pos in positions[:5]
    ]
    raise CandidateError(
        f"Generated feature '{column}' has {int(bad.sum())} non-finite or "
        f"non-numeric values on the {split_name} data. Examples: {examples}. "
        "Handle missing values, division by zero, invalid logarithms and "
        "infinities explicitly."
    )


def check_scale(
    frames: Mapping[str, pd.DataFrame],
    column: str,
    spike_factor: float = DEFAULT_SPIKE_FACTOR,
) -> None:
    """
    Reject a column whose largest magnitude spikes far above its own bulk.

    Compared against the 99th percentile rather than a fixed bound, so the test
    adapts to whatever scale the feature naturally lives on and only fires on a
    genuine outlier.
    """
    for split_name, frame in frames.items():
        if column not in frame.columns:
            continue
        values = _numeric(frame, column)
        magnitudes = np.abs(values[np.isfinite(values)])
        if magnitudes.size == 0:
            continue

        # A binary or few-valued column cannot spike: its maximum IS its
        # definition. Without this, a rare indicator - 4 ones in 6819 rows - has
        # a 99th percentile of 0, so max/p99 reads as 1e12 and every sparse flag
        # is rejected. That penalised rule-based strategies systematically.
        if np.unique(magnitudes).size <= 2:
            continue

        largest = float(magnitudes.max())
        p99 = float(np.percentile(magnitudes, 99))
        reference = max(p99, 1e-12)
        if largest <= spike_factor * reference:
            continue

        worst = int(np.nanargmax(np.abs(np.nan_to_num(values))))
        raise CandidateError(
            f"Generated feature '{column}' spikes on the {split_name} data: "
            f"largest magnitude {largest:.6g} is {largest / reference:.0f}x its "
            f"99th percentile ({p99:.6g}), at row {worst}. This is the signature "
            "of a division guarded with a tiny epsilon: adding 1e-6 to a zero "
            "denominator does not prevent a blow-up, it produces a value of "
            "order 1e6, which destroys the fit. Keep the result on a scale "
            "comparable to its inputs - mask the invalid rows, use a denominator "
            "that cannot approach zero, or clip the result."
        )


def check_redundancy(
    values: pd.Series,
    base_ranks: pd.DataFrame,
    column: str,
    max_abs_rho: float,
) -> None:
    """
    Reject a column that is a near-copy of one the table already has.

    This is the rejection that a delta cannot see. A proposal can raise the score
    and still be worthless: on a table of 95 financial ratios, a plausible-sounding
    "debt coverage ratio" came back 0.996 correlated with an existing column, so
    its gain was information the model already had under another name.

    Measured with Spearman against pre-ranked base columns, and thresholded with
    the SAME number the feature-selection gate will use - a screen that passed
    what the gate then rejects is worse than no screen, because the expensive
    stages run for nothing and the proposer learns nothing from it.

    ``base_ranks`` is the rank-transformed base matrix, computed once per run:
    ranking 95 columns per candidate would dominate a screen meant to be cheap.
    """
    ranked = values.rank()
    if ranked.nunique() <= 1:
        return      # a constant column correlates with nothing; min_unique catches it

    correlations = base_ranks.corrwith(ranked).abs().dropna()
    if correlations.empty:
        return

    worst = correlations.idxmax()
    rho = float(correlations.max())
    if rho > max_abs_rho:
        raise CandidateError(
            f"Generated feature '{column}' is redundant: |rho|={rho:.3f} against "
            f"the existing column '{worst}', above the {max_abs_rho:.2f} limit. "
            "It re-derives information the table already holds, so any gain it "
            "shows is not new. Combine columns that are not already near-copies "
            "of each other, or express a relationship none of the existing "
            "columns captures."
        )


def check_matrix_finite(frame: pd.DataFrame, split_name: str) -> None:
    """
    Last guard before fitting: the whole matrix is numeric and finite.

    In normal operation check_finite catches the generated column first; this
    exists so an unexpected interaction surfaces as a clear message rather than
    a library error from deep inside a solver.
    """
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    array = numeric.to_numpy(dtype=float)
    bad = ~np.isfinite(array)
    if not bad.any():
        return

    rows, cols = np.where(bad)
    affected = sorted({str(numeric.columns[j]) for j in cols})
    examples = [
        {"row": int(i), "column": str(frame.columns[j]), "value": repr(frame.iloc[i, j])}
        for i, j in zip(rows[:5], cols[:5])
    ]
    raise CandidateError(
        f"The {split_name} matrix has non-finite or non-numeric values. "
        f"Affected columns: {affected}. Examples: {examples}."
    )
