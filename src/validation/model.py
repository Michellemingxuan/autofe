"""Train one booster and return its predictions. Shared by both halves.

Discovery's screen and validation's variant builds ask the same question of the
same data - does this column change what the model can do - so they must ask it
of the *same* estimator. If the screen fit something other than what the verdict
fits, the feedback a proposer learns from would describe a model that is not the
one deciding, and the loop would optimise for the wrong thing.

So this is the single place that turns (matrices, params) into a fitted booster
and predictions. Everything above it differs:

    stages/modeling.py   one call per variant, plus importance, tuning, saving
    discovery/screen.py  one call on a small sample, scored for feedback

Both pass the same ``cfg.model.params``, so a screen score and a verdict score
are comparable quantities rather than coincidentally similar ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

__all__ = ["FitResult", "fit_booster"]


@dataclass
class FitResult:
    """A fitted booster and what it predicted on each split it was given."""

    booster: Any = None
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    best_iteration: Optional[int] = None
    best_score: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)


def fit_booster(
    matrices: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    feature_names: Sequence[str],
    params: Mapping[str, Any],
    *,
    num_boost_round: int,
    early_stopping_rounds: Optional[int] = None,
    weights: Optional[Mapping[str, Optional[np.ndarray]]] = None,
    nthread: int = 1,
    verbose_eval: int = 0,
) -> FitResult:
    """
    Train on ``matrices["train"]`` and predict every split supplied.

    ``nthread`` is passed explicitly rather than left to XGBoost's default,
    because the default is *all* cores: under an outer fan-out that multiplies
    instead of divides, and the machine thrashes. Callers get their share from
    ``validation.parallel.threads_per_worker``.

    Early stopping applies only when a "valid" split is present; predictions then
    use the best iteration rather than the last, so a variant that overfits late
    is not judged on its overfitted rounds.
    """
    import xgboost as xgb

    if "train" not in matrices:
        raise KeyError("fit_booster needs a 'train' entry in `matrices`")

    weights = weights or {}
    run_params = {**dict(params), "nthread": nthread}

    dmatrices = {
        split: xgb.DMatrix(
            np.asarray(matrix, dtype=np.float32),
            label=targets[split],
            weight=weights.get(split),
            feature_names=list(feature_names),
            missing=np.nan,
            nthread=nthread,
        )
        for split, matrix in matrices.items()
    }

    watchlist = [(dmatrices["train"], "train")]
    stopping = None
    if "valid" in dmatrices:
        watchlist.append((dmatrices["valid"], "valid"))
        stopping = early_stopping_rounds

    evals_result: Dict[str, Dict[str, list]] = {}
    booster = xgb.train(
        run_params,
        dmatrices["train"],
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=stopping,
        evals_result=evals_result,
        verbose_eval=verbose_eval or False,
    )

    best_iteration = getattr(booster, "best_iteration", None)
    best_score = getattr(booster, "best_score", None)

    predict_kwargs: Dict[str, Any] = {}
    if best_iteration is not None:
        predict_kwargs["iteration_range"] = (0, int(best_iteration) + 1)

    return FitResult(
        booster=booster,
        predictions={
            split: booster.predict(dmatrix, **predict_kwargs)
            for split, dmatrix in dmatrices.items()
        },
        best_iteration=int(best_iteration) if best_iteration is not None else None,
        best_score=float(best_score) if best_score is not None else None,
        params={k: v for k, v in run_params.items() if k != "nthread"},
    )
