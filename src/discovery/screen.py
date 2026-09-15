"""Score one candidate feature on the screen's rows, fast.

The screen answers a narrow question: does this column run cleanly, and does it
move the model at all? It is not the decision. The decision is made later by
`leave_one_in` over the full splits with tuning, SHAP and the verdict gates.

Two reasons it exists anyway:

* A generation loop needs feedback every round. Waiting minutes for a verdict per
  proposal would make the loop useless, and the proposer only needs a direction.
* Broken proposals should die here, with a message. A block that will not run, or
  produces values that wreck a fit, should never reach the expensive stages - and
  the reason it was rejected is exactly what the next prompt needs to say.

It deliberately uses the same estimator (:func:`validation.model.fit_booster`)
and the same parameters as the verdict, so a screen delta and a verdict delta are
the same kind of quantity. What differs is the data volume and the absence of
tuning, which is the whole point: cheap and directionally honest, not final.

The rows are the discovery config's ``screen_data``: the train and valid splits,
or rows the prepare step sampled from them. The screen fits on the first and
scores on the second - never on the rows it fit, where a boosted model rebuilds a
ratio from its own inputs and every delta reads as noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from discovery.guards import (
    check_finite,
    check_matrix_finite,
    check_redundancy,
    check_scale,
)
from discovery.sandbox import (
    CandidateError,
    apply_code,
    validate_references,
    validate_single_column,
)
from validation.model import fit_booster

__all__ = ["ScreenResult", "Screener"]


@dataclass
class ScreenResult:
    """What the screen learned about one candidate."""

    feature_name: str | None = None
    ok: bool = False
    base_score: float | None = None
    candidate_score: float | None = None
    delta: float | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0
    n_rows: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "ok": self.ok,
            "base_score": self.base_score,
            "candidate_score": self.candidate_score,
            "delta": self.delta,
            "error": self.error,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "n_rows": self.n_rows,
            **self.extras,
        }


class Screener:
    """
    Holds the sample and the baseline, and scores candidates against it.

    The baseline is fit once and reused: it does not depend on the candidate, and
    refitting it per proposal would both waste time and let noise move the
    reference between rounds, so an identical candidate could score differently
    depending on when it arrived.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target: str,
        base_features: Sequence[str],
        params: Mapping[str, Any],
        *,
        score: Callable[[pd.DataFrame, str, str], float],
        num_boost_round: int = 200,
        nthread: int = 1,
        spike_factor: float = 1000.0,
        redundancy_max_abs: float | None = None,
        column_aliases: Mapping[str, str] | None = None,
    ):
        if not len(train) or not len(valid):
            raise ValueError("the screen needs rows to fit on and rows to score on; "
                             f"got {len(train)} and {len(valid)}")
        # One frame, so a proposal's code runs once over every screen row: the
        # rows it fits on first, the rows it is scored on after them.
        self.sample = pd.concat([train, valid], ignore_index=True)
        self._train_rows = np.arange(len(train))
        self._eval_rows = np.arange(len(train), len(self.sample))
        self.target = target
        self.base_features = list(base_features)
        self.params = self._measurable(params)
        self.score = score
        self.num_boost_round = num_boost_round
        self.nthread = nthread
        self.spike_factor = spike_factor
        self.redundancy_max_abs = redundancy_max_abs
        self.column_aliases = dict(column_aliases or {})
        # Ranked once: the redundancy check runs per candidate, and rank-transforming
        # every base column each time would dominate a screen meant to be cheap.
        self._base_ranks = (
            self.sample[self.base_features].rank()
            if redundancy_max_abs is not None else None
        )

        self._y = self.sample[self.target].to_numpy()
        self._base_score = self._fit_and_score(
            self.sample[self.base_features], self.base_features
        )

    @staticmethod
    def _measurable(params: Mapping[str, Any]) -> dict[str, Any]:
        """
        Strip column subsampling, so a delta measures the feature and nothing else.

        With colsample_bytree < 1 the column *count* changes which columns each
        tree may consider, so adding any 96th column shifts the score even when
        it carries no information: a constant column measured -0.0055 here, and
        eight completely different indicators all measured an identical -0.0019.
        The baseline and the candidate must differ only by the feature, so the
        screen fits on all columns. Row subsampling is untouched - it does not
        depend on the column count.
        """
        adjusted = dict(params)
        for key in ("colsample_bytree", "colsample_bylevel", "colsample_bynode"):
            if adjusted.get(key, 1.0) != 1.0:
                adjusted[key] = 1.0
        return adjusted

    @property
    def base_score(self) -> float:
        return self._base_score

    def _fit_and_score(self, matrix: pd.DataFrame, names: Sequence[str]) -> float:
        """Fit on the screen's train rows, score on its held-out rows.

        The holdout is not optional. Scored in-sample, a gradient-boosted fit
        reconstructs a ratio from its own numerator and denominator, so the
        baseline already sits near its ceiling and the delta is noise: measured
        at -0.0132 for a feature worth +0.0143 against held-out rows. Only rows
        the fit did not see make the comparison mean anything.

        The eval split is named "eval" rather than "valid" on purpose - "valid"
        is what fit_booster watches for early stopping, and the screen must not
        tune itself on the rows it is scoring.
        """
        fitted = fit_booster(
            {
                "train": matrix.iloc[self._train_rows].to_numpy(dtype=float),
                "eval": matrix.iloc[self._eval_rows].to_numpy(dtype=float),
            },
            {
                "train": self._y[self._train_rows],
                "eval": self._y[self._eval_rows],
            },
            list(names),
            self.params,
            num_boost_round=self.num_boost_round,
            nthread=self.nthread,
        )
        scored = pd.DataFrame(
            {"y": self._y[self._eval_rows], "p": fitted.predictions["eval"]}
        )
        return float(self.score(scored, "y", "p"))

    def evaluate(self, code: str, reserved_names: Sequence[str] = ()) -> ScreenResult:
        """
        Run one candidate block and score it. Never raises on a bad candidate.

        Every rejection reason is returned as text rather than an exception,
        because the caller's next move is to put it in the following prompt.
        """
        import time

        started = time.perf_counter()
        result = ScreenResult(n_rows=len(self.sample))
        try:
            name = validate_single_column(code, self.sample.columns, reserved_names)
            result.feature_name = name
            validate_references(code, self.base_features, self.column_aliases)

            extended = apply_code(self.sample[self.base_features], [code])
            check_finite(extended, name, "sample")
            check_scale({"sample": extended}, name, self.spike_factor)
            if self._base_ranks is not None:
                check_redundancy(
                    extended[name], self._base_ranks, name, self.redundancy_max_abs
                )
            check_matrix_finite(extended, "sample")

            features = [*self.base_features, name]
            result.base_score = self._base_score
            result.candidate_score = self._fit_and_score(extended[features], features)
            result.delta = result.candidate_score - result.base_score
            result.ok = True

        except CandidateError as error:
            result.error = str(error)
        except Exception as error:  # noqa: BLE001 - any failure is feedback
            result.error = f"{type(error).__name__}: {error}"

        result.elapsed_seconds = time.perf_counter() - started
        return result
