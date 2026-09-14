import numpy as np
import pandas as pd
import pytest

from validation.config import Config
from validation.data import Dataset
from validation.stages.feature_selection import (
    mutual_info_codes,
    pairwise_complete_corr,
    quantile_codes,
    rank_columns,
    run_feature_selection,
)


@pytest.fixture
def frame():
    rng = np.random.default_rng(3)
    n = 4000
    old = pd.DataFrame({f"old_{i}": rng.normal(size=n) for i in range(3)})
    old["old_1"] = 0.8 * old["old_0"] + 0.4 * rng.normal(size=n)
    signal = rng.normal(size=n)
    df = old.assign(
        new_signal=signal,
        new_dup=old["old_0"] + rng.normal(scale=0.02, size=n),
        new_noise=rng.normal(size=n),
        y=0.7 * old["old_0"] + 0.9 * signal + rng.normal(scale=0.5, size=n),
    )
    df.loc[rng.random(n) < 0.1, "new_signal"] = np.nan
    return df


def test_pairwise_corr_matches_pandas_spearman(frame):
    cols = [c for c in frame.columns if c != "y"]
    values = frame[cols].to_numpy()
    mine = pairwise_complete_corr(rank_columns(values), rank_columns(values))
    reference = frame[cols].corr(method="spearman").to_numpy()
    assert np.nanmax(np.abs(mine - reference)) < 1e-3


def test_corr_handles_all_nan_column():
    x = np.column_stack([np.arange(10.0), np.full(10, np.nan)])
    corr = pairwise_complete_corr(x, x)
    assert corr[0, 0] == pytest.approx(1.0)
    assert np.isnan(corr[1, 1])


def test_quantile_codes_gives_missing_its_own_bucket():
    values = np.array([[1.0], [2.0], [3.0], [np.nan]])
    codes, sizes = quantile_codes(values, bins=3)
    assert codes[3, 0] == sizes[0] - 1
    assert len(set(codes[:3, 0].tolist())) > 1


def test_normalized_mi_bounds(frame):
    values = frame[["old_0", "old_1", "old_2"]].to_numpy()
    codes, sizes = quantile_codes(values, 20)
    identical = mutual_info_codes(codes[:, 0], codes[:, 0], sizes[0], sizes[0])
    correlated = mutual_info_codes(codes[:, 0], codes[:, 1], sizes[0], sizes[1])
    independent = mutual_info_codes(codes[:, 0], codes[:, 2], sizes[0], sizes[2])
    assert identical == pytest.approx(1.0)
    assert independent < correlated < identical
    assert independent < 0.05


def _dataset(frame):
    return Dataset(
        frames={"train": frame},
        target="y",
        base_features=["old_0", "old_1", "old_2"],
        new_features=["new_signal", "new_dup", "new_noise"],
    )


def test_screens_drop_redundant_and_weak_features(frame):
    cfg = Config.from_dict({
        "run": {"n_jobs": 2},
        "feature_selection": {
            "chunk_size": 2,
            "spearman": {"target_min_abs": 0.05, "redundancy_max_abs": 0.9},
            "mutual_info": {"redundancy_max": 0.85},
        },
    })
    result = run_feature_selection(_dataset(frame), cfg)
    assert result.selected == ["new_signal"]
    assert "duplicate" in result.dropped["new_dup"] or "redundant" in result.dropped["new_dup"]
    assert "weak" in result.dropped["new_noise"]
    assert set(result.spearman_matrix.index) == {"new_signal", "new_dup", "new_noise"}


def test_disabled_stage_keeps_every_candidate(frame):
    cfg = Config.from_dict({"feature_selection": {"enabled": False}})
    result = run_feature_selection(_dataset(frame), cfg)
    assert result.skipped and result.selected == ["new_signal", "new_dup", "new_noise"]


def test_redundancy_summary_accounts_for_every_candidate(frame):
    """A candidate cut by the signal screen never reaches the redundancy loop,
    but it must still appear, or the table cannot be reconciled with the count."""
    cfg = Config.from_dict({
        "run": {"n_jobs": 2},
        "feature_selection": {
            "chunk_size": 2,
            "spearman": {"target_min_abs": 0.05, "redundancy_max_abs": 0.9},
        },
    })
    result = run_feature_selection(_dataset(frame), cfg)
    summary = result.redundancy_summary.set_index("feature")

    n_candidates = len(result.selected) + len(result.dropped)
    assert len(summary) == n_candidates == 3

    # `reached_screen` is how far a candidate got, not what rejected it: a kept
    # candidate also shows "redundancy", meaning it was assessed there and survived.
    assert summary.loc["new_signal", "kept"]
    assert summary.loc["new_signal", "reached_screen"] == "redundancy"

    # signal rejects are present, marked, and carry no redundancy statistics
    assert summary.loc["new_noise", "reached_screen"] == "signal"
    assert summary.loc["new_noise", "kept"] is False or not summary.loc["new_noise", "kept"]
    assert pd.isna(summary.loc["new_noise", "max_abs_spearman"])
    assert "weak" in summary.loc["new_noise", "reason"]

    # candidates that reached the redundancy screen do carry them
    assert summary.loc["new_dup", "reached_screen"] == "redundancy"
    assert summary.loc["new_dup", "max_abs_spearman"] > 0.9
    assert summary.loc["new_signal", "reached_screen"] == "redundancy"


def test_correlation_kernel_is_immune_to_infinities():
    """+/-inf must be excluded pairwise like NaN, not turned into 1.8e308 - squaring
    that overflows and poisons the matmul with NaN."""
    import warnings

    rng = np.random.default_rng(0)
    n = 500
    column = rng.normal(size=n)
    column[:5] = np.inf
    column[5:10] = -np.inf
    column[10:15] = np.nan
    values = np.column_stack([rng.normal(size=n), column])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        corr = pairwise_complete_corr(values, values)

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert np.isfinite(corr).all()
    assert corr[1, 1] == pytest.approx(1.0)          # a column still correlates with itself

    # the answer matches simply dropping the non-finite rows first
    finite = np.isfinite(values).all(axis=1)
    expected = np.corrcoef(values[finite].T)[0, 1]
    assert corr[0, 1] == pytest.approx(expected, abs=1e-9)
