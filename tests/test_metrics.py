import numpy as np
import pandas as pd
import pytest

from validation.metrics import (
    calc_accuracy,
    calc_adj_gini,
    capture_rate,
    evaluate_predictions,
    gini_gain,
)


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    actual = rng.gamma(2.0, 1.0, size=4000)
    return pd.DataFrame({"actual": actual, "pred": actual + rng.normal(scale=0.8, size=4000)})


def test_adj_gini_is_one_for_perfect_ranking(frame):
    perfect = frame.assign(pred=frame["actual"])
    assert calc_adj_gini(perfect, "actual", "pred") == pytest.approx(1.0, abs=1e-9)


def test_adj_gini_is_minus_one_for_inverted_ranking(frame):
    inverted = frame.assign(pred=-frame["actual"])
    assert calc_adj_gini(inverted, "actual", "pred") == pytest.approx(-1.0, abs=1e-6)


def test_adj_gini_between_bounds_and_beats_noise(frame):
    signal = calc_adj_gini(frame, "actual", "pred")
    noise = calc_adj_gini(frame.assign(pred=np.arange(len(frame))), "actual", "pred")
    assert 0 < signal <= 1
    assert signal > noise


def test_capture_rate_bounds(frame):
    top5 = capture_rate(frame, "actual", "pred", 0.05)
    assert 0.05 < top5 < 1.0
    assert capture_rate(frame, "actual", "pred", 1.0) == pytest.approx(1.0)


def test_accuracy_perfect_calibration_is_one(frame):
    perfect = frame.assign(pred=frame["actual"])
    assert calc_accuracy(perfect, "actual", "pred") == pytest.approx(1.0)


def test_missing_sentinel_and_nulls_excluded():
    df = pd.DataFrame({"actual": [1.0, 2.0, -9999, np.nan], "pred": [1.0, 2.0, 5.0, 9.0]})
    assert calc_accuracy(df, "actual", "pred", k=2) == pytest.approx(1.0)


def test_degenerate_inputs_return_nan():
    empty = pd.DataFrame({"actual": [], "pred": []})
    assert np.isnan(calc_adj_gini(empty, "actual", "pred"))
    assert np.isnan(calc_accuracy(empty, "actual", "pred"))
    assert np.isnan(capture_rate(empty, "actual", "pred"))
    zeros = pd.DataFrame({"actual": [0.0, 0.0], "pred": [1.0, 2.0]})
    assert np.isnan(calc_adj_gini(zeros, "actual", "pred"))


def test_evaluate_predictions_bundle(frame):
    scores = evaluate_predictions(frame, "actual", "pred", capture_percents=[0.1])
    assert {"adj_gini", "accuracy", "capture_rate_0.1", "n_rows"} <= set(scores)
    assert scores["n_rows"] == len(frame)


def test_gini_gain():
    assert gini_gain(0.6, 0.5)["gini_gain"] == pytest.approx(0.1)
    assert gini_gain(0.6, 0.5)["gini_gain_pct"] == pytest.approx(0.2)
    assert np.isnan(gini_gain(0.6, float("nan"))["gini_gain"])


def test_accuracy_responds_to_prediction_level_not_ranking(frame):
    """calc_accuracy measures calibration: rank-preserving rescaling changes it."""
    doubled = frame.assign(pred=frame["pred"] * 2)
    assert calc_adj_gini(doubled, "actual", "pred") == pytest.approx(
        calc_adj_gini(frame, "actual", "pred"))
    assert calc_accuracy(doubled, "actual", "pred") != pytest.approx(
        calc_accuracy(frame, "actual", "pred"))
