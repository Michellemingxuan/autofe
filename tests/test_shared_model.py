"""fit_booster is the one estimator both halves use."""
import numpy as np
import pytest

from validation.model import FitResult, fit_booster

PARAMS = {"eta": 0.1, "max_depth": 3, "objective": "binary:logistic", "eval_metric": "auc"}


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    def make(n):
        X = rng.normal(size=(n, 4))
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
        return X, y
    Xtr, ytr = make(400)
    Xva, yva = make(200)
    Xte, yte = make(200)
    return (
        {"train": Xtr, "valid": Xva, "test": Xte},
        {"train": ytr, "valid": yva, "test": yte},
        ["f0", "f1", "f2", "f3"],
    )


def test_predicts_every_split_it_is_given(data):
    matrices, targets, names = data
    out = fit_booster(matrices, targets, names, PARAMS, num_boost_round=20)
    assert isinstance(out, FitResult)
    assert set(out.predictions) == {"train", "valid", "test"}
    for split, pred in out.predictions.items():
        assert pred.shape == (len(targets[split]),)
        assert ((pred >= 0) & (pred <= 1)).all()


def test_a_train_only_call_is_enough(data):
    matrices, targets, names = data
    out = fit_booster({"train": matrices["train"]}, {"train": targets["train"]},
                      names, PARAMS, num_boost_round=10)
    assert set(out.predictions) == {"train"}
    # xgboost 1.7.6 reports best_iteration even with no eval set (it is simply the
    # last round), and best_score stays None because nothing was being watched.
    assert out.best_iteration == 9
    assert out.best_score is None


def test_missing_train_split_is_an_error(data):
    matrices, targets, names = data
    with pytest.raises(KeyError, match="train"):
        fit_booster({"valid": matrices["valid"]}, targets, names, PARAMS, num_boost_round=5)


def test_early_stopping_uses_the_best_iteration(data):
    matrices, targets, names = data
    out = fit_booster(matrices, targets, names, PARAMS,
                      num_boost_round=500, early_stopping_rounds=5)
    assert out.best_iteration is not None
    assert out.best_iteration < 499        # it stopped early
    assert out.best_score is not None


def test_nthread_is_reported_out_of_params_not_into_them(data):
    matrices, targets, names = data
    out = fit_booster(matrices, targets, names, PARAMS, num_boost_round=10, nthread=2)
    assert "nthread" not in out.params     # a resource knob, not a model setting
    assert out.params["max_depth"] == 3


def test_identical_inputs_give_identical_predictions(data):
    matrices, targets, names = data
    a = fit_booster(matrices, targets, names, PARAMS, num_boost_round=25)
    b = fit_booster(matrices, targets, names, PARAMS, num_boost_round=25)
    np.testing.assert_array_equal(a.predictions["test"], b.predictions["test"])


def test_an_added_column_changes_the_fit(data):
    """The screen's whole premise: a column carrying signal moves predictions."""
    matrices, targets, names = data
    base = fit_booster(matrices, targets, names, PARAMS, num_boost_round=30)

    rng = np.random.default_rng(1)
    plus = {
        split: np.column_stack([m, targets[split] + rng.normal(scale=0.3, size=len(m))])
        for split, m in matrices.items()
    }
    with_signal = fit_booster(plus, targets, [*names, "leaky"], PARAMS, num_boost_round=30)

    from validation.metrics import calc_adj_gini
    import pandas as pd
    def gini(pred, split):
        return calc_adj_gini(pd.DataFrame({"y": targets[split], "p": pred}), "y", "p")

    assert gini(with_signal.predictions["test"], "test") > gini(base.predictions["test"], "test")
