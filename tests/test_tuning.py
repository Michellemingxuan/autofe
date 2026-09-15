"""Hyperparameter search, and the shared vs per-variant choice."""

import json

import numpy as np
import pandas as pd
import pytest

from validation.config import Config
from validation.stages.modeling import DEFAULT_SEARCH_SPACE, sample_params

xgb = pytest.importorskip("xgboost")

from synthetic import make_frame, split_by_column  # noqa: E402
from validation.pipeline import Pipeline  # noqa: E402


# --------------------------------------------------------------------------- #
# The search space
# --------------------------------------------------------------------------- #
def test_sample_params_honours_each_spec_kind():
    rng = np.random.default_rng(0)
    space = {
        "eta": [0.01, 0.05],                                   # categorical
        "max_depth": {"low": 3, "high": 8, "int": True},       # integer range
        "subsample": {"low": 0.6, "high": 1.0},                # float range
        "reg_lambda": {"low": 0.5, "high": 10.0, "log": True}, # log range
        "objective": "reg:squarederror",                       # fixed
    }
    draws = [sample_params(space, rng) for _ in range(200)]

    assert {d["eta"] for d in draws} <= {0.01, 0.05}
    assert all(isinstance(d["max_depth"], int) and 3 <= d["max_depth"] <= 8 for d in draws)
    assert all(0.6 <= d["subsample"] <= 1.0 for d in draws)
    assert all(0.5 <= d["reg_lambda"] <= 10.0 for d in draws)
    assert all(d["objective"] == "reg:squarederror" for d in draws)
    # a log range should favour the low end
    assert np.median([d["reg_lambda"] for d in draws]) < 5.25


def test_sampling_is_reproducible_from_the_seed():
    a = [sample_params(DEFAULT_SEARCH_SPACE, np.random.default_rng(7)) for _ in range(3)]
    b = [sample_params(DEFAULT_SEARCH_SPACE, np.random.default_rng(7)) for _ in range(3)]
    assert a == b


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def frames():
    return split_by_column(make_frame(n_rows=6_000, seed=11))


def _config(tmp_path, **tuning):
    return Config.from_dict({
        "run": {"name": "tune", "output_dir": str(tmp_path), "n_jobs": 2, "log_level": "ERROR"},
        "data": {"target": "y", "id_cols": ["row_id"]},
        "features": {"base_prefix": "old_", "new_prefix": "new_"},
        "feature_selection": {"chunk_size": 2, "spearman": {"target_min_abs": 0.04}},
        "model": {"num_boost_round": 30, "early_stopping_rounds": 10,
                  "variants": ["base", "base_plus_new"],
                  "params": {"eta": 0.3},
                  "tuning": {"n_trials": 3, **tuning}},
        "analysis": {"shap": {"enabled": False}},
        "verdict": {"enabled": False},
    })


def test_disabled_tuning_uses_the_configured_params(frames, tmp_path):
    result = Pipeline(_config(tmp_path, enabled=False)).run(frames=frames)
    assert all(m.tuned is False for m in result.models)
    assert all(m.params["eta"] == 0.3 for m in result.models)
    assert not (result.output_dir / "tuning_trials.csv").exists()


def test_shared_mode_gives_every_variant_identical_params(frames, tmp_path):
    result = Pipeline(_config(tmp_path, enabled=True, mode="shared",
                              tune_on="base_plus_new")).run(frames=frames)

    configs = {json.dumps(m.params, sort_keys=True, default=str) for m in result.models}
    assert len(configs) == 1, "shared mode must not leave variants on different params"
    assert all(m.tuned for m in result.models)

    trials = pd.read_csv(result.output_dir / "tuning_trials.csv")
    assert set(trials["variant"]) == {"base_plus_new"}   # only the named variant is searched
    assert len(trials) == 3


def test_per_variant_mode_searches_every_variant(frames, tmp_path):
    result = Pipeline(_config(tmp_path, enabled=True, mode="per_variant")).run(frames=frames)

    trials = pd.read_csv(result.output_dir / "tuning_trials.csv")
    assert set(trials["variant"]) == {"base", "base_plus_new"}
    assert len(trials) == 3 * 2                          # n_trials per variant

    # each variant took the best of its own trials
    for name, group in trials.groupby("variant"):
        best = json.loads(group.loc[group["score"].idxmax(), "params"])
        used = next(m.params for m in result.models if m.name == name)
        assert all(used[k] == pytest.approx(v) if isinstance(v, float) else used[k] == v
                   for k, v in best.items())


def test_the_winning_params_are_the_ones_actually_trained(frames, tmp_path):
    result = Pipeline(_config(tmp_path, enabled=True, mode="shared")).run(frames=frames)
    trials = pd.read_csv(result.output_dir / "tuning_trials.csv")
    best = json.loads(trials.loc[trials["score"].idxmax(), "params"])
    used = result.models[0].params
    for key, value in best.items():
        assert used[key] == pytest.approx(value) if isinstance(value, float) else used[key] == value
    # search-space params override the configured defaults
    assert used["eta"] != 0.3 or best["eta"] == 0.3


def test_report_records_the_mode_and_configurations(frames, tmp_path):
    result = Pipeline(_config(tmp_path, enabled=True, mode="shared")).run(frames=frames)
    report = (result.output_dir / "report.md").read_text()
    assert "## Hyperparameters" in report
    assert "Distinct configurations in use: **1**" in report
    assert "attributable to the feature" in report      # states why shared matters


def test_tuning_needs_a_valid_split(frames, tmp_path):
    cfg = _config(tmp_path, enabled=True)
    with pytest.raises(ValueError, match="valid split"):
        Pipeline(cfg).run(frames={"train": frames["train"], "test": frames["test"]})


def test_metric_cannot_be_set_to_something_unimplemented():
    """Trials are scored with calc_adj_gini; any other value would be ignored."""
    with pytest.raises(ValueError, match="only supports 'adj_gini'"):
        Config.from_dict({"model": {"tuning": {"metric": "auc"}}}).validate()
    Config.from_dict({"model": {"tuning": {"metric": "adj_gini"}}}).validate()


def test_tuning_mode_is_restricted_to_the_two_implemented_modes():
    for mode in ("shared", "per_variant"):
        Config.from_dict({"model": {"tuning": {"mode": mode}}}).validate()
    with pytest.raises(ValueError, match="shared\\|per_variant"):
        Config.from_dict({"model": {"tuning": {"mode": "bayesian"}}}).validate()
