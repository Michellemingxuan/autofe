"""Missing-value indicators: a 0/1 column recording where a feature was missing."""

import numpy as np
import pandas as pd
import pytest

from validation.config import Config
from preprocessing import stratified_split
from validation.data import add_missing_indicators, prepare_dataset_from_frames


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({
        "old_a": rng.normal(size=n),
        "old_b": rng.normal(size=n),
        "new_sparse": rng.normal(size=n),
        "new_dense": rng.normal(size=n),
        "new_never": rng.normal(size=n),
        "y": (rng.random(n) < 0.2).astype(int),
    })
    df.loc[rng.random(n) < 0.25, "new_sparse"] = np.nan     # 25% missing
    df.loc[rng.random(n) < 0.30, "old_a"] = -9999           # sentinel -> NaN
    df.loc[::300, "new_dense"] = np.nan                     # ~0.3%, below threshold
    return df


def _config(**indicators):
    return Config.from_dict({
        "data": {"target": "y", "missing_values": [-9999],
                 "missing_indicators": {"enabled": True, **indicators}},
        "features": {"base_prefix": "old_", "new_prefix": "new_"},
    })


def _indicators(dataset):
    return [c for c in dataset.all_features if c.endswith("_is_missing")]


def _dataset(frame, cfg):
    """The Dataset built from a prepare step's split of ``frame``."""
    return prepare_dataset_from_frames(stratified_split(frame, "y"), cfg)


def test_disabled_by_default(frame):
    cfg = Config.from_dict({"data": {"target": "y"},
                            "features": {"base_prefix": "old_", "new_prefix": "new_"}})
    assert cfg.data.missing_indicators.enabled is False
    assert _indicators(_dataset(frame, cfg)) == []


def test_candidates_scope_covers_only_new_features(frame):
    dataset = _dataset(frame, _config(scope="candidates"))
    assert _indicators(dataset) == ["new_sparse_is_missing"]


def test_all_scope_covers_incumbents_too(frame):
    dataset = _dataset(frame, _config(scope="all"))
    assert set(_indicators(dataset)) == {"old_a_is_missing", "new_sparse_is_missing"}


def test_indicators_are_candidates_by_default(frame):
    """The incumbent model did not carry them, so they are new features."""
    dataset = _dataset(frame, _config(scope="all"))
    assert "old_a_is_missing" in dataset.new_features
    assert "old_a_is_missing" not in dataset.base_features


def test_treat_as_source_follows_the_originating_column(frame):
    dataset = _dataset(frame, _config(scope="all", treat_as="source"))
    assert "old_a_is_missing" in dataset.base_features
    assert "new_sparse_is_missing" in dataset.new_features


def test_columns_that_carry_no_information_are_skipped(frame):
    """Never missing or always missing both give a constant indicator."""
    dataset = _dataset(frame, _config(scope="all", min_missing_rate=0.0))
    made = _indicators(dataset)
    assert "new_never_is_missing" not in made      # never missing
    assert "old_b_is_missing" not in made

    always = frame.assign(new_gone=np.nan)
    dataset = _dataset(always, _config(scope="candidates", min_missing_rate=0.0))
    assert "new_gone_is_missing" not in _indicators(dataset)


def test_min_missing_rate_is_respected(frame):
    assert "new_dense_is_missing" not in _indicators(_dataset(frame, _config()))
    loose = _dataset(frame, _config(min_missing_rate=0.0005))
    assert "new_dense_is_missing" in _indicators(loose)


def test_indicator_matches_isna_in_every_split(frame):
    dataset = _dataset(frame, _config(scope="all"))
    for split in dataset.available_splits():
        data = dataset.split(split)
        for source in ("old_a", "new_sparse"):
            expected = data[source].isna().astype("int8")
            pd.testing.assert_series_equal(
                data[f"{source}_is_missing"], expected, check_names=False)


def test_indicators_capture_sentinels_not_just_nulls(frame):
    """old_a's missingness comes from -9999, so the indicator only works if it is
    derived after cleaning."""
    dataset = _dataset(frame, _config(scope="all"))
    train = dataset.split("train")
    assert train["old_a_is_missing"].mean() > 0.2
    assert not (train["old_a"] == -9999).any()


def test_suffix_is_configurable(frame):
    dataset = _dataset(frame, _config(suffix="__nan"))
    assert "new_sparse__nan" in dataset.new_features


def test_a_name_collision_is_reported_and_skipped(frame, caplog):
    frame = frame.assign(new_sparse_is_missing=0)
    with caplog.at_level("WARNING"):
        dataset = _dataset(frame, _config(scope="candidates"))
    assert "already exists" in caplog.text
    # the pre-existing column is left as it was, not overwritten
    assert dataset.split("train")["new_sparse_is_missing"].nunique() == 1


def test_eligibility_is_decided_on_train_and_applied_everywhere(frame):
    frames = {"train": frame.iloc[:2000], "valid": frame.iloc[2000:2500],
              "test": frame.iloc[2500:]}
    cfg = _config(scope="candidates")
    prepared, base, new = add_missing_indicators(
        {k: v.copy() for k, v in frames.items()}, ["old_a", "old_b"],
        ["new_sparse", "new_dense", "new_never"], cfg)

    assert "new_sparse_is_missing" in new
    for split in prepared.values():
        assert "new_sparse_is_missing" in split.columns
