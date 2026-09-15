import numpy as np
import pandas as pd
import pytest

from validation.config import Config
from validation.data import build_dataset, clean_missing, resolve_features


@pytest.fixture
def rare_event_frame():
    rng = np.random.default_rng(0)
    n = 5000
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": (rng.random(n) < 0.03).astype(int),
        "period": rng.choice(["2023-01", "2023-02"], size=n),
    })


def test_sentinels_become_nan():
    frame = pd.DataFrame({"a": [1.0, -9999.0], "b": ["x", "y"]})
    cleaned = clean_missing(frame, ["a", "b"], [-9999])
    assert cleaned["a"].isna().sum() == 1
    assert cleaned["b"].tolist() == ["x", "y"]   # non-numeric left alone


def test_resolve_features_excludes_reserved_columns(rare_event_frame):
    cfg = Config.from_dict({
        "data": {"target": "y", "id_cols": ["period"]},
        "features": {"new": ["x"]},
    })
    base, new = resolve_features(rare_event_frame, cfg.features, cfg.data)
    assert new == ["x"] and base == []   # y/period reserved, x claimed as new


def test_resolve_features_reports_missing_columns(rare_event_frame):
    cfg = Config.from_dict({"data": {"target": "y"}, "features": {"new": ["nope"]}})
    with pytest.raises(KeyError, match="nope"):
        resolve_features(rare_event_frame, cfg.features, cfg.data)


def test_build_dataset_reads_the_three_tables(tmp_path, rare_event_frame):
    parts = {"train": rare_event_frame.iloc[:3000], "valid": rare_event_frame.iloc[3000:4000],
             "test": rare_event_frame.iloc[4000:]}
    paths = {}
    for name, part in parts.items():
        paths[name] = str(tmp_path / f"{name}.csv")
        part.to_csv(paths[name], index=False)

    cfg = Config.from_dict({
        "data": {"paths": paths, "target": "y", "id_cols": ["period"]},
        "features": {"new": ["x"]},
    })
    dataset = build_dataset(cfg)
    assert dataset.new_features == ["x"] and dataset.all_features == ["x"]
    # taken exactly as written - nothing is re-split
    assert {s: len(dataset.split(s)) for s in dataset.available_splits()} == \
        {"train": 3000, "valid": 1000, "test": 1000}


def test_build_dataset_needs_data_paths():
    with pytest.raises(ValueError, match="data.paths is empty"):
        build_dataset(Config.from_dict({"data": {"target": "y"}}))


@pytest.mark.parametrize("key, value", [("path", "t.csv"), ("split", {"mode": "random"}),
                                        ("nrows", 100)])
def test_run_time_split_keys_are_rejected(key, value):
    """A config written for run-time splitting fails loudly instead of being ignored."""
    with pytest.raises(ValueError, match="unknown config key"):
        Config.from_dict({"data": {key: value}})


def test_infinities_become_nan(caplog):
    """Ratio features with a zero denominator arrive as +/-inf; they are not
    valid model inputs and corrupt binning, PSI and the correlation kernel."""
    frame = pd.DataFrame({
        "ratio": [1.0, np.inf, -np.inf, 2.0, -9999.0],
        "clean": [1.0, 2.0, 3.0, 4.0, 5.0],
        "label": ["a", "b", "c", "d", "e"],
    })
    with caplog.at_level("WARNING"):
        cleaned = clean_missing(frame, ["ratio", "clean", "label"], [-9999])

    assert cleaned["ratio"].isna().sum() == 3          # two infinities + one sentinel
    assert np.isfinite(cleaned["ratio"].dropna()).all()
    assert cleaned["clean"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cleaned["label"].tolist() == list("abcde")  # non-numeric untouched
    assert "infinite value" in caplog.text and "ratio" in caplog.text


def test_clean_missing_leaves_finite_data_alone(caplog):
    frame = pd.DataFrame({"a": [1.0, 2.0, np.nan]})
    with caplog.at_level("WARNING"):
        cleaned = clean_missing(frame, ["a"], [])
    pd.testing.assert_frame_equal(cleaned, frame)
    assert "infinite" not in caplog.text            # no warning when there is nothing to say
