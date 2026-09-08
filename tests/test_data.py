import numpy as np
import pandas as pd
import pytest

from mllite.config import Config
from mllite.data import assign_splits, build_dataset, clean_missing, resolve_features


@pytest.fixture
def rare_event_frame():
    rng = np.random.default_rng(0)
    n = 5000
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": (rng.random(n) < 0.03).astype(int),
        "period": rng.choice(["2023-01", "2023-02"], size=n),
    })


def _data_cfg(**split):
    payload = {"data": {"target": "y", "split": {"mode": "random", **split}}}
    return Config.from_dict(payload).data


def test_stratified_split_preserves_rare_event_rate(rare_event_frame):
    frames = assign_splits(rare_event_frame, _data_cfg(stratify=True, valid_size=0.2, test_size=0.2), seed=1)
    overall = rare_event_frame["y"].mean()
    assert sum(len(f) for f in frames.values()) == len(rare_event_frame)
    for name, frame in frames.items():
        assert frame["y"].mean() == pytest.approx(overall, abs=0.003), name
    assert len(frames["test"]) == pytest.approx(0.2 * len(rare_event_frame), abs=2)


def test_splits_are_disjoint(rare_event_frame):
    frames = assign_splits(rare_event_frame, _data_cfg(stratify=True), seed=1)
    indices = [set(f.index) for f in frames.values()]
    assert not indices[0] & indices[1] and not indices[0] & indices[2] and not indices[1] & indices[2]


def test_time_split_is_ordered():
    frame = pd.DataFrame({"t": np.arange(1000), "y": np.zeros(1000)})
    cfg = Config.from_dict({"data": {"target": "y", "split": {
        "mode": "time", "time_col": "t", "valid_size": 0.2, "test_size": 0.2}}}).data
    frames = assign_splits(frame, cfg, seed=0)
    assert frames["train"]["t"].max() < frames["valid"]["t"].min()
    assert frames["valid"]["t"].max() < frames["test"]["t"].min()


def test_column_split_rejects_missing_train_rows():
    frame = pd.DataFrame({"y": [0, 1], "split": ["valid", "test"]})
    cfg = Config.from_dict({"data": {"target": "y", "split": {"mode": "column", "column": "split"}}}).data
    with pytest.raises(ValueError, match="no train rows"):
        assign_splits(frame, cfg, seed=0)


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


def test_build_dataset_end_to_end(tmp_path, rare_event_frame):
    path = tmp_path / "frame.parquet"
    rare_event_frame.to_parquet(path, index=False)
    cfg = Config.from_dict({
        "data": {"path": str(path), "target": "y", "id_cols": ["period"],
                 "split": {"mode": "random", "stratify": True}},
        "features": {"new": ["x"]},
    })
    dataset = build_dataset(cfg)
    assert dataset.new_features == ["x"] and dataset.all_features == ["x"]
    assert set(dataset.available_splits()) == {"train", "valid", "test"}
