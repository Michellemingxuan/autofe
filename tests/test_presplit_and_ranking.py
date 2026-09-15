"""Pre-split inputs, the data quality switch, and mutual-information ranking."""

import numpy as np
import pandas as pd
import pytest

from validation.config import Config
from validation.data import prepare_dataset_from_frames
from validation.stages.data_quality import run_data_quality
from validation.stages.feature_selection import run_feature_selection

xgb = pytest.importorskip("xgboost")

from synthetic import make_frame, split_by_column  # noqa: E402
from validation.pipeline import Pipeline  # noqa: E402


@pytest.fixture(scope="module")
def frames():
    return split_by_column(make_frame(n_rows=9_000, seed=4))


def _config(tmp_path, **data_overrides):
    payload = {
        "run": {"name": "t", "output_dir": str(tmp_path), "n_jobs": 2, "log_level": "ERROR"},
        "data": {"target": "y", "id_cols": ["row_id"], **data_overrides},
        "features": {"base_prefix": "old_", "new_prefix": "new_"},
        "feature_selection": {"chunk_size": 2, "spearman": {"target_min_abs": 0.04}},
        "model": {"num_boost_round": 40, "early_stopping_rounds": 10,
                  "variants": ["base", "base_plus_new"]},
        "analysis": {"shap": {"enabled": False}},
    }
    return Config.from_dict(payload)


# --------------------------------------------------------------------------- #
# 1. skipping the data quality stage
# --------------------------------------------------------------------------- #
def test_data_quality_enabled_false_skips_the_stage(frames, tmp_path):
    cfg = _config(tmp_path)
    assert cfg.data_quality.enabled is False       # skipping is the default
    dataset = prepare_dataset_from_frames(frames, cfg)

    result = run_data_quality(dataset, cfg)
    assert result.skipped is True
    assert result.report.empty and result.failed == []


def test_data_quality_enabled_true_reports_and_can_drop(frames, tmp_path):
    frames = {name: part.assign(old_constant=1.0) for name, part in frames.items()}
    cfg = _config(tmp_path)
    cfg.data_quality.enabled = True
    cfg.data_quality.drop_failed = True
    dataset = prepare_dataset_from_frames(frames, cfg)

    result = run_data_quality(dataset, cfg)
    assert result.skipped is False
    assert "old_constant" in result.failed          # min_unique catches it
    assert set(result.report.columns) >= {"feature", "missing_rate", "n_unique", "passed"}


# --------------------------------------------------------------------------- #
# 2. inputs arrive already split
# --------------------------------------------------------------------------- #
def test_prepare_dataset_from_frames_takes_the_splits_as_given(frames, tmp_path):
    dataset = prepare_dataset_from_frames(frames, _config(tmp_path))

    for name, given in frames.items():
        assert len(dataset.split(name)) == len(given)
    assert dataset.available_splits() == ["train", "valid", "test"]


def test_files_and_frames_give_the_same_run(frames, tmp_path):
    """Reading data.paths and passing the same frames in memory must agree."""
    paths = {}
    for name, part in frames.items():
        paths[name] = str(tmp_path / f"{name}.csv")
        part.to_csv(paths[name], index=False)

    from_files = Pipeline(_config(tmp_path, paths=paths)).run()
    from_frames = Pipeline(_config(tmp_path)).run(frames=frames)

    assert from_files.dataset.available_splits() == ["train", "valid", "test"]
    assert all(m.error is None for m in from_files.models)
    assert from_frames.feature_selection.selected == from_files.feature_selection.selected
    left = from_files.analysis.comparison.set_index("variant")["adj_gini_test"]
    right = from_frames.analysis.comparison.set_index("variant")["adj_gini_test"]
    pd.testing.assert_series_equal(left, right, atol=1e-9)


def test_paths_requires_train_and_rejects_unknown_keys(tmp_path):
    with pytest.raises(ValueError, match="must include a 'train'"):
        Config.from_dict({"data": {"paths": {"valid": "v.csv"}}}).validate()
    with pytest.raises(ValueError, match="train/valid/test"):
        Config.from_dict({"data": {"paths": {"train": "t.csv", "holdout": "h.csv"}}}).validate()


def test_mismatched_columns_across_splits_are_reported(frames, tmp_path):
    frames = dict(frames)
    frames["test"] = frames["test"].drop(columns="old_3")
    with pytest.raises(KeyError, match="old_3"):
        prepare_dataset_from_frames(frames, _config(tmp_path))


def test_frames_without_train_are_rejected(frames, tmp_path):
    frames = dict(frames)
    del frames["train"]
    with pytest.raises(ValueError, match="train"):
        prepare_dataset_from_frames(frames, _config(tmp_path))


# --------------------------------------------------------------------------- #
# 3. relevance vs. redundancy
# --------------------------------------------------------------------------- #
def test_both_mi_directions_are_reported(frames, tmp_path):
    dataset = prepare_dataset_from_frames(frames, _config(tmp_path))
    result = run_feature_selection(dataset, _config(tmp_path))
    stats = result.target_stats.set_index("feature")

    # relevance (vs. the outcome) and redundancy (vs. the incumbents), side by side
    assert {"mi_target", "nmi_target", "mi_redundancy_base", "mrmr_score"} <= set(stats.columns)
    assert (stats["mi_target"] >= 0).all()
    assert stats["nmi_target"].between(0, 1).all()
    assert stats["mi_redundancy_base"].between(0, 1).all()

    # new_dup_old0 duplicates an incumbent: high relevance AND high redundancy
    assert stats.loc["new_dup_old0", "mi_redundancy_base"] > stats.loc["new_signal_a", "mi_redundancy_base"]
    # new_noise is neither relevant nor redundant
    assert stats.loc["new_noise", "nmi_target"] < stats.loc["new_signal_a", "nmi_target"]


def test_mrmr_score_is_relevance_minus_redundancy(frames, tmp_path):
    cfg = _config(tmp_path)
    cfg.feature_selection.mutual_info.redundancy_stat = "mean"
    result = run_feature_selection(prepare_dataset_from_frames(frames, cfg), cfg)
    stats = result.target_stats

    expected = stats["nmi_target"] - stats["mi_redundancy_base"]
    pd.testing.assert_series_equal(stats["mrmr_score"], expected, check_names=False)

    # Both planted controls rank below the genuine signal, for opposite reasons:
    # new_noise has almost no relevance, new_dup_old0 has high relevance but pays
    # for it in redundancy. Which of the two lands last depends on how big the
    # incumbent pool is, so only their position relative to real signal is stable.
    scores = stats.set_index("feature")["mrmr_score"]
    assert scores["new_noise"] < scores["new_signal_a"]
    assert scores["new_dup_old0"] < scores["new_signal_a"]


def test_mrmr_ranking_still_drops_the_planted_controls(frames, tmp_path):
    cfg = _config(tmp_path)
    cfg.feature_selection.ranking = "mrmr"
    cfg.feature_selection.mutual_info.redundancy_stat = "mean"
    cfg.feature_selection.spearman.redundancy_max_abs = 0.9

    result = run_feature_selection(prepare_dataset_from_frames(frames, cfg), cfg)
    assert "new_dup_old0" in result.dropped     # max-based gate still catches the duplicate
    assert "new_noise" in result.dropped
    assert "new_signal_a" in result.selected


def test_ranking_choice_is_validated():
    with pytest.raises(ValueError, match="ranking"):
        Config.from_dict({"feature_selection": {"ranking": "greedy"}}).validate()
    with pytest.raises(ValueError, match="redundancy_stat"):
        Config.from_dict({"feature_selection": {"mutual_info": {"redundancy_stat": "median"}}}).validate()
