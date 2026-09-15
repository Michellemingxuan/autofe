"""End-to-end checks. Skipped when XGBoost is not installed."""

import numpy as np
import pandas as pd
import pytest

from validation.config import Config
from validation.data import Dataset, build_dataset, prepare_dataset_from_frames
from validation.pipeline import Pipeline
from validation.stages.modeling import build_variants

xgb = pytest.importorskip("xgboost")

from synthetic import make_frame, split_by_column  # noqa: E402


def test_build_variants_covers_each_kind():
    variants = build_variants(["a", "b"], ["n1", "n2"],
                              ["base", "base_plus_new", "new_only", "leave_one_in", "leave_one_out"])
    names = [v.name for v in variants]
    assert names == ["base", "base_plus_new", "new_only",
                     "loi__n1", "loi__n2", "loo__n1", "loo__n2"]
    assert dict(zip(names, (v.features for v in variants)))["loo__n1"] == ["a", "b", "n2"]


def test_build_variants_skips_empty_feature_lists():
    assert [v.name for v in build_variants(["a"], [], ["base", "new_only"])] == ["base"]


def _write(frames, folder) -> dict:
    paths = {}
    for name, part in frames.items():
        paths[name] = str(folder / f"{name}.csv")
        part.to_csv(paths[name], index=False)
    return paths


def _read(paths) -> dict:
    return {name: pd.read_csv(path, float_precision="round_trip") for name, path in paths.items()}


@pytest.fixture(scope="module")
def data_paths(tmp_path_factory):
    frames = split_by_column(make_frame(n_rows=12_000, seed=1))
    return _write(frames, tmp_path_factory.mktemp("data"))


def _config(data_paths, tmp_path, **overrides):
    payload = {
        "run": {"name": "test", "output_dir": str(tmp_path), "n_jobs": 2, "log_level": "WARNING"},
        "data": {"paths": data_paths, "target": "y", "id_cols": ["row_id"]},
        "features": {"base_prefix": "old_", "new_prefix": "new_"},
        # 12k rows -> a pure-noise feature has |rho| ~ 1/sqrt(n) ~ 0.009, so the
        # signal threshold has to sit comfortably above that to reject it.
        "feature_selection": {"sample_size": 8000, "chunk_size": 2,
                              "spearman": {"target_min_abs": 0.04, "redundancy_max_abs": 0.95}},
        "model": {"num_boost_round": 60, "early_stopping_rounds": 15,
                  "variants": ["base", "base_plus_new"]},
        "analysis": {"shap": {"sample_size": 1000}},
    }
    payload.update(overrides)
    return Config.from_dict(payload)


def test_dataset_split_and_sentinel_cleaning(data_paths, tmp_path):
    dataset = build_dataset(_config(data_paths, tmp_path))
    assert set(dataset.frames) == {"train", "valid", "test"}
    assert dataset.base_features == [f"old_{i}" for i in range(12)]
    assert "new_signal_a" in dataset.new_features
    # -9999 sentinels became NaN
    assert not (dataset.split("train")["new_signal_b"] == -9999).any()
    assert dataset.split("train")["new_signal_b"].isna().any()


def test_full_run_shows_gini_gain_from_new_features(data_paths, tmp_path):
    result = Pipeline(_config(data_paths, tmp_path)).run()

    assert result.feature_selection.selected == ["new_signal_a", "new_signal_b"]
    assert "new_dup_old0" in result.feature_selection.dropped
    assert "new_noise" in result.feature_selection.dropped
    assert all(m.error is None for m in result.models)

    comparison = result.analysis.comparison.set_index("variant")
    assert comparison.loc["base", "gini_gain_test"] == pytest.approx(0.0)
    assert comparison.loc["base_plus_new", "gini_gain_test"] > 0.05
    assert comparison.loc["base_plus_new", "adj_gini_test"] > comparison.loc["base", "adj_gini_test"]

    shap_rank = result.analysis.shap_ranking
    top_new = shap_rank[(shap_rank["variant"] == "base_plus_new")
                        & (shap_rank["feature"].str.startswith("new_"))]
    assert top_new["shap_rank"].min() <= 3
    assert result.analysis.new_feature_share["base_plus_new"] > 0.1

    for artifact in ("report.md", "summary.json", "variant_comparison.csv",
                     "shap_ranking.csv", "feature_selection_decisions.json"):
        assert (result.output_dir / artifact).exists()
    assert (result.output_dir / "models" / "base_plus_new.json").exists()


def test_sequential_backend_matches_parallel(data_paths, tmp_path):
    parallel = Pipeline(_config(data_paths, tmp_path)).run()
    sequential_cfg = _config(data_paths, tmp_path)
    sequential_cfg.run.backend = "sequential"
    sequential_cfg.run.n_jobs = 1
    sequential = Pipeline(sequential_cfg).run()

    assert parallel.feature_selection.selected == sequential.feature_selection.selected
    left = parallel.analysis.comparison.set_index("variant")["adj_gini_test"]
    right = sequential.analysis.comparison.set_index("variant")["adj_gini_test"]
    pd.testing.assert_series_equal(left, right, atol=1e-9)


def test_binary_task_runs(data_paths, tmp_path):
    frames = _read(data_paths)
    median = pd.concat(frames.values())["y"].median()
    frames = {name: part.assign(y=(part["y"] > median).astype(int))
              for name, part in frames.items()}

    cfg = _config(_write(frames, tmp_path), tmp_path)
    cfg.model.task = "binary"
    result = Pipeline(cfg).run()
    assert all(m.error is None for m in result.models)
    preds = result.models[0].predictions["test"]
    assert preds.min() >= 0 and preds.max() <= 1


def test_run_from_in_memory_frames_matches_run_from_paths(data_paths, tmp_path):
    """The discover -> verify loop passes candidates in memory, never via a file."""
    from_paths = Pipeline(_config(data_paths, tmp_path)).run()
    from_frames = Pipeline(_config(data_paths, tmp_path)).run(frames=_read(data_paths))

    assert from_frames.feature_selection.selected == from_paths.feature_selection.selected
    left = from_paths.analysis.comparison.set_index("variant")["adj_gini_test"]
    right = from_frames.analysis.comparison.set_index("variant")["adj_gini_test"]
    pd.testing.assert_series_equal(left, right, atol=1e-9)


def test_run_accepts_a_prebuilt_dataset(data_paths, tmp_path):
    cfg = _config(data_paths, tmp_path)
    dataset = prepare_dataset_from_frames(_read(data_paths), cfg)
    result = Pipeline(cfg).run(dataset=dataset)
    assert result.dataset is not None and all(m.error is None for m in result.models)


def test_run_rejects_both_frames_and_dataset(data_paths, tmp_path):
    cfg = _config(data_paths, tmp_path)
    frames = _read(data_paths)
    with pytest.raises(ValueError, match="not both"):
        Pipeline(cfg).run(frames=frames, dataset=prepare_dataset_from_frames(frames, cfg))


def test_candidates_engineered_in_memory_are_evaluated(data_paths, tmp_path):
    """Columns that exist only in the passed frames are valid candidates.

    ``new_real`` is the old_4 * old_5 interaction the generator actually uses, so
    it should survive; ``new_spurious`` multiplies two features the outcome does
    not depend on, so the signal screen should reject it.
    """
    frames = {name: part.assign(new_real=part["old_4"] * part["old_5"],
                                new_spurious=part["old_0"] * part["old_1"])
              for name, part in _read(data_paths).items()}

    cfg = _config(data_paths, tmp_path)
    cfg.run.gates = "enforce"          # this test is about the gate removing a candidate
    cfg.features.new = ["new_real", "new_spurious"]
    cfg.features.new_prefix = None
    result = Pipeline(cfg).run(frames=frames)

    considered = set(result.feature_selection.target_stats["feature"])
    assert {"new_real", "new_spurious"} <= considered

    assert result.dataset.new_features == ["new_real"]
    assert "new_spurious" in result.feature_selection.dropped
    assert "new_real" in set(result.analysis.shap_ranking["feature"])


def test_accuracy_is_kept_out_of_the_comparison_table_by_default(data_paths, tmp_path):
    result = Pipeline(_config(data_paths, tmp_path)).run()

    comparison = result.analysis.comparison
    assert not [c for c in comparison.columns if c.startswith("accuracy_")]
    assert [c for c in comparison.columns if c.startswith("adj_gini_")]
    # still computed and written per split, just not surfaced in the comparison
    assert "accuracy" in result.analysis.metrics.columns


def test_accuracy_can_be_switched_back_on(data_paths, tmp_path):
    cfg = _config(data_paths, tmp_path)
    cfg.analysis.include_accuracy = True
    result = Pipeline(cfg).run()
    assert [c for c in result.analysis.comparison.columns if c.startswith("accuracy_")]


def test_report_never_truncates_away_a_new_feature(data_paths, tmp_path, monkeypatch):
    """The ranking table is capped, but new features are the point of the run.

    The cut is shrunk to 2 so truncation is guaranteed on this small dataset:
    with three selected candidates, at least one must fall below it.
    """
    import validation.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "REPORT_TOP_N", 2)
    result = Pipeline(_config(data_paths, tmp_path)).run()
    section = (result.output_dir / "report.md").read_text().split("## Feature ranking")[1]

    ranked = result.analysis.feature_ranking
    ranked = ranked[ranked["variant"] == "base_plus_new"].set_index("feature")
    new_features = set(result.dataset.new_features)

    below_cut = ranked[ranked["shap_rank"] > 2]
    new_below = [f for f in below_cut.index if f in new_features]
    assert new_below, "expected at least one new feature below the cut"

    for feature in new_features:
        assert feature in section, f"{feature} was truncated out of the report"

    # incumbents below the cut are still dropped, so the table stays readable
    incumbents_below = [f for f in below_cut.index if f not in new_features]
    assert incumbents_below
    assert not any(f in section for f in incumbents_below)
    assert "Full ranking in `feature_ranking.csv`" in section


def test_capture_rate_reaches_the_comparison_table(data_paths, tmp_path):
    cfg = _config(data_paths, tmp_path)
    result = Pipeline(cfg).run()
    comparison = result.analysis.comparison.set_index("variant")

    for split in ("train", "valid", "test"):
        assert f"capture_top5_{split}" in comparison.columns
        assert f"capture_gain_top5_{split}" in comparison.columns

    # the baseline is its own reference
    assert comparison.loc["base", "capture_gain_top5_test"] == pytest.approx(0.0)
    # and the gain is exactly the difference from it
    champion = comparison.loc["base", "capture_top5_test"]
    for variant, row in comparison.iterrows():
        assert row["capture_gain_top5_test"] == pytest.approx(
            row["capture_top5_test"] - champion)

    report = (result.output_dir / "report.md").read_text()
    assert "capture_top5_test" in report


def test_headline_capture_percent_need_not_be_in_capture_rate_percents(data_paths, tmp_path):
    """The two lists are unioned, so a headline percent is never missing."""
    cfg = _config(data_paths, tmp_path)
    cfg.analysis.capture_rate_percents = [0.10]
    cfg.analysis.comparison_capture_percents = [0.02]
    result = Pipeline(cfg).run()

    assert "capture_top2_test" in result.analysis.comparison.columns
    assert "capture_rate_0.02" in result.analysis.metrics.columns
    assert "capture_rate_0.1" in result.analysis.metrics.columns


def test_several_capture_percents_can_be_surfaced(data_paths, tmp_path):
    cfg = _config(data_paths, tmp_path)
    cfg.analysis.comparison_capture_percents = [0.01, 0.10]
    comparison = Pipeline(cfg).run().analysis.comparison
    assert {"capture_top1_test", "capture_top10_test"} <= set(comparison.columns)
    assert "capture_top5_test" not in comparison.columns
