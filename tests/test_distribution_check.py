"""Distribution consistency across splits: PSI against a reference split."""

import numpy as np
import pandas as pd
import pytest

from validation.config import Config
from validation.data import prepare_dataset
from validation.stages.data_quality import population_stability_index as psi
from validation.stages.data_quality import run_data_quality


# --------------------------------------------------------------------------- #
# The statistic
# --------------------------------------------------------------------------- #
def test_identical_samples_score_zero():
    values = np.random.default_rng(0).normal(size=5000)
    assert psi(values, values) == pytest.approx(0.0, abs=1e-12)


def test_shift_is_ordered_by_severity():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=20000)
    same = psi(reference, rng.normal(size=20000))
    mean_shift = psi(reference, rng.normal(0.5, 1.0, 20000))
    variance_shift = psi(reference, rng.normal(0.0, 2.0, 20000))

    assert same < 0.01 < mean_shift < variance_shift
    assert mean_shift > 0.1        # a half-sd move is a real shift


def test_missingness_shift_is_caught():
    """A variable 0% null in train and 40% null in test has genuinely changed."""
    rng = np.random.default_rng(1)
    reference = rng.normal(size=10000)
    comparison = rng.normal(size=10000)
    comparison[:4000] = np.nan
    assert psi(reference, comparison) > 1.0


def test_degenerate_inputs_return_nan():
    values = np.random.default_rng(0).normal(size=100)
    assert np.isnan(psi(values, np.array([])))
    assert np.isnan(psi(np.array([]), values))
    assert np.isnan(psi(np.full(50, np.nan), values))


def test_a_constant_reference_does_not_blow_up():
    constant = np.ones(1000)
    assert psi(constant, np.ones(1000)) == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(psi(constant, np.random.default_rng(0).normal(size=1000)))


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #
@pytest.fixture
def frame():
    rng = np.random.default_rng(3)
    n = 9000
    df = pd.DataFrame({
        "old_stable": rng.normal(size=n),
        "old_drifting": rng.normal(size=n),
        "new_a": rng.normal(size=n),
        "y": (rng.random(n) < 0.2).astype(int),
    })
    df["split"] = np.where(np.arange(n) < 5000, "train",
                           np.where(np.arange(n) < 7000, "valid", "test"))
    # make one variable genuinely different outside train
    df.loc[df["split"] != "train", "old_drifting"] += 1.5
    return df


def _config(**dq):
    return Config.from_dict({
        "run": {"n_jobs": 2, "log_level": "ERROR"},
        "data": {"target": "y", "split": {"mode": "column", "column": "split"}},
        "features": {"base_prefix": "old_", "new_prefix": "new_"},
        "data_quality": {"enabled": True, **dq},
    })


def test_psi_columns_are_added_per_non_reference_split(frame):
    cfg = _config()
    result = run_data_quality(prepare_dataset(frame, cfg), cfg)
    report = result.report.set_index("feature")

    assert {"psi_valid", "psi_test", "psi_max", "psi_worst_split"} <= set(report.columns)
    assert "psi_train" not in report.columns      # the reference is not compared to itself
    assert report.loc["old_drifting", "psi_max"] > 0.25
    assert report.loc["old_stable", "psi_max"] < 0.05
    assert report.loc["old_drifting", "psi_max"] == pytest.approx(
        max(report.loc["old_drifting", ["psi_valid", "psi_test"]]))


def test_shifted_feature_fails_the_quality_check(frame):
    result = run_data_quality(prepare_dataset(frame, _config(max_psi=0.25)), _config(max_psi=0.25))
    assert "old_drifting" in result.failed
    assert "old_stable" not in result.failed


def test_threshold_is_configurable(frame):
    cfg = _config(max_psi=99.0)
    result = run_data_quality(prepare_dataset(frame, cfg), cfg)
    assert result.failed == []                     # nothing exceeds an enormous threshold


def test_check_can_be_switched_off(frame):
    cfg = _config(distribution_check=False)
    result = run_data_quality(prepare_dataset(frame, cfg), cfg)
    assert "psi_max" not in result.report.columns
    assert "missing_rate" in result.report.columns  # the other checks still run


def test_reference_split_is_configurable(frame):
    cfg = _config(distribution_reference="test")
    result = run_data_quality(prepare_dataset(frame, cfg), cfg)
    report = result.report.set_index("feature")
    assert {"psi_train", "psi_valid"} <= set(report.columns)
    assert "psi_test" not in report.columns
    # train now looks like the odd one out, since test is the reference
    assert report.loc["old_drifting", "psi_worst_split"] == "train"


def test_missing_reference_split_is_reported_not_fatal(frame):
    cfg = _config(distribution_reference="valid")
    frame = frame[frame["split"] != "valid"]
    result = run_data_quality(prepare_dataset(frame, cfg), cfg)
    assert "psi_max" not in result.report.columns   # skipped, stage still returns
    assert len(result.report) > 0


# --------------------------------------------------------------------------- #
# Thresholds: visible, attributed, adjustable
# --------------------------------------------------------------------------- #
def test_report_names_the_check_that_rejected_each_feature(frame):
    frame = frame.assign(old_constant=1.0)
    cfg = _config(max_psi=0.25, min_unique=2)
    result = run_data_quality(prepare_dataset(frame, cfg), cfg)
    report = result.report.set_index("feature")

    assert report.loc["old_constant", "failed_checks"] == "n_unique < 2"
    assert report.loc["old_drifting", "failed_checks"] == "psi_max > 0.25"
    assert report.loc["old_stable", "failed_checks"] == ""
    assert report.loc["old_stable", "passed"]


def test_thresholds_are_reported_alongside_the_result(frame):
    cfg = _config(max_missing_rate=0.9, min_unique=3, max_psi=0.2)
    result = run_data_quality(prepare_dataset(frame, cfg), cfg)

    assert result.thresholds == {"max_missing_rate": 0.9, "min_unique": 3, "max_psi": 0.2}
    assert result.summary()["thresholds"] == result.thresholds
    assert result.failed_by_check()["psi_max > 0.2"] >= 1


def test_a_feature_can_fail_several_checks_at_once(frame):
    frame = frame.assign(old_broken=np.nan)
    cfg = _config(max_missing_rate=0.5, min_unique=2)
    result = run_data_quality(prepare_dataset(frame, cfg), cfg)
    checks = result.report.set_index("feature").loc["old_broken", "failed_checks"]

    assert "missing_rate" in checks and "n_unique" in checks


def test_tightening_a_threshold_rejects_more(frame):
    dataset = prepare_dataset(frame, _config())
    loose = run_data_quality(dataset, _config(max_psi=0.25))
    tight = run_data_quality(dataset, _config(max_psi=0.001))

    assert len(tight.failed) > len(loose.failed)
    assert set(loose.failed) <= set(tight.failed)


def test_thresholds_appear_in_the_report_file(tmp_path):
    """The run must record the thresholds it actually applied."""
    import pandas as pd
    from validation.pipeline import Pipeline

    rng = np.random.default_rng(5)
    n = 4000
    df = pd.DataFrame({f"old_{i}": rng.normal(size=n) for i in range(4)})
    df["new_a"] = rng.normal(size=n)
    df["y"] = (rng.random(n) < 0.2).astype(int)
    df["split"] = np.where(np.arange(n) < 2400, "train",
                           np.where(np.arange(n) < 3200, "valid", "test"))

    cfg = Config.from_dict({
        "run": {"name": "dq", "output_dir": str(tmp_path), "n_jobs": 2, "log_level": "ERROR"},
        "data": {"target": "y", "split": {"mode": "column", "column": "split"}},
        "features": {"base_prefix": "old_", "new_prefix": "new_"},
        "data_quality": {"enabled": True, "max_psi": 0.33, "min_unique": 7},
        "model": {"num_boost_round": 30, "variants": ["base", "base_plus_new"]},
        "analysis": {"shap": {"enabled": False}},
    })
    result = Pipeline(cfg).run(frame=df)
    report = (result.output_dir / "report.md").read_text()

    assert "max_psi" in report and "0.33" in report
    assert "min_unique" in report and "Adjust these under" in report
