"""`run.gates: open` - measure every candidate at every stage, remove nothing."""

import pandas as pd
import pytest

from validation.config import Config
from validation.stages.verdict import FAIL, NOT_REACHED, PASS

xgb = pytest.importorskip("xgboost")

from synthetic import make_frame, split_by_column  # noqa: E402
from validation.pipeline import Pipeline  # noqa: E402


@pytest.fixture(scope="module")
def frames():
    return split_by_column(make_frame(n_rows=9_000, seed=12).assign(new_constant=1.0))


def _config(tmp_path, gates):
    return Config.from_dict({
        "run": {"name": f"gates_{gates}", "output_dir": str(tmp_path), "n_jobs": 2,
                "log_level": "ERROR", "gates": gates},
        "data": {"target": "y", "id_cols": ["row_id"]},
        "features": {"base_prefix": "old_", "new_prefix": "new_"},
        "data_quality": {"enabled": True, "drop_failed": True},
        "feature_selection": {"chunk_size": 2, "spearman": {"target_min_abs": 0.04,
                                                            "redundancy_max_abs": 0.9}},
        "model": {"num_boost_round": 60, "early_stopping_rounds": 15,
                  "variants": ["base", "base_plus_new", "leave_one_in"]},
        "analysis": {"shap": {"sample_size": 800}},
        "verdict": {"min_gini_gain": 0.005},
    })


def test_open_is_the_default():
    assert Config.from_dict({}).run.gates == "open"
    assert Config.from_dict({}).gates_enforced is False


def test_bad_gate_mode_is_rejected():
    with pytest.raises(ValueError, match="run.gates"):
        Config.from_dict({"run": {"gates": "half"}}).validate()


def test_open_gates_remove_nothing(frames, tmp_path):
    result = Pipeline(_config(tmp_path, "open")).run(frames=frames)

    proposed = {"new_signal_a", "new_signal_b", "new_dup_old0", "new_noise", "new_constant"}
    assert set(result.dataset.new_features) == proposed, "no candidate may be removed"

    # the screens still ran and still recorded what they would have cut
    assert result.data_quality.failed == ["new_constant"]
    assert {"new_noise", "new_dup_old0"} <= set(result.feature_selection.dropped)

    # every candidate therefore reached a model of its own
    variants = {m.name for m in result.models}
    assert {f"loi__{f}" for f in proposed} <= variants


def test_open_gates_measure_every_candidate_at_every_gate(frames, tmp_path):
    result = Pipeline(_config(tmp_path, "open")).run(frames=frames)
    verdicts = result.verdicts.set_index("feature")

    # nothing is short-circuited: no cell says "not reached"
    gates = ["data quality", "feature selection", "gini gain", "shap rank"]
    assert NOT_REACHED not in verdicts[gates].to_numpy()

    # a candidate the selection screen rejected still carries later-stage evidence
    rejected = verdicts.loc["new_noise"]
    assert rejected["feature selection"] == FAIL
    assert rejected["gini gain"] in {PASS, FAIL}
    assert rejected["shap rank"] in {PASS, FAIL}
    assert pd.notna(rejected["shap_rank_pct"])

    assert (verdicts["n_gates_failed"] >= 1).any()


def test_enforce_stops_at_the_first_failing_gate(frames, tmp_path):
    result = Pipeline(_config(tmp_path, "enforce")).run(frames=frames)
    verdicts = result.verdicts.set_index("feature")

    assert verdicts.loc["new_constant", "data quality"] == FAIL
    assert verdicts.loc["new_constant", "feature selection"] == NOT_REACHED
    assert verdicts.loc["new_noise", "gini gain"] == NOT_REACHED
    assert "new_constant" not in result.dataset.new_features
    assert "new_noise" not in result.dataset.new_features


def test_batch_verdict_is_advisory_while_gates_are_open(frames, tmp_path):
    result = Pipeline(_config(tmp_path, "open")).run(frames=frames)
    assert "advisory" in result.batch.note
    assert result.summary()["gates"] == "open"

    report = (result.output_dir / "report.md").read_text()
    assert "nothing was removed" in report
