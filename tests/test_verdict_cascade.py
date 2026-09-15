"""The four-gate cascade and the batch-level call."""

import numpy as np
import pandas as pd
import pytest

from validation.config import Config
from validation.stages.verdict import (
    FAIL,
    NOT_EVALUABLE,
    NOT_REACHED,
    PASS,
    build_verdicts,
    decide_batch,
)
from validation.stages.analysis import AnalysisResult
from validation.stages.data_quality import DataQualityResult
from validation.stages.feature_selection import FeatureSelectionResult

xgb = pytest.importorskip("xgboost")

from synthetic import make_frame, split_by_column  # noqa: E402
from validation.pipeline import Pipeline  # noqa: E402


def _cfg(**verdict):
    return Config.from_dict({"verdict": verdict, "model": {"task": "regression"}})


def _analysis(gains, shap_rows):
    comparison = pd.DataFrame(
        [{"variant": v, "gini_gain_test": t, "gini_gain_valid": va} for v, (t, va) in gains.items()]
    )
    shap = pd.DataFrame(shap_rows)
    return AnalysisResult(comparison=comparison, shap_ranking=shap)


def _selection(selected, dropped=None, stages=None):
    return FeatureSelectionResult(
        selected=list(selected), dropped=dict(dropped or {}), dropped_stage=dict(stages or {})
    )


# --------------------------------------------------------------------------- #
# The cascade
# --------------------------------------------------------------------------- #
def test_each_gate_can_be_the_one_that_fails():
    candidates = ["dq_bad", "sel_bad", "gini_bad", "shap_bad", "good"]
    dq = DataQualityResult(failed=["dq_bad"], skipped=False)
    fs = _selection(
        ["gini_bad", "shap_bad", "good"],
        {"sel_bad": "weak spearman vs target (0.001)"},
        {"sel_bad": "signal screen"},
    )
    analysis = _analysis(
        gains={"base_plus_new": (0.02, 0.02),
               "loi__gini_bad": (0.000, 0.000),
               "loi__shap_bad": (0.02, 0.02),
               "loi__good": (0.03, 0.03)},
        shap_rows=[{"variant": "base_plus_new", "feature": f, "shap_rank": r, "mean_abs_shap": 1.0}
                   for f, r in [("good", 1), ("a", 2), ("shap_bad", 4), ("b", 3)]],
    )
    verdicts = build_verdicts(candidates, _cfg(min_gini_gain=0.005, max_shap_rank_pct=0.5),
                              dq=dq, fs=fs, analysis=analysis).set_index("feature")

    assert verdicts.loc["dq_bad", "failed_at"] == "data quality"
    assert verdicts.loc["sel_bad", "failed_at"] == "feature selection"
    assert verdicts.loc["gini_bad", "failed_at"] == "gini gain"
    assert verdicts.loc["shap_bad", "failed_at"] == "shap rank"
    assert verdicts.loc["good", "verdict"] == PASS
    assert verdicts.loc["good", "failed_at"] == ""


def test_a_failed_gate_stops_later_gates_being_evaluated():
    fs = _selection([], {"x": "redundant with old_0"}, {"x": "redundancy screen"})
    verdicts = build_verdicts(["x"], _cfg(), fs=fs).set_index("feature")
    assert verdicts.loc["x", "feature selection"] == FAIL
    # never entered a model, so there is nothing to read at the last two gates
    assert verdicts.loc["x", "gini gain"] == NOT_REACHED
    assert verdicts.loc["x", "shap rank"] == NOT_REACHED


def test_missing_leave_one_in_makes_the_gini_gate_not_evaluable():
    """Without loi variants the gain cannot be attributed to one feature."""
    fs = _selection(["x"])
    analysis = _analysis({"base_plus_new": (0.02, 0.02)}, [])
    verdicts = build_verdicts(["x"], _cfg(), fs=fs, analysis=analysis).set_index("feature")
    assert verdicts.loc["x", "gini gain"] == NOT_EVALUABLE
    assert verdicts.loc["x", "verdict"] == PASS   # not punished for missing infrastructure


def test_valid_split_must_agree_when_required():
    fs = _selection(["x"])
    analysis = _analysis({"loi__x": (0.02, -0.01)}, [])
    strict = build_verdicts(["x"], _cfg(min_gini_gain=0.005, require_valid_too=True),
                            fs=fs, analysis=analysis).set_index("feature")
    lenient = build_verdicts(["x"], _cfg(min_gini_gain=0.005, require_valid_too=False),
                             fs=fs, analysis=analysis).set_index("feature")
    assert strict.loc["x", "gini gain"] == FAIL
    assert lenient.loc["x", "gini gain"] == PASS


def test_shap_gate_is_a_percentile_not_an_absolute_rank():
    fs = _selection(["x"])
    analysis = _analysis(
        {"loi__x": (0.02, 0.02)},
        [{"variant": "base_plus_new", "feature": f, "shap_rank": i + 1, "mean_abs_shap": 1.0}
         for i, f in enumerate(["a", "b", "x", "c"])],   # rank 3 of 4 -> 75th pct
    )
    loose = build_verdicts(["x"], _cfg(max_shap_rank_pct=0.8), fs=fs, analysis=analysis).set_index("feature")
    tight = build_verdicts(["x"], _cfg(max_shap_rank_pct=0.5), fs=fs, analysis=analysis).set_index("feature")
    assert loose.loc["x", "shap rank"] == PASS
    assert tight.loc["x", "shap rank"] == FAIL


# --------------------------------------------------------------------------- #
# The batch call
# --------------------------------------------------------------------------- #
def test_all_failing_means_try_a_new_batch():
    verdicts = pd.DataFrame([
        {"feature": "a", "verdict": FAIL, "failed_at": "feature selection"},
        {"feature": "b", "verdict": FAIL, "failed_at": "gini gain"},
        {"feature": "c", "verdict": FAIL, "failed_at": "feature selection"},
    ])
    batch = decide_batch(verdicts, _cfg())
    assert batch.verdict == "TRY A NEW BATCH"
    assert batch.n_passed == 0
    assert batch.failed_at == {"feature selection": 2, "gini gain": 1}
    assert "feature selection" in batch.note      # names where most fell


def test_one_survivor_is_enough_to_keep_the_batch():
    verdicts = pd.DataFrame([
        {"feature": "a", "verdict": PASS, "failed_at": ""},
        {"feature": "b", "verdict": FAIL, "failed_at": "gini gain"},
    ])
    batch = decide_batch(verdicts, _cfg())
    assert batch.verdict == "KEEP" and batch.passed == ["a"]


def test_no_candidates_at_all():
    batch = decide_batch(pd.DataFrame(), _cfg())
    assert batch.verdict == "TRY A NEW BATCH" and batch.n_candidates == 0


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_pipeline_emits_the_cascade_and_the_batch_call(tmp_path):
    frame = make_frame(n_rows=9_000, seed=8)
    cfg = Config.from_dict({
        "run": {"name": "cascade", "output_dir": str(tmp_path), "n_jobs": 2, "log_level": "ERROR"},
        "data": {"target": "y", "id_cols": ["row_id"]},
        "features": {"base_prefix": "old_", "new_prefix": "new_"},
        # redundancy_max_abs defaults to 1.01 (screen off), so set it explicitly -
        # otherwise the duplicate survives selection and is only caught a gate later.
        "feature_selection": {"chunk_size": 2, "spearman": {"target_min_abs": 0.04,
                                                            "redundancy_max_abs": 0.9}},
        "model": {"num_boost_round": 80, "early_stopping_rounds": 20,
                  "variants": ["base", "base_plus_new", "leave_one_in"]},
        "analysis": {"shap": {"sample_size": 800}},
        "verdict": {"min_gini_gain": 0.005},
    })
    result = Pipeline(cfg).run(frames=split_by_column(frame))

    assert result.batch is not None
    assert result.batch.verdict in {"KEEP", "TRY A NEW BATCH"}
    assert result.batch.n_candidates == 4          # every proposed feature is accounted for

    verdicts = result.verdicts.set_index("feature")
    assert verdicts.loc["new_noise", "failed_at"] == "feature selection"
    assert verdicts.loc["new_dup_old0", "failed_at"] == "feature selection"
    # the genuine signals reach the later gates
    assert verdicts.loc["new_signal_a", "feature selection"] == PASS
    assert verdicts.loc["new_signal_a", "gini gain"] in {PASS, FAIL}

    assert (result.output_dir / "candidate_verdicts.csv").exists()
    assert (result.output_dir / "batch_verdict.json").exists()
    report = (result.output_dir / "report.md").read_text()
    assert f"## Verdict: {result.batch.verdict}" in report


def test_verdict_stage_can_be_disabled(tmp_path):
    frame = make_frame(n_rows=6_000, seed=9)
    cfg = Config.from_dict({
        "run": {"name": "no_verdict", "output_dir": str(tmp_path), "n_jobs": 2, "log_level": "ERROR"},
        "data": {"target": "y", "id_cols": ["row_id"]},
        "features": {"base_prefix": "old_", "new_prefix": "new_"},
        "model": {"num_boost_round": 40, "variants": ["base", "base_plus_new"]},
        "analysis": {"shap": {"enabled": False}},
        "verdict": {"enabled": False},
    })
    result = Pipeline(cfg).run(frames=split_by_column(frame))
    assert result.batch is None
    assert "verdict" in result.verdicts.columns    # falls back to the selection view
