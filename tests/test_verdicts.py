"""The selection stage's own view: did each proposed feature get past the screens?

``Pipeline.verdicts`` carries the full four-gate cascade (see test_verdict_cascade);
this file covers the selection-only table, written to
``feature_selection_verdicts.csv``.
"""

import numpy as np
import pandas as pd
import pytest

from validation.config import Config
from validation.data import prepare_dataset_from_frames
from validation.stages.feature_selection import build_verdict_table, run_feature_selection

xgb = pytest.importorskip("xgboost")

from synthetic import make_frame, split_by_column  # noqa: E402
from validation.pipeline import Pipeline  # noqa: E402


@pytest.fixture(scope="module")
def frames():
    return split_by_column(make_frame(n_rows=9_000, seed=6))


def _config(tmp_path, **overrides):
    payload = {
        "run": {"name": "verdicts", "output_dir": str(tmp_path), "n_jobs": 2, "log_level": "ERROR"},
        "data": {"target": "y", "id_cols": ["row_id"]},
        "features": {"base_prefix": "old_", "new_prefix": "new_"},
        "feature_selection": {"chunk_size": 2, "spearman": {"target_min_abs": 0.04,
                                                            "redundancy_max_abs": 0.9}},
        "model": {"num_boost_round": 40, "early_stopping_rounds": 10,
                  "variants": ["base", "base_plus_new"]},
        "analysis": {"shap": {"enabled": False}},
    }
    payload.update(overrides)
    return Config.from_dict(payload)


def test_every_candidate_gets_exactly_one_verdict(frames, tmp_path):
    cfg = _config(tmp_path)
    dataset = prepare_dataset_from_frames(frames, cfg)
    candidates = list(dataset.new_features)
    result = run_feature_selection(dataset, cfg)

    verdicts = build_verdict_table(candidates, result)
    assert list(verdicts["feature"].sort_values()) == sorted(candidates)
    assert set(verdicts["verdict"]) <= {"IN", "OUT"}
    assert (verdicts["verdict"] == "IN").sum() == len(result.selected)


def test_verdict_names_the_gate_that_decided(frames, tmp_path):
    cfg = _config(tmp_path)
    dataset = prepare_dataset_from_frames(frames, cfg)
    result = run_feature_selection(dataset, cfg)
    verdicts = build_verdict_table(list(dataset.new_features), result).set_index("feature")

    assert verdicts.loc["new_signal_a", "verdict"] == "IN"
    assert verdicts.loc["new_signal_a", "decided_by"] == "-"

    assert verdicts.loc["new_noise", "verdict"] == "OUT"
    assert verdicts.loc["new_noise", "decided_by"] == "signal screen"

    assert verdicts.loc["new_dup_old0", "verdict"] == "OUT"
    assert verdicts.loc["new_dup_old0", "decided_by"] == "redundancy screen"
    # the redundancy verdict says what it collided with
    assert verdicts.loc["new_dup_old0", "closest_existing"] == "old_0"


def test_kept_candidates_are_listed_first(frames, tmp_path):
    cfg = _config(tmp_path)
    dataset = prepare_dataset_from_frames(frames, cfg)
    verdicts = build_verdict_table(list(dataset.new_features), run_feature_selection(dataset, cfg))
    positions = verdicts["verdict"].tolist()
    assert positions == sorted(positions, key=lambda v: v != "IN")


def test_a_candidate_cut_by_data_quality_still_gets_a_verdict(frames, tmp_path):
    """The case that used to disappear: dropped before the screens ever ran."""
    frames = {name: part.assign(new_constant=1.0) for name, part in frames.items()}
    cfg = _config(tmp_path)
    cfg.run.gates = "enforce"          # removal only happens when gates are enforced
    cfg.data_quality.enabled = True
    cfg.data_quality.drop_failed = True

    result = Pipeline(cfg).run(frames=frames)

    selection = pd.read_csv(result.output_dir / "feature_selection_verdicts.csv").set_index("feature")
    assert "new_constant" in selection.index, "a data-quality drop must still be reported"
    assert selection.loc["new_constant", "verdict"] == "OUT"
    assert selection.loc["new_constant", "decided_by"] == "data quality"

    cascade = result.verdicts.set_index("feature")
    assert cascade.loc["new_constant", "failed_at"] == "data quality"
    assert "new_constant" not in result.dataset.new_features


def test_verdicts_are_written_and_summarised(frames, tmp_path):
    result = Pipeline(_config(tmp_path)).run(frames=frames)

    selection_csv = result.output_dir / "feature_selection_verdicts.csv"
    cascade_csv = result.output_dir / "candidate_verdicts.csv"
    assert selection_csv.exists() and cascade_csv.exists()

    written = pd.read_csv(cascade_csv)
    text_cols = ["reason", "failed_at"]        # empty strings come back as NaN
    pd.testing.assert_frame_equal(
        written.fillna({c: "" for c in text_cols}),
        result.verdicts.fillna({c: "" for c in text_cols}),
        check_dtype=False,
    )

    report = (result.output_dir / "report.md").read_text()
    assert f"## Verdict: {result.batch.verdict}" in report
    for feature in result.verdicts["feature"]:
        assert feature in report

    selection = pd.read_csv(selection_csv).set_index("feature")["verdict"].to_dict()
    assert selection["new_noise"] == "OUT" and selection["new_signal_a"] == "IN"
    assert result.summary()["candidate_verdicts"]["new_noise"] == "FAIL"


def test_selection_disabled_lets_every_candidate_through(frames, tmp_path):
    cfg = _config(tmp_path)
    cfg.feature_selection.enabled = False
    result = Pipeline(cfg).run(frames=frames)

    selection = pd.read_csv(result.output_dir / "feature_selection_verdicts.csv")
    assert set(selection["verdict"]) == {"IN"}
    # they still have to face the later gates
    assert set(result.verdicts["feature selection"]) == {"PASS"}
