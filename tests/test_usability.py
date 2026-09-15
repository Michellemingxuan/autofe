import json

import pandas as pd
import pytest
import yaml

from validation.cli import main
from validation.config import Config
from validation.data import prepare_dataset_from_frames
from validation.logging_utils import setup_logging
from validation.preflight import run_preflight
from validation.status import RunStatus, render_plan, stage_specs


def _frames(overlap=False):
    train = pd.DataFrame({"row_id": [1, 2, 3, 4], "y": [0, 1, 0, 1],
                          "base_a": [0.0, 1.0, 2.0, 3.0],
                          "cand_x": [1.0, 1.5, 2.0, 2.5]})
    valid = pd.DataFrame({"row_id": [4 if overlap else 5, 6], "y": [0, 1],
                          "base_a": [0.5, 2.5], "cand_x": [1.1, 2.4]})
    test = pd.DataFrame({"row_id": [7, 8], "y": [0, 1],
                         "base_a": [0.2, 2.8], "cand_x": [1.2, 2.6]})
    return {"train": train, "valid": valid, "test": test}


def _config():
    return Config.from_dict({
        "data": {"target": "y", "id_cols": ["row_id"]},
        "features": {"base_prefix": "base_", "new_prefix": "cand_"},
        "model": {"task": "binary", "variants": ["base", "base_plus_new", "leave_one_in"]},
    })


def test_plan_keeps_disabled_stages_visible():
    plan = render_plan(_config())
    assert "[Data]" in plan
    assert "[Discover (off)]" in plan
    assert "[Quality (off)]" in plan
    assert plan.endswith("[Verdict]")


def test_preflight_reports_cross_split_leakage():
    cfg = _config()
    report = run_preflight(cfg, prepare_dataset_from_frames(_frames(overlap=True), cfg))
    check = next(check for check in report.checks if check.name == "split IDs are disjoint")
    assert check.status == "FAIL"
    assert "train/valid=1" in check.detail
    assert report.ok is False


def test_live_board_persists_success_and_checks(tmp_path):
    cfg = _config()
    setup_logging("INFO", tmp_path / "run.log")
    status = RunStatus(tmp_path, "demo", stage_specs(cfg))
    with status.stage("data") as stage:
        stage.detail = "10 rows"
        stage.check("shape", True, "ok")
    status.finish()

    payload = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert payload["status"] == "succeeded"
    assert payload["stages"][0]["status"] == "passed"
    assert payload["stages"][0]["checks"][0]["name"] == "shape"
    assert not (tmp_path / "pipeline.html").exists()
    assert "pipeline |" in (tmp_path / "run.log").read_text()


def test_live_board_preserves_failure_and_marks_later_stages_not_reached(tmp_path):
    cfg = _config()
    status = RunStatus(tmp_path, "broken", stage_specs(cfg))
    with pytest.raises(RuntimeError, match="bad input"):
        with status.stage("data"):
            raise RuntimeError("bad input")

    payload = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert payload["status"] == "failed"
    assert payload["stages"][0]["error"] == "RuntimeError: bad input"
    assert all(stage["status"] == "skipped" for stage in payload["stages"][1:])
    selection = next(stage for stage in payload["stages"] if stage["key"] == "feature_selection")
    assert "not reached" in selection["detail"]


def test_init_writes_a_minimal_valid_config(tmp_path):
    path = tmp_path / "churn.yaml"
    code = main([
        "init", str(path), "--train", "train.csv", "--valid", "valid.csv",
        "--test", "test.csv", "--target", "churned", "--task", "binary",
        "--id", "customer_id", "--new-prefix", "cand_",
    ])
    assert code == 0
    payload = yaml.safe_load(path.read_text())
    assert payload["features"] == {"base": [], "new_prefix": "cand_"}
    assert payload["model"]["variants"] == ["base", "base_plus_new", "leave_one_in"]
    Config.from_dict(payload).validate()
