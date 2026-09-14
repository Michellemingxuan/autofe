"""The full loop: discovery proposes, the validation stages decide."""
import json

import numpy as np
import pandas as pd
import pytest

from validation.config import Config
from validation.pipeline import Pipeline

REPLY = """```python
# ('A over B', 'ratio of a to b')
# Usefulness: relative magnitude drives the outcome
df["ratio"] = df["a"] / df["b"].clip(lower=0.1)
```end

```python
# ('Dead weight', 'a constant')
# Usefulness: none
df["dead"] = df["noise"] * 0.0
```end
"""


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 2400
    a = rng.uniform(1, 20, n); b = rng.uniform(1, 20, n); noise = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(1.5 * (a / b) - 3.0 + 0.4 * noise)))).astype(int)
    return pd.DataFrame({"y": y, "a": a, "b": b, "noise": noise})


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []

    def _call(prompt, **kwargs):
        calls.append(prompt)
        return REPLY, {"backend": "stub", "model": "stub", "attempts": 1}

    # patched on the shared base, so one fixture covers every strategy
    monkeypatch.setattr("discovery.strategies.base.call_llm", _call)
    return calls


def _config(tmp_path, **discovery):
    return Config.from_dict({
        "run": {"name": "loop", "output_dir": str(tmp_path), "seed": 42,
                "n_jobs": 1, "log_level": "WARNING"},
        "data": {"target": "y", "split": {"valid_size": 0.25, "test_size": 0.25,
                                          "stratify": True}},
        "features": {"base": ["a", "b", "noise"]},
        "discovery": {"enabled": True, "strategy": "caafe", "batch_size": 2,
                      "max_rounds": 1, "task_description": "signal is in a/b",
                      "sample_size": 500, "shots": 6, "screen_boost_rounds": 30,
                      **discovery},
        "model": {"num_boost_round": 40, "variants": ["base", "base_plus_new",
                                                      "leave_one_in"],
                  "params": {"eta": 0.1, "max_depth": 3,
                             "objective": "binary:logistic", "eval_metric": "auc"}},
        "data_quality": {"enabled": True},
        "feature_selection": {"enabled": False},
        "analysis": {"metrics_on": ["train", "valid", "test"]},
        "verdict": {"enabled": True},
    })


def test_discovered_features_flow_through_every_stage(tmp_path, frame, fake_llm):
    cfg = _config(tmp_path)
    result = Pipeline(cfg).run(frame=frame)

    assert result.discovery is not None
    kept = result.discovery.kept_features
    assert kept, result.discovery.summary()

    # stage 3 built one variant per discovered feature
    names = [m.name for m in result.models]
    assert "base" in names
    for feature in kept:
        assert f"loi__{feature}" in names

    # stage 5 reached a verdict on each of them
    assert not result.verdicts.empty
    judged = set(result.verdicts["feature"])
    assert set(kept) <= judged, judged


def test_the_ledger_is_written_with_rationale_and_scores(tmp_path, frame, fake_llm):
    result = Pipeline(cfg := _config(tmp_path)).run(frame=frame)
    ledger = result.output_dir / "discovered_features.csv"
    assert ledger.exists()

    rows = pd.read_csv(ledger)
    assert set(rows["feature_name"]) == {"ratio", "dead"}
    ratio = rows[rows["feature_name"] == "ratio"].iloc[0]
    assert ratio["display_name"] == "A over B"
    assert ratio["rationale"] == "relative magnitude drives the outcome"
    assert ratio["input_columns"] == "a, b"
    assert not pd.isna(ratio["delta"])
    assert "df[" in ratio["code"]


def test_the_summary_records_how_the_run_stopped(tmp_path, frame, fake_llm):
    result = Pipeline(_config(tmp_path)).run(frame=frame)
    payload = json.loads((result.output_dir / "discovery_summary.json").read_text())
    assert payload["proposed"] == 2
    assert "max_rounds reached (1)" in payload["stopped_because"]
    assert payload["rounds"] == 1
    assert "discovery" in result.summary()


def test_prompts_are_saved_under_the_run_directory(tmp_path, frame, fake_llm):
    result = Pipeline(_config(tmp_path)).run(frame=frame)
    saved = result.output_dir / "discovery" / "round_001"
    assert (saved / "prompt.txt").exists()
    assert (saved / "reply.txt").exists()


def test_discovery_appends_to_hand_written_candidates(tmp_path, frame, fake_llm):
    """A config may already list candidates; discovery adds to them."""
    cfg = _config(tmp_path)
    cfg.features.new = ["noise"]           # pretend this was proposed by hand
    cfg.features.base = ["a", "b"]
    result = Pipeline(cfg).run(frame=frame)
    assert "noise" in result.discovery.dataset.new_features
    assert set(result.discovery.kept_features) <= set(result.discovery.dataset.new_features)


def test_disabled_discovery_leaves_the_pipeline_untouched(tmp_path, frame):
    cfg = _config(tmp_path)
    cfg.discovery.enabled = False
    cfg.features.new = ["noise"]
    cfg.features.base = ["a", "b"]
    result = Pipeline(cfg).run(frame=frame)
    assert result.discovery is None
    assert not (result.output_dir / "discovered_features.csv").exists()
    assert [m.name for m in result.models]          # the rest of the run still happened


def test_validation_imports_without_the_llm_extra(monkeypatch):
    """A validation-only install must not need openai/dotenv present."""
    import sys
    for name in ("openai", "dotenv", "nest_asyncio"):
        monkeypatch.setitem(sys.modules, name, None)
    for name in [m for m in list(sys.modules) if m.startswith(("validation", "discovery"))]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    import validation  # noqa: F401
    import validation.pipeline  # noqa: F401
