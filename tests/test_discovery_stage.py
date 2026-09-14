"""Stage 0.5 end to end: propose -> screen -> materialise -> validation stages."""
import numpy as np
import pandas as pd
import pytest

from discovery.stage import DiscoveryResult, run_discovery_stage
from validation.config import Config
from validation.data import Dataset

REPLY = """Here are three features.

```python
# ('A over B', 'ratio of a to b')
# Usefulness: relative magnitude drives the outcome
df["ratio"] = df["a"] / df["b"].clip(lower=0.1)
```end

```python
# ('A times B', 'product of a and b')
# Usefulness: joint magnitude
df["prod"] = df["a"] * df["b"]
```end

```python
# ('Dead', 'a constant')
# Usefulness: none, honestly
df["dead"] = df["noise"] * 0.0
```end
"""


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    frames = {}
    for split, n in (("train", 1500), ("valid", 600), ("test", 600)):
        a = rng.uniform(1, 20, n); b = rng.uniform(1, 20, n); noise = rng.normal(size=n)
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(1.5 * (a / b) - 3.0 + 0.4 * noise)))).astype(int)
        frames[split] = pd.DataFrame({"y": y, "a": a, "b": b, "noise": noise})
    return Dataset(frames=frames, target="y",
                   base_features=["a", "b", "noise"], new_features=[])


@pytest.fixture
def cfg():
    return Config.from_dict({
        "run": {"seed": 42},
        "discovery": {
            "enabled": True, "strategy": "caafe", "batch_size": 3, "max_rounds": 1,
            "task_description": "synthetic; the signal is in a/b",
            "sample_size": 600, "shots": 8, "screen_boost_rounds": 40,
            "metric": "adj_gini", "min_delta": 0.0,
        },
        "model": {"params": {"eta": 0.1, "max_depth": 3, "objective": "binary:logistic"}},
    })


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the LLM call; everything else is real."""
    calls = []

    def _call(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return REPLY, {"backend": "stub", "model": "stub", "attempts": 1}

    # patched on the shared base, so one fixture covers every strategy
    monkeypatch.setattr("discovery.strategies.base.call_llm", _call)
    return calls


def test_disabled_discovery_passes_the_dataset_through(dataset):
    cfg = Config.from_dict({"discovery": {"enabled": False}})
    out = run_discovery_stage(cfg, dataset)
    assert out.enabled is False
    assert out.dataset is dataset
    assert out.kept_features == []


def test_every_proposal_is_recorded_with_its_rationale(cfg, dataset, fake_llm):
    out = run_discovery_stage(cfg, dataset)
    frame = out.to_frame()
    assert len(frame) == 3
    assert set(frame["feature_name"]) == {"ratio", "prod", "dead"}
    row = frame[frame["feature_name"] == "ratio"].iloc[0]
    assert row["display_name"] == "A over B"
    assert row["rationale"] == "relative magnitude drives the outcome"
    assert row["input_columns"] == "a, b"
    assert not pd.isna(row["delta"])


def test_kept_features_are_materialised_on_every_split(cfg, dataset, fake_llm):
    out = run_discovery_stage(cfg, dataset)
    assert out.kept_features, out.summary()
    for split, frame in out.dataset.frames.items():
        for name in out.kept_features:
            assert name in frame.columns, f"{name} missing from {split}"
            assert np.isfinite(frame[name]).all()


def test_the_widened_dataset_keeps_base_and_declares_new(cfg, dataset, fake_llm):
    out = run_discovery_stage(cfg, dataset)
    assert out.dataset.base_features == ["a", "b", "noise"]
    assert out.dataset.new_features == out.kept_features
    assert out.dataset.meta["discovered_features"] == out.kept_features


def test_a_column_is_computed_by_the_same_code_on_every_split(cfg, dataset, fake_llm):
    out = run_discovery_stage(cfg, dataset)
    if "ratio" not in out.kept_features:
        pytest.skip("ratio was not kept on this fixture")
    for frame in out.dataset.frames.values():
        expected = frame["a"] / frame["b"].clip(lower=0.1)
        pd.testing.assert_series_equal(frame["ratio"], expected, check_names=False)


def test_the_proposer_only_ever_sees_train(cfg, dataset, fake_llm):
    run_discovery_stage(cfg, dataset)
    prompt = fake_llm[0]["prompt"]
    # a handful of test-only values must not appear in the prompt
    test_values = dataset.split("test")["a"].head(20).round(3).astype(str)
    leaked = [v for v in test_values if v in prompt]
    assert not leaked, f"test rows leaked into the prompt: {leaked[:3]}"


def test_the_prompt_asks_for_the_batch_size(cfg, dataset, fake_llm):
    run_discovery_stage(cfg, dataset)
    assert "3 additive columns" in fake_llm[0]["prompt"]


def test_the_stopping_reason_is_recorded(cfg, dataset, fake_llm):
    out = run_discovery_stage(cfg, dataset)
    assert "max_rounds reached (1)" in out.stopped_because
    assert out.summary()["rounds"] == 1


def test_an_unknown_strategy_is_rejected(dataset):
    cfg = Config.from_dict({"discovery": {"enabled": True, "task_description": "x",
                                          "strategy": "nope"}})
    with pytest.raises(ValueError, match="Unknown discovery strategy"):
        run_discovery_stage(cfg, dataset)


def test_capture_rate_metric_is_explained_in_the_prompt(dataset, fake_llm):
    cfg = Config.from_dict({
        "run": {"seed": 42},
        "discovery": {"enabled": True, "task_description": "x", "batch_size": 1,
                      "metric": "capture_rate", "capture_percent": 0.05,
                      "sample_size": 400, "screen_boost_rounds": 20},
        "model": {"params": {"eta": 0.1, "max_depth": 3, "objective": "binary:logistic"}},
    })
    run_discovery_stage(cfg, dataset)
    prompt = fake_llm[0]["prompt"]
    assert "capture rate at the top 5%" in prompt
    assert "highest-scoring 5% of rows" in prompt


def test_prompts_and_replies_are_saved_for_audit(cfg, dataset, fake_llm, tmp_path):
    run_discovery_stage(cfg, dataset, output_dir=tmp_path)
    saved = tmp_path / "discovery" / "round_001"
    assert (saved / "prompt.txt").exists() and (saved / "reply.txt").exists()
    assert "additive columns" in (saved / "prompt.txt").read_text()


def test_the_result_hands_off_to_the_validation_pipeline(cfg, dataset, fake_llm):
    """The seam: stages 1-5 must accept the widened dataset unchanged."""
    from validation.stages.modeling import build_variants

    out = run_discovery_stage(cfg, dataset)
    variants = build_variants(out.dataset.base_features, out.dataset.new_features,
                              ["base", "leave_one_in"])
    names = [v.name for v in variants]
    assert "base" in names
    for feature in out.kept_features:
        assert f"loi__{feature}" in names        # one variant per discovered feature
