"""Stage 0.5 end to end: propose -> screen -> materialise -> validation stages."""
import numpy as np
import pandas as pd
import pytest

from discovery.stage import DiscoveryResult, run_discovery_stage
from preprocessing import build_shot_batches
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
    offset = 0
    for split, n in (("train", 1500), ("valid", 600), ("test", 600)):
        a = rng.uniform(1, 20, n); b = rng.uniform(1, 20, n); noise = rng.normal(size=n)
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(1.5 * (a / b) - 3.0 + 0.4 * noise)))).astype(int)
        frames[split] = pd.DataFrame({"row_id": np.arange(n) + offset,
                                      "y": y, "a": a, "b": b, "noise": noise})
        offset += n
    return Dataset(frames=frames, target="y",
                   base_features=["a", "b", "noise"], new_features=[])


@pytest.fixture
def few_shot(tmp_path, dataset):
    """What a prepare step writes: example rows clustered on train, as full rows."""
    batches = build_shot_batches(dataset.split("train"), "y",
                                 columns=["a", "b", "noise"], shots=8)
    path = tmp_path / "few_shot.csv"
    pd.concat([b.assign(batch=i) for i, b in enumerate(batches)]).to_csv(path, index=False)
    return str(path)


def _config(few_shot, **discovery):
    return Config.from_dict({
        "run": {"seed": 42},
        "data": {"target": "y", "id_cols": ["row_id"]},
        "discovery": {"enabled": True, "task_description": "synthetic; the signal is in a/b",
                      "few_shot_path": few_shot, **discovery},
        "model": {"params": {"eta": 0.1, "max_depth": 3, "objective": "binary:logistic"}},
    })


@pytest.fixture
def cfg(few_shot):
    return _config(few_shot, strategy="caafe", batch_size=3, max_rounds=1,
                   screen_boost_rounds=40, metric="adj_gini", min_delta=0.0)


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


@pytest.fixture
def zero_on_test(dataset):
    """b >= 1 everywhere except one test row, which the screen never sees."""
    frames = {split: frame.copy() for split, frame in dataset.frames.items()}
    frames["test"].loc[0, "b"] = 0.0
    return Dataset(frames=frames, target="y",
                   base_features=["a", "b", "noise"], new_features=[])


def test_a_feature_that_breaks_on_another_split_is_dropped(zero_on_test):
    from discovery.loop import Candidate
    from discovery.screen import ScreenResult
    from discovery.stage import _materialise

    def passed(name, code):
        return Candidate(round_index=1, code=code, feature_name=name,
                         screen=ScreenResult(feature_name=name, ok=True))

    kept = [
        passed("safe", 'df["safe"] = df["a"] / df["b"].clip(lower=0.1)'),
        passed("masked", 'df["masked"] = df["a"] / df["b"].where(df["b"] != 0)'),
        passed("inf", 'df["inf"] = df["a"] / df["b"]'),
        passed("spike", 'df["spike"] = df["a"] / (df["b"] + 1e-6)'),
        passed("row_id", 'df["row_id"] = df["a"] * 2'),   # clobbers an id column
    ]
    out, dropped = _materialise(zero_on_test, kept, spike_factor=1000.0)

    assert out.new_features == ["safe", "masked"]
    assert out.meta["discovered_features"] == ["safe", "masked"]
    assert [c.feature_name for c, _ in dropped] == ["inf", "spike", "row_id"]
    for split, frame in out.frames.items():
        assert np.isfinite(frame["safe"]).all()
        assert not {"inf", "spike"} & set(frame.columns), split
        pd.testing.assert_series_equal(frame["row_id"], zero_on_test.frames[split]["row_id"])
    # the undefined row is missing, not rejected
    assert out.frames["test"]["masked"].isna().sum() == 1
    assert out.frames["train"]["masked"].notna().all()


def test_a_dropped_feature_is_recorded_as_rejected(zero_on_test, few_shot, monkeypatch):
    """Nothing downstream may still count a feature validation never received."""
    reply = '''```python
# ('A over B', 'ratio of a to b')
# Usefulness: relative magnitude drives the outcome
df["ratio"] = df["a"] / df["b"]
```end
'''
    monkeypatch.setattr("discovery.strategies.base.call_llm",
                        lambda prompt, **kw: (reply, {"backend": "stub"}))
    cfg = _config(few_shot, strategy="caafe", batch_size=1, max_rounds=1,
                  screen_boost_rounds=40, metric="adj_gini")

    out = run_discovery_stage(cfg, zero_on_test)

    assert out.kept_features == []
    assert out.dataset.new_features == []
    assert all("ratio" not in f.columns for f in out.dataset.frames.values())
    record = out.to_frame().iloc[0]
    assert record["feature_name"] == "ratio"
    assert record["error"].startswith("Dropped on the full splits")
    assert "test data" in record["error"]
    assert out.rounds[0]["kept"] == 0 and out.rounds[0]["rejected"] == 1
    assert out.summary()["kept"] == 0


def test_the_proposer_only_ever_sees_train(cfg, dataset, fake_llm):
    run_discovery_stage(cfg, dataset)
    prompt = fake_llm[0]["prompt"]
    # a handful of test-only values must not appear in the prompt
    test_values = dataset.split("test")["a"].head(20).round(3).astype(str)
    leaked = [v for v in test_values if v in prompt]
    assert not leaked, f"test rows leaked into the prompt: {leaked[:3]}"


def test_the_prompt_shows_the_precomputed_rows(cfg, dataset, few_shot, fake_llm):
    run_discovery_stage(cfg, dataset)
    prompt = fake_llm[0]["prompt"]
    ids = pd.read_csv(few_shot)["row_id"]
    shown = dataset.split("train").set_index("row_id").loc[ids, "a"].round(1).astype(str)
    assert any(value in prompt for value in shown)


def test_a_stale_few_shot_file_is_refused(cfg, dataset, tmp_path, fake_llm):
    stale = tmp_path / "stale.csv"
    pd.DataFrame({"batch": [0], "row_id": [999_999]}).to_csv(stale, index=False)
    cfg.discovery.few_shot_path = str(stale)
    with pytest.raises(ValueError, match="not in train"):
        run_discovery_stage(cfg, dataset)


def test_the_task_context_file_reaches_the_prompt(cfg, dataset, tmp_path, fake_llm):
    """A page of domain background lives in a file; the run reads it in."""
    path = tmp_path / "task_context.md"
    path.write_text("Ratios of a to b are the known driver.", encoding="utf-8")
    cfg.discovery.task_context_path = str(path)
    run_discovery_stage(cfg, dataset)
    prompt = fake_llm[-1]["prompt"]
    assert "Domain context for this task" in prompt
    assert "Ratios of a to b are the known driver." in prompt


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


def test_capture_rate_metric_is_explained_in_the_prompt(dataset, few_shot, fake_llm):
    cfg = _config(few_shot, batch_size=1, metric="capture_rate", capture_percent=0.05,
                  screen_boost_rounds=20)
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


# --------------------------------------------------------------------------- #
# the screen's rows: the splits themselves, or rows the prepare step sampled
# --------------------------------------------------------------------------- #
COLUMNS = ["y", "a", "b", "noise"]


def _use_files(cfg, tmp_path, train_rows, valid_rows):
    """Screen samples written as a prepare step writes them: full rows."""
    cfg.discovery.screen_data = "files"
    cfg.discovery.screen_paths = {}
    for part, rows in (("train", train_rows), ("valid", valid_rows)):
        path = tmp_path / f"screen_{part}.csv"
        rows.to_csv(path, index=False)
        cfg.discovery.screen_paths[part] = str(path)


def test_the_screen_fits_on_train_and_scores_on_valid_by_default(cfg, dataset):
    from discovery.stage import _screen_frames

    train, valid, _ = _screen_frames(cfg, dataset)
    pd.testing.assert_frame_equal(train, dataset.split("train")[COLUMNS])
    pd.testing.assert_frame_equal(valid, dataset.split("valid")[COLUMNS])


def test_the_screen_can_use_sample_files(cfg, dataset, tmp_path):
    from discovery.stage import _screen_frames

    _use_files(cfg, tmp_path, dataset.split("train").iloc[:400],
               dataset.split("valid").iloc[:200])
    train, valid, _ = _screen_frames(cfg, dataset)
    pd.testing.assert_frame_equal(
        train, dataset.split("train")[COLUMNS].iloc[:400].reset_index(drop=True))
    pd.testing.assert_frame_equal(
        valid, dataset.split("valid")[COLUMNS].iloc[:200].reset_index(drop=True))


def test_sample_files_are_cleaned_like_the_splits(cfg, dataset, tmp_path):
    from discovery.stage import _screen_frames

    rows = dataset.split("train").iloc[:400].copy()
    rows.loc[rows.index[0], "a"] = -9999          # a sentinel, as a raw extract has
    _use_files(cfg, tmp_path, rows, dataset.split("valid").iloc[:200])
    train, _, _ = _screen_frames(cfg, dataset)
    assert np.isnan(train.loc[0, "a"])


def test_a_column_the_files_lack_is_joined_from_the_splits(cfg, dataset, tmp_path):
    """A column the dataset derives - a missing indicator, say - is not in the file."""
    from discovery.stage import _screen_frames

    frames = {name: frame.assign(carried=frame["a"] * 2)
              for name, frame in dataset.frames.items()}
    widened = Dataset(frames=frames, target="y",
                      base_features=["a", "b", "noise", "carried"], new_features=[])
    _use_files(cfg, tmp_path, dataset.split("train").iloc[:400],     # no 'carried'
               dataset.split("valid").iloc[:200])
    train, valid, _ = _screen_frames(cfg, widened)
    np.testing.assert_allclose(train["carried"], train["a"] * 2)
    np.testing.assert_allclose(valid["carried"], valid["a"] * 2)


def test_a_run_screens_on_the_sample_files(cfg, dataset, tmp_path, fake_llm):
    _use_files(cfg, tmp_path, dataset.split("train").iloc[:600],
               dataset.split("valid").iloc[:300])
    out = run_discovery_stage(cfg, dataset)
    assert out.kept_features, out.summary()


def test_the_screen_never_sees_test_rows(cfg, dataset, tmp_path):
    from discovery.stage import _screen_frames

    _use_files(cfg, tmp_path, dataset.split("train").iloc[:400],
               dataset.split("test").iloc[:100])
    with pytest.raises(ValueError, match="test rows"):
        _screen_frames(cfg, dataset)


def test_the_screen_never_scores_on_rows_it_fit(cfg, dataset, tmp_path):
    from discovery.stage import _screen_frames

    rows = dataset.split("train").iloc[:400]
    _use_files(cfg, tmp_path, rows, rows.iloc[:50])
    with pytest.raises(ValueError, match="in both screen files"):
        _screen_frames(cfg, dataset)


# --------------------------------------------------------------------------- #
# across runs: the history of verdicts, and the rotation through example rows
# --------------------------------------------------------------------------- #
EARLIER = {"run": "earlier", "round": 3, "feature_name": "old_idea",
           "code": 'df["old_idea"] = df["a"] + df["b"]', "base_score": 0.5,
           "candidate_score": 0.51, "delta": 0.01, "outcome": "rejected",
           "failed_at": "gini gain", "reason": "gini gain +0.0010 on test, below +0.0050"}


def _history(tmp_path, rows):
    path = tmp_path / "history.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_earlier_verdicts_and_their_reasons_reach_the_prompt(cfg, dataset, tmp_path, fake_llm):
    cfg.discovery.history_path = _history(tmp_path, [EARLIER])
    run_discovery_stage(cfg, dataset)
    prompt = fake_llm[0]["prompt"]
    assert 'df["old_idea"]' in prompt
    assert "REJECTED at the gini gain gate - gini gain +0.0010 on test, below +0.0050" in prompt


def test_round_numbers_continue_after_the_history(cfg, dataset, tmp_path, fake_llm):
    cfg.discovery.history_path = _history(tmp_path, [EARLIER])
    out = run_discovery_stage(cfg, dataset, output_dir=tmp_path / "run")
    assert {r["round"] for r in out.records} == {4}
    assert (tmp_path / "run" / "discovery" / "round_004").exists()


def test_an_earlier_name_is_not_reused(cfg, dataset, tmp_path, fake_llm):
    cfg.discovery.history_path = _history(tmp_path, [{**EARLIER, "feature_name": "ratio"}])
    out = run_discovery_stage(cfg, dataset)
    # refused before its name is recorded, so find it by the message
    errors = [r.get("error") or "" for r in out.records]
    assert any("'ratio' was already proposed in an earlier round" in e for e in errors)


@pytest.fixture
def three_batches(tmp_path, dataset):
    batches = build_shot_batches(dataset.split("train"), "y",
                                 columns=["a", "b", "noise"], shots=8, batches=3)
    path = tmp_path / "few_shot_3.csv"
    pd.concat([b.assign(batch=i) for i, b in enumerate(batches)]).to_csv(path, index=False)
    return str(path)


def test_the_example_rows_keep_rotating_across_runs(cfg, dataset, tmp_path,
                                                    three_batches, fake_llm):
    def shown(prompt):
        return [l for l in prompt.splitlines()
                if l.startswith("| r") and not l.startswith("| row")]

    cfg.discovery.few_shot_path = three_batches
    run_discovery_stage(cfg, dataset)                            # round 1: batch 0
    first = fake_llm[-1]["prompt"]
    cfg.discovery.history_path = _history(tmp_path, [{**EARLIER, "round": 1}])
    run_discovery_stage(cfg, dataset)                            # round 2: batch 1
    second = fake_llm[-1]["prompt"]
    assert shown(first) and shown(second) and shown(first) != shown(second)


def test_the_history_records_each_proposals_verdict_and_reason(tmp_path):
    from discovery.stage import _read_history, append_history

    discovered = DiscoveryResult(records=[
        {"round": 1, "feature_name": "kept_one", "code": "c1", "delta": 0.02},
        {"round": 1, "feature_name": "weak_one", "code": "c2", "delta": 0.01},
        {"round": 1, "feature_name": "broken", "code": "c3", "error": "Forbidden syntax"},
    ])
    verdicts = pd.DataFrame({"feature": ["kept_one", "weak_one"], "verdict": ["PASS", "FAIL"],
                             "failed_at": ["", "gini gain"],
                             "reason": ["", "gini gain +0.0010 on test"]})
    path = tmp_path / "history.csv"
    assert append_history(path, discovered, verdicts, run="r1") == 3
    append_history(path, discovered, verdicts, run="r2")        # appends, never overwrites

    rows = pd.DataFrame(_read_history(path))
    assert list(rows["run"]) == ["r1"] * 3 + ["r2"] * 3
    first = rows[rows["run"] == "r1"].set_index("feature_name")
    assert first.loc["kept_one", "outcome"] == "accepted"
    assert (first.loc["weak_one", "outcome"], first.loc["weak_one", "failed_at"],
            first.loc["weak_one", "reason"]) == ("rejected", "gini gain", "gini gain +0.0010 on test")
    assert (first.loc["broken", "failed_at"], first.loc["broken", "reason"]) == \
        ("screen", "Forbidden syntax")


def test_screen_files_need_a_train_and_a_valid_path():
    cfg = Config.from_dict({"discovery": {
        "enabled": True, "task_description": "x", "few_shot_path": "few_shot.csv",
        "screen_data": "files", "screen_paths": {"train": "screen_train.csv"}}})
    with pytest.raises(ValueError, match="screen_paths"):
        cfg.validate()
