"""The shared parts of a dataset's prepare step.

The checks matter more than the mechanics here. A prepare step that writes tables
disagreeing with its config produces a run whose verdict is about a feature set
nobody chose, and nothing downstream notices - so most of these tests are about
refusing to write rather than about writing correctly.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    SPLITS,
    declared_candidates,
    fetch_archive,
    resolve_path,
    sanitize,
    sanitize_columns,
    stratified_split,
    write_splits,
)


# --------------------------------------------------------------------------- #
# a config stand-in: only the fields write_splits reads
# --------------------------------------------------------------------------- #
@dataclass
class FakeData:
    paths: dict
    target: str = "y"
    id_cols: list = field(default_factory=lambda: ["row_id"])


@dataclass
class FakeFeatures:
    new: list = field(default_factory=list)
    new_prefix: str = ""


@dataclass
class FakeDiscovery:
    enabled: bool = False


@dataclass
class FakeConfig:
    data: FakeData
    features: FakeFeatures = field(default_factory=FakeFeatures)
    discovery: FakeDiscovery = field(default_factory=FakeDiscovery)


def cfg_for(tmp_path, **kw):
    features = FakeFeatures(new=list(kw.pop("new", ["cand_a"])),
                            new_prefix=kw.pop("new_prefix", ""))
    discovery = FakeDiscovery(enabled=kw.pop("discovery", False))
    paths = {name: str(tmp_path / f"{name}.csv") for name in SPLITS}
    return FakeConfig(data=FakeData(paths=paths, **kw), features=features,
                      discovery=discovery)


@pytest.fixture
def frames():
    frame = pd.DataFrame({
        "row_id": range(12),
        "y": [0, 1] * 6,
        "incumbent": np.arange(12.0),
        "cand_a": np.arange(12) / 10,
    })
    return {"train": frame.iloc[:6].reset_index(drop=True),
            "valid": frame.iloc[6:9].reset_index(drop=True),
            "test": frame.iloc[9:].reset_index(drop=True)}


def _mapping(frame):
    return pd.DataFrame({"original": [c.upper() for c in frame.columns],
                         "column": list(frame.columns)})


def _write(cfg, frames, tmp_path, **kw):
    kw.setdefault("mapping", _mapping(frames["train"]))
    return write_splits(cfg, frames, root=tmp_path, **kw)


# --------------------------------------------------------------------------- #
# sanitize
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw, expected", [
    ("Bankrupt?", "bankrupt"),
    (" Net Income Flag ", "net_income_flag"),
    ("ROA(C) before interest", "roa_c_before_interest"),
    ("android.permission.GET_ACCOUNTS", "android_permission_get_accounts"),
    ("Cash Flow to Liability", "cash_flow_to_liability"),
    ("Research & Development", "research_and_development"),
    ("Operating Profit Rate %", "operating_profit_rate_pct"),
])
def test_sanitize_produces_a_usable_identifier(raw, expected):
    assert sanitize(raw) == expected


def test_sanitize_columns_records_the_original_name():
    raw = pd.DataFrame({"Bankrupt?": [0], "Cash Flow Rate": [1.0]})
    renamed, mapping = sanitize_columns(raw)
    assert list(renamed.columns) == ["bankrupt", "cash_flow_rate"]
    assert dict(zip(mapping["column"], mapping["original"])) == {
        "bankrupt": "Bankrupt?", "cash_flow_rate": "Cash Flow Rate"}


def test_a_name_collision_is_refused_rather_than_dropping_a_column():
    """Two headers sanitizing to one name would silently lose a column."""
    raw = pd.DataFrame({"Total Assets": [1], "total assets": [2]})
    with pytest.raises(ValueError, match="duplicate column name"):
        sanitize_columns(raw)


# --------------------------------------------------------------------------- #
# the split
# --------------------------------------------------------------------------- #
@pytest.fixture
def rare_event_frame():
    rng = np.random.default_rng(0)
    n = 5000
    return pd.DataFrame({"row_id": np.arange(n), "x": rng.normal(size=n),
                         "y": (rng.random(n) < 0.03).astype(int)})


def test_the_split_keeps_the_event_rate_in_every_part(rare_event_frame):
    parts = stratified_split(rare_event_frame, "y", valid_size=0.2, test_size=0.2, seed=1)
    overall = rare_event_frame["y"].mean()
    for name, part in parts.items():
        assert part["y"].mean() == pytest.approx(overall, abs=0.003), name
    assert len(parts["test"]) == pytest.approx(0.2 * len(rare_event_frame), abs=2)
    assert len(parts["valid"]) == pytest.approx(0.2 * len(rare_event_frame), abs=2)


def test_the_split_is_disjoint_complete_and_reproducible(rare_event_frame):
    parts = stratified_split(rare_event_frame, "y", seed=1)
    ids = {name: set(part["row_id"]) for name, part in parts.items()}
    assert not ids["train"] & ids["valid"] and not ids["train"] & ids["test"] \
        and not ids["valid"] & ids["test"]
    assert set().union(*ids.values()) == set(rare_event_frame["row_id"])

    again = stratified_split(rare_event_frame, "y", seed=1)
    other = stratified_split(rare_event_frame, "y", seed=2)
    assert all(parts[n]["row_id"].equals(again[n]["row_id"]) for n in SPLITS)
    assert not parts["test"]["row_id"].equals(other["test"]["row_id"])


# --------------------------------------------------------------------------- #
# the checks that stop bad tables being written
# --------------------------------------------------------------------------- #
def test_a_candidate_that_was_not_built_is_refused(frames, tmp_path):
    """The failure this exists for: the run would silently evaluate nothing."""
    cfg = cfg_for(tmp_path, new=["cand_a", "cand_never_built"])
    with pytest.raises(KeyError, match="cand_never_built"):
        _write(cfg, frames, tmp_path)
    assert not (tmp_path / "train.csv").exists()


def test_a_missing_target_is_refused(frames, tmp_path):
    with pytest.raises(KeyError, match="not_a_column"):
        _write(cfg_for(tmp_path, target="not_a_column"), frames, tmp_path)


def test_a_non_binary_target_is_refused(frames, tmp_path):
    frames["test"] = frames["test"].assign(y=[0, 1, 2])
    with pytest.raises(ValueError, match="not binary"):
        _write(cfg_for(tmp_path), frames, tmp_path)


def test_nothing_to_judge_is_refused(frames, tmp_path):
    """No declared candidates and no discovery means a run with no question."""
    with pytest.raises(ValueError, match="nothing to judge"):
        _write(cfg_for(tmp_path, new=[]), frames, tmp_path)


def test_no_candidates_is_allowed_when_discovery_will_supply_them(frames, tmp_path):
    written = _write(cfg_for(tmp_path, new=[], discovery=True), frames, tmp_path)
    assert written["train"].exists()


def test_an_id_in_two_splits_is_refused_as_leakage(frames, tmp_path):
    frames["test"].loc[0, "row_id"] = frames["train"].loc[0, "row_id"]
    with pytest.raises(ValueError, match="leakage"):
        _write(cfg_for(tmp_path), frames, tmp_path)
    assert not (tmp_path / "train.csv").exists()


def test_splits_with_different_columns_are_refused(frames, tmp_path):
    frames["valid"] = frames["valid"].drop(columns="incumbent")
    with pytest.raises(ValueError, match="incumbent"):
        _write(cfg_for(tmp_path), frames, tmp_path)


def test_a_config_without_a_path_for_every_split_is_refused(frames, tmp_path):
    cfg = cfg_for(tmp_path)
    del cfg.data.paths["test"]
    with pytest.raises(ValueError, match="data.paths"):
        _write(cfg, frames, tmp_path)


# --------------------------------------------------------------------------- #
# what gets written
# --------------------------------------------------------------------------- #
def test_the_tables_and_sidecars_are_written(frames, tmp_path):
    written = _write(cfg_for(tmp_path), frames, tmp_path)
    for name in SPLITS:
        pd.testing.assert_frame_equal(pd.read_csv(written[name]), frames[name])
    assert written["mapping"].exists()
    descriptions = json.loads(written["descriptions"].read_text())
    # The id and target are not features, so they carry no description.
    assert set(descriptions) == {"incumbent", "cand_a"}
    assert descriptions["cand_a"] == "CAND_A"          # the original header


def test_every_split_is_written_in_trains_column_order(frames, tmp_path):
    frames["test"] = frames["test"][list(reversed(frames["test"].columns))]
    written = _write(cfg_for(tmp_path), frames, tmp_path)
    assert list(pd.read_csv(written["test"]).columns) == list(frames["train"].columns)


def test_explicit_descriptions_win_over_the_mapping(frames, tmp_path):
    written = _write(
        cfg_for(tmp_path), frames, tmp_path,
        descriptions={"incumbent": "an existing ratio", "cand_a": "the proposal"},
        extra_descriptions={"cand_a": "corrected"},
    )
    descriptions = json.loads(written["descriptions"].read_text())
    assert descriptions == {"incumbent": "an existing ratio", "cand_a": "corrected"}


def test_new_prefix_candidates_are_picked_up(frames, tmp_path):
    cfg = cfg_for(tmp_path, new=[], new_prefix="cand_")
    assert declared_candidates(cfg, frames["train"]) == ["cand_a"]


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def test_fetch_archive_extracts_a_zip_member(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", "a,b\n1,2\n")
    _serve(monkeypatch, buffer.getvalue())

    path = fetch_archive("https://example.invalid/x.zip", tmp_path, "data.csv")
    assert path.read_text() == "a,b\n1,2\n"


def test_fetch_archive_handles_a_bare_file(tmp_path, monkeypatch):
    _serve(monkeypatch, b'{"variables": []}')
    path = fetch_archive("https://example.invalid/api", tmp_path, "variables.json")
    assert json.loads(path.read_text()) == {"variables": []}


def test_fetch_archive_does_not_download_twice(tmp_path, monkeypatch):
    """A re-run of a prepare step should do no network I/O at all."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", "cached\n")
    calls = _serve(monkeypatch, buffer.getvalue())

    fetch_archive("https://example.invalid/x.zip", tmp_path, "data.csv")
    fetch_archive("https://example.invalid/x.zip", tmp_path, "data.csv")
    assert calls == [1]


def test_a_missing_zip_member_names_what_the_archive_holds(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("actual.csv", "x\n")
    _serve(monkeypatch, buffer.getvalue())
    with pytest.raises(KeyError, match="actual.csv"):
        fetch_archive("https://example.invalid/x.zip", tmp_path, "expected.csv")


def _serve(monkeypatch, payload: bytes) -> list:
    """Stand in for urlopen, counting how many times it was called."""
    calls = []

    class Response:
        def read(self): return payload
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    def urlopen(url, timeout=None):
        calls.append(1)
        return Response()

    monkeypatch.setattr("preprocessing.dataset.urllib.request.urlopen", urlopen)
    return calls


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def test_a_relative_config_path_resolves_against_the_repo_root(tmp_path):
    assert resolve_path("data/x/train.csv", tmp_path) == tmp_path / "data/x/train.csv"


def test_an_absolute_config_path_is_left_alone(tmp_path):
    absolute = tmp_path / "elsewhere.csv"
    assert resolve_path(absolute, Path("/ignored")) == absolute
