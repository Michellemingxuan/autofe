"""The shared parts of a data/<dataset>/prepare.py script.

The checks matter more than the mechanics here. A prepare script that writes a
table disagreeing with its config produces a run whose verdict is about a
feature set nobody chose, and nothing downstream notices - so most of these
tests are about refusing to write rather than about writing correctly.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from preprocessing import (
    fetch_archive,
    resolve_path,
    sanitize,
    sanitize_columns,
    write_modeling_table,
)


# --------------------------------------------------------------------------- #
# a config stand-in: only the fields write_modeling_table reads
# --------------------------------------------------------------------------- #
@dataclass
class FakeData:
    path: str
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
    return FakeConfig(
        data=FakeData(path=str(tmp_path / "modeling.parquet"), **kw),
        features=features,
        discovery=discovery,
    )


@pytest.fixture
def frame():
    return pd.DataFrame({
        "row_id": [0, 1, 2, 3],
        "y": [0, 1, 0, 1],
        "incumbent": [1.0, 2.0, 3.0, 4.0],
        "cand_a": [0.5, 0.6, 0.7, 0.8],
    })


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
# the checks that stop a bad table being written
# --------------------------------------------------------------------------- #
def test_a_candidate_the_script_did_not_build_is_refused(frame, tmp_path):
    """The failure this exists for: the run would silently evaluate nothing."""
    cfg = cfg_for(tmp_path, new=["cand_a", "cand_never_built"])
    with pytest.raises(KeyError, match="cand_never_built"):
        write_modeling_table(cfg, frame, root=tmp_path, id_col="row_id",
                             mapping=_mapping(frame))
    assert not (tmp_path / "modeling.parquet").exists()


def test_a_missing_target_is_refused(frame, tmp_path):
    cfg = cfg_for(tmp_path, target="not_a_column")
    with pytest.raises(KeyError, match="not_a_column"):
        write_modeling_table(cfg, frame, root=tmp_path, id_col="row_id",
                             mapping=_mapping(frame))


def test_a_non_binary_target_is_refused(tmp_path):
    frame = pd.DataFrame({"row_id": [0, 1, 2], "y": [0, 1, 2],
                          "incumbent": [1.0, 2.0, 3.0], "cand_a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not binary"):
        write_modeling_table(cfg_for(tmp_path), frame, root=tmp_path,
                             id_col="row_id", mapping=_mapping(frame))


def test_nothing_to_judge_is_refused(frame, tmp_path):
    """No declared candidates and no discovery means a run with no question."""
    cfg = cfg_for(tmp_path, new=[])
    with pytest.raises(ValueError, match="nothing to judge"):
        write_modeling_table(cfg, frame, root=tmp_path, id_col="row_id",
                             mapping=_mapping(frame))


def test_no_candidates_is_allowed_when_discovery_will_supply_them(frame, tmp_path):
    cfg = cfg_for(tmp_path, new=[], discovery=True)
    report = write_modeling_table(cfg, frame, root=tmp_path, id_col="row_id",
                                  mapping=_mapping(frame))
    assert report.candidates == []


# --------------------------------------------------------------------------- #
# what gets written
# --------------------------------------------------------------------------- #
def _mapping(frame):
    return pd.DataFrame({"original": [c.upper() for c in frame.columns],
                         "column": list(frame.columns)})


def test_the_three_artifacts_are_written(frame, tmp_path):
    report = write_modeling_table(cfg_for(tmp_path), frame, root=tmp_path,
                                  id_col="row_id", mapping=_mapping(frame))
    assert pd.read_parquet(report.paths["table"]).equals(frame)
    assert report.paths["mapping"].exists()
    written = json.loads(report.paths["descriptions"].read_text())
    # The id and target are not features, so they carry no description.
    assert set(written) == {"incumbent", "cand_a"}
    assert written["cand_a"] == "CAND_A"          # the original header


def test_the_report_counts_incumbents_and_candidates_apart(frame, tmp_path):
    report = write_modeling_table(cfg_for(tmp_path), frame, root=tmp_path,
                                  id_col="row_id", mapping=_mapping(frame))
    assert report.rows == 4
    assert report.base_features == 1               # incumbent
    assert report.candidates == ["cand_a"]
    assert report.positives == 2
    assert report.positive_rate == 0.5


def test_new_prefix_candidates_are_picked_up(frame, tmp_path):
    cfg = cfg_for(tmp_path, new=[], new_prefix="cand_")
    report = write_modeling_table(cfg, frame, root=tmp_path, id_col="row_id",
                                  mapping=_mapping(frame))
    assert report.candidates == ["cand_a"]


def test_explicit_descriptions_win_over_the_mapping(frame, tmp_path):
    report = write_modeling_table(
        cfg_for(tmp_path), frame, root=tmp_path, id_col="row_id",
        mapping=_mapping(frame),
        descriptions={"incumbent": "an existing ratio", "cand_a": "the proposal"},
        extra_descriptions={"cand_a": "corrected"},
    )
    written = json.loads(report.paths["descriptions"].read_text())
    assert written == {"incumbent": "an existing ratio", "cand_a": "corrected"}


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
    """A re-run of a prepare script should do no network I/O at all."""
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
    assert resolve_path("data/x/modeling.parquet", tmp_path) == \
        tmp_path / "data/x/modeling.parquet"


def test_an_absolute_config_path_is_left_alone(tmp_path):
    absolute = tmp_path / "elsewhere.parquet"
    assert resolve_path(absolute, Path("/ignored")) == absolute
