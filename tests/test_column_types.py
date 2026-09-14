"""How a config says which columns are measurements and which are codes.

This decides what the proposer is shown: a range, or a list of levels. Getting
it wrong is not a cosmetic problem - a column shown as a range invites
arithmetic, so an ECG finding presented as continuous is an invitation to
compute the average of a diagnostic code.

Two declarations exist because the natural one differs per table, and the
important case is the one that looks like a default but is not: an empty
`continuous_columns` means "nothing here is continuous", which is a real
statement about a table of binary flags, not an absence of one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from discovery.stage import _categorical_columns


@dataclass
class FakeDiscovery:
    categorical_columns: list = field(default_factory=list)
    continuous_columns: list | None = None


BASE = ["age", "bp", "ecg_rhythm", "drug_given"]


def test_not_declared_leaves_everything_continuous():
    """The default for a table of ratios, where a range is the useful view."""
    assert _categorical_columns(FakeDiscovery(), BASE) == []


def test_an_empty_continuous_list_makes_everything_categorical():
    """The case that breaks if [] is read as "not declared": all-binary tables."""
    cfg = FakeDiscovery(continuous_columns=[])
    assert _categorical_columns(cfg, BASE) == BASE


def test_declaring_the_continuous_columns_makes_the_rest_categorical():
    cfg = FakeDiscovery(continuous_columns=["age", "bp"])
    assert _categorical_columns(cfg, BASE) == ["ecg_rhythm", "drug_given"]


def test_declaring_the_categorical_columns_directly_still_works():
    cfg = FakeDiscovery(categorical_columns=["ecg_rhythm"])
    assert _categorical_columns(cfg, BASE) == ["ecg_rhythm"]


def test_a_categorical_name_not_in_the_table_is_ignored():
    cfg = FakeDiscovery(categorical_columns=["ecg_rhythm", "not_a_column"])
    assert _categorical_columns(cfg, BASE) == ["ecg_rhythm"]


def test_a_continuous_candidate_column_is_not_reported_as_a_typo(caplog):
    """
    A config declares the type of its candidate columns too.

    Discovery only builds over the incumbents, so a candidate name is absent
    from `base_features` while being a perfectly correct declaration - warning
    about it would train the reader to ignore the warning.
    """
    cfg = FakeDiscovery(continuous_columns=["age", "k_blood"])
    with caplog.at_level("WARNING"):
        result = _categorical_columns(cfg, BASE, known=[*BASE, "k_blood"])
    assert result == ["bp", "ecg_rhythm", "drug_given"]
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_a_continuous_name_in_no_list_at_all_is_reported(caplog):
    cfg = FakeDiscovery(continuous_columns=["age", "typo_here"])
    with caplog.at_level("WARNING"):
        _categorical_columns(cfg, BASE, known=BASE)
    assert any("typo_here" in r.getMessage() for r in caplog.records)
