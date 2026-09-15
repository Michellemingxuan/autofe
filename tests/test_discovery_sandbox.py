"""Sandbox and value guards: what generated code is allowed to do, and produce."""
import numpy as np
import pandas as pd
import pytest

from discovery import (
    CandidateError,
    apply_code,
    assigned_columns,
    check_finite,
    check_matrix_finite,
    check_scale,
    referenced_columns,
    validate_code,
    validate_single_column,
)


@pytest.fixture
def frame():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 0.0, 8.0]})


# --------------------------------------------------------------------------- #
# validate_code
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code", [
    'import os',
    'from os import path',
    'def f(): pass',
    'class C: pass',
    'with open("x") as f: pass',
    'try:\n    pass\nexcept Exception:\n    pass',
    'raise ValueError()',
    'f = lambda x: x',
    'del df',
    'df["x"] = eval("1")',
    'df["x"] = exec("1")',
    'df["x"] = open("f")',
    'df["x"] = __import__("os")',
    'df["x"] = df.__class__',
])
def test_forbidden_syntax_is_rejected(code):
    with pytest.raises(CandidateError):
        validate_code(code)


@pytest.mark.parametrize("code", [
    'df["x"] = df["a"] / df["b"].clip(lower=0.1)',
    'df["x"] = np.log1p(df["a"].abs())',
    'df["x"] = (df["a"] > 2).astype(int)',
    'df["x"] = df["a"].where(df["b"] > 0, 0.0)',
])
def test_ordinary_feature_expressions_are_allowed(code):
    validate_code(code)


# --------------------------------------------------------------------------- #
# column analysis
# --------------------------------------------------------------------------- #
def test_reads_and_writes_are_told_apart():
    code = 'df["ratio"] = df["a"] / df["b"]'
    assert assigned_columns(code) == ["ratio"]
    assert referenced_columns(code) == ["a", "b"]


def test_column_analysis_survives_unparseable_code():
    assert assigned_columns("df[ = (((") == []
    assert referenced_columns("df[ = (((") == []


def test_exactly_one_new_column_is_required():
    assert validate_single_column('df["x"] = df["a"]', ["a", "b"]) == "x"

    with pytest.raises(CandidateError, match="exactly one"):
        validate_single_column('df["x"] = df["a"]\ndf["y"] = df["b"]', ["a"])
    with pytest.raises(CandidateError, match="exactly one"):
        validate_single_column('z = df["a"]', ["a"])
    with pytest.raises(CandidateError, match="already exists"):
        validate_single_column('df["a"] = df["b"]', ["a", "b"])
    with pytest.raises(CandidateError, match="earlier round"):
        validate_single_column('df["x"] = df["a"]', ["a"], reserved_names=["x"])


# --------------------------------------------------------------------------- #
# apply_code
# --------------------------------------------------------------------------- #
def test_apply_code_adds_the_column_without_mutating_the_input(frame):
    before = frame.copy()
    out = apply_code(frame, ['df["r"] = df["a"] * 2'])
    assert list(out["r"]) == [2.0, 4.0, 6.0, 8.0]
    pd.testing.assert_frame_equal(frame, before)


def test_apply_code_applies_blocks_in_order(frame):
    out = apply_code(frame, ['df["x"] = df["a"] + 1', 'df["y"] = df["x"] * 10'])
    assert list(out["y"]) == [20.0, 30.0, 40.0, 50.0]


@pytest.mark.parametrize("code", [
    # all seen in real proposals; emptying __builtins__ rejected every one
    'df["x"] = df["a"] / np.maximum(df["b"], np.finfo(float).eps)',
    'df["x"] = df["a"].astype(float) * 2',
    'df["x"] = abs(df["a"] - df["b"])',
    'df["x"] = round(df["a"], 2)',
    'df["x"] = df["a"].clip(lower=float(0.1))',
    'df["x"] = int(1) + df["a"]',
])
def test_safe_builtins_are_available_to_feature_code(frame, code):
    out = apply_code(frame, [code])
    assert "x" in out.columns


@pytest.mark.parametrize("code", [
    'df["x"] = open("f")',
    'df["x"] = eval("1")',
    'df["x"] = exec("1")',
    'df["x"] = getattr(df, "values")',
    'df["x"] = vars(df)',
    'df["x"] = __import__("os")',
])
def test_dangerous_builtins_stay_blocked(frame, code):
    with pytest.raises(CandidateError):
        apply_code(frame, [code])


def test_the_builtins_allowlist_excludes_the_filesystem_and_import_system(frame):
    from discovery.sandbox import _SAFE_BUILTINS
    forbidden = {"open", "eval", "exec", "compile", "__import__", "globals",
                 "locals", "input", "getattr", "setattr", "vars", "dir"}
    assert not (forbidden & set(_SAFE_BUILTINS))


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_infinite_values_are_rejected(frame):
    out = apply_code(frame, ['df["r"] = df["a"] / df["b"]'])   # b has a zero
    with pytest.raises(CandidateError, match="infinite"):
        check_finite(out, "r", "train")


def test_finite_column_passes(frame):
    out = apply_code(frame, ['df["r"] = df["a"] / df["b"].clip(lower=0.5)'])
    check_finite(out, "r", "train")


def test_nan_passes_because_it_means_missing(frame):
    out = apply_code(frame, ['df["r"] = df["a"] / df["b"].where(df["b"] != 0)'])
    assert out["r"].isna().sum() == 1
    check_finite(out, "r", "train")                  # must not raise


def test_non_numeric_values_are_rejected():
    frame = pd.DataFrame({"r": [1.0, "high", np.nan]}, dtype=object)
    with pytest.raises(CandidateError, match="non-numeric"):
        check_finite(frame, "r", "train")


def test_missing_column_is_reported(frame):
    with pytest.raises(CandidateError, match="missing"):
        check_finite(frame, "nope", "train")


def test_epsilon_guard_spike_is_caught():
    # the real failure: one zero denominator, "guarded" with 1e-6
    values = pd.DataFrame({"f": list(np.linspace(0.1, 20, 200)) + [1.58e6]})
    with pytest.raises(CandidateError, match="spikes"):
        check_scale({"valid": values}, "f")


def test_heavy_tail_is_not_a_spike():
    rng = np.random.default_rng(0)
    values = pd.DataFrame({"f": rng.lognormal(0, 2.0, 5000)})
    check_scale({"valid": values}, "f")          # must not raise


def test_spike_is_caught_on_any_split():
    clean = pd.DataFrame({"f": np.linspace(0.1, 20, 200)})
    dirty = pd.DataFrame({"f": list(np.linspace(0.1, 20, 200)) + [1e9]})
    with pytest.raises(CandidateError, match="test data"):
        check_scale({"train": clean, "valid": clean, "test": dirty}, "f")


def test_constant_column_does_not_trip_the_spike_guard():
    check_scale({"train": pd.DataFrame({"f": np.zeros(100)})}, "f")


def test_matrix_guard_names_the_offending_column():
    bad = pd.DataFrame({"ok": [1.0, 2.0], "bad": [1.0, np.inf]})
    with pytest.raises(CandidateError, match="bad"):
        check_matrix_finite(bad, "train")


def test_matrix_guard_lets_missing_values_through():
    check_matrix_finite(pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]}), "train")
