"""The screen: cheap, honest about direction, and never raises on bad input."""
import numpy as np
import pandas as pd
import pytest

from discovery.prompt import build_prompt, describe_columns, extract_code, format_history, parse_candidate
from discovery.sandbox import CandidateError
from discovery.screen import Screener, build_sample
from validation.metrics import calc_adj_gini

PARAMS = {"eta": 0.1, "max_depth": 3, "objective": "binary:logistic"}


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 3000
    a = rng.uniform(1, 20, n)
    b = rng.uniform(1, 20, n)
    noise = rng.normal(size=n)
    # the signal lives in the RATIO, which no single column exposes
    logit = 1.5 * (a / b) - 3.0 + 0.4 * noise
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"class": y, "a": a, "b": b, "noise": noise})


@pytest.fixture
def screener(frame):
    sample = build_sample(frame, "class", size=800, seed=1)
    return Screener(sample, "class", ["a", "b", "noise"], PARAMS,
                    score=calc_adj_gini, num_boost_round=60)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def test_sample_is_class_balanced_by_default(frame):
    sample = build_sample(frame, "class", size=400, seed=1)
    counts = sample["class"].value_counts()
    assert abs(counts[0] - counts[1]) <= 1, counts.to_dict()


def test_unbalanced_sample_keeps_the_base_rate(frame):
    sample = build_sample(frame, "class", size=400, seed=1, balance=False)
    assert abs(sample["class"].mean() - frame["class"].mean()) < 0.06


def test_sampling_is_reproducible(frame):
    a = build_sample(frame, "class", size=300, seed=7)
    b = build_sample(frame, "class", size=300, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_sample_never_exceeds_available_rows(frame):
    small = frame.head(20)
    assert len(build_sample(small, "class", size=10_000, seed=1)) <= 20


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_the_screen_scores_on_held_out_rows(screener, frame):
    """In-sample the baseline saturates and the delta is meaningless; see screen.py."""
    assert 0.0 < screener.base_score < 0.95


def test_a_useful_feature_scores_above_the_baseline(screener):
    out = screener.evaluate('df["ratio"] = df["a"] / df["b"].clip(lower=0.1)')
    assert out.ok and out.error is None
    assert out.feature_name == "ratio"
    assert out.delta > 0, out.as_dict()


def test_a_constant_column_changes_nothing(screener):
    out = screener.evaluate('df["dead"] = df["noise"] * 0.0')
    assert out.ok
    assert abs(out.delta) < 1e-9, out.as_dict()


def test_the_baseline_is_fit_once_and_stays_fixed(screener):
    first = screener.evaluate('df["x"] = df["a"] + df["b"]')
    second = screener.evaluate('df["y"] = df["a"] - df["b"]')
    assert first.base_score == second.base_score == screener.base_score


def test_identical_candidates_score_identically(screener):
    code = 'df["r"] = df["a"] / df["b"].clip(lower=0.1)'
    assert screener.evaluate(code).delta == screener.evaluate(code).delta


# --------------------------------------------------------------------------- #
# rejections come back as text, not exceptions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code, expected", [
    ('import os\ndf["x"] = 1',                      "Forbidden syntax"),
    ('df["x"] = df["a"]\ndf["y"] = df["b"]',        "exactly one"),
    ('df["a"] = df["b"]',                           "already exists"),
    ('df["x"] = df["a"] / (df["b"] - df["b"])',     "non-finite"),
    ('df["x"] = (((',                               "SyntaxError"),
    ('df["x"] = df["nope"] * 2',                    "does not exist"),
])
def test_bad_candidates_return_an_error_message(screener, code, expected):
    out = screener.evaluate(code)
    assert not out.ok
    assert out.error and expected in out.error
    assert out.delta is None


def test_the_epsilon_guard_spike_is_rejected(frame):
    # one row with b == 0 -> "+ 1e-6" produces ~1e6
    poisoned = frame.copy()
    poisoned.loc[0, "b"] = 0.0
    sample = build_sample(poisoned, "class", size=800, seed=1)
    sample.loc[0, "b"] = 0.0                       # make sure it is in the sample
    screener = Screener(sample, "class", ["a", "b", "noise"], PARAMS,
                        score=calc_adj_gini, num_boost_round=40)
    out = screener.evaluate('df["r"] = df["a"] / (df["b"] + 1e-6)')
    assert not out.ok and "spikes" in out.error


def test_an_already_proposed_name_is_refused(screener):
    out = screener.evaluate('df["r"] = df["a"] + 1', reserved_names=["r"])
    assert not out.ok and "earlier round" in out.error


def test_result_serialises_for_the_run_record(screener):
    payload = screener.evaluate('df["r"] = df["a"] * 2').as_dict()
    assert set(payload) >= {"feature_name", "ok", "base_score", "candidate_score",
                            "delta", "error", "elapsed_seconds", "n_rows"}


# --------------------------------------------------------------------------- #
# prompt round trip
# --------------------------------------------------------------------------- #
def test_column_context_shows_types_ranges_and_values(frame):
    sample = frame.head(4)
    text = describe_columns(sample, ["a", "b"], {"a": "numerator"}, categorical=[])
    # the identifier leads, quoted exactly as it must be typed
    assert 'df["a"] (' in text and "numerator" in text
    assert "observed range=" in text and "Samples [" in text


def test_categorical_columns_list_their_values(frame):
    sample = pd.DataFrame({"flag": [0, 1, 1, 0]})
    text = describe_columns(sample, ["flag"], categorical=["flag"])
    assert "categorical; values seen=[0, 1]" in text


def test_indexing_by_description_is_answered_with_the_identifier(frame):
    """The failure that cost a whole run: df["Debt ratio %"] instead of df["X36"]."""
    sample = build_sample(frame, "class", size=400, seed=1)
    screener = Screener(sample, "class", ["a", "b", "noise"], PARAMS,
                        score=calc_adj_gini, num_boost_round=20,
                        column_aliases={"Total debt over net worth": "a"})
    out = screener.evaluate('df["x"] = df["Total debt over net worth"] * 2')
    assert not out.ok
    assert "is a description, not a key" in out.error
    assert 'df["a"]' in out.error


def test_the_prompt_spells_out_that_descriptions_are_not_keys():
    text = build_prompt("task", 'df["X36"] (float64) - Total debt/Total net worth',
                        "none", "adjusted Gini", redundancy_max_abs=0.95)
    assert "never by a column's description" in text
    assert 'df["X36"]`, not `df["Total debt/Total net worth"]' in text


def test_history_reports_scores_and_failures():
    text = format_history([
        {"round": 1, "code": 'df["x"] = 1', "base_score": 0.5,
         "candidate_score": 0.55, "delta": 0.05},
        {"round": 2, "code": 'df["y"] = 2', "error": "Forbidden syntax: Import"},
    ], "adjusted Gini")
    assert "Change (adjusted Gini): +0.0500" in text
    assert "rejected: Forbidden syntax" in text
    assert "not present in `df`" in text


def test_history_is_explicit_when_there_is_none():
    assert "No previous code blocks" in format_history([], "adjusted Gini")


def test_prompt_carries_the_pieces_and_the_epsilon_warning():
    text = build_prompt("predict bankruptcy", "a (float64)\nSamples [1, 2]",
                        "No previous code blocks or feedback are available.",
                        "adjusted Gini", already_proposed=["old_ratio"], n_rows=1234)
    assert "predict bankruptcy" in text
    assert "1,234 rows" in text
    assert "adjusted Gini" in text
    assert '"old_ratio"' in text and "do not propose them again" in text
    assert 'df["b"] + 1e-6' in text          # the trap is spelled out
    assert "```end" in text


def test_code_is_extracted_from_either_fence():
    assert extract_code('blah\n```python\ndf["x"] = 1\n```end\ntrailing') == 'df["x"] = 1'
    assert extract_code('```\ndf["x"] = 2\n```') == 'df["x"] = 2'
    assert extract_code('df["x"] = 3') == 'df["x"] = 3'


def test_rationale_is_lifted_out_of_the_comments():
    code = (
        "# ('Debt to Equity', 'total debt over total equity')\n"
        "# Usefulness: leverage drives distress risk\n"
        "# Input samples: 'a': [1], 'b': [2]\n"
        'df["de"] = df["a"] / df["b"]'
    )
    parsed = parse_candidate(code)
    assert parsed.display_name == "Debt to Equity"
    assert parsed.description == "total debt over total equity"
    assert parsed.rationale == "leverage drives distress risk"
    assert parsed.input_columns == ["a", "b"]        # from the AST, not the comment
    assert parsed.expression == "df['a'] / df['b']"


def test_rationale_parsing_degrades_without_comments():
    parsed = parse_candidate('df["x"] = df["a"] + df["b"]')
    assert parsed.display_name is None and parsed.rationale is None
    assert parsed.input_columns == ["a", "b"]


def test_rationale_parsing_survives_broken_code():
    parsed = parse_candidate('df["x"] = (((')
    assert parsed.input_columns == [] and parsed.expression is None


# --------------------------------------------------------------------------- #
# redundancy: the rejection a delta cannot see
# --------------------------------------------------------------------------- #
def test_a_near_copy_of_an_existing_column_is_rejected(frame):
    sample = build_sample(frame, "class", size=600, seed=1)
    screener = Screener(sample, "class", ["a", "b", "noise"], PARAMS,
                        score=calc_adj_gini, num_boost_round=30,
                        redundancy_max_abs=0.95)
    # a monotone rescale of an existing column: perfectly correlated, no new info
    out = screener.evaluate('df["copy"] = df["a"] * 3.0 + 1.0')
    assert not out.ok
    assert "redundant" in out.error and "'a'" in out.error


def test_a_genuinely_new_combination_passes_the_redundancy_check(frame):
    sample = build_sample(frame, "class", size=600, seed=1)
    screener = Screener(sample, "class", ["a", "b", "noise"], PARAMS,
                        score=calc_adj_gini, num_boost_round=30,
                        redundancy_max_abs=0.95)
    out = screener.evaluate('df["ratio"] = df["a"] / df["b"].clip(lower=0.1)')
    assert out.ok, out.error


def test_redundancy_checking_is_off_unless_a_threshold_is_given(frame):
    sample = build_sample(frame, "class", size=600, seed=1)
    screener = Screener(sample, "class", ["a", "b", "noise"], PARAMS,
                        score=calc_adj_gini, num_boost_round=30)
    assert screener.evaluate('df["copy"] = df["a"] * 3.0').ok


def test_a_constant_column_is_not_called_redundant(frame):
    """It correlates with nothing; min_unique is the gate that catches it."""
    sample = build_sample(frame, "class", size=600, seed=1)
    screener = Screener(sample, "class", ["a", "b", "noise"], PARAMS,
                        score=calc_adj_gini, num_boost_round=30,
                        redundancy_max_abs=0.95)
    out = screener.evaluate('df["dead"] = df["noise"] * 0.0')
    assert out.ok, out.error


def test_the_prompt_states_the_real_redundancy_threshold():
    text = build_prompt("task", "a (float64)\nSamples [1, 2]", "none",
                        "adjusted Gini", redundancy_max_abs=0.95)
    assert "|rho| = 0.95" in text
    assert "most common rejection" in text
    assert "hard constraints, not advice" in text


def test_the_prompt_omits_the_redundancy_rule_when_it_is_off():
    text = build_prompt("task", "a (float64)\nSamples [1, 2]", "none", "adjusted Gini")
    assert "REDUNDANT WITH AN EXISTING COLUMN" not in text
    assert "NON-FINITE VALUES" in text        # the other rules still stand
    assert "EXTREME VALUES" in text


# --------------------------------------------------------------------------- #
# the delta must measure the feature, not the column count
# --------------------------------------------------------------------------- #
def test_column_subsampling_is_stripped_so_a_delta_is_attributable(frame):
    """With colsample_bytree < 1 the column COUNT moves the score.

    A constant column carries no information, so its delta must be exactly zero.
    Before this was stripped it measured -0.0055, and eight different indicators
    all measured an identical -0.0019 - the offset, not the features.
    """
    sample = build_sample(frame, "class", size=800, seed=1)
    params = {**PARAMS, "colsample_bytree": 0.8, "colsample_bylevel": 0.7,
              "subsample": 0.8}
    screener = Screener(sample, "class", ["a", "b", "noise"], params,
                        score=calc_adj_gini, num_boost_round=60)

    # column subsampling stripped, because the column count is what moves
    assert screener.params["colsample_bytree"] == 1.0
    assert screener.params["colsample_bylevel"] == 1.0
    # row subsampling untouched: it does not depend on the column count
    assert screener.params["subsample"] == 0.8

    out = screener.evaluate('df["zero_information"] = 1')
    assert out.ok and out.delta == 0.0, out.as_dict()


def test_two_uninformative_columns_do_not_score_identically_nonzero(frame):
    """The signature of the offset: different columns, same non-zero delta."""
    sample = build_sample(frame, "class", size=800, seed=1)
    screener = Screener(sample, "class", ["a", "b", "noise"], PARAMS,
                        score=calc_adj_gini, num_boost_round=60)
    first = screener.evaluate('df["c1"] = 1').delta
    second = screener.evaluate('df["c2"] = 0').delta
    assert first == second == 0.0


def test_a_sparse_binary_flag_is_not_called_a_spike(frame):
    """4 ones in 6819 rows has p99 == 0; max/p99 must not read as 1e12."""
    sample = build_sample(frame, "class", size=800, seed=1)
    screener = Screener(sample, "class", ["a", "b", "noise"], PARAMS,
                        score=calc_adj_gini, num_boost_round=30,
                        spike_factor=1000.0)
    out = screener.evaluate('df["rare"] = (df["a"] > df["a"].quantile(0.999)).astype(int)')
    assert out.ok, out.error


def test_a_binary_column_never_trips_the_spike_guard():
    from discovery.guards import check_scale
    for ones in (1, 4, 100, 399):
        values = np.zeros(400)
        values[:ones] = 1
        check_scale({"sample": pd.DataFrame({"flag": values})}, "flag")   # must not raise


def test_a_genuine_spike_is_still_caught_in_a_many_valued_column():
    values = pd.DataFrame({"f": list(np.linspace(0.1, 20, 300)) + [1.58e6]})
    from discovery.guards import check_scale
    with pytest.raises(CandidateError, match="spikes"):
        check_scale({"sample": values}, "f")


# --------------------------------------------------------------------------- #
# near-constant columns: the mechanism behind most redundancy rejections
# --------------------------------------------------------------------------- #
def test_near_constant_columns_are_detected():
    from discovery.prompt import low_variation_columns
    frame = pd.DataFrame({
        "flat": np.full(500, 10.0) + np.random.default_rng(0).normal(0, 0.01, 500),
        "varying": np.random.default_rng(1).uniform(1, 100, 500),
    })
    found = low_variation_columns(frame, ["flat", "varying"])
    assert found == ["flat"]


def test_a_ratio_against_a_near_constant_column_reproduces_the_other_input():
    """Why the flag matters: this is not a conceptual duplication, it is arithmetic."""
    rng = np.random.default_rng(0)
    flat = np.full(500, 10.0) + rng.normal(0, 0.01, 500)
    varying = rng.uniform(1, 100, 500)
    ratio = flat / varying
    rho = pd.Series(ratio).rank().corr(pd.Series(varying).rank())
    assert abs(rho) > 0.99, abs(rho)


def test_the_prompt_flags_near_constant_columns_inline():
    frame = pd.DataFrame({"flat": [10.0, 10.0, 10.001, 10.0], "v": [1.0, 50.0, 3.0, 90.0]})
    text = describe_columns(frame, ["flat", "v"], low_variation=["flat"])
    flat_line = [l for l in text.splitlines() if l.startswith('df["flat"]')][0]
    v_line = [l for l in text.splitlines() if l.startswith('df["v"]')][0]
    assert "NEARLY CONSTANT" in flat_line
    assert "NEARLY CONSTANT" not in v_line


def test_the_prompt_explains_the_ratio_mechanism_not_just_the_rule():
    text = build_prompt("task", 'df["a"] (float64; NEARLY CONSTANT across rows)',
                        "none", "adjusted Gini", redundancy_max_abs=0.95)
    assert "monotone function of whichever input actually varies" in text
    assert "Check that BOTH inputs vary substantially" in text
    assert "Safer shapes" in text


def test_the_prompt_shows_a_safe_division_not_only_an_unsafe_one():
    """Told only what to avoid, the model substituted NaN instead of an epsilon."""
    text = build_prompt("task", 'df["a"] (float64)', "none", "adjusted Gini")
    assert "np.nan)   # NaN is not a number" in text
    assert "These work:" in text
    assert "Substituting zero" in text


def test_the_prompt_forbids_continuing_the_tables_naming_scheme():
    """Emphasising identifiers made the model name its features X96..X105."""
    text = build_prompt("task", 'df["X95"] (float64)', "none", "adjusted Gini")
    assert "`X96` is not an acceptable name" in text
    assert "snake_case" in text
