"""The loop: batches, per-feature feedback, and the stopping rule."""
import numpy as np
import pandas as pd
import pytest

from discovery.loop import DiscoveryRun, RoundRecord, StoppingRule, run_discovery
from discovery.screen import Screener, build_sample
from validation.metrics import calc_adj_gini

PARAMS = {"eta": 0.1, "max_depth": 3, "objective": "binary:logistic"}


class StubScreener:
    """A screener with scripted deltas.

    The loop's job is batching, per-feature feedback and stopping - not signal
    detection, which test_discovery_screen covers against a real fit. Scripting
    the deltas keeps these tests about the loop and free of the sampling noise
    that a cheap screen legitimately carries.
    """

    def __init__(self, deltas):
        self.deltas = dict(deltas)
        self.base_score = 0.50
        self.seen = []

    def evaluate(self, code, reserved_names=()):
        from discovery.sandbox import CandidateError, validate_code, validate_single_column
        from discovery.screen import ScreenResult

        self.seen.append((code, list(reserved_names)))
        result = ScreenResult(n_rows=100)
        try:
            # Same gate order the real screener applies, so a block rejected here
            # is a block that would be rejected for real.
            name = validate_single_column(code, ["a", "b", "noise", "class"], reserved_names)
            result.feature_name = name       # recorded before syntax is judged,
            validate_code(code)              # exactly as the real screener does
            if name not in self.deltas:
                raise CandidateError(f"stub has no delta for '{name}'")
            result.base_score = self.base_score
            result.delta = self.deltas[name]
            result.candidate_score = self.base_score + result.delta
            result.ok = True
        except CandidateError as error:
            result.error = str(error)
        return result


@pytest.fixture
def screener():
    return StubScreener({"ratio": 0.05, "ratio2": 0.02, "dead": 0.0, "x": 0.01})


class ScriptedProposer:
    """Yields pre-written batches; records what history it was shown."""

    name = "scripted"

    def __init__(self, batches):
        self.batches = list(batches)
        self.seen_history = []
        self.seen_names = []
        self.calls = 0

    def propose(self, *, history, proposed_names, n_features, round_index):
        self.calls += 1
        self.seen_history.append(list(history))
        self.seen_names.append(list(proposed_names))
        blocks = self.batches.pop(0) if self.batches else []
        return blocks, {"round": round_index, "requested": n_features}


USEFUL = 'df["ratio"] = df["a"] / df["b"].clip(lower=0.1)'
USEFUL2 = 'df["ratio2"] = (df["a"] + 1) / (df["b"] + 1)'
DEAD = 'df["dead"] = df["noise"] * 0.0'
BROKEN = 'import os\ndf["x"] = 1'


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #
def test_a_round_screens_every_block_in_the_batch(screener):
    proposer = ScriptedProposer([[USEFUL, DEAD, BROKEN]])
    run = run_discovery(proposer, screener, batch_size=3)
    assert len(run.candidates) == 3
    # the broken block's name is still parsed and reported - useful feedback
    assert [c.feature_name for c in run.candidates] == ["ratio", "dead", "x"]
    assert run.rounds[0].returned == 3
    assert run.rounds[0].rejected == 1          # the broken one


def test_the_default_is_a_single_round(screener):
    proposer = ScriptedProposer([[USEFUL], [USEFUL2]])
    run = run_discovery(proposer, screener, batch_size=1)
    assert proposer.calls == 1
    assert len(run.rounds) == 1
    assert "max_rounds reached (1)" in run.stopped_because


def test_rationale_and_numbers_land_on_the_same_record(screener):
    code = (
        "# ('A over B', 'ratio of a to b')\n"
        "# Usefulness: captures relative magnitude\n"
        'df["ratio"] = df["a"] / df["b"].clip(lower=0.1)'
    )
    run = run_discovery(ScriptedProposer([[code]]), screener, batch_size=1)
    row = run.records()[0]
    assert row["display_name"] == "A over B"
    assert row["rationale"] == "captures relative magnitude"
    assert row["input_columns"] == "a, b"
    assert row["feature_name"] == "ratio"
    assert row["delta"] is not None and row["base_score"] is not None
    assert row["code"] == code


# --------------------------------------------------------------------------- #
# feedback
# --------------------------------------------------------------------------- #
def test_feedback_is_per_feature_not_per_batch(screener):
    proposer = ScriptedProposer([[USEFUL, DEAD], [USEFUL2]])
    run_discovery(proposer, screener, batch_size=2, stopping=StoppingRule(max_rounds=2))
    shown = proposer.seen_history[1]
    assert len(shown) == 2                       # one entry per feature, not one per round
    assert {r["feature_name"] for r in shown} == {"ratio", "dead"}
    assert all("delta" in r for r in shown)


def test_the_second_round_is_told_which_names_are_taken(screener):
    proposer = ScriptedProposer([[USEFUL], [USEFUL2]])
    run_discovery(proposer, screener, batch_size=1, stopping=StoppingRule(max_rounds=2))
    assert proposer.seen_names[0] == []
    assert proposer.seen_names[1] == ["ratio"]


def test_a_failed_block_does_not_reserve_its_name(screener):
    """A broken block leaves its name free: the idea may be right, the code wrong."""
    proposer = ScriptedProposer([[BROKEN, 'df["x"] = df["a"] + 1']])
    run = run_discovery(proposer, screener, batch_size=2)
    assert not run.candidates[0].ok            # import os
    assert run.candidates[1].ok                # same name, now allowed
    assert run.candidates[1].feature_name == "x"


def test_a_repeat_inside_one_batch_is_refused(screener):
    proposer = ScriptedProposer([[USEFUL, USEFUL]])
    run = run_discovery(proposer, screener, batch_size=2)
    assert run.candidates[0].ok
    assert not run.candidates[1].ok
    assert "earlier round" in run.candidates[1].screen.error


# --------------------------------------------------------------------------- #
# stopping rules
# --------------------------------------------------------------------------- #
def test_max_rounds_bounds_the_run(screener):
    proposer = ScriptedProposer([[USEFUL], [USEFUL2], [DEAD], [DEAD]])
    run = run_discovery(proposer, screener, batch_size=1, stopping=StoppingRule(max_rounds=3))
    assert proposer.calls == 3 and len(run.rounds) == 3


def test_target_features_stops_early(screener):
    proposer = ScriptedProposer([[USEFUL], [USEFUL2], [DEAD]])
    run = run_discovery(proposer, screener, batch_size=1,
                        stopping=StoppingRule(max_rounds=10, target_features=1))
    assert len(run.kept) >= 1
    assert "target_features reached (1)" in run.stopped_because
    assert proposer.calls == 1


def test_patience_stops_a_run_that_keeps_nothing(screener):
    proposer = ScriptedProposer([[DEAD], [DEAD], [USEFUL]])
    run = run_discovery(proposer, screener, batch_size=1,
                        stopping=StoppingRule(max_rounds=10, patience=2), min_delta=0.001)
    assert proposer.calls == 2
    assert "no features kept in the last 2 round(s)" in run.stopped_because


def test_max_candidates_caps_total_screening(screener):
    proposer = ScriptedProposer([[USEFUL, DEAD], [USEFUL2], [USEFUL2]])
    run = run_discovery(proposer, screener, batch_size=2,
                        stopping=StoppingRule(max_rounds=10, max_candidates=2))
    assert len(run.candidates) == 2
    assert "max_candidates reached (2)" in run.stopped_because


def test_by_default_everything_that_ran_is_carried_forward(screener):
    """The screen filters broken code, not weak deltas - see run_discovery."""
    run = run_discovery(ScriptedProposer([[USEFUL, DEAD, BROKEN]]), screener, batch_size=3)
    assert len(run.candidates) == 3
    assert {c.feature_name for c in run.kept} == {"ratio", "dead"}   # dead ran; broken did not
    assert run.rounds[0].kept == 2 and run.rounds[0].rejected == 1


def test_a_negative_delta_is_still_forwarded_by_default(screener):
    screener.deltas["ratio"] = -0.02        # noise-dominated, as a real screen often is
    run = run_discovery(ScriptedProposer([[USEFUL]]), screener, batch_size=1)
    assert [c.feature_name for c in run.kept] == ["ratio"]


def test_min_delta_controls_what_is_carried_forward(screener):
    permissive = run_discovery(ScriptedProposer([[USEFUL, DEAD]]), screener,
                               batch_size=2, min_delta=0.0)
    strict = run_discovery(ScriptedProposer([[USEFUL, DEAD]]), screener,
                           batch_size=2, min_delta=0.99)
    assert len(permissive.kept) >= 1
    assert len(strict.kept) == 0
    # everything is still recorded either way - filtering is not forgetting
    assert len(permissive.candidates) == len(strict.candidates) == 2


def test_a_failing_proposer_ends_the_run_without_losing_earlier_work(screener):
    class Exploding(ScriptedProposer):
        def propose(self, **kwargs):
            if self.calls >= 1:
                self.calls += 1
                raise RuntimeError("backend down")
            return super().propose(**kwargs)

    proposer = Exploding([[USEFUL]])
    run = run_discovery(proposer, screener, batch_size=1, stopping=StoppingRule(max_rounds=5))
    assert len(run.candidates) == 1 and run.candidates[0].ok
    assert "backend down" in run.stopped_because


def test_an_empty_batch_is_survivable(screener):
    run = run_discovery(ScriptedProposer([[]]), screener, batch_size=2)
    assert run.candidates == [] and run.rounds[0].returned == 0


def test_base_score_is_recorded_once_for_the_run(screener):
    run = run_discovery(ScriptedProposer([[USEFUL]]), screener, batch_size=1)
    assert run.base_score == screener.base_score


def test_on_candidate_is_called_as_each_one_is_screened(screener):
    seen = []
    run_discovery(ScriptedProposer([[USEFUL, DEAD]]), screener, batch_size=2,
                  on_candidate=seen.append)
    assert [c.feature_name for c in seen] == ["ratio", "dead"]
