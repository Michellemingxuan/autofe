"""The agent's tools end to end on the synthetic use case, with no LLM.

The synthetic data plants a real signal (share of 90-day spend paid back) and
a leak (collections spends after as_of for defaulters). These tests drive the
session the way the agent would and check that the first verifies through a
point-in-time linkage, and that the second cannot get through.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent.execution import check_code
from agent.session import AutoApprover, RunParams, Session
from agent.tools import propose_linkage, screen_request, screen_feature
from agent.synthetic import generate
from agent.workspace import Workspace
from validation.config import load_config

LINK = '''
def link(base_ids, source):
    ids = base_ids.copy()
    parts = ids["id"].str.split("_")
    ids["customer_id"] = parts.str[0]
    ids["as_of"] = pd.to_datetime(parts.str[1], format="%Y%m%d")
    out = ids.merge(source, on="customer_id")
    out["event_dt"] = pd.to_datetime(out["event_dt"])
    return out[out["event_dt"] < out["as_of"]]
'''

LEAKY_LINK = LINK.replace('out[out["event_dt"] < out["as_of"]]', "out")

PAY_TO_SPEND = '''
def build(spark, sources, base):
    def window_sum(frame):
        recent = frame[frame["event_dt"] >= frame["as_of"] - pd.Timedelta(days=90)]
        return recent.groupby("id")["amount"].sum()
    paid, spent = window_sum(sources["payments"]), window_sum(sources["spends"])
    out = base[["id"]].copy()
    p = out["id"].map(paid).fillna(0.0)
    s = out["id"].map(spent).fillna(0.0)
    out["pay_to_spend_90d"] = np.where(s > 0, p / s, np.nan)
    return out
'''


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = generate(tmp_path_factory.mktemp("syn") / "data", n_customers=2500, seed=1)
    raw = yaml.safe_load(open("configs/synthetic_agent.yaml"))
    text = json.dumps(raw).replace("data/synthetic_agent", str(root))
    payload = json.loads(text)
    payload["agent"]["run_dir"] = str(root.parent / "runs")
    path = root.parent / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload))
    return Workspace.from_config(load_config(path))


def test_workspace_sees_sources_scope_and_no_target(workspace, tmp_path):
    assert sorted(workspace.sources()) == ["balances", "payments", "spends"]
    scope = workspace.scope()
    assert set(scope["status"]) == {"in_model", "unused_raw"}
    hits = workspace.catalog("payment returned")["matches"]
    assert any(h["where"] == "source:payments" and h["column"] == "returned" for h in hits)
    import pandas as pd
    base = pd.read_parquet(workspace.write_base(tmp_path / "base.parquet"))
    assert "default" not in base.columns


def test_guard_refuses_file_access():
    assert check_code("df = pd.read_csv('data/x/test.csv')")
    assert check_code("import os\n")
    assert check_code("open('x')")
    assert check_code("def build(spark, sources, base):\n    return base") is None


@pytest.fixture
def no_history(workspace, tmp_path):
    """A run folder with no earlier directions in it - for tests that repeat a feature
    or a request across sessions on purpose, which the memory would otherwise refuse."""
    old = workspace.cfg.agent.run_dir
    workspace.cfg.agent.run_dir = str(tmp_path / "runs")
    yield workspace
    workspace.cfg.agent.run_dir = old


def _session(workspace, direction, **kwargs):
    """A session for tests that are not about the level split: no shares, so a
    test's L1 feature or L3 request is never refused by the random draw."""
    session = Session(workspace, direction, **kwargs)
    session.quota = {}
    return session


def test_each_kind_of_run_is_briefed_with_its_skills(workspace):
    from agent.composer import compose as brief
    linkage_brief = brief

    direction = Session(workspace, "x", params=RunParams(K=1, min_capture_gain=0.01))
    assert list(direction.skills) == ["ideas", "data_sourcing", "feature", "evaluate"]
    text = brief(direction)
    assert "## Skill: feature" in text and "def build(spark, sources, base)" in text
    assert "capture-rate gain" in text and "report_findings" in text
    linkage = Session(workspace, "Link spends", kind="linkage", source="spends")
    assert list(linkage.skills) == ["data_sourcing"]
    assert "## Skill: data_sourcing" in linkage_brief(linkage)


def test_planted_feature_verifies_through_linkage(workspace):
    session = _session(workspace, "payments vs spend", approver=AutoApprover(), params=RunParams(K=3))
    for source in ("payments", "spends"):
        reply = propose_linkage(session, source, LINK, "event_dt")
        assert reply["ok"], reply
        assert reply["checks"]["point_in_time_violations"] == 0
    reply = screen_feature(session, "pay_to_spend_90d", "share of spend repaid", "L2",
                                   PAY_TO_SPEND)
    assert reply["verified"], reply
    assert reply["gini_gain"] > 0.03 and reply["capture_gain"] is not None
    events = [e["event"] for e in session.events]
    assert "approval_required" in events and "feature_verified" in events
    assert (session.run_dir / "features" / "pay_to_spend_90d.py").exists()


def test_leaky_linkage_is_refused_before_the_user_sees_it(workspace):
    asked = []

    class Recorder(AutoApprover):
        def request(self, kind, payload):
            asked.append(kind)
            return super().request(kind, payload)

    session = _session(workspace, "collections", approver=Recorder(), params=RunParams(K=2))
    reply = propose_linkage(session, "spends", LEAKY_LINK, "event_dt")
    assert not reply["ok"]
    assert reply["point_in_time_violations"] > 0
    assert asked == []


def test_unlinked_source_costs_nothing_and_target_and_attempts_are_enforced(workspace, tmp_path):
    workspace.cfg.discovery.additional_data.linkage_dir = str(tmp_path / "empty_linkage")
    session = _session(workspace, "x", params=RunParams(K=1))
    reply = screen_feature(session, "f1", "d", "L2",
                                   'def build(spark, sources, base):\n    return sources["spends"]')
    assert not reply["ok"] and "no confirmed linkage" in reply["error"]
    assert session.attempts == 0

    ratio = '''
def build(spark, sources, base):
    out = base[["id"]].copy()
    out["util_x_delinq"] = base["utilization"] * (1 + base["num_delinq_12m"])
    return out
'''
    first = screen_feature(session, "util_x_delinq", "d", "L1", ratio)
    assert first["intent"] == "I1" and session.max_attempts == 3
    # K is a target of results: once one is verified, no more are taken.
    session.ledger[0]["verified"] = True
    met = screen_feature(session, "again", "d", "L1", ratio.replace("util_x_delinq", "again"))
    assert not met["ok"] and "the target of 1 is met" in met["error"]
    # Short of the target, the attempts still end the run.
    session.ledger[0]["verified"] = False
    session.attempts = session.max_attempts
    capped = screen_feature(session, "again", "d", "L1", ratio.replace("util_x_delinq", "again"))
    assert not capped["ok"] and "all 3 attempts are used" in capped["error"]


def test_a_failed_script_comes_back_short_with_the_columns_it_had(workspace):
    session = _session(workspace, "x", params=RunParams(K=3))
    broken = """
def build(spark, sources, base):
    out = base[["id", "as_of"]].copy()
    return out
"""
    first = screen_feature(session, "f1", "d", "L1", broken)
    # Its own line and the error - not a page of pandas frames.
    assert first["reason"] == ("script failed: line 3: out = base[[\"id\", \"as_of\"]].copy()\n"
                               "KeyError: \"['as_of'] not in index\"")
    assert first["frames"]["base"][0] == "id" and "as_of" not in first["frames"]["base"]
    assert "next" not in first
    # Two in a row: stop patching and look.
    second = screen_feature(session, "f2", "d", "L1", broken.replace("f1", "f2"))
    assert "2 scripts in a row failed" in second["next"] and "run_probe" in second["next"]
    assert "lessons" not in second                              # the gate handles scripts
    # ... and the next screen waits for a look at the data, at no cost.
    waits = screen_feature(session, "f3", "d", "L1", broken.replace("f1", "f3"))
    assert not waits["ok"] and "look before the next attempt" in waits["error"]
    assert session.attempts == 2
    from agent.tools import run_probe

    run_probe(session, "print(base.columns.tolist())", "what base holds")
    assert screen_feature(session, "f3", "d", "L1", broken.replace("f1", "f3"))["intent"] == "I3"


def test_c1_makes_the_redundancy_call_not_the_screen_guard(workspace):
    from agent.checks import REBUILT

    # The guard sits above C1's limit, so a feature C2 saves from C1 is fitted.
    assert workspace.screener.redundancy_max_abs == REBUILT > workspace.cfg.agent.checks.max_model_rho


def test_repeated_failures_become_a_lesson_and_old_results_are_cut():
    from agents.run import CallModelData, ModelInputData

    from agent.agent import KEEP_FULL, _trim_old_outputs
    from agent.tools.screen import lessons

    redundant = "is redundant: |rho|=0.99 against the existing column 'tenure_months', above"
    explained = "explained by the model: |ρ| 0.93 with the model column `tenure_months` (above 0.9)"
    ledger = [{"reason": redundant}, {"reason": explained}, {"reason": "Gini gain -0.01 is not above"}]
    assert lessons(ledger) == ["2 features were redundant with `tenure_months` - a ratio, "
                               "difference or rescaling of it re-derives it; build on other columns"]

    outputs = [{"type": "function_call_output", "call_id": str(i), "output": "x" * 2000}
               for i in range(KEEP_FULL + 2)]
    items = [{"role": "user", "content": "go"}, *outputs]
    sent = _trim_old_outputs(CallModelData(model_data=ModelInputData(input=items, instructions="i"),
                                           agent=None, context=None)).input
    assert [len(i["output"]) < 2000 for i in sent[1:]] == [True, True] + [False] * KEEP_FULL
    assert items[1]["output"] == "x" * 2000                       # the conversation keeps it whole


def test_a_running_script_is_stopped_at_once_and_a_slow_one_times_out(tmp_path):
    import time

    import pandas as pd

    from agent.execution import run_code

    base = tmp_path / "base.parquet"
    pd.DataFrame({"id": ["a"]}).to_parquet(base)
    forever = "while True:\n    pass"
    common = dict(workdir=tmp_path / "w", engine="pandas", id_col="id", base_path=base)
    start = time.perf_counter()
    stopped = run_code(forever, "probe", tag="s", timeout_s=60,
                       should_stop=lambda: time.perf_counter() - start > 1, **common)
    assert stopped.error == "stopped by the user" and time.perf_counter() - start < 5
    slow = run_code(forever, "probe", tag="t", timeout_s=1, **common)
    assert slow.error.startswith("timed out after 1s - vectorise")


def test_an_earlier_rounds_idea_stays_in_the_order(workspace):
    from agent.agent import _work_order
    from agent.board import round_board

    session = _session(workspace, "x", params=RunParams(K=3))
    session.ideas = [{"name": "pay_gap", "engine": "analogy", "round": 1},
                     {"name": "tenure_curve", "engine": "mechanism", "round": 2}]
    session.round_ideas, session.round = session.ideas[1:], 2
    assert set(_work_order(session)) == {"pay_gap", "tenure_curve"}
    assert {i["name"]: i["round"] for i in round_board(session)} == {"pay_gap": 1, "tenure_curve": 2}


def test_an_idea_like_an_earlier_rounds_is_sent_back_to_work_that_one(no_history):
    from agent.tools.ideas import _same_as

    session = _session(no_history, "x", params=RunParams(K=3))
    session.ideas = [{"name": "payment_shortfall", "round": 1,
                      "description": "months the payment falls short of the statement balance"}]
    again = [{"name": "payment_gap", "description": "months the payment falls short of the statement balance"}]
    problem = _same_as(session, again, [])
    assert "'payment_shortfall' of round 1" in problem and 'idea="payment_shortfall"' in problem


def test_an_idea_or_request_name_of_an_earlier_run_is_taken(no_history):
    import json

    from agent.tools.data_pull import _can_propose
    from agent.tools.ideas import record_ideas

    first = _session(no_history, "x", params=RunParams(K=2, levels=["L3"]))
    first.ideas = [{"name": "decline_streaks", "round": 1, "description": "d"}]
    (first.run_dir / "ideas.json").write_text(json.dumps(first.ideas))
    (first.run_dir / "data_requests.json").write_text(json.dumps(
        [{"intent": "R1", "source_name": "cas_declines", "status": "dropped"}]))
    second = _session(no_history, "y", params=RunParams(K=2, levels=["L3"]))
    reply = record_ideas(second, [{"name": "decline_streaks", "engine": "mechanism", "lens": "trend",
                                   "level": "L3", "beyond_scope": True, "description": "runs of declines",
                                   "measures": ["a b", "c d"], "data": "declines from the auth system monthly"}])
    assert not reply["ok"] and f"idea of run {first.run_id}" in reply["error"]
    second.ideas = [{"name": "other"}]                         # past the ideas stage
    taken = _can_propose(second, "cas_declines")
    assert not taken["ok"] and f"run {first.run_id} (dropped)" in taken["error"]


def test_an_idea_whose_features_carry_no_signal_is_worked_later_not_refused(workspace):
    from agent.board import idea_standing, order_ideas
    from agent.tools.screen import idea_of

    assert idea_of("pay_gap_v3", []) == idea_of("pay_gap_final", []) == "pay_gap"
    assert idea_of("pay_gap_60d_fixed", [{"name": "pay_gap_60d"}]) == "pay_gap_60d"
    session = _session(workspace, "x", params=RunParams(K=3))
    session.ideas = session.round_ideas = [{"name": "pay_gap", "engine": "analogy"},
                                           {"name": "tenure_curve", "engine": "mechanism"}]
    copy = "def build(spark, sources, base):\n    out = base[['id']].copy()\n    out['{n}'] = base['{c}'] * 2\n    return out"
    # A failed script says nothing about the idea: it does not count against it.
    broken = "def build(spark, sources, base):\n    return base[['id', 'as_of']]"
    assert screen_feature(session, "pay_gap", "d", "L1", broken)["reason"].startswith("script failed")
    session.failed_streak = 0
    # Two features the model already has (C1): the idea draws low ...
    for name, column in (("pay_gap_v1", "income_est"), ("pay_gap_v2", "credit_limit")):
        reply = screen_feature(session, name, "d", "L1", copy.format(n=name, c=column),
                               differs="another base column")
        assert reply["reason"].startswith("explained by the model") and reply["checks"]["max_model_rho"] > 0.9
        assert reply["checks"]["nearest"] == column
    standing = {i["name"]: i for i in reply["ideas"]}
    assert standing["pay_gap"]["status"] == "low" and standing["pay_gap"]["p_useful"] == 0.333   # misses count half
    assert standing["pay_gap"]["measures_left"] == 0 and standing["tenure_curve"]["measures_left"] == 0
    assert set(reply["work_next"]) == {"pay_gap", "tenure_curve"}
    # ... and is worked later - Thompson sampling, not a ban:
    low, fresh = (dict(name="pay_gap", **idea_standing(["none", "none"])),
                  dict(name="tenure_curve", **idea_standing([])))
    firsts = [order_ideas([low, fresh], seed=str(i))[0] for i in range(400)]
    assert 0 < firsts.count("pay_gap") < firsts.count("tenure_curve")
    third = screen_feature(session, "pay_gap_v3", "d", "L1", copy.format(n="pay_gap_v3", c="tenure_months"),
                           differs="tenure, not money")
    assert "intent" in third and session.attempts == 4           # still screened


def test_bad_contract_is_reported_not_raised(workspace):
    session = _session(workspace, "x", params=RunParams(K=2))
    reply = screen_feature(session, "two_cols", "d", "L1",
                                   "def build(spark, sources, base):\n    return base")
    assert reply["verified"] is False
    assert "exactly one feature column" in reply["reason"]
    states = [e["state"] for e in session.events if e["event"] == "code_status"]
    assert states == ["running", "error"]


def test_evaluation_pools_features_across_runs_and_combines_them(no_history):
    workspace = no_history
    from agent.evaluate import Evaluation, feature_pool

    workspace.cfg.analysis.shap.enabled = False
    first = _session(workspace, "payments vs spend", params=RunParams(K=1))
    for source in ("payments", "spends"):
        assert propose_linkage(first, source, LINK, "event_dt")["ok"]
    assert screen_feature(first, "pay_to_spend_90d", "d", "L2", PAY_TO_SPEND)["verified"]

    # A second direction may not reuse the name - a name is one feature, across runs.
    second = _session(workspace, "utilisation", params=RunParams(K=1, min_gini_gain=-1.0))
    util = """
def build(spark, sources, base):
    out = base[["id"]].copy()
    out["{n}"] = base["income_est"] / base["credit_limit"]
    return out
"""
    taken = screen_feature(second, "pay_to_spend_90d", "d", "L1", util.format(n="pay_to_spend_90d"))
    assert not taken["ok"] and f"run {first.run_id}" in taken["error"]
    checks = workspace.cfg.agent.checks
    limit, checks.max_model_rho = checks.max_model_rho, None  # this test is about pooling, not C1
    try:
        assert screen_feature(second, "income_to_limit", "d", "L1", util.format(n="income_to_limit"),
                              differs="income against the limit, not payments against spend")["verified"]
    finally:
        checks.max_model_rho = limit

    pool = {f["key"] for f in feature_pool(workspace)}
    a, b = f"{first.run_id}:pay_to_spend_90d", f"{second.run_id}:income_to_limit"
    assert {a, b} <= pool

    evaluation = Evaluation(workspace, [a, b], {"both": [a, b]})
    out = evaluation.run()
    variants = {row["variant"]: row for row in out["comparison"]}
    col_a = "pay_to_spend_90d"                      # names are unique across runs: no suffix
    assert {"base", f"loi__{col_a}", "combo__both"} <= set(variants)
    assert variants[f"loi__{col_a}"]["gini_gain_test"] > 0.03
    # The variable Gini on test: the feature alone as a score, as check C2 reads it.
    assert 0.05 < out["feature_stats"][col_a]["variable_gini"] <= 1
    from agent.evaluate import evaluation_results
    # Beside it on the results, the same on the screen rows, as discovery measured it.
    row = next(r for r in evaluation_results(workspace)
               if r["eval_id"] == evaluation.eval_id and r["variant"] == f"loi__{col_a}")
    assert 0 < row["screen_variable_gini"] <= 1 and row["variable_gini"] is not None
    assert evaluation.events[-1]["event"] == "eval_done"

    # A combination by itself: its members get no model alone, and no row says
    # they were "not trained".
    from agent.evaluate import evaluation_results

    alone = Evaluation(workspace, [], {"both": [a, b]})
    trained = {row["variant"] for row in alone.run()["comparison"]}
    assert trained == {"base", "combo__both"}
    rows = [r for r in evaluation_results(workspace) if r["eval_id"] == alone.eval_id]
    assert [r["variant"] for r in rows] == ["combo__both"]

    # Base + every passing feature, together: one more combination, added by the
    # evaluation itself after the verdicts.
    from agent.evaluate import TOGETHER, evaluation_status, list_evaluations

    verdict = workspace.cfg.verdict
    kept = verdict.min_gini_gain, verdict.max_shap_rank_pct
    verdict.min_gini_gain, verdict.max_shap_rank_pct = -1.0, 1.0      # both pass
    try:
        together = Evaluation(workspace, [a, b], {}, together="passing")
        done = together.run()
    finally:
        verdict.min_gini_gain, verdict.max_shap_rank_pct = kept
    passed = [v["feature"] for v in done["verdicts"] if v["verdict"] == "PASS"]
    assert len(passed) == 2 and not done["together_note"]
    name = TOGETHER["passing"]
    assert f"combo__{name}" in {r["variant"] for r in done["comparison"]}
    assert f"combo__{name}" in done["score_cutoffs"]
    assert (Path(done["output_dir"]) / "models" / f"combo__{name}.json").exists()
    listed = next(e for e in list_evaluations(workspace) if e["eval_id"] == together.eval_id)
    assert listed["combinations"][name]["auto"] and listed["combinations"][name]["columns"] == passed
    # "all": every single that reached modelling - here without lowering any bar.
    latest = Evaluation(workspace, [a, b], {}, together="all")
    every = latest.run()
    assert f"combo__{TOGETHER['all']}" in {r["variant"] for r in every["comparison"]}
    # It trains in the evaluation's own run, beside the singles - no second run.
    assert not (latest.folder / "together").exists()
    assert (Path(every["output_dir"]) / "models" / f"combo__{TOGETHER['all']}.json").exists()
    # The pool knows what became of each feature.
    status = evaluation_status(workspace)
    assert status[a]["single"]["eval_id"] == latest.eval_id and "both" in status[a]["combinations"]

    # The board: an evaluated feature says so - and nothing measured out of time;
    # a removed one too. Its id is the one the metrics results carry.
    from agent.board import board_rows

    def board(name, run):
        return next(r for r in board_rows(workspace) if r["name"] == name and r["run_id"] == run)
    row = board("pay_to_spend_90d", first.run_id)
    assert row["sent"] == "evaluated" and "verdict" not in row
    assert not any(k.startswith("oot") for k in row)
    assert row["feature_id"] in {r["key"] for r in evaluation_results(workspace)}
    assert first.delete_intent("pay_to_spend_90d")
    assert board("pay_to_spend_90d", first.run_id)["sent"] == "evaluated · removed"

    # The base model is the same in the next evaluation: kept, then reused - not retrained.
    logs = lambda e: " ".join(x.get("message", "") for x in e.events if x["event"] == "eval_log")  # noqa: E731
    assert "reusing the kept model" not in logs(evaluation)
    again = Evaluation(workspace, [b], {})
    rerun = again.run()
    assert "base: reusing the kept model" in logs(again)
    base_row = lambda o: next(r for r in o["comparison"] if r["variant"] == "base")  # noqa: E731
    assert base_row(rerun)["adj_gini_test"] == base_row(out)["adj_gini_test"]


def test_a_request_in_a_mixed_run_is_challenged_not_put_to_the_analyst(no_history):
    sql = ("SELECT customer_id, trans_dt, auth_decline_cnt_30d FROM wwcas_synthetic "
           "WHERE trans_dt >= '2023-01-01'")
    asked = []

    class Recorder:
        def ask(self, kind, payload):
            asked.append(kind)

    session = _session(no_history, "x", approver=Recorder(),
                       params=RunParams(K=2, levels=["L1", "L3"]))
    reply = screen_request(session, "declines are missing", sql, "declines")
    assert reply["recorded"] and "challenge R1" in reply["next"] and asked == []
    assert sql in (session.run_dir / "data_requests" / "R1_declines.sql").read_text()
    assert session.pending_at("L3") == 1 and session.results_at("L3") == 0


def test_a_dropped_source_is_seen_schema_first_then_usable(workspace):
    import pandas as pd

    extra = workspace.extra_dir
    (extra / "declines_data_sample.json").write_text(json.dumps(
        {"customer_id": ["Customer ID", ["1"]], "event_dt": ["Date", ["2024-01-01"]]}))
    try:
        assert workspace.sources()["declines"].usable is False
        pd.DataFrame({"customer_id": ["1"], "event_dt": ["2024-01-01"]}).to_parquet(
            extra / "declines.parquet")
        assert workspace.sources()["declines"].usable is True
    finally:
        for name in ("declines_data_sample.json", "declines.parquet"):
            (extra / name).unlink(missing_ok=True)


def test_discovery_reads_a_sources_fe_cut_and_evaluation_the_whole(no_history):
    import pandas as pd

    workspace = no_history                   # no screen cut from an earlier test's run
    extra = workspace.extra_dir
    whole = workspace.sources()["payments"].data_path
    cut = extra / f"payments_fe{whole.suffix}"
    session = _session(workspace, "payments cut")
    assert session.raw_paths()["payments"] == str(whole)
    pd.read_parquet(whole).head(10).to_parquet(cut)
    try:
        src = workspace.sources()["payments"]
        assert "payments_fe" not in workspace.sources()          # not a source of its own
        assert src.data_path == whole and src.discovery_path == cut
        assert session.raw_paths()["payments"] == str(cut)
    finally:
        cut.unlink(missing_ok=True)


def test_the_analysts_capture_gate_can_hold_back_a_gini_win(no_history):
    workspace = no_history
    session = _session(workspace, "payments vs spend",
                       params=RunParams(K=1, min_gini_gain=0.0, min_capture_gain=1.0))
    for source in ("payments", "spends"):
        assert propose_linkage(session, source, LINK, "event_dt")["ok"]
    reply = screen_feature(session, "pay_to_spend_90d", "d", "L2", PAY_TO_SPEND)
    assert reply["gini_gain"] > 0.03 and not reply["verified"]
    assert "capture-rate gain" in reply["reason"]
    assert reply["gates"] == {"min_gini_gain": 0.0, "min_capture_gain": 1.0}


SPEC = """# Early cures
## 1. Context
Customers who went past due and cured.
## 2. IDs
- {a}
- {b}, {c}
```
{d}
```
## 3. Same examples each discovery?
No - rotate, 2 per batch
"""


def test_a_shot_spec_parses_its_sections():
    from agent.tools.shots import parse_spec

    spec = parse_spec(SPEC.format(a="x1", b="x2", c="x3", d="x4"), "fallback")
    assert spec["name"] == "Early cures"
    assert spec["context"].startswith("Customers who went past due")
    assert spec["ids"] == ["x1", "x2", "x3", "x4"]
    assert spec["rotate"] and spec["batch_size"] == 2
    same = parse_spec("## IDs\nx1 x2\n## Same examples each discovery?\nYes", "mine")
    assert same["name"] == "mine" and not same["rotate"] and same["batch_size"] is None
    with pytest.raises(ValueError, match="no examples found"):
        parse_spec("## Context\nnothing else", "empty")


def test_shot_categories_append_rotate_and_reach_the_agent(workspace, tmp_path):
    from agent.tools import shots
    from agent.tools.shots import categories

    fit = workspace.screen.iloc[: workspace.n_screen_train]
    ids = fit[workspace.id_col].astype(str).tolist()[:4]
    spec = tmp_path / "early_cures.md"
    spec.write_text(SPEC.format(a=ids[0], b=ids[1], c=ids[2], d="not_an_id"))
    fixed = tmp_path / "fixed.md"
    fixed.write_text(f"# Fixed\n## IDs\n{ids[3]}\n## Same examples each discovery?\nyes\n")
    workspace.cfg.discovery.shot_spec_paths = [str(spec), str(fixed)]
    try:
        cats = categories(workspace)
        assert [c.key for c in cats] == ["clustering", "early_cures", "fixed"]
        cures = cats[1]
        assert cures.missing == ["not_an_id"] and cures.batches == 2
        assert cures.for_round(0)[workspace.id_col].tolist() == ids[:2]
        assert cures.for_round(1)[workspace.id_col].tolist() == ids[2:3]
        assert cats[2].for_round(5)[workspace.id_col].tolist() == [ids[3]]

        first, second = _session(workspace, "a"), _session(workspace, "b")
        listing = shots(first, "")
        assert "early_cures" in listing and "fixed" in listing
        assert shots(second, "early_cures") != shots(first, "early_cures")
        assert "Customers who went past due" in shots(first, "Early cures")
    finally:
        workspace.cfg.discovery.shot_spec_paths = []


def test_the_scope_is_read_once_until_a_file_changes(workspace):
    import os

    import pandas as pd

    first = workspace.scope()
    assert workspace.scope() is first                         # no second read
    _, path = workspace.scope_files()[0]
    original = path.read_bytes()                              # the fixture is shared
    try:
        frame = pd.read_csv(path)
        pd.concat([frame, frame.tail(1).assign(NAME="new_var")]).to_csv(path, index=False)
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
        assert len(workspace.scope()) == len(first) + 1       # the change is seen
    finally:
        path.write_bytes(original)
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 2 * 10**9))


def test_a_probe_reads_a_sample_of_a_raw_source_and_linkage_reads_it_all(tmp_path):
    import pandas as pd

    from agent.execution import run_code

    csv = tmp_path / "events.csv"
    # a mixed-type column: chunked parsing would read it as ints, then strings
    pd.DataFrame({"key": list(range(500)) + ["x"] * 500, "v": range(1000)}).to_csv(csv, index=False)
    base = tmp_path / "base.parquet"
    pd.DataFrame({"id": ["a", "b"]}).to_parquet(base)
    common = dict(workdir=tmp_path / "w", engine="pandas", id_col="id", base_path=base,
                  raw={"events": str(csv)}, probe_rows=100)
    probe = run_code("print(len(raw['events']))", "probe", tag="p", **common)
    assert probe.ok and probe.stdout.strip() == "100"
    link = run_code("""
def link(base_ids, source):
    print(len(source), source['key'].map(type).nunique())
    return base_ids.assign(as_of=pd.Timestamp('2024-01-01'))
""", "linkage", source="events", tag="l", **common)
    assert link.ok and link.stdout.split() == ["1000", "1"]          # every row, one type


def test_a_probe_joins_a_confirmed_source_it_names(workspace):
    from agent.tools import run_probe

    first = _session(workspace, "link", approver=AutoApprover(), params=RunParams(K=1))
    assert propose_linkage(first, "payments", LINK, "event_dt")["ok"]
    fresh = _session(workspace, "probe", params=RunParams(K=1))      # nothing joined yet
    reply = run_probe(fresh, 'print(len(sources["payments"]) > 0)', "check")
    assert reply["ok"] and "True" in reply["stdout"]


def test_the_scope_tool_lists_cas_variables_and_sampling_one_explains(workspace):
    from agent.tools import sample_rows, scope

    session = _session(workspace, "x", params=RunParams(K=1, levels=["L3"]))
    unused = scope(session, status="unused_raw")
    assert unused["total"] == 6 and all(v["status"] == "unused_raw" for v in unused["variables"])
    roles = {v["variable"]: v.get("role", "") for v in unused["variables"]}
    assert roles["trans_dt"].startswith("partition") and roles["customer_id"].startswith("identifier")
    assert unused["table_keys"]["wwcas_synthetic"] == {"partition": ["trans_dt"],
                                                       "identifiers": ["customer_id"]}
    assert {v["variable"] for v in scope(session, query="decline")["variables"]} == {"auth_decline_cnt_30d"}
    assert scope(session, table="wwcas_synthetic")["total"] == 14
    assert "no rows in this workspace" in sample_rows(session, "wwcas_synthetic")
    overview = session.ws.catalog("")["scopes"]["CAS"]
    assert overview["tables"]["wwcas_synthetic"]["unused_raw"] == 6
    assert overview["tables"]["wwcas_synthetic"]["partition"] == ["trans_dt"]
    assert overview["sql"] == "BigQuery"
    assert scope(session, scope_name="CAS")["total"] == 14 and scope(session, scope_name="X")["total"] == 0
    assert {v["scope"] for v in scope(session)["variables"]} == {"CAS"}
    assert session.ws.catalog("unused_raw")["total_matches"] == 6     # status is searchable


def test_the_sql_must_use_the_provided_cas_columns(workspace):
    session = _session(workspace, "x", params=RunParams(K=2, levels=["L3"]))
    invented = screen_request(session, "why", "SELECT customer_id, as_of_date, auth_decline_cnt_30d "
                                 "FROM wwcas_synthetic WHERE trans_dt > '2024-01-01'", "a")
    assert not invented["ok"] and "as_of_date" in invented["error"]
    no_id = screen_request(session, "why", "SELECT trans_dt, auth_decline_cnt_30d FROM "
                              "wwcas_synthetic WHERE trans_dt > '2024-01-01'", "b")
    assert not no_id["ok"] and "identifier" in no_id["error"]
    no_filter = screen_request(session, "why", "SELECT customer_id, auth_decline_cnt_30d "
                                  "FROM wwcas_synthetic", "c")
    assert not no_filter["ok"] and "partition date" in no_filter["error"]
    good = screen_request(session, "why", """
        SELECT s.customer_id, DATE_TRUNC(s.trans_dt, MONTH) AS mth, SUM(s.cash_adv_amt_90d) amt
        FROM `proj.cas.wwcas_synthetic` s
        WHERE s.trans_dt BETWEEN @start AND @end GROUP BY s.customer_id, mth""", "d")
    assert good["recorded"] and session.attempts == 1      # refusals spent nothing


def test_a_data_request_run_is_an_sql_run(workspace):
    l3 = Session(workspace, "x", params=RunParams(K=1, levels=["L3"]))
    assert l3.params.engine == "sql" and l3.local_engine == "pandas"     # constructions: screen rows
    with pytest.raises(ValueError, match="data-request runs"):
        Session(workspace, "x", params=RunParams(K=1, levels=["L1", "L3"], engine="sql"))


def test_an_agent_that_stops_without_reporting_is_nudged_to_carry_on(workspace, monkeypatch):
    import asyncio

    import agent.agent as runner
    from agent.composer import message
    from agent.tools import report_findings

    session = _session(workspace, "x", params=RunParams(K=2, levels=["L3"]))
    calls = []

    class Stream:
        def __init__(self, items):
            self.items = items

        def to_input_list(self):
            return [{"role": "user", "content": str(self.items)}]

    async def fake_run(s, items):
        calls.append(items)
        if len(calls) == 3:                          # the third turn reports
            report_findings(s, "done")
        return Stream(items)

    monkeypatch.setattr(runner, "_run", fake_run)
    asyncio.run(runner._run_stage(session, "Direction: x"))
    assert len(calls) == 3 and message('nudge') in str(calls[1])
    assert [e["attempt"] for e in session.events if e["event"] == "agent_nudged"] == [1, 2]
    assert session.finished and session.events[-1]["summary"].startswith("done")


WINDOW = """
def build(spark, sources, base):
    s = sources["spends"]
    recent = s[s["event_dt"] >= s["as_of"] - pd.Timedelta(days={days})]
    out = base[["id"]].copy()
    out["{name}"] = out["id"].map(recent.groupby("id")["amount"].sum()).fillna(0.0)
    return out
"""


def test_a_later_direction_remembers_what_was_proposed(no_history):
    from agent.composer import compose as brief

    workspace = no_history
    first = _session(workspace, "spend level", params=RunParams(K=3, min_gini_gain=-1.0))
    assert propose_linkage(first, "spends", LINK, "event_dt")["ok"]
    assert screen_feature(first, "spend_90d", "90-day spend", "L2",
                          WINDOW.format(days=90, name="spend_90d"))["verified"]

    later = _session(workspace, "spend again", params=RunParams(K=3, min_gini_gain=-1.0))
    assert "`spend_90d`" in brief(later) and "Earlier directions" in brief(later)

    # The same code under a new name is refused before it runs, and costs nothing (C0).
    twin = screen_feature(later, "total_spend_last_quarter", "d", "L2",
                          WINDOW.format(days=90, name="total_spend_last_quarter"))
    assert not twin["ok"] and "the same code as `spend_90d`" in twin["error"]
    assert later.attempts == 0
    # Other code with the same values is refused once built, and costs nothing too.
    rewritten = WINDOW.format(days=90, name="quarter_outlay").replace(
        'out["id"].map(recent.groupby("id")["amount"].sum()).fillna(0.0)',
        'out["id"].map(recent["amount"].groupby(recent["id"]).sum()).fillna(0) * 1.0')
    values_twin = screen_feature(later, "quarter_outlay", "d", "L2", rewritten)
    assert not values_twin["ok"] and "identical to `spend_90d`" in values_twin["error"]
    assert later.attempts == 0
    # So is one that means much the same, until it says what differs.
    alike = screen_feature(later, "spend_90d_total", "90-day spend total", "L2",
                           WINDOW.format(days=60, name="spend_90d_total"))
    assert not alike["ok"] and "differs" in alike["error"] and later.attempts == 0

    # Another window measures something else, and is screened as usual.
    other = screen_feature(later, "spend_30d", "30-day spend", "L2",
                           WINDOW.format(days=30, name="spend_30d"), differs="30 days, not 90")
    # C0 lets it through on its word, and says what it resembled.
    assert later.ledger[-1]["checks"]["seen_before"].startswith("like `spend_90d`")
    assert "verified" in other and later.attempts == 1

    # A deleted feature is forgotten.
    assert first.delete_intent("spend_90d")
    again = screen_feature(later, "spend_quarter", "d", "L2",
                           WINDOW.format(days=90, name="spend_quarter"))
    assert "verified" in again


def test_a_later_data_request_for_the_same_data_is_refused(no_history):
    workspace = no_history
    sql = ("SELECT customer_id, trans_dt, auth_decline_cnt_30d FROM wwcas_synthetic "
           "WHERE trans_dt BETWEEN '2024-01-01' AND '2024-12-31'")
    first = _session(workspace, "declines", approver=AutoApprover())
    assert screen_request(first, "declines are missing", sql, "declines")["recorded"]

    later = _session(workspace, "declines again", approver=AutoApprover())
    narrower = sql.replace("'2024-01-01'", "'2024-06-01'")      # another range, same data
    refused = screen_request(later, "declines", narrower, "decline_counts")
    assert not refused.get("ok", True) and "the same data as `declines`" in refused["error"]
    other = sql.replace("auth_decline_cnt_30d", "cash_adv_amt_90d")
    assert screen_request(later, "cash advances", other, "cash_advances")["recorded"]


def test_a_later_run_is_told_which_cas_columns_are_already_requested(no_history):
    from agent.composer.sections import memory as memory_brief
    from agent.tools import scope

    sql = ("SELECT customer_id, trans_dt, auth_decline_cnt_30d FROM wwcas_synthetic "
           "WHERE trans_dt BETWEEN '2024-01-01' AND '2024-12-31'")
    first = _session(no_history, "declines", approver=AutoApprover())
    assert screen_request(first, "declines are missing", sql, "declines")["recorded"]

    # Up front, before any proposal: the brief names the column, scope() marks it.
    later = _session(no_history, "declines again", approver=AutoApprover())
    assert "`wwcas_synthetic.auth_decline_cnt_30d`" in memory_brief(later)
    marked = {v["variable"]: v.get("requested") for v in scope(later, table="wwcas_synthetic")["variables"]}
    assert marked["auth_decline_cnt_30d"] and "declines" in marked["auth_decline_cnt_30d"][0]
    assert not marked["cash_adv_amt_90d"]
    assert not marked["trans_dt"]                 # the partition date every request selects
    assert "used up" not in memory_brief(later)     # other raw columns are still open


def _ideas(lenses, engines=None):
    from agent.engines import ENGINES

    engines = engines or [ENGINES[i % len(ENGINES)] for i in range(len(lenses))]
    return [{"name": f"idea_{'abcdefghijklmnop'[i]}_{lens}", "engine": engine, "lens": lens,
             "measures": ["first way to measure it", "second way to measure it"],
             "description": f"an idea through {lens}",
             "data": "auth_decline_cnt_30d from the CAS"}
            for i, (lens, engine) in enumerate(zip(lenses, engines))]


def test_proposals_wait_for_ideas_with_a_spread(no_history):
    from agent import tools
    from agent.agent import build_agent
    from agent.composer import compose as brief
    from agent.engines import ENGINES
    from agent.tools.ideas import LENSES, focus_engines, record_ideas

    session = _session(no_history, "x", params=RunParams(K=2, levels=["L3"]))
    session.ideas_required = True                      # as the runner sets it
    text = brief(session, "ideas")                     # the ideas stage's brief
    assert "Ideas first" in text and "| `mechanism` | Mechanism simulation |" in text
    assert "## Skill: ideas" in text
    assert "## Ideas first" not in brief(session)      # not the proposals" brief
    sql = ("SELECT customer_id, trans_dt, auth_decline_cnt_30d FROM wwcas_synthetic "
           "WHERE trans_dt > '2024-01-01'")
    early = screen_request(session, "why", sql, "declines")
    assert not early["ok"] and "the ideas come first" in early["error"] and session.attempts == 0

    focus = focus_engines(session)                     # this round's, drawn and recorded
    assert len(focus) == 2 and focus_engines(session) == focus
    assert [e["focus"] for e in session.events if e["event"] == "engines_drawn"] == [focus]
    others = [k for k in ENGINES if k not in focus]
    lenses = list(LENSES)[:4]
    narrow = record_ideas(session, _ideas(lenses[:1] * 4, focus * 2))         # one lens only
    assert not narrow["ok"] and "lenses" in narrow["error"]
    one_engine = record_ideas(session, _ideas(lenses, [focus[0]] * 4))        # one engine only
    assert not one_engine["ok"] and "[MISSING] at least 3 different engines" in one_engine["error"]
    unfocused = record_ideas(session, _ideas(lenses, others + others[:1]))    # misses the focus
    assert not unfocused["ok"] and "focus engines" in unfocused["error"]
    assert not record_ideas(session, [{"name": "x", "engine": "mechanism", "lens": "vibes",
                                       "description": "x"}])["ok"]
    vague = record_ideas(session, [{"name": "x", "engine": "brainstorm", "lens": "trend",
                                    "description": "x"}])
    assert not vague["ok"] and "engine 'brainstorm'" in vague["error"]
    unnamed = record_ideas(session, [{"name": "Not A Name", "engine": "mechanism", "lens": "trend",
                                      "description": "x"}])
    assert not unnamed["ok"] and "snake_case name" in unnamed["error"]
    spread = record_ideas(session, _ideas(lenses, focus + others[:2]))
    assert spread["ok"] and len(spread["lenses"]) == 4 and len(spread["engines"]) == 4
    saved = json.loads((session.run_dir / "ideas.json").read_text())    # the board reads them
    assert [i["engine"] for i in saved] == focus + others[:2] and saved[0]["round"] == 1
    # With ideas recorded, a proposal says which idea it tests.
    unlinked = screen_request(session, "why", sql, "declines")
    assert not unlinked["ok"] and "say which idea this tests" in unlinked["error"]
    assert not screen_request(session, "why", sql, "declines", idea="no_such")["ok"]
    assert screen_request(session, "why", sql, "declines", idea=saved[0]["name"])["recorded"]
    assert session.data_requests[0]["idea"] == saved[0]["name"]
    assert any(e["event"] == "ideas_recorded" for e in session.events)
    # The ideas are the first stage's structured answer, not a tool.
    assert "brainstorm" not in [t.name for t in tools.for_agent(session)]
    stage = build_agent(session, "ideas")
    assert stage.output_type.output_type is tools.Ideas          # read leniently: FirstAnswer
    assert {t.name for t in stage.tools} == {"engine", "catalog", "scope", "sample_rows", "shots",
                                             "run_probe"}


def test_each_engine_has_its_guide_and_the_focus_follows_their_yield():
    from agent.engines import ENGINES, draw_focus, engine_guide

    assert ENGINES == ("mechanism", "contrastive", "analogy", "morphological", "others")
    assert "Minsky" in engine_guide("mechanism")["guide"]                 # the theory library
    assert "population" in engine_guide("morphological")["guide"]         # the seven axes
    assert not engine_guide("vibes")["ok"]
    board = ([{"engine": "analogy", "status": "active"}] * 12
             + [{"engine": "mechanism", "status": "low"}] * 12)
    draws = [draw_focus(board, seed=str(i))["focus"] for i in range(200)]
    assert sum("analogy" in f for f in draws) > 180            # a strong engine is asked more
    weak = sum("mechanism" in f for f in draws)
    untried = [sum(e in f for f in draws) for e in ("contrastive", "morphological", "others")]
    assert weak < min(untried) and min(untried) > 30           # untried engines still get turns
    assert draw_focus(board, "x") == draw_focus(board, "x")    # seeded, so on record


def test_ideas_that_say_the_same_thing_are_sent_back_and_a_retired_one_refused(no_history, monkeypatch):
    from agent import board
    from agent.tools.ideas import LENSES, focus_engines, record_ideas

    def round_of(session, descriptions):
        focus = focus_engines(session)
        engines = [*focus, *[e for e in ("analogy", "others", "contrastive", "morphological")
                             if e not in focus]][:4]
        return [{"name": name, "engine": engine, "lens": lens, "level": "L1", "description": text,
                 "measures": ["first way to measure it", "second way to measure it"],
                 "data": "x"} for (name, text), engine, lens
                in zip(descriptions, engines, list(LENSES)[:4])]

    first = _session(no_history, "x", params=RunParams(K=2, levels=["L1"]))
    alike = round_of(first, [("pay_drift", "payment day drifting toward the due date"),
                             ("pay_late_drift", "payment day drifting toward the due date each cycle"),
                             ("spend_vol", "volatility of monthly spend"),
                             ("inquiry_burst", "bureau inquiries in a short window")])
    sent_back = record_ideas(first, alike)
    assert not sent_back["ok"] and "say much the same thing" in sent_back["error"]
    alike[1] = {**alike[1], "name": "fee_rise", "description": "fees and charges rising"}
    assert record_ideas(first, alike)["ok"]
    # spend_vol's two features carry no signal: low, with the reason on the board.
    first.ledger += [{"name": "spend_vol", "reason": "explained by the model: ... `spend_3m`"},
                     {"name": "spend_vol_v2", "reason": "no new signal: |ρ| ..."}]
    first.save_ledger()

    def later_round(tag=""):                  # a run that records takes its names
        session = _session(no_history, "y", params=RunParams(K=2, levels=["L1"]))
        return session, round_of(session, [
            (f"monthly_spend_volatility{tag}", "volatility of monthly spend"),
            (f"pay_channel_switch{tag}", "switch of payment channel"),
            (f"limit_raise_surge{tag}", "spend surge after a limit increase"),
            (f"min_pay_cluster{tag}", "payments at exactly the minimum due")])

    # Close in words to an earlier idea and silent on it: asked which it is, with the
    # candidate. Here spend_vol is in the new run's draw already: worked by its name.
    later, again = later_round()
    asked = record_ideas(later, again)
    assert not asked["ok"] and "`monthly_spend_volatility` reads like `spend_vol`" in asked["error"]
    again[0]["same_as"] = "spend_vol"
    in_draw = record_ideas(later, again)
    assert not in_draw["ok"] and 'work it with idea="spend_vol"' in in_draw["error"]
    # A different claim says so, and is new.
    again[0] = {**again[0], "same_as": "", "differs": "volatility net of seasonal peaks"}
    assert "linked" not in record_ideas(later, again)
    # An unknown name is sent back.
    later, again = later_round("_b")
    again[1]["same_as"] = "nope"
    assert "same_as names no earlier idea ('nope')" in record_ideas(later, again)["error"]

    # Not in the draw: linked to it, and starting from its record - so it is drawn late.
    monkeypatch.setattr(board, "PRELOAD", 0)
    later, again = later_round("_c")
    again[0]["same_as"] = "spend_vol"
    # The others are worded as the run above's: as close as that, they must say.
    assert "say for each of these ideas" in record_ideas(later, again)["error"]
    for idea in again[1:]:
        idea["differs"] = "a new population"
    linked = record_ideas(later, again)
    assert linked["ok"] and linked["linked"][0]["same_as"] == f"{first.run_id}:spend_vol", linked
    assert linked["linked"][0]["p_useful"] == 0.333 and "no new signal" in linked["linked"][0]["why_not"]
    from agent.board import session_ideas

    standing = {i["name"]: i for i in session_ideas(later)}
    assert standing["monthly_spend_volatility_c"]["status"] == "low"
    assert standing["pay_channel_switch_c"]["status"] == "explore"


def test_an_idea_keeps_one_record_across_runs_and_earlier_ideas_join_the_draw(no_history, monkeypatch):
    from agent import board
    from agent.board import board_rows, idea_for, idea_records, session_ideas
    from agent.tools.ideas import LENSES, focus_engines, record_ideas

    def round_of(session, descriptions):
        focus = focus_engines(session)
        engines = [*focus, *[e for e in ("analogy", "others", "contrastive", "morphological")
                             if e not in focus]][:4]
        return [{"name": name, "engine": engine, "lens": lens, "level": "L1", "description": text,
                 "measures": ["first way to measure it", "second way to measure it"],
                 "data": "x"} for (name, text), engine, lens
                in zip(descriptions, engines, list(LENSES)[:4])]

    def run(direction):
        session = _session(no_history, direction, params=RunParams(K=2, levels=["L1"]))
        session.start()
        return session

    first = run("cash stress")
    assert record_ideas(first, round_of(first, [
        ("pay_drift", "payment day drifting toward the due date"),
        ("spend_vol", "volatility of monthly spend"),
        ("fee_rise", "fees and charges rising"),
        ("inquiry_burst", "bureau inquiries in a short window")]))["ok"]
    first.ledger += [{"name": "spend_vol_6m", "idea": "spend_vol", "reason": "no new signal: |ρ| ..."}]
    first.save_ledger()

    # Another direction gives an idea like spend_vol: linked, and its feature that
    # works counts for spend_vol too - in the first run's row of the board as well.
    monkeypatch.setattr(board, "PRELOAD", 0)
    other = run("spending habits")
    restated = round_of(other, [
        ("monthly_spend_volatility", "volatility of monthly spend"),
        ("pay_channel_switch", "switch of payment channel"),
        ("limit_raise_surge", "spend surge after a limit increase"),
        ("min_pay_cluster", "payments at exactly the minimum due")])
    restated[0]["same_as"] = "spend_vol"
    # The others read a little like earlier ones: asked once - silent again, they are new.
    assert "say for each of these ideas" in record_ideas(other, restated)["error"]
    assert record_ideas(other, restated)["ok"]
    monkeypatch.setattr(board, "PRELOAD", 20)
    other.ledger += [{"name": "spend_cv_12m", "idea": "monthly_spend_volatility", "verified": True}]
    other.save_ledger()
    records = {r["key"]: r for r in idea_records(no_history)}
    for key in (f"{first.run_id}:spend_vol", f"{other.run_id}:monthly_spend_volatility"):
        assert records[key]["signal"] == 1 and records[key]["none"] == 1
        assert records[key]["p_useful"] == 0.571 and records[key]["status"] == "active"
    row = next(r for r in board_rows(no_history) if r["name"] == "spend_vol_6m")
    assert row["p_useful"] == 0.571
    # A run's own ideas don't pool with others: an unlinked idea keeps its own record.
    assert records[f"{first.run_id}:pay_drift"]["status"] == "explore"

    # A new run starts with the best earlier ideas in its draw, before any are given -
    # each family once, and those of runs with the same direction first.
    again = run("Cash  stress")
    drawn = {i["name"]: i for i in session_ideas(again)}
    assert {"pay_drift", "fee_rise", "inquiry_burst", "pay_channel_switch"} <= set(drawn)
    assert not {"spend_vol", "monthly_spend_volatility"} <= set(drawn)
    assert drawn["pay_drift"]["carried_from"] == first.run_id
    monkeypatch.setattr(board, "PRELOAD", 2)
    again = run("Cash  stress")
    carried = [i for i in session_ideas(again) if i.get("carried_from")]
    assert len(carried) == 2 and all(i["carried_from"] == first.run_id for i in carried)
    drawn = {i["name"]: i for i in carried}
    if "spend_vol" in drawn:                              # the family's best, when it is first's
        assert drawn["spend_vol"]["p_useful"] == 0.571
    drawn = {i["name"]: i for i in session_ideas(again)}
    if "pay_drift" not in drawn:
        again.carried.append(f"{first.run_id}:pay_drift")
    # A feature that names a carried idea brings it into the run, linked to its family.
    assert idea_for(again, "pay_drift", "pay_day_gap_3m") == ("pay_drift", None)
    assert again.ideas[-1]["same_as"] == f"{first.run_id}:pay_drift" and again.ideas[-1]["carried"]
    again.ledger += [{"name": "pay_day_gap_3m", "idea": "pay_drift", "reason": "no new signal: ..."}]
    drawn = {i["name"]: i for i in session_ideas(again)}
    assert "carried_from" not in drawn["pay_drift"] and drawn["pay_drift"]["status"] == "low"

    # Beyond the ones it starts with, the agent names an earlier idea in carry to draw
    # it; an unknown name is sent back.
    monkeypatch.setattr(board, "PRELOAD", 1)
    new = run("bureau signals")
    ideas = round_of(new, [("score_slide", "bureau score sliding"),
                           ("new_trades", "many new trade lines opened"),
                           ("util_jump", "utilisation jumping across lenders"),
                           ("closed_accounts", "accounts closed by lenders")])
    assert "no earlier idea named 'nope'" in record_ideas(new, ideas, carry=["nope"])["error"]
    assert record_ideas(new, ideas, carry=["inquiry_burst"])["ok"]
    drawn = {i["name"]: i for i in session_ideas(new)}
    assert drawn["inquiry_burst"]["carried_from"] == first.run_id and "pay_drift" not in drawn


def test_ideas_come_in_rounds_and_a_round_ends_when_its_ideas_are_used(no_history):
    from agent.engines import ENGINES
    from agent.tools.ideas import LENSES, focus_engines, ideas_in_round, record_ideas

    big = _session(no_history, "x", params=RunParams(K=100, levels=["L1"], ideas_per_round=10))
    assert ideas_in_round(big) == 10                      # a large K: ten at a time
    too_many = record_ideas(big, _ideas((list(LENSES) * 2)[:11]))
    assert "[MISSING] 10 to 10 ideas (you gave 11)" in too_many["error"]
    session = _session(no_history, "x", params=RunParams(K=2, levels=["L1"], ideas_per_round=10))
    assert ideas_in_round(session) == 4                   # a few more than still wanted
    lenses = list(LENSES)[:4]
    engines = list(dict.fromkeys([*focus_engines(session), *ENGINES]))[:4]
    # Every requirement comes back, met or not - fixing one must not break another.
    short = record_ideas(session, _ideas(lenses[:1] * 2, engines[:2]))
    assert "[MISSING] 4 to 10 ideas (you gave 2)" in short["error"]
    assert "[MISSING] at least 4 different lenses" in short["error"]
    assert record_ideas(session, _ideas(lenses, engines))["round"] == 1
    for _ in range(3):
        session.take_attempt()
    assert session.round_over() is None                   # one attempt of the round left
    session.take_attempt()
    assert "round 1's ideas are used" in session.round_over() and session.round_spent
    again = record_ideas(session, _ideas(lenses, engines))       # the same names as round 1
    assert not again["ok"] and "earlier round" in again["error"]


def test_the_report_check_stays_on_when_the_ideas_never_pass(workspace):
    from agent.tools import report_findings

    session = _session(workspace, "x", params=RunParams(K=2, levels=["L1"]))
    session.gated, session.ideas_required = True, False   # ideas failed: proposals ungated
    assert not report_findings(session, "early")["ok"]


def test_an_l3_idea_must_name_cas_data(no_history):
    from agent.tools.ideas import record_ideas

    session = _session(no_history, "x", params=RunParams(K=2, levels=["L3"]))
    from_sources = record_ideas(session, [{"name": "spend_trend", "engine": "mechanism", "measures": ["first way to measure it", "second way to measure it"], 
                                         "lens": "trend", "level": "L3",
                                         "description": "spend trend over 3 months",
                                         "data": "spends source"}])
    assert not from_sources["ok"] and "L1/L2 idea" in from_sources["error"]


def test_an_l3_idea_on_variables_requested_already_is_sent_back(no_history):
    from agent.tools.ideas import record_ideas

    session = _session(no_history, "x", params=RunParams(K=2, levels=["L3"]))
    session.data_requests = [{"intent": "R1", "source_name": "cas_declines", "status": "kept",
                              "tables": ["wwcas_synthetic"],
                              "columns": ["customer_id", "trans_dt", "auth_decline_cnt_30d"]}]
    again = record_ideas(session, [{"name": "decline_streaks", "engine": "mechanism", "lens": "trend",
                                    "level": "L3", "measures": ["streak of months with declines",
                                                                "longest run of declines"],
                                    "description": "runs of months with declines",
                                    "data": "auth_decline_cnt_30d"}])
    assert not again["ok"] and "requested already" in again["error"] and "cas_declines" in again["error"]


def test_a_mixed_run_splits_its_target_by_level(workspace):
    from collections import Counter

    from agent.session import level_quota

    workspace.cfg.discovery.additional_data.linkage_dir = str(workspace.linkage_dir)      # payments/spends linked
    params = RunParams(K=2000, levels=["L1", "L2", "L3"]).resolve(workspace)
    split = level_quota(workspace, params, "seed")
    assert list(split) == ["L2", "L1", "L3"] and sum(split.values()) == 2000
    assert split["L2"] > split["L1"] > split["L3"]                     # p1 > p2 > p3
    assert level_quota(workspace, params, "seed") == split           # fixed by the run id

    no_sources = RunParams(K=2000, levels=["L1", "L2", "L3"], sources=[]).resolve(workspace)
    assert list(level_quota(workspace, no_sources, "seed")) == ["L1", "L3"]  # no L2 without data

    session = Session(workspace, "x", params=RunParams(K=3, levels=["L1", "L3"]))
    session.quota = {"L1": 1, "L3": 2}
    ratio = """
def build(spark, sources, base):
    out = base[["id"]].copy()
    out["inc_lim"] = base["income_est"] / base["credit_limit"]
    return out
"""
    assert screen_feature(session, "inc_lim", "d", "L1", ratio)["intent"] == "I1"
    # An L1 attempt that is not verified does not use up L1: its target is a result.
    session.ledger[0]["verified"] = False
    other = ratio.replace("inc_lim", "lim_x_inc").replace('base["income_est"] / base["credit_limit"]',
                                                          'base["credit_limit"] * base["income_est"]')
    assert screen_feature(session, "lim_x_inc", "d", "L1", other, differs="a product")["intent"] == "I2"
    # Once L1 has its result, L1 takes no more - the rest of the target is L3.
    session.ledger[0]["verified"] = True
    third = screen_feature(session, "util_sq", "d", "L1", ratio.replace("inc_lim", "util_sq"))
    assert not third["ok"] and "L1 target is met" in third["error"]
    assert session.attempts == 2 and "L3 2" in session.budget()


def test_once_the_cas_scope_is_used_up_l3_looks_beyond_it(no_history):
    from agent.composer import compose as brief
    from agent.session import level_quota

    params = RunParams(K=200, levels=["L1", "L3"]).resolve(no_history)
    first = _session(no_history, "everything", approver=AutoApprover())
    for column in ("auth_decline_cnt_30d", "cash_adv_amt_90d", "merchant_country"):
        sql = (f"SELECT customer_id, trans_dt, {column} FROM wwcas_synthetic "
               "WHERE trans_dt BETWEEN '2024-01-01' AND '2024-12-31'")
        assert screen_request(first, "why", sql, column)["recorded"]

    # Every unused_raw column is asked for - L3 keeps its share: the room is beyond scope.
    assert "L3" in level_quota(no_history, params, "seed")
    later = Session(no_history, "more", params=RunParams(K=3, levels=["L3"]))
    text = brief(later)
    assert "the room is beyond scope" in text and "propose_new_data" in text


def test_a_request_beyond_the_cas_scope_is_an_idea_with_no_sql_and_is_challenged(no_history):
    from agent.tools import challenge_request, propose_new_data
    from agent.tools.ideas import record_ideas
    from agent.tools.report import validated_sql

    session = _session(no_history, "x", params=RunParams(K=3, levels=["L3"]))
    vague = propose_new_data(session, "why", "rla_treatments", "RLA data")
    assert not vague["ok"] and session.attempts == 0          # say what, and from where
    reply = propose_new_data(session, "Line actions tell us what the bank already saw.",
                             "rla_treatments", "RLA strategy log: the treatment applied to "
                             "each account per month, 24 months back", "rla_cut_last_6m")
    assert reply["recorded"] and reply["intent"] == "R1" and session.attempts == 1
    record = session.data_requests[0]
    assert record["scope"] == "beyond_scope" and record["sql"] == "" and record["tables"] == []
    assert not (session.run_dir / "data_requests" / "R1_rla_treatments.sql").exists()

    verdict = challenge_request(session, "R1", "new", "no source carries treatments")
    assert verdict["status"] == "kept"
    summary = validated_sql(session)
    assert "Beyond scope" in summary and "RLA strategy log" in summary
    assert "Beyond scope - the data it needs" in \
        (session.run_dir / "data_requests.md").read_text()

    # An L3 idea names scope variables, or is marked beyond scope and says what it needs.
    base = {"engine": "mechanism", "lens": "trend", "level": "L3", "description": "d",
            "measures": ["first way to measure it", "second way to measure it"]}
    named = record_ideas(session, [{**base, "name": "a", "data": "rla"}])
    assert not named["ok"] and "Within a scope (CAS)" in named["error"]
    for field in ("beyond_scope", "beyond_cas"):                   # the old name still reads
        beyond = record_ideas(session, [{**base, "name": "a", field: True, "data": "x"}])
        assert not beyond["ok"] and "where it would come from" in beyond["error"]


def test_an_l3_report_with_requests_left_is_sent_back_while_the_scope_has_room(no_history):
    from agent.tools import report_findings

    session = Session(no_history, "x", params=RunParams(K=2, levels=["L3"]))
    session.gated = True
    early = report_findings(session, "nothing to ask")
    assert not early["ok"] and "2 more result(s) wanted" in early["error"]


def test_a_feature_reading_a_source_that_does_not_exist_costs_nothing(workspace):
    session = _session(workspace, "x", params=RunParams(K=2, levels=["L1"]))
    code = """
def build(spark, sources, base):
    s = sources["cash_advance_temporal"]
    return base[["id"]].assign(x=1.0)
"""
    reply = screen_feature(session, "x", "d", "L1", code)
    assert not reply["ok"] and "A data request is not data" in reply["error"]
    assert session.attempts == 0


def test_a_report_with_feature_intents_left_is_sent_back(workspace):
    from agent.tools import report_findings

    session = _session(workspace, "x", params=RunParams(K=3, levels=["L1"]))
    session.gated = True                         # as the runner sets it
    first = report_findings(session, "done early")
    assert not first["ok"] and "3 more result(s) wanted" in first["error"]
    assert not session.finished
    assert not report_findings(session, "still early")["ok"]
    assert report_findings(session, "exhausted, because ...")["ok"]       # it may insist
    assert session.finished


def test_insisting_on_an_early_report_after_a_round_that_found_something_asks_for_a_new_round(no_history):
    from agent.tools import report_findings

    session = Session(no_history, "x", params=RunParams(K=10, levels=["L3"]))
    session.gated, session.ideas_required = True, True      # as the runner sets them
    session.data_requests = [{"intent": f"R{i}", "status": "kept", "source_name": "t"}
                             for i in range(8)]              # this round kept 8
    for _ in range(2):
        assert not report_findings(session, "exhausted")["ok"]
    insisted = report_findings(session, "exhausted, because ...")
    assert not insisted["ok"] and "new round of ideas" in insisted["error"]
    assert session.round_spent and not session.finished
    # A round that finds nothing may end the run.
    session.round_results, session.report_refusals, session.round_spent = 8, 0, False
    for _ in range(2):
        assert not report_findings(session, "exhausted")["ok"]
    assert report_findings(session, "exhausted, because ...")["ok"] and session.finished


def test_a_dropped_request_is_not_a_result_and_another_is_proposed(no_history):
    from agent.tools import challenge_request

    session = _session(no_history, "x", params=RunParams(K=1, levels=["L3"]))
    sql = ("SELECT customer_id, trans_dt, utilization FROM wwcas_synthetic "
           "WHERE trans_dt > '2024-01-01'")
    assert screen_request(session, "why", sql, "util")["recorded"]
    assert session.attempts == 1
    construction = """
def build(spark, sources, base):
    out = base[["id"]].copy()
    out["proxy"] = base["utilization"]
    return out
"""
    dropped = challenge_request(session, "R1", "constructible", "a base column", ["utilization"],
                                construction)
    assert dropped["status"] == "dropped" and "not a result" in dropped["next"]
    assert session.results() == 0 and session.wanted() == 1         # still one to find
    again = screen_request(session, "why", sql.replace("utilization", "auth_decline_cnt_30d"),
                              "declines")
    assert again["recorded"] and again["intent"] == "R2" and session.attempts == 2


def test_a_feature_brief_lists_the_columns_it_can_read(workspace):
    from agent.composer import compose

    session = Session(workspace, "x", params=RunParams(K=2, levels=["L1", "L2"]))
    text = compose(session)
    section = text[text.index("## The columns you can use"):text.index("## This run")]
    for column in workspace.base_features:
        assert f"`{column}` - " in section
    assert f"`{workspace.base_features[0]}` - {workspace.descriptions[workspace.base_features[0]]}" in section
    source = session.params.sources[0]
    for column, description in workspace.sources()[source].columns.items():
        assert f"`{column}` - {description}" in section


def test_the_composer_puts_the_prompt_together_and_can_print_it(workspace, capsys):
    from agent.composer import TEMPLATES_DIR, compose, message, template_for
    from agent.composer.__main__ import main
    from agent.session import SKILLS_DIR

    names = sorted(p.name for p in TEMPLATES_DIR.glob("*.md"))
    assert names == ["data_scout.md", "feature_engineer.md", "linkage_writer.md", "messages.md"]
    assert SKILLS_DIR.name == "skills" and SKILLS_DIR.parent.name == "agent"     # beside tools/
    assert sorted(p.stem for p in SKILLS_DIR.glob("*.md")) == ["data_sourcing", "evaluate", "feature",
                                                                "ideas"]
    l3 = Session(workspace, "x", params=RunParams(K=2, levels=["L3"]))
    assert template_for(l3) == "data_scout" and "data scout" in compose(l3)
    assert message("propose", round=1, budget="0/2 results", targets_left="2 results",
                   round_ideas="(ideas)", gates="Gini gain above +0.0000").startswith("Round 1 - proposals")

    from pathlib import Path

    runs = Path(workspace.cfg.agent.run_dir)
    before = sorted(p.name for p in runs.rglob("*"))
    main(["-c", str(workspace.extra_dir.parent.parent / "cfg.yaml"), "--levels", "L3", "-k", "3"])
    out = capsys.readouterr().out
    assert "# brief (templates/data_scout.md)" in out and "Stage 1 of 2" in out
    assert sorted(p.name for p in runs.rglob("*")) == before   # a dry run leaves no trace


def test_a_script_the_guard_refuses_spends_nothing(workspace):
    session = _session(workspace, "x", params=RunParams(K=1))
    reply = screen_feature(session, "peek", "d", "L1",
                           "def build(spark, sources, base):\n    return pd.read_csv('data/test.csv')")
    assert not reply["ok"] and "Nothing was spent" in reply["error"] and session.attempts == 0
    assert not any(e["event"] == "code_status" for e in session.events)       # never ran


def test_a_feature_loads_only_the_source_columns_it_names(tmp_path):
    import pandas as pd

    from agent._child import named_columns
    from agent.execution import run_code

    cols = ["id", "as_of", "amount", "merchant", "channel"]
    assert named_columns('s["amount"]', cols, {"id", "as_of"}) == ["id", "as_of", "amount"]
    assert named_columns("s.select_dtypes('number')", cols, {"id", "as_of"}) is None

    linked = tmp_path / "linked.parquet"
    pd.DataFrame({"id": ["a", "b"], "as_of": pd.to_datetime(["2024-01-01"] * 2),
                  "amount": [1.0, 2.0], "merchant": ["m", "n"]}).to_parquet(linked)
    base = tmp_path / "base.parquet"
    pd.DataFrame({"id": ["a", "b"]}).to_parquet(base)
    result = run_code('''
def build(spark, sources, base):
    s = sources["spends"]
    print(sorted(s.columns))
    return s.groupby("id")["amount"].sum().rename("f").reset_index()
''', "feature", workdir=tmp_path / "w", engine="pandas", id_col="id", base_path=base,
                      sources={"spends": str(linked)}, tag="f")
    assert result.ok, result.error
    assert result.stdout.strip() == "['amount', 'as_of', 'id']"
    assert result.peak_mb is not None


def test_the_brief_gives_the_full_size_and_a_screen_warns_what_will_not_scale(workspace,
                                                                              monkeypatch):
    from agent.composer import compose
    from agent.tools import screen as screen_tool

    size = workspace.data_size()
    assert size["full_rows"] > len(workspace.screen)
    assert size["sources"]["spends"]["rows"] > 0
    session = _session(workspace, "x", params=RunParams(K=2))
    assert "## Data size - write for the full data" in compose(session)

    class Run:
        ok, elapsed_s, setup_s, peak_mb = True, 1.0, 0.0, 10.0

    assert screen_tool.scale_note(session, Run()) is None
    monkeypatch.setattr(workspace.cfg.agent.timeouts, "eval_s", 1.0)
    assert "min (the limit is" in screen_tool.scale_note(session, Run())


def test_the_skills_pandas_linkage_example_links_point_in_time(workspace):
    import re as _re

    skill = open("src/agent/skills/data_sourcing.md").read()
    example = _re.findall(r"```python\n(def link.*?)```", skill, _re.S)[-1]
    code = (example.replace("<id>", "id").replace("<key>", "customer_id")
            .replace("<event_date>", "event_dt").replace("<column>", "amount"))
    session = _session(workspace, "link", approver=AutoApprover(), params=RunParams(K=1))
    reply = propose_linkage(session, "spends", code, "event_dt")
    assert reply["ok"], reply
    assert reply["checks"]["point_in_time_violations"] == 0


def test_a_spark_script_without_pyspark_is_refused_with_a_reason(tmp_path, monkeypatch):
    from agent import execution

    monkeypatch.setattr(execution, "_SPARK", False)
    result = execution.run_code("def link(base_ids, source): return source", "linkage",
                                workdir=tmp_path, engine="spark", id_col="id",
                                base_path=tmp_path / "base.parquet")
    assert not result.ok and "pyspark is not installed" in result.error


def test_a_merge_is_sized_before_it_runs():
    import pandas as pd

    from agent._child import size_merge

    events = pd.DataFrame({"id": ["a"] * 3 + ["b"] * 2, "v": range(5)})
    per_id = pd.DataFrame({"id": ["a", "b"], "w": [1, 2]})
    assert size_merge(events, events, on="id") == {
        "rows": 3 * 3 + 2 * 2, "many_to_many": True, "left_per_key": 3.0, "right_per_key": 3.0}
    one = size_merge(events, per_id, on="id", how="left")
    assert one["many_to_many"] is False and one["rows"] == 5 + 5


def test_a_feature_may_not_join_events_to_events_but_a_linkage_joins_ids_to_events(tmp_path):
    import pandas as pd

    from agent.execution import run_code

    linked = tmp_path / "linked.parquet"
    pd.DataFrame({"id": ["a"] * 4 + ["b"] * 4, "as_of": pd.to_datetime(["2024-01-01"] * 8),
                  "amount": range(8)}).to_parquet(linked)
    base = tmp_path / "base.parquet"
    pd.DataFrame({"id": ["a", "b"]}).to_parquet(base)
    common = dict(workdir=tmp_path / "w", engine="pandas", id_col="id", base_path=base)
    self_join = run_code('''
def build(spark, sources, base):
    s = sources["spends"][["id", "amount"]]
    pairs = s.merge(s, on="id")
    return pairs.groupby("id")["amount_x"].sum().rename("f").reset_index()
''', "feature", sources={"spends": str(linked)}, tag="f", **common)
    assert not self_join.ok and "many-to-many merge" in self_join.error

    raw = tmp_path / "events.csv"
    pd.DataFrame({"key": ["a", "a", "b"], "v": [1, 2, 3]}).to_csv(raw, index=False)
    link = run_code('''
def link(base_ids, source):
    ids = pd.concat([base_ids, base_ids]).assign(key=lambda d: d["id"],
        as_of=pd.Timestamp("2024-01-01"))
    return ids.merge(source, on="key")
''', "linkage", raw={"events": str(raw)}, source="events", tag="l", **common)
    assert link.ok, link.error


def test_an_exploration_draws_a_new_theme_each_round(no_history):
    from agent import themes
    from agent.tools.ideas import record_ideas

    themes.save(no_history, ["payment counts", "spend volatility", "income relative to the line"])
    directed = _session(no_history, "x", params=RunParams(K=2, levels=["L1"]))
    assert not themes.draw_theme(directed)["ok"]                 # a direction is followed

    session = _session(no_history, themes.OPEN, explore=True, params=RunParams(K=2, levels=["L1"]))
    ideas = [{"name": f"i{n}", "engine": "analogy", "lens": lens, "level": "L1", "description": "d", "measures": ["first way to measure it", "second way to measure it"], 
              "data": "x"}
             for n, lens in enumerate(["trend", "ratio", "volatility", "recency"])]
    refused = record_ideas(session, ideas)
    assert not refused["ok"] and "draw_theme" in refused["error"]  # the theme comes first
    first = themes.draw_theme(session)
    assert first["ok"] and first["round"] == 1 and first["explored_before"] == 0
    assert themes.draw_theme(session)["theme"] == first["theme"]  # one per round
    record_ideas(session, ideas)
    session.round = 1
    second = themes.draw_theme(session)
    assert second["round"] == 2 and second["theme"] != first["theme"]
    drawn = [e["theme"] for e in session.events if e["event"] == "theme_drawn"]
    assert drawn == [first["theme"], second["theme"]]
    assert themes.explored(no_history)[first["theme"].lower()] == 1


def test_the_cheap_checks_see_code_meaning_and_what_the_model_misses():
    import numpy as np
    import pandas as pd

    from agent.checks import Sketch, code_fingerprint, model_checks, similarity
    from validation.config import ChecksConfig

    # C0 - code: the same computation under other names is the same fingerprint.
    a = 'def build(spark, sources, base):\n    x = base["a"] / base["b"]  # ratio\n    return base.assign(r1=x)[["id", "r1"]]'
    b = 'def build(spark, sources, base):\n    """Doc."""\n    ratio = base["a"] / base["b"]\n    return base.assign(r2=ratio)[["id", "r2"]]'
    assert code_fingerprint(a, "r1") == code_fingerprint(b, "r2")
    assert code_fingerprint(a, "r1") != code_fingerprint(a.replace('"b"', '"c"'), "r1")
    assert code_fingerprint("def build(:", "x") is None
    # C0 - meaning: another window reads alike; another measure does not.
    near, far = similarity("spend_vol_90d volatility of monthly spend over 90 days",
                           ["spend_vol_30d volatility of monthly spend over 30 days",
                            "days past due count in the last year"])
    assert near > 0.6 > far

    rng = np.random.default_rng(0)
    n = 2000
    base = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    hidden = rng.normal(size=n)
    y = (base["x1"] + hidden + rng.normal(size=n) > 0).astype(float).to_numpy()
    eval_rows = np.arange(1000, n)
    sketch = Sketch(base, eval_rows, y, 1 / (1 + np.exp(-base["x1"].to_numpy()[eval_rows])))
    screener = type("S", (), {"_agent_sketch": sketch})()
    # C2 is information only by default: measured, never a failure.
    noise = rng.normal(size=n)
    _, default = model_checks(screener, noise, ChecksConfig())
    assert default == "" and ChecksConfig().min_variable_gini is None
    checks = ChecksConfig(min_variable_gini=0.05)                # made a gate, as the config can
    # C1: a rescaled model column is a near-copy - |rho| 1 with it, named.
    measured, reason = model_checks(screener, (base["x1"] * 3).to_numpy(), checks)
    assert measured["max_model_rho"] > 0.99 and measured["nearest"] == "x1"
    assert reason.startswith("explained by the model")
    # Pairwise by design: the sum of two columns is close to neither - the fit judges it.
    mixed, ok_mixed = model_checks(screener, (base["x1"] + base["x2"]).to_numpy(), checks)
    assert mixed["max_model_rho"] < 0.9 and not ok_mixed.startswith("explained")
    # C2: noise alone ranks nothing; the hidden driver does - its variable Gini is
    # |2·AUC - 1| of it as a score for y.
    _, gated = model_checks(screener, noise, checks)
    assert gated.startswith("no signal")
    found, ok = model_checks(screener, hidden, checks)
    assert not ok and found["variable_gini"] > 0.3 and found["max_model_rho"] < 0.1
    from sklearn.metrics import roc_auc_score
    assert found["variable_gini"] == pytest.approx(abs(2 * roc_auc_score(y, hidden) - 1), abs=1e-4)
    # C1's exception still reads the residual: the hidden driver tracks it.
    assert found["residual_rho"] > 0.2


def test_the_model_history_reaches_the_brief_and_check_c0(no_history, tmp_path):
    from agent.composer import compose as brief
    from agent.model_history import read

    from pathlib import Path

    empty = read(Path(__file__).resolve().parents[1] / "data/cdss_us_sbs/model_history.md")         # the template: Gen9, no rows
    assert empty == {"current": "Gen9", "variables": []}
    doc = tmp_path / "history.md"
    doc.write_text("Current model: Gen9\n\n| generation | variable | description | status | why |\n"
                   "| --- | --- | --- | --- | --- |\n"
                   "| Gen7 | income_to_limit | income relative to the credit limit | dropped | unstable across vintages |\n")
    no_history.cfg.agent.model_history = str(doc)
    try:
        session = _session(no_history, "x", params=RunParams(K=2, levels=["L1"]))
        assert "`income_to_limit` (Gen7, dropped: unstable across vintages)" in brief(session)
        code = "def build(spark, sources, base):\n    out = base[['id']].copy()\n    out['inc_lim'] = base['income_est'] / base['credit_limit']\n    return out"
        seen = screen_feature(session, "inc_lim", "income relative to the credit limit", "L1", code)
        assert not seen["ok"] and "Gen7 - dropped (unstable across vintages)" in seen["error"]
        assert session.attempts == 0
        fixed = screen_feature(session, "inc_lim", "income relative to the credit limit", "L1", code,
                               differs="six-month median income, which is stable across vintages")
        assert "intent" in fixed
    finally:
        no_history.cfg.agent.model_history = None


def test_a_request_on_the_board_is_kept_or_dropped_and_says_why():
    from agent.board import _evaluation, _request_outcome

    challenged = {"status": "dropped", "challenge": {"reasoning": "utilisation is a base column"}}
    assert _request_outcome(challenged, True) == (
        "dropped", "challenge: the current data already supplies it - utilisation is a base column")
    assert _request_outcome({"status": "kept", "deleted": True}, True) == ("dropped", "removed by the analyst")
    assert _request_outcome({"approved": False, "note": "too costly"}, True) == (
        "dropped", "rejected by the analyst: too costly")
    assert _request_outcome({"approved": True}, True)[0] == "kept"
    assert _request_outcome({"status": "proposed"}, True)[0] == "dropped"     # the run ended first
    assert _request_outcome({"status": "proposed"}, False)[0] == "pending"    # still running
    # Evaluation is for verified features only - whether it was evaluated, not how it did.
    assert _evaluation({"verified": False}, "r:f", {}) == ""
    assert _evaluation({"verified": True}, "r:f", {}) == "not evaluated"
    assert _evaluation({"verified": True, "deleted": True}, "r:f", {}) == "removed"
    assert _evaluation({"verified": True}, "r:f", {"r:f": {"removed": False}}) == "evaluated"
    assert _evaluation({"verified": True}, "r:f", {"r:f": {"removed": True}}) == "evaluated · removed"


def test_a_round_that_keeps_missing_the_spread_still_keeps_its_ideas(no_history):
    from agent.tools.ideas import focus_engines, record_ideas

    # The run that lost its ideas: three per round, L1 and L2 both wanted.
    session = _session(no_history, "x", params=RunParams(K=10, levels=["L1", "L2"], ideas_per_round=3))
    session.quota = {"L2": 6, "L1": 4}
    focus = focus_engines(session)
    narrow = [{"name": f"i{n}", "engine": focus[n % 2], "lens": lens, "level": "L1", "measures": ["first way to measure it", "second way to measure it"], 
               "description": f"idea {n}", "data": "x"}
              for n, lens in enumerate(["calendar", "recency", "trend"])]
    sent_back = record_ideas(session, narrow)
    assert not sent_back["ok"]
    # It says what to change, not only what is missing.
    assert "To fix: change the engine of 1 idea(s) to one not used yet" in sent_back["error"]
    assert "make one idea level L2" in sent_back["error"]
    # The last resort: recorded anyway, with the rules it missed - never lost.
    kept = record_ideas(session, narrow + [{"name": "Bad Name", "engine": "x", "lens": "trend"}],
                        lenient=True)
    assert kept["ok"] and [i["name"] for i in session.ideas] == ["i0", "i1", "i2"]
    recorded = next(e for e in session.events if e["event"] == "ideas_recorded")
    assert recorded["lenient"] and any("engines" in m for m in recorded["missed"])
    assert any("snake_case" in m for m in recorded["missed"])           # the invalid one, skipped
    assert (session.run_dir / "ideas.json").exists()                     # on the board


def test_an_idea_is_a_hypothesis_its_features_name_it(workspace):
    from agent.board import board_rows
    from agent.tools.ideas import focus_engines, record_ideas

    session = _session(workspace, "x", params=RunParams(K=2, levels=["L1"]))
    focus = focus_engines(session)
    engines = [*focus, *[e for e in ("analogy", "others", "contrastive", "mechanism") if e not in focus]][:4]
    good = [{"name": n, "engine": e, "lens": l, "level": "L1", "description": f"the {n} hypothesis",
             "measures": [f"{n} measured one way", f"{n} measured another way"], "data": "base"}
            for n, e, l in zip(["capacity_strain", "limit_dependence", "tenure_maturity", "income_mismatch"],
                               engines, ["capacity", "level", "trend", "peer"])]
    # A feature's name is not an idea's, and one way to measure it is not enough.
    feature_like = record_ideas(session, [{**good[0], "name": "income_to_limit_6m"}, *good[1:]])
    assert not feature_like["ok"] and "reads like a feature" in feature_like["error"]
    one_measure = record_ideas(session, [{**good[0], "measures": ["just one"]}, *good[1:]])
    assert not one_measure["ok"] and "2 to 5 measures" in one_measure["error"]
    assert record_ideas(session, good)["ok"]
    # Features are named for what they compute, and say which idea they test.
    code = "def build(spark, sources, base):\n    out = base[['id']].copy()\n    out['{n}'] = base['income_est'] / base['credit_limit'] ** 0.5\n    return out"
    reply = screen_feature(session, "income_over_limit", "income per unit of limit", "L1",
                           code.format(n="income_over_limit"), idea="capacity_strain")
    assert "intent" in reply and session.ledger[0]["idea"] == "capacity_strain"
    assert session.ledger[0]["checks"]["seen_before"].startswith("new")      # C0, on record
    row = next(r for r in board_rows(workspace) if r["name"] == "income_over_limit")
    assert row["idea"] == "capacity_strain" and len(row["idea_measures"]) == 2


def test_the_proposal_message_carries_each_idea_in_the_order_to_work_it(no_history):
    from agent.agent import _propose
    from agent.tools.ideas import focus_engines, record_ideas

    session = _session(no_history, "x", params=RunParams(K=2, levels=["L1"]))
    focus = focus_engines(session)
    engines = [*focus, *[e for e in ("analogy", "others", "contrastive", "mechanism") if e not in focus]][:4]
    ideas = [{"name": n, "engine": e, "lens": l, "level": "L1", "description": f"the {n} hypothesis",
              "measures": [f"{n} measured one way", f"{n} measured another way"], "data": "base columns"}
             for n, e, l in zip(["capacity_stress", "limit_reliance", "tenure_seasoning", "income_gap"],
                                engines, ["capacity", "level", "trend", "peer"])]
    assert record_ideas(session, ideas)["ok"]
    session.ledger.append({"intent": "I1", "name": "limit_doubled", "idea": "limit_reliance",
                           "verified": False, "reason": "explained by the model: R² 1.00; nearest `credit_limit`"})
    text = _propose(session)
    assert text.startswith("Round 1 - proposals") and "Still wanted: 2 results" in text
    for n in ("capacity_stress", "limit_reliance", "tenure_seasoning", "income_gap"):
        assert f"`{n}` - " in text and f"{n} measured another way" in text
    assert "Built so far: limit_doubled ✗ explained by the model" in text
    assert "Why not yet: explained by the model" in text
    assert "Gini gain above" in text and 'idea="<idea>"' in text
    order = next(e["order"] for e in session.events if e["event"] == "ideas_ordered")
    assert text.index(f"`{order[0]}`") < text.index(f"`{order[-1]}`")      # in the drawn order


def test_every_model_call_is_traced_with_what_went_in_and_came_out(tmp_path):
    import asyncio

    from agents import Agent, ModelResponse, Runner, Usage
    from agents.models.interface import Model
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    from agent.llm_trace import TraceHooks, TraceStore

    class Scripted(Model):
        """Answers with text, reports usage - as the openai backend does."""
        async def get_response(self, system_instructions, input, model_settings, tools,
                               output_schema, handoffs, tracing, **kwargs):
            text = ResponseOutputText(type="output_text", text="the answer", annotations=[])
            message = ResponseOutputMessage(id="m1", type="message", role="assistant",
                                            status="completed", content=[text])
            return ModelResponse(output=[message], usage=Usage(requests=1, input_tokens=120,
                                                               output_tokens=7, total_tokens=127),
                                 response_id="r1")

        def stream_response(self, *args, **kwargs):
            raise NotImplementedError

    store = TraceStore(tmp_path / "traces.db")
    agent = Agent(name="feature_engineer", instructions="THE BRIEF", model=Scripted())
    hooks = TraceHooks(store, "run_1", "propose", kind="direction", round=2)
    result = asyncio.run(Runner.run(agent, "propose a feature", hooks=hooks))
    assert result.final_output == "the answer"

    [call] = store.calls("run_1")
    assert (call["stage"], call["round"], call["call"], call["outcome"]) == ("propose", 2, 1, "ok")
    assert (call["input_tokens"], call["output_tokens"], call["tokens_estimated"]) == (120, 7, 0)
    assert call["duration_ms"] is not None and "the answer" in call["output_preview"]
    full = store.call(call["id"])
    assert full["system_prompt"] == "THE BRIEF"                       # the brief, kept once
    assert "propose a feature" in json.dumps(full["input_json"])       # what the model saw
    assert "the answer" in json.dumps(full["output_json"])
    # A second call numbers on; the brief is stored once.
    asyncio.run(Runner.run(agent, "again", hooks=TraceHooks(store, "run_1", "propose", round=2)))
    assert [c["call"] for c in store.calls("run_1")] == [1, 2]
    assert store._conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == 1
    store.delete_run("run_1")
    assert store.calls("run_1") == []


def test_the_trace_viewer_shows_runs_calls_and_their_messages(tmp_path):
    import time

    from agent.llm_trace import TraceStore
    from agent.trace_viewer import create_viewer

    store = TraceStore(tmp_path / "llm_traces.db")
    row = store.begin(run_id="run_1", stage="propose", round=1, call=1, model="gpt-4.1",
                      started_at=time.time(), input_items=1, system_sha=store.prompt("THE BRIEF"),
                      input_json=json.dumps([{"role": "user", "content": "propose <a feature>"}]),
                      outcome="unfinished")
    store.finish(row, duration_ms=1200, input_tokens=50, output_tokens=9, outcome="ok",
                 tool_calls=json.dumps(["screen_feature"]),
                 output_json=json.dumps([{"type": "function_call", "name": "screen_feature",
                                          "arguments": "{\"name\": \"x\"}"}]))
    client = create_viewer(tmp_path / "llm_traces.db").test_client()
    index = client.get("/").get_data(as_text=True)
    assert "run_1" in index and "propose" in index
    page = client.get("/run/run_1").get_data(as_text=True)     # the conversation
    assert "propose &lt;a feature&gt;" in page                 # the runner's turn, escaped
    assert "Runner (user message)" in page                     # left: the runner's turns
    assert "LLM model (assistant message) → tool call · screen_feature" in page    # right
    assert "Tool (run by the runner) · screen_feature" in page
    assert "model call 1" in page and "1.2 s" in page
    assert page.count("THE BRIEF") == 1                        # each brief once
    assert client.get("/run/run_1/lanes").status_code == 404   # one view: the conversation
    sha = store._conn.execute("SELECT sha FROM prompts").fetchone()[0]
    assert "THE BRIEF" in client.get(f"/prompt/{sha}").get_data(as_text=True)
    assert client.get("/run/nope").status_code == 404



def test_each_stage_reads_its_own_brief_and_only_what_this_run_can_need(workspace, tmp_path):
    from agent.composer import compose

    session = _session(workspace, "x", params=RunParams(K=2, levels=["L1", "L2"]))
    ideas, propose = compose(session, "ideas"), compose(session, "propose")
    # The ideas stage: engines, lenses, the ideas skill - no code contract.
    assert "| `mechanism` |" in ideas and "## An idea is a hypothesis" in ideas
    assert "## The contract" not in ideas and "## Scale - the screen is a sample" not in ideas
    assert "## Levels" in ideas                               # what L1 / L2 mean
    # The proposals: the code contract, scale, evaluation - not the engine table.
    assert "## The contract" in propose and "## Skill: evaluate" in propose
    assert "| `mechanism` |" not in propose and "## Then, per feature" in propose
    assert "## Data size" in propose and "## Data size" not in ideas
    # L3 is off: its section is in neither. Linkage: in the proposals only, and only
    # while a source the run may use is unlinked.
    from agent.composer import sections

    for text in (ideas, propose):
        assert "## L3 - when the data you need is not there" not in text
    assert "## Linkage - once per source" not in ideas
    assert ("## Linkage - once per source" in propose) == sections._needs_linkage(session)
    with_l3 = _session(workspace, "x", params=RunParams(K=2, levels=["L1", "L3"]))
    assert "## L3 - when the data you need is not there" in compose(with_l3, "propose")
    # Forced either way: present when a source needs linking, absent when none does.
    original = sections._needs_linkage
    try:
        sections.SECTION_WHEN[("data_sourcing", "Linkage")] = lambda s: True
        assert "## Linkage - once per source" in compose(session, "propose")
        assert "## Linkage - once per source" not in compose(session, "ideas")
        sections.SECTION_WHEN[("data_sourcing", "Linkage")] = lambda s: False
        assert "## Linkage - once per source" not in compose(session, "propose")
    finally:
        sections.SECTION_WHEN[("data_sourcing", "Linkage")] = original



def test_a_failed_script_leaves_its_name_free_for_the_fix(workspace):
    session = _session(workspace, "x", params=RunParams(K=3))
    broken = "def build(spark, sources, base):\n    return base[['id', 'as_of']]"
    assert screen_feature(session, "util_band_x", "d", "L1", broken)["reason"].startswith("script failed")
    fixed = ("def build(spark, sources, base):\n    out = base[['id']].copy()\n"
             "    out['util_band_x'] = (base['income_est'] % 997).astype(float)\n    return out")
    again = screen_feature(session, "util_band_x", "d", "L1", fixed)       # the same name, fixed
    assert "intent" in again and again["intent"] == "I2"
    # A built one keeps its name.
    taken = screen_feature(session, "util_band_x", "d", "L1", fixed.replace("997", "991"))
    assert not taken["ok"] and "already taken" in taken["error"]


def _evaluated_pay_to_spend(workspace, **kwargs):
    """A run that verifies the planted feature, then an evaluation of it."""
    from agent.evaluate import Evaluation

    workspace.cfg.analysis.shap.enabled = False
    session = _session(workspace, "payments vs spend", params=RunParams(K=1))
    for source in ("payments", "spends"):
        assert propose_linkage(session, source, LINK, "event_dt")["ok"]
    assert screen_feature(session, "pay_to_spend_90d", "d", "L2", PAY_TO_SPEND)["verified"]
    evaluation = Evaluation(workspace, [f"{session.run_id}:pay_to_spend_90d"], {}, **kwargs)
    return evaluation, evaluation.run()


def test_a_feature_the_enforced_gates_drop_still_shows_in_the_results(no_history):
    import copy as copying

    from agent.evaluate import evaluation_results

    workspace = no_history
    saved = copying.deepcopy(workspace.cfg)
    workspace.cfg.run.gates = "enforce"
    workspace.cfg.feature_selection.spearman.target_min_abs = 0.99     # nothing gets through
    try:
        evaluation, out = _evaluated_pay_to_spend(workspace)
    finally:
        workspace.cfg = saved
    assert {r["variant"] for r in out["comparison"]} == {"base"}       # only base trained ...
    [row] = evaluation_results(workspace)                               # ... and the feature still shows
    assert row["variant"] == "loi__pay_to_spend_90d" and row["trained"] is False
    assert row["gini_gain"] is None and row["verdict"] == "FAIL"
    assert "not trained: dropped at feature selection" in row["not_trained"]
    assert "run.gates: enforce" in row["not_trained"]


def test_inference_scores_the_target_cases_with_base_and_the_variant(no_history):
    import pandas as pd

    from agent.evaluate import evaluation_results
    from agent.infer import Inference, inference_result, list_inferences, preview

    workspace = no_history
    cases = preview(workspace)
    assert cases["error"] is None and cases["cases"] == 24 and cases["missing_base"] == []
    assert cases["id_example"].count("_") == cases["model_id_example"].count("_")
    assert {n: s["state"] for n, s in cases["sources"].items()} == \
        {"balances": "provided", "payments": "provided", "spends": "provided"}

    evaluation, out = _evaluated_pay_to_spend(workspace)
    assert (evaluation.folder / "features.json").exists()
    assert set(out["score_cutoffs"]) == {"base", "loi__pay_to_spend_90d"}
    # The results: the score level of base and of the model, and the nearest base
    # feature's SHAP rank in each.
    [row] = evaluation_results(workspace)
    assert row["score_base"] == pytest.approx(out["mean_scores"]["base"])
    assert row["score_change"] == pytest.approx(row["score_mean"] / row["score_base"] - 1)
    [listed] = list_inferences(workspace)
    assert listed["status"] == "none" and [v["variant"] for v in listed["variants"]] == \
        ["loi__pay_to_spend_90d"]

    # A source the features read needs the cases' own history: without it, Setup
    # marks it and the inference stops, saying which.
    tc = workspace.cfg.discovery.additional_data.target_cases
    given = dict(tc.sources)
    tc.sources["payments"] = "no_such_file.parquet"           # by default: payments.parquet in dir
    try:
        lacking = preview(workspace)["sources"]["payments"]
        assert lacking["state"] == "not_found" and lacking["needed_by"] == 1
        with pytest.raises(ValueError, match="no history of payments"):
            Inference(workspace, evaluation.eval_id).run()
    finally:
        tc.sources = given

    job = Inference(workspace, evaluation.eval_id, flag_top=0.05)      # every model it trained
    result = job.run()
    summary = result["summaries"]["loi__pay_to_spend_90d"]
    assert summary["cases"] == 24 and result["cutoffs"]["base"] == out["score_cutoffs"]["base"]["0.05"]
    # The planted story: defaulters' payback slides long before base sees anything,
    # so base + the payback feature flags more of them, and sooner.
    defaulted = [c for c in summary["per_case"] if c["case"] < "900012"]    # the synthetic defaulters
    assert sum(c["first_variant"] is not None for c in defaulted) > \
        sum(c["first_base"] is not None for c in defaulted)
    assert summary["outcomes"]["earlier"] + summary["outcomes"]["only_variant"] >= 3
    assert summary["outcomes"]["later"] + summary["outcomes"]["only_base"] == 0
    scores = pd.read_parquet(job.folder / "scores.parquet")
    assert scores["pay_to_spend_90d"].notna().mean() > 0.9             # the cases' own histories
    detail = inference_result(workspace, job.infer_id)
    assert len(detail["timelines"]) == 24 and detail["status"] == "done"
    assert set(detail["timelines"]["900000"][0]["scores"]) == {"base", "loi__pay_to_spend_90d"}
    assert list_inferences(workspace)[0]["status"] == "done"

    # An evaluation from before cutoffs were kept: test is scored again to set them.
    done = next(e for e in evaluation.events if e["event"] == "eval_done")
    done.pop("score_cutoffs")
    lines = (evaluation.folder / "events.jsonl").read_text().splitlines()
    (evaluation.folder / "events.jsonl").write_text("\n".join(
        json.dumps({k: v for k, v in json.loads(line).items() if k != "score_cutoffs"})
        for line in lines) + "\n")
    again = Inference(workspace, evaluation.eval_id, flag_top=0.05).run()   # replaces the first
    assert again["cutoffs"]["base"] == pytest.approx(result["cutoffs"]["base"], rel=1e-3)


def test_ids_repeated_in_the_full_data_are_built_once_and_go_to_every_row(no_history, tmp_path):
    """The screen sample has each id once; the full train can repeat one. A script
    sees each id once - so a linked history is not doubled - and every row with the
    id gets its value."""
    import pandas as pd

    from agent.evaluate import Evaluation

    workspace = no_history
    session = _session(workspace, "payments vs spend", params=RunParams(K=1))
    for source in ("payments", "spends"):
        assert propose_linkage(session, source, LINK, "event_dt")["ok"]
    assert screen_feature(session, "pay_to_spend_90d", "d", "L2", PAY_TO_SPEND)["verified"]
    key = f"{session.run_id}:pay_to_spend_90d"
    once = Evaluation(workspace, [key], {}).materialise()["train"]

    train = pd.read_csv(workspace.cfg.data.paths["train"])
    repeated = tmp_path / "train.csv"
    pd.concat([train, train.head(50)]).to_csv(repeated, index=False)
    original = workspace.cfg.data.paths["train"]
    workspace.cfg.data.paths["train"] = str(repeated)
    try:
        evaluation = Evaluation(workspace, [key], {})
        twice = evaluation.materialise()["train"]
    finally:
        workspace.cfg.data.paths["train"] = original
    assert len(twice) == len(train) + 50
    value = "pay_to_spend_90d"
    assert twice[value].iloc[-50:].tolist() == pytest.approx(once[value].iloc[:50].tolist(), nan_ok=True)
    assert twice[value].iloc[:len(train)].tolist() == pytest.approx(once[value].tolist(), nan_ok=True)
    assert any("repeat an id" in e.get("message", "") for e in evaluation.events)


def test_a_feature_with_no_data_out_of_time_says_so():
    """A feature that varies on train but is one value on valid and test - its data
    stops before them - has no variable Gini there, and the reason says why."""
    import pandas as pd

    from agent.evaluate import _no_signal, variable_gini

    frames = {"train": pd.DataFrame({"x": [0.1, 0.5, 0.9, 0.0], "y": [0, 1, 1, 0]}),
              "valid": pd.DataFrame({"x": [0.0] * 4, "y": [0, 1, 0, 0]}),
              "test": pd.DataFrame({"x": [0.0] * 4, "y": [0, 1, 0, 0]})}
    assert variable_gini(frames["test"]["x"], frames["test"]["y"]) is None
    note = _no_signal("x", frames)
    assert note.startswith("one value (0) on valid and test") and "varies on train" in note


def test_exhaustion_at_half_the_target_is_sent_back_once_to_rethink(workspace):
    """The agent insists the direction is exhausted at half the target or less: once a
    run, it is sent back with where the run stands and why its attempts failed - and
    that analysis leads the next round of ideas. The second time, the run ends."""
    from agent.agent import _next_round
    from agent.tools import report_findings

    session = _session(workspace, "x", params=RunParams(K=4, levels=["L1"]))
    session.gated, session.ideas_required = True, True
    session.ledger = [
        {"name": "good_one", "level": "L1", "verified": True, "delta": 0.012, "reason": ""},
        {"name": "flat_one", "level": "L1", "verified": False, "delta": -0.002,
         "reason": "Gini gain -0.0020 is not above +0.0000"},
        {"name": "broken_one", "level": "L1", "verified": False, "reason": "script failed: KeyError"}]
    session.round_results = session.results()           # this round found nothing
    for _ in range(2):                                   # the usual early-report refusals
        assert not report_findings(session, "exhausted")["ok"]
    back = report_findings(session, "exhausted")
    assert not back["ok"] and session.round_spent and not session.finished
    text = back["error"]
    assert "quite challenging" in text and "`good_one` (L1, +0.0120)" in text
    assert "1 did not beat base on the screen" in text and "1 the script failed" in text
    # The analysis leads the next round's request, once.
    assert _next_round(session).startswith("Not yet.") and not session.rethink
    # Insisting again ends the run.
    session.round_spent = False
    report_findings(session, "still exhausted")
    assert session.finished


BY_MONTH = '''
def aggregate(spark, raw):
    r = raw[["customer_id", "period_end", "amount"]]
    return (r.groupby(["customer_id", "period_end"], as_index=False)
             .agg(spend=("amount", "sum"), n_txn=("amount", "size")))
'''

# The last three calendar months before as_of - from the events, then from the months.
SPEND_3M = '''
def build(spark, sources, base):
    s = sources["spends"]
    recent = s[s["event_dt"] >= s["as_of"] - pd.DateOffset(months=3)]
    out = base[["id"]].copy()
    out["{name}"] = out["id"].map(recent.groupby("id")["amount"].sum()).fillna(0.0)
    return out
'''
SPEND_3M_BY_MONTH = '''
def build(spark, sources, base):
    m = sources["spends_by_month"]
    recent = m[m["event_dt"] >= m["as_of"] - pd.DateOffset(months=3)]
    out = base[["id"]].copy()
    out["{name}"] = out["id"].map(recent.groupby("id")["spend"].sum()).fillna(0.0)
    return out
'''


def test_an_aggregate_joins_through_a_linkage_that_names_the_source_columns(no_history):
    import pandas as pd

    from agent.tools.aggregate import define_aggregate

    session = _session(no_history, "spend by month", params=RunParams(K=3))
    columns = list(no_history.sources()["spends"].columns)
    # Written for the raw source: it picks its columns by name - most of them are not
    # in the aggregate, and reach it empty.
    selecting = LINK.replace('out = ids.merge(source, on="customer_id")',
                             f'events = source[{columns!r}].copy()\n'
                             '    out = ids.merge(events, on="customer_id")')
    assert propose_linkage(session, "spends", selecting, "event_dt")["ok"]
    made = define_aggregate(session, "spends_by_month", "spends", "month",
                            "monthly spend", BY_MONTH, key="customer_id")
    assert made["ok"], made
    assert set(made["columns"]) == {"customer_id", "event_dt", "spend", "n_txn"}  # no empty columns
    # A filter on a column the aggregate lacks drops every row: said, not an empty table.
    other = next(c for c in columns if c not in ("customer_id", "event_dt"))
    filtering = selecting.replace('return out[out["event_dt"] < out["as_of"]]',
                                  f'return out[(out["event_dt"] < out["as_of"]) & out[{other!r}].notna()]')
    session = _session(no_history, "spend by month", params=RunParams(K=3))
    assert propose_linkage(session, "spends", filtering, "event_dt")["ok"]
    dropped = define_aggregate(session, "spends_by_month", "spends", "month", "monthly spend",
                               BY_MONTH, key="customer_id")
    assert not dropped["ok"] and "kept none of the aggregate's" in dropped["error"]


def test_an_aggregate_summarises_a_source_before_it_is_joined(no_history):
    from agent.composer import compose
    from agent.tools.aggregate import define_aggregate

    workspace = no_history
    session = _session(workspace, "spend by month", params=RunParams(K=3, min_gini_gain=-1.0))
    assert propose_linkage(session, "spends", LINK, "event_dt")["ok"]
    assert "is taken" in define_aggregate(session, "spends", "spends", "month", "d", BY_MONTH)["error"]
    assert "period must be" in define_aggregate(
        session, "spends_by_month", "spends", "year", "d", BY_MONTH)["error"]
    # The runner sets period_end; the script may not drop it or move it.
    dropped = define_aggregate(session, "spends_by_month", "spends", "month", "d",
                               BY_MONTH.replace('"customer_id", "period_end"]', '"customer_id"]')
                               .replace(', "period_end", "amount"]', ', "amount"]'), key="customer_id")
    assert not dropped["ok"] and "must keep 'period_end'" in dropped["error"]
    shifted = define_aggregate(session, "spends_by_month", "spends", "month", "d",
                               BY_MONTH.replace("    return (r.groupby",
                                                "    r = r.assign(period_end=r['period_end'] + pd.Timedelta(days=1))\n"
                                                "    return (r.groupby"), key="customer_id")
    assert not shifted["ok"] and "no event has" in shifted["error"]
    assert session.attempts == 0 and not session.aggregates

    assert "name the key" in define_aggregate(
        session, "spends_by_month", "spends", "month", "d", BY_MONTH)["error"]
    assert "is not a column" in define_aggregate(
        session, "spends_by_month", "spends", "month", "d", BY_MONTH, key="cust")["error"]
    made = define_aggregate(session, "spends_by_month", "spends", "month",
                            "monthly spend and transactions", BY_MONTH, key="customer_id")
    assert made["ok"], made
    a = session.aggregates["spends_by_month"]
    assert a["rows"] < a["source_rows"]
    assert set(a["columns"]) >= {"event_dt", "spend", "n_txn"}
    assert a["rows_per_id"] <= 20 and session.attempts == 0       # a month per row, not an event
    assert (session.run_dir / "aggregates" / "spends_by_month.py").exists()
    assert "spends_by_month" in compose(session)                   # the brief lists it
    assert not define_aggregate(session, "spends_by_month", "spends", "month", "d", BY_MONTH,
                                key="customer_id")["ok"]

    # A feature on it is L2: it reads additional data.
    as_l1 = screen_feature(session, "spend_3m_m", "d", "L1", SPEND_3M_BY_MONTH.format(name="spend_3m_m"))
    assert not as_l1["ok"] and "that is L2" in as_l1["error"] and session.attempts == 0
    # Months that end before as_of are exactly the events before it: the same values.
    assert screen_feature(session, "spend_3m", "3-month spend", "L2",
                          SPEND_3M.format(name="spend_3m"))["verified"]
    twin = screen_feature(session, "spend_3m_m", "spend over the last three months, by month",
                          "L2", SPEND_3M_BY_MONTH.format(name="spend_3m_m"), differs="from months")
    assert not twin["ok"] and "identical to `spend_3m`" in twin["error"], twin


def test_a_model_only_feature_is_l1_and_a_wrong_level_is_sent_back(no_history):
    session = _session(no_history, "x", params=RunParams(K=2, levels=["L1", "L2"]))
    product = 'def build(spark, sources, base):\n    out = base[["id"]].copy()\n' \
              '    out["util_by_tenure"] = base["utilization"] * base["tenure_months"]\n    return out\n'
    wrong = screen_feature(session, "util_by_tenure", "d", "L2", product)
    assert not wrong["ok"] and "which is L1" in wrong["error"] and session.attempts == 0
    assert screen_feature(session, "util_by_tenure", "d", "L1", product)["intent"] == "I1"


def _in_batches(monkeypatch):
    """A long source, as the runner meets it in the private environment: read a batch
    of rows at a time, split by key on disk, features built a chunk of ids at a time."""
    monkeypatch.setenv("AUTOFE_BATCH_ROWS", "300000")         # ~6 batches of spends
    monkeypatch.setenv("AUTOFE_PART_BYTES", str(8 * 2**20))   # ~4 partitions of it
    monkeypatch.setenv("AUTOFE_CHUNK_IDS", "1500")            # several chunks of ids


@pytest.mark.parametrize("batched", [False, True], ids=["whole", "in_batches"])
def test_evaluation_recomputes_an_aggregate_on_the_full_splits(no_history, monkeypatch, batched):
    import pandas as pd

    from agent.evaluate import Evaluation
    from agent.tools.aggregate import define_aggregate

    if batched:
        _in_batches(monkeypatch)
    workspace = no_history
    session = _session(workspace, "spend by month", params=RunParams(K=1, min_gini_gain=-1.0))
    assert propose_linkage(session, "spends", LINK, "event_dt")["ok"]
    assert define_aggregate(session, "spends_by_month", "spends", "month", "d", BY_MONTH,
                            key="customer_id")["ok"]
    assert screen_feature(session, "spend_3m_m", "d", "L2",
                          SPEND_3M_BY_MONTH.format(name="spend_3m_m"))["verified"]
    assert session.ledger[-1]["aggregates"]["spends_by_month"]["period"] == "month"
    assert session.ledger[-1]["sources"] == []                 # the long source is not linked

    key = f"{session.run_id}:spend_3m_m"
    train = Evaluation(workspace, [key], {}).materialise()["train"]
    evaluation = Evaluation(workspace, [key], {})
    evaluation.run()
    titles = [e.get("title", "") for e in evaluation.log.events if e["event"] == "code_status"]
    assert any(t.startswith("aggregate spends_by_month") and "full splits" in t for t in titles)
    assert not any(t.startswith("linkage for spends ") for t in titles)

    # The same sum straight from the raw events, on the full train split.
    spends = pd.read_parquet(workspace.sources()["spends"].data_path)
    spends["event_dt"] = pd.to_datetime(spends["event_dt"])
    ids = train[["id"]].drop_duplicates().copy()
    ids["customer_id"] = ids["id"].str.split("_").str[0]
    ids["as_of"] = pd.to_datetime(ids["id"].str.split("_").str[1], format="%Y%m%d")
    joined = ids.merge(spends, on="customer_id")
    joined = joined[(joined["event_dt"] < joined["as_of"])
                    & (joined["event_dt"] >= joined["as_of"] - pd.DateOffset(months=3))]
    expected = ids["id"].map(joined.groupby("id")["amount"].sum()).fillna(0.0)
    got = ids["id"].map(train.drop_duplicates("id").set_index("id")["spend_3m_m"])
    assert (got - expected).abs().max() < 1e-6

    # Inference summarises the cases' own history the same way.
    from agent.infer import Inference, preview

    assert preview(workspace)["sources"]["spends"]["needed_by"] == 1
    result = Inference(workspace, evaluation.eval_id).run()
    assert result["summaries"]["loi__spend_3m_m"]["cases"] == 24


# Per transaction: each id's spend in the 90 days before its as-of date.
SPEND_90D_PER_ID = '''
def aggregate(spark, raw):
    return (raw.groupby("id", as_index=False)
               .agg(spend_90d=("amount", "sum"), n_txn_90d=("amount", "size")))
'''


def test_a_per_transaction_aggregate_summarises_each_ids_window_in_chunks(no_history, monkeypatch):
    from agent.evaluate import Evaluation
    from agent.tools.aggregate import define_aggregate

    monkeypatch.setenv("AUTOFE_CHUNK_IDS", "700")             # several chunks on the screen ids
    monkeypatch.setenv("AUTOFE_BATCH_ROWS", "300000")         # and batches of the source
    workspace = no_history
    session = _session(workspace, "recent spend", params=RunParams(K=3, min_gini_gain=-1.0))
    assert propose_linkage(session, "spends", LINK, "event_dt")["ok"]
    assert "window_days is for" in define_aggregate(
        session, "spends_90d", "spends", "month", "d", BY_MONTH, window_days=90)["error"]
    made = define_aggregate(session, "spends_90d", "spends", "transaction",
                            "spend in the 90 days before", SPEND_90D_PER_ID, window_days=90)
    assert made["ok"], made
    a = session.aggregates["spends_90d"]
    assert a["rows_per_id"] == 1 and a["rows"] <= len(workspace.screen)
    assert set(a["columns"]) == {"spend_90d", "n_txn_90d"}
    one_row = SPEND_90D_PER_ID.replace('groupby("id", as_index=False)', 'groupby("customer_id", as_index=False)')
    assert "must return 'id'" in define_aggregate(session, "by_customer", "spends", "transaction",
                                                    "d", one_row, window_days=90)["error"]

    # The same values as the 90-day sum over the joined events.
    assert screen_feature(session, "spend_90d_raw", "90-day spend", "L2",
                          WINDOW.format(days=90, name="spend_90d_raw"))["verified"]
    from_agg = """
def build(spark, sources, base):
    out = base[["id"]].copy()
    out["{name}"] = out["id"].map(sources["spends_90d"].set_index("id")["spend_90d"]).fillna(0.0)
    return out
"""
    twin = screen_feature(session, "spend_90d_agg", "spend in the last 90 days, per transaction",
                          "L2", from_agg.format(name="spend_90d_agg"), differs="from an aggregate")
    assert not twin["ok"] and "identical to `spend_90d_raw`" in twin["error"], twin

    # Evaluation summarises the full splits the same way, without linking the source whole.
    ratio = """
def build(spark, sources, base):
    a = sources["spends_90d"].set_index("id")
    out = base[["id", "credit_limit"]].copy()
    spend = out["id"].map(a["spend_90d"]).fillna(0.0)
    out["{name}"] = np.where(out["credit_limit"] > 0, spend / out["credit_limit"], np.nan)
    return out[["id", "{name}"]]
"""
    assert screen_feature(session, "spend_90d_to_limit", "recent spend against the line", "L2",
                          ratio.format(name="spend_90d_to_limit"))["verified"]
    key = f"{session.run_id}:spend_90d_to_limit"
    evaluation = Evaluation(workspace, [key], {})
    train = evaluation.materialise()["train"]
    titles = [e.get("title", "") for e in evaluation.log.events if e["event"] == "code_status"]
    assert any(t.startswith("aggregate spends_90d (per transaction") for t in titles)
    assert not any(t.startswith("linkage for spends") for t in titles)
    assert train["spend_90d_to_limit"].notna().mean() > 0.5



def test_a_long_source_is_linked_in_batches_and_evaluated_by_chunk(no_history, monkeypatch):
    import pandas as pd

    from agent.evaluate import Evaluation

    _in_batches(monkeypatch)
    workspace = no_history
    session = _session(workspace, "recent spend", params=RunParams(K=1, min_gini_gain=-1.0))
    # A link that dedupes across events gives other rows batch by batch: refused.
    latest = LINK.replace('return out[out["event_dt"] < out["as_of"]]',
                          'out = out[out["event_dt"] < out["as_of"]]\n'
                          '    return out.drop_duplicates("id", keep="last")')
    refused = propose_linkage(session, "spends", latest, "event_dt")
    assert not refused["ok"] and "not row by row" in refused["error"], refused
    assert propose_linkage(session, "spends", LINK, "event_dt")["ok"]
    assert screen_feature(session, "spend_90d", "90-day spend", "L2",
                          WINDOW.format(days=90, name="spend_90d"))["verified"]

    key = f"{session.run_id}:spend_90d"
    evaluation = Evaluation(workspace, [key], {})
    train = evaluation.materialise()["train"]
    linked = [e for e in evaluation.log.events if e["event"] == "code_status"
              and e.get("state") == "ok" and e.get("mode") == "linkage"]
    assert linked and linked[0]["result"]["chunks"] > 1          # on disk by chunk of ids

    spends = pd.read_parquet(workspace.sources()["spends"].data_path)
    spends["event_dt"] = pd.to_datetime(spends["event_dt"])
    ids = train[["id"]].drop_duplicates().copy()
    ids["customer_id"] = ids["id"].str.split("_").str[0]
    ids["as_of"] = pd.to_datetime(ids["id"].str.split("_").str[1], format="%Y%m%d")
    joined = ids.merge(spends, on="customer_id")
    joined = joined[(joined["event_dt"] < joined["as_of"])
                    & (joined["event_dt"] >= joined["as_of"] - pd.Timedelta(days=90))]
    expected = ids["id"].map(joined.groupby("id")["amount"].sum()).fillna(0.0)
    got = ids["id"].map(train.drop_duplicates("id").set_index("id")["spend_90d"])
    assert (got - expected).abs().max() < 1e-6


def test_discovery_builds_from_a_cut_of_the_screen_customers_but_proposes_from_the_source(
        no_history, monkeypatch):
    import os

    import pandas as pd

    from agent._child import is_parts
    from agent.tools.linkage import ensure_linked, read_linked

    monkeypatch.setenv("AUTOFE_BATCH_ROWS", "300000")
    monkeypatch.setenv("AUTOFE_CHUNK_IDS", "700")             # several chunks of screen ids
    workspace = no_history
    full = str(workspace.sources()["spends"].discovery_path)
    first = _session(workspace, "recent spend", params=RunParams(K=2, min_gini_gain=-1.0))
    assert first.build_paths()["spends"] == full
    assert propose_linkage(first, "spends", LINK, "event_dt", key="customer_id")["ok"]
    assert "key=customer_id" in workspace.linkage_path("spends").read_text()
    # The screen ids' linked rows are on disk by chunk of ids, as at evaluation.
    assert is_parts(first.linked["spends"])
    chunked = screen_feature(first, "spend_90d", "90-day spend", "L2",
                             WINDOW.format(days=90, name="spend_90d"))
    assert chunked["verified"], chunked

    # Here the screen's customers hold nearly every row: a cut would save nothing,
    # so none is kept and the next run reads the source.
    assert _session(workspace, "x", params=RunParams(K=1)).build_paths()["spends"] == full
    # Where it is smaller, the next run builds from the cut - every row of the
    # screen's customers - while the agent still sees the source itself.
    import agent.tools.linkage as linkage

    monkeypatch.setattr(linkage, "CUT_MAX_SHARE", 1.0)
    assert ensure_linked(_session(workspace, "x", params=RunParams(K=1)), "spends") is None
    second = _session(workspace, "recent spend", params=RunParams(K=1))
    cut = second.build_paths()["spends"]
    assert cut != full and "extracts" in cut
    assert second.raw_paths()["spends"] == full
    rows, source = pd.read_parquet(cut), pd.read_parquet(full)
    linked = set(read_linked(first.linked["spends"])["customer_id"].astype(str))
    assert len(rows) < len(source)
    assert len(rows) == source["customer_id"].astype(str).isin(linked).sum()
    # ... and links to the same rows as the whole file did.
    assert ensure_linked(second, "spends") is None
    assert len(read_linked(second.linked["spends"])) == len(read_linked(first.linked["spends"]))

    # A changed source invalidates it.
    os.utime(full, (os.stat(full).st_atime, os.stat(full).st_mtime + 10))
    assert _session(workspace, "x", params=RunParams(K=1)).build_paths()["spends"] == full


def test_a_loop_over_rows_is_refused_before_it_runs(no_history):
    from agent.execution import check_loops

    looping = """
def build(spark, sources, base):
    s = sources["spends"]
    out = base[["id"]].copy()
    values = []
    for _, row in base.iterrows():
        values.append(s[s["id"] == row["id"]]["amount"].sum())
    out["{name}"] = values
    return out
"""
    session = _session(no_history, "spend", params=RunParams(K=2, min_gini_gain=-1.0))
    assert propose_linkage(session, "spends", LINK, "event_dt")["ok"]
    refused = screen_feature(session, "spend_loop", "d", "L2", looping.format(name="spend_loop"))
    assert not refused["ok"] and "iterates over rows" in refused["error"] and "line 6" in refused["error"]
    assert session.attempts == 0                                    # nothing spent
    # Groups and row-wise apply are loops too; a loop over a few fixed windows is not.
    assert check_loops("for _id, g in s.groupby('id'):\n    pass")
    assert check_loops("base['f'] = base.apply(lambda r: r['a'] > r['b'], axis=1)")
    assert check_loops("vals = [x * 2 for x in s['amount']]")
    assert check_loops("for i in range(len(base)):\n    pass")
    assert check_loops("for days in (30, 90, 180):\n    pass") is None
    assert check_loops("for c in ['a', 'b']:\n    pass") is None
    assert check_loops("# for _, r in base.iterrows():\nx = 1") is None      # a comment
    # The vectorised feature is screened as usual.
    assert screen_feature(session, "spend_90d", "90-day spend", "L2",
                          WINDOW.format(days=90, name="spend_90d"))["intent"]
