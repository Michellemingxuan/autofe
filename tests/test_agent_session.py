"""The agent's tools end to end on the synthetic use case, with no LLM.

The synthetic data plants a real signal (share of 90-day spend paid back) and
a leak (collections spends after as_of for defaulters). These tests drive the
session the way the agent would and check that the first verifies through a
point-in-time linkage, and that the second cannot get through.
"""

from __future__ import annotations

import json

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
    assert list(direction.skills) == ["data_sourcing", "feature", "evaluate"]
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
    reply = screen_feature(session, "pay_to_spend_90d", "share of spend repaid", "L1",
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


def test_unlinked_source_costs_no_intent_and_budget_is_enforced(workspace, tmp_path):
    workspace.cfg.agent.linkage_dir = str(tmp_path / "empty_linkage")
    session = _session(workspace, "x", params=RunParams(K=1))
    reply = screen_feature(session, "f1", "d", "L1",
                                   'def build(spark, sources, base):\n    return sources["spends"]')
    assert not reply["ok"] and "no confirmed linkage" in reply["error"]
    assert session.intents_used == 0

    ratio = '''
def build(spark, sources, base):
    out = base[["id"]].copy()
    out["util_x_delinq"] = base["utilization"] * (1 + base["num_delinq_12m"])
    return out
'''
    first = screen_feature(session, "util_x_delinq", "d", "L1", ratio)
    assert first["intent"] == "I1" and "next" in first
    second = screen_feature(session, "again", "d", "L1", ratio)
    assert "intents are used" in second["error"]


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
    assert screen_feature(first, "pay_to_spend_90d", "d", "L1", PAY_TO_SPEND)["verified"]

    # A second direction that happens to reuse the name: both must survive.
    second = _session(workspace, "utilisation", params=RunParams(K=1, min_gini_gain=-1.0))
    util = """
def build(spark, sources, base):
    out = base[["id"]].copy()
    out["pay_to_spend_90d"] = base["income_est"] / base["credit_limit"]
    return out
"""
    assert screen_feature(second, "pay_to_spend_90d", "d", "L1", util)["verified"]

    pool = {f["key"] for f in feature_pool(workspace)}
    a, b = f"{first.run_id}:pay_to_spend_90d", f"{second.run_id}:pay_to_spend_90d"
    assert {a, b} <= pool

    evaluation = Evaluation(workspace, [a, b], {"both": [a, b]})
    out = evaluation.run()
    variants = {row["variant"]: row for row in out["comparison"]}
    col_a = f"pay_to_spend_90d__{first.run_id[-4:]}"
    assert {"base", f"loi__{col_a}", "combo__both"} <= set(variants)
    assert variants[f"loi__{col_a}"]["gini_gain_test"] > 0.03
    assert evaluation.events[-1]["event"] == "eval_done"


def test_data_pull_waits_for_the_user_and_keeps_the_sql(no_history):
    workspace = no_history
    sql = ("SELECT customer_id, trans_dt, auth_decline_cnt_30d FROM wwcas_synthetic "
           "WHERE trans_dt >= '2023-01-01'")
    rejected = _session(workspace, "x", approver=AutoApprover(False, "too broad"))
    reply = screen_request(rejected, "declines are missing", sql, "declines")
    assert reply == {"approved": False, "user_note": "too broad"}

    approved = _session(workspace, "x", approver=AutoApprover())
    reply = screen_request(approved, "declines are missing", sql, "declines")
    assert reply["approved"] and "declines.parquet" in reply["next"]
    assert sql in (approved.run_dir / "data_requests" / "R1_declines.sql").read_text()


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


def test_the_analysts_capture_gate_can_hold_back_a_gini_win(no_history):
    workspace = no_history
    session = _session(workspace, "payments vs spend",
                       params=RunParams(K=1, min_gini_gain=0.0, min_capture_gain=1.0))
    for source in ("payments", "spends"):
        assert propose_linkage(session, source, LINK, "event_dt")["ok"]
    reply = screen_feature(session, "pay_to_spend_90d", "d", "L1", PAY_TO_SPEND)
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
    workspace.cfg.agent.shot_spec_paths = [str(spec), str(fixed)]
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
        workspace.cfg.agent.shot_spec_paths = []


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
    overview = session.ws.catalog("")["cas_scope"]
    assert overview["tables"]["wwcas_synthetic"]["unused_raw"] == 6
    assert overview["tables"]["wwcas_synthetic"]["partition"] == ["trans_dt"]
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
    assert good["recorded"] and session.intents_used == 1      # refusals spent nothing


def test_a_data_request_run_is_an_sql_run(workspace):
    l3 = Session(workspace, "x", params=RunParams(K=1, levels=["L3"]))
    assert l3.params.engine == "sql" and l3.local_engine == workspace.cfg.agent.engine
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
    assert screen_feature(first, "spend_90d", "90-day spend", "L1",
                          WINDOW.format(days=90, name="spend_90d"))["verified"]

    later = _session(workspace, "spend again", params=RunParams(K=3, min_gini_gain=-1.0))
    assert "`spend_90d`" in brief(later) and "Earlier directions" in brief(later)

    # The same quantity under a new name is refused, and costs nothing.
    twin = screen_feature(later, "total_spend_last_quarter", "d", "L1",
                          WINDOW.format(days=90, name="total_spend_last_quarter"))
    assert not twin["ok"] and "identical to `spend_90d`" in twin["error"]
    assert later.intents_used == 0

    # Another window measures something else, and is screened as usual.
    other = screen_feature(later, "spend_30d", "30-day spend", "L1",
                           WINDOW.format(days=30, name="spend_30d"))
    assert "verified" in other and later.intents_used == 1

    # A deleted feature is forgotten.
    assert first.delete_intent("spend_90d")
    again = screen_feature(later, "spend_quarter", "d", "L1",
                           WINDOW.format(days=90, name="spend_quarter"))
    assert "verified" in again


def test_a_later_data_request_for_the_same_data_is_refused(no_history):
    workspace = no_history
    sql = ("SELECT customer_id, trans_dt, auth_decline_cnt_30d FROM wwcas_synthetic "
           "WHERE trans_dt BETWEEN '2024-01-01' AND '2024-12-31'")
    first = _session(workspace, "declines", approver=AutoApprover())
    assert screen_request(first, "declines are missing", sql, "declines")["approved"]

    later = _session(workspace, "declines again", approver=AutoApprover())
    narrower = sql.replace("'2024-01-01'", "'2024-06-01'")      # another range, same data
    refused = screen_request(later, "declines", narrower, "decline_counts")
    assert not refused.get("ok", True) and "the same data as `declines`" in refused["error"]
    other = sql.replace("auth_decline_cnt_30d", "cash_adv_amt_90d")
    assert screen_request(later, "cash advances", other, "cash_advances")["approved"]


def test_a_later_run_is_told_which_cas_columns_are_already_requested(no_history):
    from agent.composer.sections import memory as memory_brief
    from agent.tools import scope

    sql = ("SELECT customer_id, trans_dt, auth_decline_cnt_30d FROM wwcas_synthetic "
           "WHERE trans_dt BETWEEN '2024-01-01' AND '2024-12-31'")
    first = _session(no_history, "declines", approver=AutoApprover())
    assert screen_request(first, "declines are missing", sql, "declines")["approved"]

    # Up front, before any proposal: the brief names the column, scope() marks it.
    later = _session(no_history, "declines again", approver=AutoApprover())
    assert "`wwcas_synthetic.auth_decline_cnt_30d`" in memory_brief(later)
    marked = {v["variable"]: v.get("requested") for v in scope(later, table="wwcas_synthetic")["variables"]}
    assert marked["auth_decline_cnt_30d"] and "declines" in marked["auth_decline_cnt_30d"][0]
    assert not marked["cash_adv_amt_90d"]
    assert not marked["trans_dt"]                 # the partition date every request selects
    assert "used up" not in memory_brief(later)     # other raw columns are still open


def test_a_rejected_request_does_not_block_a_revised_one(no_history):
    sql = ("SELECT customer_id, trans_dt, cash_adv_amt_90d FROM wwcas_synthetic "
           "WHERE trans_dt BETWEEN '2023-01-01' AND '2024-12-31'")
    first = _session(no_history, "cash", approver=AutoApprover(False, "too broad"))
    assert screen_request(first, "cash advances", sql, "cash_adv")["approved"] is False
    later = _session(no_history, "cash, narrower", approver=AutoApprover())
    revised = sql.replace("'2023-01-01'", "'2024-06-01'")
    assert screen_request(later, "cash advances, last 6 months", revised, "cash_adv_6m")["approved"]


def _ideas(lenses):
    return [{"name": f"idea_{i}_{lens}", "lens": lens, "description": f"an idea through {lens}",
             "data": "auth_decline_cnt_30d from the CAS"} for i, lens in enumerate(lenses)]


def test_proposals_wait_for_ideas_with_a_spread(no_history):
    from agent import tools
    from agent.agent import build_agent
    from agent.composer import compose as brief
    from agent.tools.ideas import LENSES, MIN_FOCUS, record_ideas

    session = _session(no_history, "x", params=RunParams(K=2, levels=["L3"]))
    session.ideas_required = True                      # as the runner sets it
    assert "Ideas first" in brief(session) and session.focus   # this run's focus, in the brief
    sql = ("SELECT customer_id, trans_dt, auth_decline_cnt_30d FROM wwcas_synthetic "
           "WHERE trans_dt > '2024-01-01'")
    early = screen_request(session, "why", sql, "declines")
    assert not early["ok"] and "the ideas come first" in early["error"] and session.intents_used == 0

    others = [k for k in LENSES if k not in session.focus]
    narrow = record_ideas(session, _ideas([others[0]] * 4))                    # one lens only
    assert not narrow["ok"] and "lenses" in narrow["error"]
    unfocused = record_ideas(session, _ideas(others[:4]))                     # misses the focus
    assert not unfocused["ok"] and "focus" in unfocused["error"]
    assert not record_ideas(session, [{"name": "x", "lens": "vibes", "description": "x"}])["ok"]
    unnamed = record_ideas(session, [{"name": "Not A Name", "lens": "trend", "description": "x"}])
    assert not unnamed["ok"] and "snake_case name" in unnamed["error"]
    spread = record_ideas(session, _ideas(session.focus[:MIN_FOCUS] + others[:2]))
    assert spread["ok"] and len(spread["lenses"]) == 4
    assert screen_request(session, "why", sql, "declines")["recorded"]
    assert any(e["event"] == "ideas_recorded" for e in session.events)
    # The ideas are the first stage's structured answer, not a tool.
    assert "brainstorm" not in [t.name for t in tools.for_agent(session)]
    stage = build_agent(session, "ideas")
    assert stage.output_type is tools.Ideas
    assert {t.name for t in stage.tools} == {"catalog", "scope", "sample_rows", "shots", "run_probe"}


def test_each_run_leans_on_different_lenses(no_history):
    from agent.tools.ideas import focus_lenses

    a = _session(no_history, "x")
    b = _session(no_history, "x")
    assert focus_lenses(a) != focus_lenses(b)
    assert focus_lenses(a) == focus_lenses(a)                 # taken once per run



def test_an_l3_idea_must_name_cas_data(no_history):
    from agent.tools.ideas import record_ideas

    session = _session(no_history, "x", params=RunParams(K=2, levels=["L3"]))
    from_sources = record_ideas(session, [{"name": "spend_trend_3m", "lens": "trend", "level": "L3",
                                         "description": "spend trend over 3 months",
                                         "data": "spends source"}])
    assert not from_sources["ok"] and "L1/L2 idea" in from_sources["error"]


def test_a_mixed_run_splits_its_intents_by_level(workspace):
    from collections import Counter

    from agent.session import level_quota

    workspace.cfg.agent.linkage_dir = str(workspace.linkage_dir)      # payments/spends linked
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
    second = screen_feature(session, "inc_lim2", "d", "L1", ratio.replace("inc_lim", "inc_lim2"))
    assert not second["ok"] and "L1 share is used" in second["error"]
    assert session.intents_used == 1 and "L3 2" in session.budget()


def test_l3_gets_no_share_once_the_scope_is_used_up(no_history):
    from agent.composer import compose as brief
    from agent.session import level_quota

    params = RunParams(K=200, levels=["L1", "L3"]).resolve(no_history)
    assert "L3" in level_quota(no_history, params, "seed")
    first = _session(no_history, "everything", approver=AutoApprover())
    for column in ("auth_decline_cnt_30d", "cash_adv_amt_90d", "merchant_country"):
        sql = (f"SELECT customer_id, trans_dt, {column} FROM wwcas_synthetic "
               "WHERE trans_dt BETWEEN '2024-01-01' AND '2024-12-31'")
        assert screen_request(first, "why", sql, column)["approved"]

    # Every unused_raw column is asked for: the K goes to the levels that can use it.
    assert level_quota(no_history, params, "seed") == {"L1": 200}
    later = Session(no_history, "more", params=RunParams(K=3, levels=["L1", "L3"]))
    assert later.quota == {"L1": 3} and "L3 has no share" in brief(later)
    # Asked for alone, L3 keeps the run - the agent reports the scope is used up.
    assert level_quota(no_history, RunParams(K=2, levels=["L3"]).resolve(no_history), "s") == {"L3": 2}


def test_an_l3_report_with_requests_left_is_sent_back_while_the_scope_has_room(no_history):
    from agent.tools import report_findings

    session = Session(no_history, "x", params=RunParams(K=2, levels=["L3"]))
    session.ideas_required = True
    early = report_findings(session, "nothing to ask")
    assert not early["ok"] and "2 intent(s) left" in early["error"]


def test_a_feature_reading_a_source_that_does_not_exist_costs_nothing(workspace):
    session = _session(workspace, "x", params=RunParams(K=2, levels=["L1"]))
    code = """
def build(spark, sources, base):
    s = sources["cash_advance_temporal"]
    return base[["id"]].assign(x=1.0)
"""
    reply = screen_feature(session, "x", "d", "L1", code)
    assert not reply["ok"] and "A data request is not data" in reply["error"]
    assert session.intents_used == 0


def test_a_report_with_feature_intents_left_is_sent_back(workspace):
    from agent.tools import report_findings

    session = _session(workspace, "x", params=RunParams(K=3, levels=["L1"]))
    session.ideas_required = True                         # as the runner sets it
    first = report_findings(session, "done early")
    assert not first["ok"] and "3 intent(s) left" in first["error"]
    assert not session.finished
    assert not report_findings(session, "still early")["ok"]
    assert report_findings(session, "exhausted, because ...")["ok"]       # it may insist
    assert session.finished


def test_a_dropped_request_is_refunded_and_must_be_replaced(no_history):
    from agent.tools import challenge_request

    session = _session(no_history, "x", params=RunParams(K=1, levels=["L3"]))
    sql = ("SELECT customer_id, trans_dt, utilization FROM wwcas_synthetic "
           "WHERE trans_dt > '2024-01-01'")
    assert screen_request(session, "why", sql, "util")["recorded"]
    assert session.intents_used == 1
    construction = """
def build(spark, sources, base):
    out = base[["id"]].copy()
    out["proxy"] = base["utilization"]
    return out
"""
    dropped = challenge_request(session, "R1", "constructible", "a base column", ["utilization"],
                                construction)
    assert dropped["status"] == "dropped" and "re-propose" in dropped["next"]
    assert session.intents_used == 0                                 # the intent came back
    again = screen_request(session, "why", sql.replace("utilization", "auth_decline_cnt_30d"),
                              "declines")
    assert again["recorded"] and again["intent"] == "R2"             # numbered by proposal


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
    assert sorted(p.stem for p in SKILLS_DIR.glob("*.md")) == ["data_sourcing", "evaluate", "feature"]
    l3 = Session(workspace, "x", params=RunParams(K=2, levels=["L3"]))
    assert template_for(l3) == "data_scout" and "data scout" in compose(l3)
    assert message("propose", names="a, b").startswith("Stage 2 of 2")

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
    assert not reply["ok"] and "Nothing was spent" in reply["error"] and session.intents_used == 0
    assert not any(e["event"] == "code_status" for e in session.events)       # never ran
