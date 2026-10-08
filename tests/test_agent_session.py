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


def test_unlinked_source_costs_nothing_and_target_and_attempts_are_enforced(workspace, tmp_path):
    workspace.cfg.discovery.additional_data.linkage_dir = str(tmp_path / "empty_linkage")
    session = _session(workspace, "x", params=RunParams(K=1))
    reply = screen_feature(session, "f1", "d", "L1",
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


def test_repeated_failures_become_a_lesson_and_old_results_are_cut():
    from agents.run import CallModelData, ModelInputData

    from agent.agent import KEEP_FULL, _trim_old_outputs
    from agent.tools.screen import lessons

    redundant = "is redundant: |rho|=0.99 against the existing column 'tenure_months', above"
    ledger = [{"reason": redundant}, {"reason": redundant}, {"reason": "Gini gain -0.01 is not above"}]
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


def test_an_idea_that_failed_twice_takes_no_third_attempt(workspace):
    from agent.tools.screen import idea_of

    assert idea_of("pay_gap_v3", []) == idea_of("pay_gap_final", []) == "pay_gap"
    assert idea_of("pay_gap_60d_fixed", [{"name": "pay_gap_60d"}]) == "pay_gap_60d"
    session = _session(workspace, "x", params=RunParams(K=3))
    broken = "def build(spark, sources, base):\n    return base[['id', 'as_of']]"
    for name in ("pay_gap", "pay_gap_v2"):
        assert screen_feature(session, name, "d", "L1", broken)["intent"]
        session.failed_streak = 0                              # it probed in between
    third = screen_feature(session, "pay_gap_v3", "d", "L1", broken)
    assert not third["ok"] and "has failed 2 times" in third["error"] and session.attempts == 2


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
    assert screen_feature(first, "spend_90d", "90-day spend", "L1",
                          WINDOW.format(days=90, name="spend_90d"))["verified"]

    later = _session(workspace, "spend again", params=RunParams(K=3, min_gini_gain=-1.0))
    assert "`spend_90d`" in brief(later) and "Earlier directions" in brief(later)

    # The same quantity under a new name is refused, and costs nothing.
    twin = screen_feature(later, "total_spend_last_quarter", "d", "L1",
                          WINDOW.format(days=90, name="total_spend_last_quarter"))
    assert not twin["ok"] and "identical to `spend_90d`" in twin["error"]
    assert later.attempts == 0

    # Another window measures something else, and is screened as usual.
    other = screen_feature(later, "spend_30d", "30-day spend", "L1",
                           WINDOW.format(days=30, name="spend_30d"))
    assert "verified" in other and later.attempts == 1

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
    assert not early["ok"] and "the ideas come first" in early["error"] and session.attempts == 0

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
    assert stage.output_type.output_type is tools.Ideas          # read leniently: FirstAnswer
    assert {t.name for t in stage.tools} == {"catalog", "scope", "sample_rows", "shots", "run_probe"}


def test_ideas_come_in_rounds_and_a_round_ends_when_its_ideas_are_used(no_history):
    from agent.tools.ideas import LENSES, focus_lenses, ideas_in_round, record_ideas

    big = _session(no_history, "x", params=RunParams(K=100, levels=["L1"], ideas_per_round=10))
    assert ideas_in_round(big) == 10                      # a large K: ten at a time
    too_many = record_ideas(big, _ideas((list(LENSES) * 2)[:11]))
    assert "[MISSING] 10 to 10 ideas (you gave 11)" in too_many["error"]
    session = _session(no_history, "x", params=RunParams(K=2, levels=["L1"], ideas_per_round=10))
    assert ideas_in_round(session) == 4                   # a few more than still wanted
    lenses = list(dict.fromkeys([*focus_lenses(session), *LENSES]))[:4]
    # Every requirement comes back, met or not - fixing one must not break another.
    short = record_ideas(session, _ideas(lenses[:1] * 2))
    assert "[MISSING] 4 to 10 ideas (you gave 2)" in short["error"]
    assert "[MISSING] at least 4 different lenses" in short["error"]
    assert record_ideas(session, _ideas(lenses))["round"] == 1
    for _ in range(3):
        session.take_attempt()
    assert session.round_over() is None                   # one idea is still untried
    session.take_attempt()
    assert "round 1's ideas are used" in session.round_over() and session.round_spent
    again = record_ideas(session, _ideas(lenses))       # the same names as round 1
    assert not again["ok"] and "earlier round" in again["error"]


def test_the_report_check_stays_on_when_the_ideas_never_pass(workspace):
    from agent.tools import report_findings

    session = _session(workspace, "x", params=RunParams(K=2, levels=["L1"]))
    session.gated, session.ideas_required = True, False   # ideas failed: proposals ungated
    assert not report_findings(session, "early")["ok"]


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
    assert screen_feature(session, "lim_x_inc", "d", "L1", other)["intent"] == "I2"
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
    base = {"lens": "trend", "level": "L3", "description": "d"}
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
    ideas = [{"name": f"i{n}", "lens": lens, "level": "L1", "description": "d", "data": "x"}
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
