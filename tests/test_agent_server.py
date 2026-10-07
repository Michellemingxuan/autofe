"""The frontend's API, end to end, on the synthetic use case - no LLM.

``create_app`` takes the function that runs an agent job; here it is a script
that calls the same session tools the agent would. Everything else is real:
the routes, the background threads, the approval round-trip, the evaluation
pipeline. The first test reads the frontend's source and checks every
``/api/...`` path it calls exists on the server, so the two cannot drift.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import pytest
import yaml

from agent.synthetic import generate

ROOT = Path(__file__).resolve().parents[1]

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

PAY_COUNT = '''
def build(spark, sources, base):
    n = sources["payments"].groupby("id").size()
    out = base[["id"]].copy()
    out["n_payments"] = out["id"].map(n).fillna(0.0)
    return out
'''

RATIO = '''
def build(spark, sources, base):
    out = base[["id"]].copy()
    out["income_to_limit"] = base["income_est"] / base["credit_limit"]
    return out
'''


def scripted(session):
    """What the agent would do, without a model."""
    from agent.tools import propose_linkage, report_findings, screen_feature

    session.start()
    if session.kind == "linkage":
        propose_linkage(session, session.source, LINK, "event_dt")
    elif session.l3_only:
        from agent.tools import challenge_request, screen_request

        sql = ("SELECT customer_id, trans_dt, auth_decline_cnt_30d FROM proj.cas.wwcas_synthetic "
               "WHERE trans_dt >= '2023-01-01'")
        screen_request(session, "Declines are not in the model.", sql.replace(
            "wwcas_synthetic", "not_a_cas_table"), "declines")               # refused, free
        # Each proposal: propose (validated) -> challenge it -> kept or dropped.
        screen_request(session, "Declines are not in the model.", sql, "declines",
                          "decline_rate_30d\ndays_since_last_decline")
        challenge_request(session, "R1", "new", "No decline data exists.")
        screen_request(session, "Utilisation squared might matter.",
                          "SELECT customer_id, utilization FROM wwcas_synthetic WHERE trans_dt > '2023-01-01'",
                          "util_sq")
        session.scripted_refusal = challenge_request(session, "R2", "constructible", "x")  # no code
        challenge_request(session, "R2", "constructible", "Utilisation is a base column.",
                          ["utilization"], RATIO.replace("income_to_limit", "proxy"))
        screen_request(session, "Income stability.",
                          "WITH c AS (SELECT customer_id, income_est FROM wwcas_synthetic "
                          "WHERE trans_dt > '2023-01-01') SELECT customer_id, income_est FROM c",
                          "income_stability")
        challenge_request(session, "R3", "constructible", "Maybe from income.",
                          ["income_est"], "def build(spark, sources, base):\n    return 1/0\n")
        report_findings(session, "Declines are worth pulling.")
        return
    else:
        screen_feature(session, "income_to_limit", "income relative to the line", "L1", RATIO)
        screen_feature(session, "n_payments", "how many payments were made", "L1", PAY_COUNT)
    report_findings(session, "scripted")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from agent.server import create_app

    root = generate(tmp_path_factory.mktemp("syn") / "data", n_customers=1500, seed=3)
    payload = json.loads(json.dumps(yaml.safe_load(open(ROOT / "configs/synthetic_agent.yaml")))
                         .replace("data/synthetic_agent", str(root)))
    payload["agent"]["run_dir"] = str(root.parent / "runs")
    payload["agent"]["min_gini_gain"] = -1.0          # verify whatever the screen says
    payload["analysis"] = {"shap": {"enabled": True, "sample_size": 500}}
    path = root.parent / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload))
    app = create_app(str(path), run_session=scripted)
    return app.test_client(), root, Path(payload["agent"]["run_dir"])


def wait_for(folder: Path, event: str, timeout: float = 120.0) -> list[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        path = folder / "events.jsonl"
        if path.exists():
            events = [json.loads(line) for line in open(path)]
            if any(e["event"] == event for e in events):
                return events
        time.sleep(0.2)
    raise AssertionError(f"no {event} in {folder} after {timeout}s")


def test_every_path_the_frontend_calls_exists(env):
    client, _, _ = env
    rules = {re.sub(r"<[^>]+>", "<>", r.rule) for r in client.application.url_map.iter_rules()}
    api_ts = (ROOT / "web" / "src" / "api.ts").read_text()
    # `${kind}` is a path segment chosen from the JobKind union - expand it.
    kinds = re.findall(r"'(\w+)'", re.search(r"type JobKind = ([^\n]+)", api_ts).group(1))
    called = set()
    for source in (ROOT / "web" / "src").rglob("*.ts*"):
        for path in re.findall(r"[`'\"](/api/[^`'\"?]*)", source.read_text()):
            for expanded in ([path.replace("${kind}", k) for k in kinds]
                             if "${kind}" in path else [path]):
                called.add(re.sub(r"\$\{[^}]+\}", "<>", expanded))
    assert called, "found no API calls in the frontend"
    missing = sorted(p for p in called if p not in rules)
    assert not missing, f"the frontend calls routes the server lacks: {missing}"


def test_every_event_the_views_fold_is_streamed_to_them():
    # An SSE event the page does not subscribe to never reaches it - silently.
    api_ts = (ROOT / "web" / "src" / "api.ts").read_text()
    subscribed = set(re.findall(r"'(\w+)'", re.search(r"const EVENTS = \[(.*?)\]", api_ts, re.S).group(1)))
    derive = (ROOT / "web" / "src" / "derive.ts").read_text()
    folded = set(re.findall(r"case '(\w+)':", derive))
    assert folded and not folded - subscribed, f"not subscribed: {sorted(folded - subscribed)}"


def test_setup_validates_before_it_saves(env):
    client, root, _ = env
    view = client.get("/api/setup").get_json()
    assert [b["key"] for b in view["blocks"]] == ["model", "context", "shots", "evaluation", "additional", "scope"]
    fields = {f["key"]: f for b in view["blocks"] for f in b["fields"]}
    assert fields["data.target"]["value"] == "default"

    bad = client.post("/api/setup", json={"values": {"data.paths.test": "/nowhere/test.csv"}})
    assert bad.status_code == 400 and "not found" in bad.get_json()["error"]
    assert client.get("/api/workspace").status_code == 200

    good = client.post("/api/setup", json={"values": {"discovery.task_description": "Find risk."}})
    assert good.status_code == 200
    changed = {f["key"] for b in good.get_json()["blocks"] for f in b["fields"] if f["changed"]}
    assert changed == {"discovery.task_description"}
    assert client.post("/api/setup/reset").status_code == 200


def test_a_source_registered_by_path_then_forgotten(env, tmp_path):
    client, _, _ = env
    schema = tmp_path / "bureau_sample.json"
    schema.write_text(json.dumps({"customer_id": ["Customer", ["1"]],
                                  "month_date": ["Snapshot month", ["2024-01-01"]]}))
    bad = client.post("/api/sources", json={"name": "bureau", "schema": str(tmp_path / "x.json")})
    assert bad.status_code == 400
    sources = client.post("/api/sources", json={"name": "bureau", "schema": str(schema)}
                          ).get_json()["sources"]
    assert {s["name"]: s["state"] for s in sources}["bureau"] == "schema_only"
    assert "sources" not in {s["name"] for s in sources}      # the registry is not a source
    assert client.delete("/api/sources/bureau").status_code == 200
    assert client.delete("/api/sources/spends").status_code == 404     # a folder file


def test_linkage_job_waits_for_approval_then_links(env):
    client, _, runs = env
    reply = client.post("/api/linkage", json={"source": "payments"})
    assert reply.status_code == 202
    job = reply.get_json()["run_id"]
    events = wait_for(runs / "linkage" / job, "approval_required")
    req = next(e for e in events if e["event"] == "approval_required")
    assert req["point_in_time_violations"] == 0

    busy = client.post("/api/runs", json={"direction": "x"})
    assert busy.status_code == 409                       # one agent job at a time

    ok = client.post(f"/api/linkage/{job}/approvals/{req['req_id']}",
                     json={"approved": True, "note": ""})
    assert ok.status_code == 200
    wait_for(runs / "linkage" / job, "run_done")
    states = {s["name"]: s["state"] for s in client.get("/api/workspace").get_json()["sources"]}
    assert states["payments"] == "linked"
    assert "def link" in client.get("/api/sources/payments/linkage").get_json()["code"]


def test_direction_then_evaluation_then_deletes(env):
    client, _, runs = env
    reply = client.post("/api/runs", json={"direction": "income and payments",
                                           "params": {"K": 3, "levels": ["L1"]}})
    assert reply.status_code == 202
    run = reply.get_json()["run_id"]
    events = wait_for(runs / run, "run_done")
    assert events[0]["params"]["levels"] == ["L1"]
    listed = {r["run_id"]: r for r in client.get("/api/runs").get_json()}
    assert listed[run]["verified"] == 2
    # The process log, in the output folder: the run as an account, and its
    # attempts in the cross-run log.
    assert "## What worked" in (runs / run / "process.md").read_text()
    attempts = [json.loads(line) for line in (runs / "attempts.jsonl").read_text().splitlines()]
    mine = [a for a in attempts if a["run_id"] == run and a["kind"] == "feature"]
    assert [a["outcome"] for a in mine].count("verified") == 2

    pool = {f["key"]: f for f in client.get("/api/features").get_json()}
    a, b = f"{run}:income_to_limit", f"{run}:n_payments"
    assert {a, b} <= set(pool) and pool[b]["linkage"]["payments"]

    reply = client.post("/api/evaluations", json={"features": [a, b], "combinations": {"both": [a, b]}})
    assert reply.status_code == 202
    ev = reply.get_json()["eval_id"]
    events = wait_for(runs / "evaluations" / ev, "eval_done", timeout=300)
    kinds = {e["event"] for e in events}
    assert {"eval_status", "eval_log", "code_status"} <= kinds
    done = events[-1]
    test_cols = set(done["comparison"][0])
    assert {"gini_gain_test", "capture_gain_top10_test", "capture_gain_top5_test",
            "capture_gain_top1_test"} <= test_cols
    assert done["shap_ranks"][f"loi__income_to_limit"][0]["rank"] >= 1
    assert len(done["shap_ranks"]["combo__both"]) == 2
    # The SHAP gate reads each feature's own model (base + it): it applies.
    assert {v["shap rank"] for v in done["verdicts"]} <= {"PASS", "FAIL"}
    assert client.get(f"/api/evaluations/{ev}/stream", buffered=False).status_code == 200
    # Each stage says, in words, why it warned.
    stages = [e for e in events if e["event"] == "eval_status"][-1]["stages"]
    assert all(st["warnings"] for st in stages if st["status"] == "warning")

    # The results page: one row per variant, base left out.
    results = [r for r in client.get("/api/results").get_json() if r["eval_id"] == ev]
    by_name = {r["name"]: r for r in results}
    assert set(by_name) == {"income_to_limit", "n_payments", "both"}
    assert by_name["both"]["kind"] == "combination"
    assert set(by_name["both"]["members"]) == {"income_to_limit", "n_payments"}
    assert by_name["income_to_limit"]["direction"] and by_name["income_to_limit"]["verdict"]
    assert set(by_name["n_payments"]["capture_gain"]) == {"top10", "top5", "top1"}
    stats = by_name["income_to_limit"]
    assert 0 <= stats["missing_rate"] <= 1 and 0 <= stats["max_corr"] <= 1
    assert stats["max_corr_with"] not in {None, "income_to_limit", "n_payments"}   # a base feature
    assert "missing_rate" not in by_name["both"]                  # a combination has members'
    # Checks every evaluation fails by design (no all-features model) are not warnings.
    assert not any("batch verdict" in w for st in stages for w in st["warnings"])

    # A row comes off the results; the last one takes the evaluation with it.
    assert client.delete(f"/api/evaluations/{ev}/variants/loi__n_payments").get_json() == \
        {"ok": True, "evaluation_deleted": False}
    assert client.delete(f"/api/evaluations/{ev}/variants/loi__n_payments").status_code == 404
    left = {r["variant"] for r in client.get("/api/results").get_json() if r["eval_id"] == ev}
    assert left == {"loi__income_to_limit", "combo__both"}
    assert {e["eval_id"]: e["status"] for e in client.get("/api/evaluations").get_json()}[ev] == "done"

    assert client.delete(f"/api/runs/{run}/intents/n_payments").status_code == 200
    assert b not in {f["key"] for f in client.get("/api/features").get_json()}
    assert client.delete(f"/api/runs/{run}").status_code == 200
    assert run not in {r["run_id"] for r in client.get("/api/runs").get_json()}
    # The evaluation outlives the run it drew on, and is deleted on its own.
    assert ev in {e["eval_id"] for e in client.get("/api/evaluations").get_json()}
    copy = runs / "evaluations" / f"{ev}_copy"                 # a second finished evaluation
    shutil.copytree(runs / "evaluations" / ev, copy)
    assert client.delete(f"/api/evaluations/{ev}/variants/combo__both").get_json()["evaluation_deleted"] is False
    assert client.delete(f"/api/evaluations/{ev}/variants/loi__income_to_limit").get_json()["evaluation_deleted"]
    assert client.delete(f"/api/evaluations/{ev}").status_code == 404
    # Clear all takes every finished evaluation.
    assert client.delete("/api/results").get_json()["cleared"] == [copy.name]
    assert client.get("/api/results").get_json() == [] and client.get("/api/evaluations").get_json() == []


def _docx(text: str) -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("word/document.xml",
                   '<w:document><w:body><w:p><w:r><w:t>' + text + '</w:t></w:r></w:p>'
                   '</w:body></w:document>')
    return buffer.getvalue()


def test_small_files_upload_and_scope_notes_reach_the_agent(env):
    import io

    from agent.composer import compose as brief
    from agent.session import Session

    client, root, _ = env
    reply = client.post("/api/uploads", data={"kind": "context", "file": (
        io.BytesIO(b"# Portfolio\nSmall-business cards."), "task_context.md")},
        content_type="multipart/form-data")
    assert reply.status_code == 200
    context_path = reply.get_json()["path"]

    bad = client.post("/api/uploads", data={"kind": "context", "file": (
        io.BytesIO(b"x"), "notes.exe")}, content_type="multipart/form-data")
    assert bad.status_code == 400

    note = client.post("/api/uploads", data={"kind": "scope_note", "file": (
        io.BytesIO(_docx("Prefer wwcas_auth_analytics_02 for declines.")), "cas_guide.docx")},
        content_type="multipart/form-data").get_json()
    assert note["path"].endswith(".docx.txt")

    check = client.post("/api/setup/check", json={"paths": [context_path, "/no/such"]}).get_json()
    assert check[context_path]["exists"] and not check["/no/such"]["exists"]

    applied = client.post("/api/setup", json={"values": {
        "discovery.task_context_path": context_path,
        "agent.scope_notes_paths": [note["path"]]}})
    assert applied.status_code == 200, applied.get_json()
    workspace = client.get("/api/workspace").get_json()
    assert [n["name"] for n in workspace["scope_notes"]] == ["cas_guide.docx.txt"]
    assert workspace["scope_files"]                       # the default CAS file is found

    from agent.setup import Setup

    cfg, loaded = Setup(str(root.parent / "cfg.yaml")).load()     # what a restart would load
    assert cfg.discovery.task_context_path == context_path
    assert "Prefer wwcas_auth_analytics_02" in brief(Session(loaded, "declines"))
    assert client.post("/api/setup/reset").status_code == 200


def test_shot_specs_upload_append_show_and_delete(env):
    import io

    client, root, _ = env
    no_ids = client.post("/api/uploads", data={"kind": "shot_spec", "file": (
        io.BytesIO(b"# Empty\n## Context\nnothing"), "empty.md")},
        content_type="multipart/form-data")
    assert no_ids.status_code == 400 and "no examples found" in no_ids.get_json()["error"]

    import pandas as pd
    fit = pd.read_csv(root / "screen_train.csv")["id"].astype(str).tolist()
    paths = []
    for name, chosen in (("cures.md", fit[:3]), ("stress.md", fit[3:5])):
        body = f"# {name[:-3]}\n## Context\nabout {name}\n## IDs\n" + "\n".join(chosen)
        reply = client.post("/api/uploads", data={"kind": "shot_spec", "file": (
            io.BytesIO(body.encode()), name)}, content_type="multipart/form-data")
        assert reply.status_code == 200, reply.get_json()
        paths.append(reply.get_json()["path"])

    assert client.post("/api/setup", json={"values": {
        "agent.shot_spec_paths": paths}}).status_code == 200
    shots = client.get("/api/workspace").get_json()["shots"]
    assert [c["key"] for c in shots] == ["clustering", "cures", "stress"]
    assert shots[1]["found"] == 3 and shots[1]["context"] == "about cures.md"

    # Delete one category, then the clustering shots.
    assert client.post("/api/setup", json={"values": {
        "agent.shot_spec_paths": paths[1:], "discovery.few_shot_path": ""}}).status_code == 200
    assert [c["key"] for c in client.get("/api/workspace").get_json()["shots"]] == ["stress"]

    # Generate the clustering shots again, from the screen's fit rows.
    made = client.post("/api/shots/clustering", json={"shots": 8, "batches": 2})
    assert made.status_code == 200, made.get_json()
    clustering = made.get_json()["shots"][0]
    assert clustering["key"] == "clustering" and clustering["rotate"]
    assert clustering["batches"] == 2 and clustering["found"] == 16
    assert client.post("/api/setup/reset").status_code == 200


def test_shot_categories_from_a_table_or_ids_with_their_statistics(env):
    import io

    import pandas as pd

    client, root, _ = env
    train = pd.read_csv(root / "screen_train.csv")
    valid = pd.read_csv(root / "screen_valid.csv")
    table = pd.concat([train.head(6), valid.head(1)])          # one held-out row slips in
    upload = client.post("/api/uploads", data={"kind": "shot_table", "file": (
        io.BytesIO(table.to_csv(index=False).encode()), "high_utilisation.csv")},
        content_type="multipart/form-data").get_json()

    made = client.post("/api/shots/categories", json={
        "name": "High utilisation", "context": "Accounts near their limit.",
        "table": upload["path"], "rotate": True, "batch_size": 3})
    assert made.status_code == 200, made.get_json()
    ids = client.post("/api/shots/categories", json={
        "name": "Two defaulters", "context": "", "ids": train[train["default"] == 1]["id"].head(2).tolist()})
    assert ids.status_code == 200, ids.get_json()
    bad = client.post("/api/shots/categories", json={"name": "Nothing", "context": "x"})
    assert bad.status_code == 400

    shots = {c["name"]: c for c in client.get("/api/workspace").get_json()["shots"]}
    table_cat = shots["High utilisation"]
    assert table_cat["kind"] == "table" and table_cat["found"] == 6
    assert table_cat["n_missing"] == 1 and "held out" in table_cat["missing"][0]
    assert table_cat["batches"] == 2 and table_cat["per_batch"] == [3, 3]
    assert sum(table_cat["classes"].values()) == 6
    assert shots["Two defaulters"]["kind"] == "ids" and shots["Two defaulters"]["classes"] == {"1": 2}
    assert client.post("/api/setup/reset").status_code == 200


def test_evaluation_settings_are_typed_and_checked(env):
    client, _, _ = env
    ok = client.post("/api/setup", json={"values": {
        "model.params.max_depth": "4", "analysis.capture_rate_percents": "0.2, 0.05",
        "analysis.shap.enabled": False, "run.gates": "enforce"}})
    assert ok.status_code == 200, ok.get_json()
    fields = {f["key"]: f["value"] for b in ok.get_json()["blocks"] for f in b["fields"]}
    assert fields["model.params.max_depth"] == 4
    assert fields["analysis.capture_rate_percents"] == [0.2, 0.05]
    assert fields["run.gates"] == "enforce"
    assert client.post("/api/setup", json={"values": {"run.gates": "sometimes"}}).status_code == 400
    assert client.post("/api/setup", json={"values": {"model.params.eta": ""}}).status_code == 400
    assert client.post("/api/setup/reset").status_code == 200


def test_a_table_path_relative_to_the_project_root_resolves(env, monkeypatch):
    from agent.tools.shots import categories, write_spec
    from agent.workspace import Workspace
    from validation.config import load_config

    client, root, runs = env
    monkeypatch.chdir(root.parent)                      # as the server runs: from the root
    (root.parent / "tables").mkdir(exist_ok=True)
    (root.parent / "tables" / "rows.csv").write_text("utilization,default\n0.9,1\n0.2,0\n")
    spec = write_spec(runs / "specs", "Relative", "x", table="tables/rows.csv")
    broken = write_spec(runs / "specs", "Broken", "x", table="tables/none.csv")
    cfg = load_config(root.parent / "cfg.yaml")
    cfg.agent.shot_spec_paths = [str(spec), str(broken)]
    cats = {c.key: c for c in categories(Workspace.from_config(cfg))}
    assert cats["relative"].kind == "table" and len(cats["relative"].rows) == 2
    assert cats["broken"].kind == "error"               # reported, and the rest still load


def test_an_l3_only_run_challenges_each_proposal_as_it_goes(env):
    client, _, runs = env
    reply = client.post("/api/runs", json={"direction": "authorization behaviour",
                                           "params": {"K": 3, "levels": ["L3"]}})
    assert reply.status_code == 202
    run = reply.get_json()["run_id"]
    events = wait_for(runs / run, "run_done")
    kinds = [e["event"] for e in events]
    assert "feature_screened" not in kinds and "approval_required" not in kinds
    proposed = [e for e in events if e["event"] == "data_request"]
    assert [r["intent"] for r in proposed] == ["R1", "R2", "R3"]   # the refused one spent nothing
    assert proposed[0]["tables"] == ["proj.cas.wwcas_synthetic"]
    # one loop per proposal: R1 is challenged before R2 is proposed
    assert [k for k in kinds if k in ("data_request", "request_challenged")] == \
        ["data_request", "request_challenged"] * 3

    decided = {e["intent"]: e for e in events if e["event"] == "request_challenged"}
    assert decided["R1"]["status"] == "kept" and decided["R1"]["verdict"] == "new"
    assert decided["R2"]["status"] == "dropped" and decided["R2"]["code_ok"] is True
    assert decided["R3"]["status"] == "kept" and decided["R3"]["code_ok"] is False

    # The summary carries the validated SQL of the kept requests - and only theirs.
    done = next(e for e in events if e["event"] == "run_done")["summary"]
    assert done.startswith("Declines are worth pulling.") and "## Validated SQL" in done
    assert "auth_decline_cnt_30d FROM proj.cas.wwcas_synthetic" in done
    assert "SELECT customer_id, utilization" not in done and "Dropped: R2 util_sq" in done

    summary = {r["run_id"]: r for r in client.get("/api/runs").get_json()}[run]
    assert summary["mode"] == "l3" and summary["requests"] == 3
    doc = client.get(f"/api/runs/{run}/requests.md").get_data(as_text=True)
    assert "# Kept" in doc and "# Dropped" in doc and "```sql" in doc
    assert doc.index("util_sq") > doc.index("# Dropped")
    folder = runs / run / "data_requests"
    assert (folder / "R1_declines.sql").exists() and not (folder / "R2_util_sq.sql").exists()
    assert client.delete(f"/api/runs/{run}").status_code == 200


def test_a_constructible_verdict_needs_its_construction(env):
    from agent.session import RunParams, Session
    from agent.tools import challenge_request, screen_request

    from agent.setup import Setup

    _, loaded = Setup(str(env[1].parent / "cfg.yaml")).load()
    session = Session(loaded, "x", params=RunParams(K=1, levels=["L3"]))
    screen_request(session, "why", "SELECT customer_id, trans_dt FROM wwcas_synthetic "
                      "WHERE trans_dt > '2024-01-01'", "thing")
    refused = challenge_request(session, "R1", "constructible", "trust me")
    assert not refused["ok"] and "needs the construction" in refused["error"]
    assert session.data_requests[0]["status"] == "proposed"

    from agent.tools import report_findings

    early = report_findings(session, "R1 is new")                 # a verdict in prose only
    assert not early["ok"] and "challenge_request first" in early["error"]
    assert not session.finished


def test_a_rerun_replaces_the_earlier_run_and_its_features(env):
    client, _, runs = env
    first = client.post("/api/runs", json={"direction": "ratios",
                                           "params": {"K": 2, "levels": ["L1"]}}).get_json()["run_id"]
    wait_for(runs / first, "run_done")
    assert any(f["run_id"] == first for f in client.get("/api/features").get_json())

    again = client.post("/api/runs", json={"direction": "ratios, again", "replaces": first,
                                           "params": {"K": 2, "levels": ["L1"]}})
    assert again.status_code == 202 and again.get_json()["replaced"] == first
    second = again.get_json()["run_id"]
    wait_for(runs / second, "run_done")

    listed = {r["run_id"] for r in client.get("/api/runs").get_json()}
    assert first not in listed and second in listed
    pool = client.get("/api/features").get_json()
    assert not any(f["run_id"] == first for f in pool)
    assert any(f["run_id"] == second for f in pool)
    assert (runs / "replaced" / first / "events.jsonl").exists()      # kept, not deleted
    missing = client.post("/api/runs", json={"direction": "x", "replaces": "no_such_run"})
    assert missing.status_code == 404
    assert client.delete(f"/api/runs/{second}").status_code == 200
