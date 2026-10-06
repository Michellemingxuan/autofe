"""The backend the frontend talks to.

    PYTHONPATH=src python -m agent.server -c configs/synthetic_agent.yaml   # :49010
    cd web && npm run build      # served at /     (or: npm run dev, :5173)

Work happens in background threads; every event is buffered and fanned out to
SSE subscribers, and a client that connects late (or reconnects) gets the
buffer replayed first. One agent job - a direction or a linkage - at a time.

The contract, page by page (``web/src/api.ts`` calls exactly these;
``tests/test_agent_server.py`` checks both sides agree):

1. Setup
   GET    /api/workspace                      what is loaded, sources, run defaults
   GET    /api/setup                          the editable data settings
   POST   /api/setup              {values}    apply - only if the workspace loads
   POST   /api/setup/reset                    back to the config file
   POST   /api/setup/check        {paths}     which paths exist
   POST   /api/uploads            file, kind  keep a small file; returns its path
   POST   /api/shots/clustering   {shots, batches}   generate the clustering shots
   POST   /api/shots/categories   {name, context, ids | table, rotate, batch_size}
                                               add a shot category from the form
   POST   /api/sources            {name, schema, data}   register a source by path
   DELETE /api/sources/<name>                 forget a registered source
   GET    /api/sources/<name>/linkage         the confirmed linkage code
   POST   /api/linkage            {source}    the agent proposes a linkage
   GET    /api/linkage/<id>/stream            SSE
   POST   /api/linkage/<id>/approvals/<req>   {approved, note}
   POST   /api/linkage/<id>/cancel

2. Discover
   GET    /api/runs
   POST   /api/runs               {direction, params, replaces?}   start - or re-run, replacing
                                               an earlier run: its features leave the pool
   DELETE /api/runs/<id>
   GET    /api/runs/<id>/stream               SSE
   POST   /api/runs/<id>/approvals/<req>      {approved, note}
   POST   /api/runs/<id>/cancel
   DELETE /api/runs/<id>/intents/<name>       drop one feature from the pool
   GET    /api/runs/<id>/requests.md          a data-request run's requests, to download

3. Evaluate
   GET    /api/features                       every verified feature of every run
   GET    /api/evaluations
   POST   /api/evaluations        {features: [run_id:name], combinations}
   DELETE /api/evaluations/<id>
   DELETE /api/evaluations/<id>/variants/<variant>   one row off the results
   GET    /api/results                       every evaluated variant, every evaluation
   DELETE /api/results                       clear them all: every finished evaluation
   GET    /api/evaluations/<id>/stream        SSE
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, jsonify, request, send_from_directory

from agent.evaluate import (Evaluation, evaluation_results, feature_pool, list_evaluations,
                            removed_variants)
from agent.events import EventLog, read_events
from agent.session import LEVELS, Decision, RunParams, Session
from agent.setup import UPLOAD_LIMIT, Setup
from agent.tools.shots import categories as shot_categories, generate_clustering, write_spec

__all__ = ["create_app"]


class Cancelled(RuntimeError):
    pass


class ServerApprover:
    """A request blocks its job's thread until the user answers in the UI."""

    def __init__(self) -> None:
        self._waiting: dict[str, tuple[threading.Event, list[Decision]]] = {}
        self.cancelled = threading.Event()

    def request(self, kind: str, payload: dict[str, Any]) -> Decision:
        done, answer = threading.Event(), []
        self._waiting[payload["req_id"]] = (done, answer)
        while not done.wait(0.5):
            if self.cancelled.is_set():
                raise Cancelled("the run was stopped while waiting for approval")
        return answer[0]

    def resolve(self, req_id: str, decision: Decision) -> bool:
        entry = self._waiting.pop(req_id, None)
        if entry is None:
            return False
        entry[1].append(decision)
        entry[0].set()
        return True


class Hub:
    """One job's events, their subscribers, and the worker thread."""

    def __init__(self, job: Any, approver: ServerApprover | None = None):
        self.job = job
        self.approver = approver or ServerApprover()
        self.subscribers: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        job.listeners.append(self.publish)

    def publish(self, record: dict[str, Any]) -> None:
        with self.lock:
            for q in list(self.subscribers):
                q.put(record)

    def start(self, work: Callable[[], Any]) -> None:
        def wrapped() -> None:
            try:
                work()
            except Exception:  # noqa: BLE001 - already recorded as an error event
                pass

        self.thread = threading.Thread(target=wrapped, daemon=True)
        self.thread.start()

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def stream(self) -> Response:
        q: queue.Queue = queue.Queue()
        with self.lock:
            self.subscribers.append(q)
            replay = list(self.job.events)

        def generate():
            try:
                last = 0
                for record in replay:
                    last = record["seq"]
                    yield _frame(record)
                while True:
                    try:
                        record = q.get(timeout=5)
                    except queue.Empty:
                        yield "event: ping\ndata: {}\n\n"
                        continue
                    if record["seq"] > last:
                        yield _frame(record)
            finally:
                with self.lock:
                    if q in self.subscribers:
                        self.subscribers.remove(q)

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _frame(record: dict[str, Any]) -> str:
    return f"event: {record['event']}\ndata: {json.dumps(record, default=str)}\n\n"


def shot_summaries(ws: Any) -> list[dict[str, Any]]:
    """The shot categories for the Setup page; a spec that will not load says why."""
    try:
        return [c.summary(ws.target) for c in shot_categories(ws)]
    except Exception as error:  # noqa: BLE001 - shown on the page, not fatal
        return [{"key": "error", "name": "Shots could not load", "context": str(error),
                 "kind": "error", "rotate": False, "batch_size": None, "batches": 0,
                 "per_batch": [], "path": None, "ids": 0, "found": 0, "columns": 0,
                 "classes": {}, "missing": [], "n_missing": 0}]


def _run_summary(folder: Path) -> dict[str, Any] | None:
    events = read_events(folder / "events.jsonl")
    started = next((e for e in events if e["event"] == "run_started"), None)
    if started is None:
        return None
    done = next((e for e in reversed(events) if e["event"] == "run_done"), None)
    ledger = folder / "ledger.json"
    entries = json.loads(ledger.read_text()) if ledger.exists() else []
    requests = folder / "data_requests.json"
    n_requests = len(json.loads(requests.read_text())) if requests.exists() else 0
    levels = (started.get("params") or {}).get("levels") or []
    return {"run_id": folder.name, "direction": started["direction"], "K": started["K"],
            "mode": "l3" if levels == ["L3"] else "features", "requests": n_requests,
            "started": started["ts"], "params": started.get("params"),
            "verified": sum(1 for e in entries if e.get("verified") and not e.get("deleted")),
            "status": "done" if done else "unfinished",
            "stopped_because": done["stopped_because"] if done else None}


def create_app(config_path: str, static_dir: str | None = None,
               run_session: Callable[[Session], Any] | None = None) -> Flask:
    """``run_session`` runs an agent job; the default drives the LLM. Tests pass
    their own, so the whole API can be exercised without a model."""
    setup = Setup(config_path)
    cfg, ws = setup.load()
    ctx: dict[str, Any] = {"cfg": cfg, "ws": ws, "active": None}
    hubs: dict[str, dict[str, Hub]] = {"direction": {}, "linkage": {}, "evaluation": {}}
    app = Flask(__name__, static_folder=None)
    # Uploads are for small files; refuse anything bigger before reading it.
    app.config["MAX_CONTENT_LENGTH"] = UPLOAD_LIMIT + 2**20

    def _default_runner(session: Session) -> Any:
        from agent.agent import run_direction

        return run_direction(session)

    runner = run_session or _default_runner

    # ------------------------------------------------------------ helpers
    def busy_job() -> str | None:
        active = ctx["active"]
        if active and any(active in h and h[active].busy for h in hubs.values()):
            return active
        return None

    def job_hub(kind: str, job_id: str) -> Hub:
        if job_id not in hubs[kind]:
            if kind == "evaluation":
                # A past evaluation is only replayed, so its log is enough -
                # and it stays viewable after the runs it drew on are deleted.
                folder = Path(ctx["cfg"].agent.run_dir) / "evaluations" / job_id
                if not (folder / "events.jsonl").exists():
                    raise FileNotFoundError(f"no evaluation {job_id}")
                job: Any = EventLog(folder, job_id)
            else:
                job = Session.resume(ctx["ws"], job_id, kind=kind)
            hubs[kind][job_id] = Hub(job)
        return hubs[kind][job_id]

    def source_states() -> list[dict[str, Any]]:
        w = ctx["ws"]
        linked, registered = set(w.linked()), w.registered()
        return [{**s.summary(), "registered": s.name in registered,
                 "state": ("linked" if s.name in linked else
                           "needs_linkage" if s.usable else "schema_only")}
                for s in w.sources().values()]

    def start_session(session: Session, approver: ServerApprover) -> Hub:
        hub = Hub(session, approver)
        hubs[session.kind][session.run_id] = hub
        ctx["active"] = session.run_id
        hub.start(lambda: runner(session))
        return hub

    def watch_sources() -> None:
        seen: dict[str, bool] = {}
        while True:
            time.sleep(3)
            try:
                now = ctx["ws"].sources()
            except Exception:  # noqa: BLE001 - a half-written file; try again
                continue
            active = busy_job()
            for name, src in now.items():
                if seen and seen.get(name) != src.usable and active \
                        and active in hubs["direction"]:
                    hubs["direction"][active].job.emit("source_detected", **src.summary())
            seen = {n: s.usable for n, s in now.items()}

    threading.Thread(target=watch_sources, daemon=True).start()

    def error(message: str, status: int = 400):
        return jsonify(error=message), status

    # ------------------------------------------------------------- 1. setup
    @app.get("/api/workspace")
    def get_workspace():
        c, w = ctx["cfg"], ctx["ws"]
        agent = c.agent
        scope = w.scope()
        return jsonify({
            "name": c.run.name, "base_features": len(w.base_features),
            "id_column": w.id_col, "id_format": agent.id_format, "target": w.target,
            "screen_rows": {"fit": w.n_screen_train, "scored": len(w.screen) - w.n_screen_train},
            "additional_data_dir": agent.additional_data_dir,
            "sources": source_states(),
            "scope": scope["status"].value_counts().to_dict() if len(scope) else {},
            "scope_files": [{"name": p.name, "path": str(p)} for p in w.scope_files()],
            "scope_notes": [{"name": n, "chars": len(t)} for n, t in w.scope_notes()],
            "shots": shot_summaries(w),
            "defaults": {"K": agent.max_intents, "model": agent.llm.model,
                         "engine": agent.engine, "min_gini_gain": agent.min_gini_gain,
                         "min_capture_gain": agent.min_capture_gain,
                         "capture_percent": c.discovery.capture_percent,
                         "levels": list(LEVELS), "level_weights": list(agent.level_weights)},
            "choices": {"models": list(dict.fromkeys([agent.llm.model, *agent.models])),
                        "engines": ["pandas", "spark", "sql"], "levels": list(LEVELS)},
            "active_run": busy_job() if busy_job() in hubs["direction"] else None,
            "active_linkage": busy_job() if busy_job() in hubs["linkage"] else None,
        })

    @app.get("/api/setup")
    def get_setup():
        return jsonify(setup.view(ctx["cfg"]))

    @app.post("/api/setup")
    def apply_setup():
        if busy_job():
            return error("an agent job is running; change the data when it is done", 409)
        try:
            ctx["cfg"], ctx["ws"] = setup.apply(request.get_json(force=True).get("values") or {})
        except Exception as exc:  # noqa: BLE001 - every reason goes back to the form
            return error(f"{type(exc).__name__}: {exc}")
        return jsonify(setup.view(ctx["cfg"]))

    @app.post("/api/shots/clustering")
    def make_clustering_shots():
        """Pick the clustering shots from the screen's fit rows and point the setup at them."""
        if busy_job():
            return error("an agent job is running; change the data when it is done", 409)
        body = request.get_json(force=True)
        try:
            path = generate_clustering(ctx["ws"], shots=int(body.get("shots") or 32),
                                       batches=int(body.get("batches") or 4))
            ctx["cfg"], ctx["ws"] = setup.apply({"discovery.few_shot_path": str(path)})
        except Exception as exc:  # noqa: BLE001 - every reason goes back to the form
            return error(f"{type(exc).__name__}: {exc}")
        return jsonify(path=str(path), shots=shot_summaries(ctx["ws"]))

    @app.post("/api/shots/categories")
    def add_shot_category():
        """Write a category's spec from the form and append it to the shot list."""
        if busy_job():
            return error("an agent job is running; change the data when it is done", 409)
        body = request.get_json(force=True)
        raw_ids = body.get("ids") or []
        ids = raw_ids if isinstance(raw_ids, list) else re.split(r"[\s,;]+", str(raw_ids))
        try:
            path = write_spec(setup.root / "uploads" / "shot_spec",
                              str(body.get("name") or ""), str(body.get("context") or ""),
                              ids=[i for i in ids if i.strip()] or None,
                              table=str(body.get("table") or "").strip() or None,
                              rotate=bool(body.get("rotate")),
                              batch_size=int(body.get("batch_size") or 0) or None)
            current = list(ctx["cfg"].agent.shot_spec_paths)
            ctx["cfg"], ctx["ws"] = setup.apply(
                {"agent.shot_spec_paths": [*[p for p in current if p != str(path)], str(path)]})
        except Exception as exc:  # noqa: BLE001 - every reason goes back to the form
            return error(f"{type(exc).__name__}: {exc}")
        return jsonify(path=str(path), shots=shot_summaries(ctx["ws"]))

    @app.post("/api/setup/reset")
    def reset_setup():
        if busy_job():
            return error("an agent job is running", 409)
        ctx["cfg"], ctx["ws"] = setup.reset()
        return jsonify(setup.view(ctx["cfg"]))

    @app.post("/api/setup/check")
    def check_paths():
        return jsonify(setup.check(list(request.get_json(force=True).get("paths") or [])))

    @app.post("/api/uploads")
    def upload():
        file = request.files.get("file")
        if file is None:
            return error("send the file as multipart field 'file'")
        try:
            return jsonify(setup.save_upload(request.form.get("kind", ""), file.filename or "",
                                             file.read()))
        except (ValueError, KeyError, zipfile.BadZipFile) as exc:
            return error(str(exc))

    @app.post("/api/sources")
    def register_source():
        body = request.get_json(force=True)
        try:
            ctx["ws"].register_source(str(body.get("name") or "").strip(),
                                      str(body.get("schema") or "").strip(),
                                      str(body.get("data") or "").strip() or None)
        except ValueError as exc:
            return error(str(exc))
        return jsonify(sources=source_states())

    @app.delete("/api/sources/<name>")
    def unregister_source(name: str):
        if not ctx["ws"].unregister_source(name):
            return error(f"{name} is not a registered source (files in the folder are "
                         "removed by deleting them)", 404)
        return jsonify(sources=source_states())

    @app.get("/api/sources/<name>/linkage")
    def source_linkage(name: str):
        code = ctx["ws"].linkage_code(name)
        return jsonify(code=code) if code is not None else error("no confirmed linkage", 404)

    @app.post("/api/linkage")
    def start_linkage():
        body = request.get_json(force=True)
        source = str(body.get("source") or "")
        src = ctx["ws"].sources().get(source)
        if src is None or not src.usable:
            return error(f"{source!r} is not a source with data")
        if busy_job():
            return error(f"{busy_job()} is still running", 409)
        approver = ServerApprover()
        session = Session(ctx["ws"], f"Link {source}", approver=approver, kind="linkage",
                          source=source,
                          params=RunParams(model=body.get("model"), engine=body.get("engine"),
                                           sources=[source]))
        start_session(session, approver)
        return jsonify(run_id=session.run_id), 202

    # ---------------------------------------------- shared job endpoints
    def job_routes(prefix: str, kind: str) -> None:
        @app.get(f"/api/{prefix}/<job_id>/stream", endpoint=f"{kind}_stream")
        def stream(job_id: str):
            try:
                return job_hub(kind, job_id).stream()
            except FileNotFoundError as exc:
                return error(str(exc), 404)

        if kind == "evaluation":
            return

        @app.post(f"/api/{prefix}/<job_id>/approvals/<req_id>", endpoint=f"{kind}_approve")
        def approve(job_id: str, req_id: str):
            body = request.get_json(force=True)
            ok = job_hub(kind, job_id).approver.resolve(
                req_id, Decision(bool(body.get("approved")), str(body.get("note") or "")))
            return jsonify(ok=True) if ok else error("no such pending request", 404)

        @app.post(f"/api/{prefix}/<job_id>/cancel", endpoint=f"{kind}_cancel")
        def cancel(job_id: str):
            hub = job_hub(kind, job_id)
            hub.approver.cancelled.set()
            hub.job.cancelled = True
            return jsonify(ok=True)

    job_routes("linkage", "linkage")
    job_routes("runs", "direction")
    job_routes("evaluations", "evaluation")

    # ---------------------------------------------------------- 2. discover
    @app.get("/api/runs")
    def list_runs():
        root = Path(ctx["cfg"].agent.run_dir)
        folders = [d for d in sorted(root.glob("*"), reverse=True)
                   if d.is_dir() and d.name not in ("evaluations", "linkage", "replaced")] \
            if root.exists() else []
        out = [r for r in (_run_summary(d) for d in folders) if r]
        for r in out:
            if r["run_id"] in hubs["direction"] and hubs["direction"][r["run_id"]].busy:
                r["status"] = "running"
        return jsonify(out)

    @app.post("/api/runs")
    def start_run():
        body = request.get_json(force=True)
        direction = (body.get("direction") or "").strip()
        if not direction:
            return error("a direction is required")
        if busy_job():
            return error(f"{busy_job()} is still running", 409)
        raw = body.get("params") or {}
        replaces = str(body.get("replaces") or "").strip()
        old = Session.folder_for(ctx["ws"], replaces) if replaces else None
        if old is not None:
            if replaces in hubs["direction"] and hubs["direction"][replaces].busy:
                return error(f"{replaces} is still running; stop it before re-running", 409)
            if not (old / "events.jsonl").exists():
                return error(f"no run {replaces} to replace", 404)
        try:
            params = RunParams(**{k: raw.get(k) for k in RunParams.__dataclass_fields__})
            approver = ServerApprover()
            session = Session(ctx["ws"], direction, approver=approver, params=params)
        except (TypeError, ValueError) as exc:
            return error(str(exc))
        if old is not None:
            # Replaced, not deleted: out of the list and the pool, kept on disk.
            archive = Path(ctx["cfg"].agent.run_dir) / "replaced"
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(archive / replaces))
            hubs["direction"].pop(replaces, None)
            session.emit("run_replaces", replaces=replaces)
        start_session(session, approver)
        return jsonify(run_id=session.run_id, replaced=replaces or None), 202

    @app.delete("/api/runs/<run_id>")
    def delete_run(run_id: str):
        hub = hubs["direction"].get(run_id)
        if hub and hub.busy:
            return error("stop the run before deleting it", 409)
        folder = Session.folder_for(ctx["ws"], run_id)
        if not (folder / "events.jsonl").exists():
            return error(f"no run {run_id}", 404)
        shutil.rmtree(folder)
        hubs["direction"].pop(run_id, None)
        return jsonify(ok=True)

    @app.get("/api/runs/<run_id>/requests.md")
    def requests_report(run_id: str):
        path = Session.folder_for(ctx["ws"], run_id) / "data_requests.md"
        if not path.exists():
            return error("this run has no data requests", 404)
        return Response(path.read_text(), mimetype="text/markdown", headers={
            "Content-Disposition": f'attachment; filename="data_requests_{run_id}.md"'})

    @app.delete("/api/runs/<run_id>/intents/<name>")
    def delete_intent(run_id: str, name: str):
        hub = job_hub("direction", run_id)
        if hub.busy:
            return error("the run is still going; delete intents when it is done", 409)
        if not hub.job.delete_intent(name):
            return error(f"no intent {name} in {run_id}", 404)
        return jsonify(ok=True)

    # ---------------------------------------------------------- 3. evaluate
    @app.get("/api/features")
    def features():
        return jsonify(feature_pool(ctx["ws"]))

    @app.get("/api/evaluations")
    def evaluations():
        out = list_evaluations(ctx["ws"])
        for e in out:
            if e["eval_id"] in hubs["evaluation"] and hubs["evaluation"][e["eval_id"]].busy:
                e["status"] = "running"
        return jsonify(out)

    @app.get("/api/results")
    def results():
        return jsonify(evaluation_results(ctx["ws"]))

    @app.delete("/api/results")
    def clear_results():
        """Clear the results: delete every finished evaluation. A running one stays."""
        cleared = []
        for e in list_evaluations(ctx["ws"]):
            hub = hubs["evaluation"].get(e["eval_id"])
            if e["status"] == "running" or (hub and hub.busy):
                continue
            shutil.rmtree(Path(ctx["cfg"].agent.run_dir) / "evaluations" / e["eval_id"])
            hubs["evaluation"].pop(e["eval_id"], None)
            cleared.append(e["eval_id"])
        return jsonify(ok=True, cleared=cleared)

    @app.post("/api/evaluations")
    def start_evaluation():
        body = request.get_json(force=True)
        try:
            job = Evaluation(ctx["ws"], list(body.get("features") or []),
                             dict(body.get("combinations") or {}))
        except (KeyError, ValueError) as exc:
            return error(str(exc))
        hub = Hub(job)
        hubs["evaluation"][job.eval_id] = hub
        hub.start(job.run)
        return jsonify(eval_id=job.eval_id), 202

    @app.delete("/api/evaluations/<eval_id>")
    def delete_evaluation(eval_id: str):
        hub = hubs["evaluation"].get(eval_id)
        if hub and hub.busy:
            return error("the evaluation is still running", 409)
        folder = Path(ctx["cfg"].agent.run_dir) / "evaluations" / eval_id
        if not folder.exists():
            return error(f"no evaluation {eval_id}", 404)
        shutil.rmtree(folder)
        hubs["evaluation"].pop(eval_id, None)
        return jsonify(ok=True)

    @app.delete("/api/evaluations/<eval_id>/variants/<variant>")
    def remove_variant(eval_id: str, variant: str):
        """Take one variant off the results. Its record stays in the evaluation's log;
        once every variant is off, the evaluation itself is deleted."""
        try:
            hub = job_hub("evaluation", eval_id)
        except FileNotFoundError:
            return error(f"no evaluation {eval_id}", 404)
        if hub.busy:
            return error("the evaluation is still running", 409)
        events = hub.job.events
        done = next((e for e in reversed(events) if e["event"] == "eval_done"), None)
        variants = {r["variant"] for r in (done or {}).get("comparison", [])} - {"base"}
        removed = removed_variants(events)
        if variant not in variants or variant in removed:
            return error(f"no result {variant} in {eval_id}", 404)
        if variants <= removed | {variant}:
            shutil.rmtree(Path(ctx["cfg"].agent.run_dir) / "evaluations" / eval_id)
            hubs["evaluation"].pop(eval_id, None)
            return jsonify(ok=True, evaluation_deleted=True)
        hub.job.emit("variant_removed", variant=variant)
        return jsonify(ok=True, evaluation_deleted=False)

    # ------------------------------------------------------- built frontend
    static_dir = str(Path(static_dir).resolve()) if static_dir else None
    if static_dir and Path(static_dir).exists():
        @app.get("/")
        @app.get("/<path:path>")
        def frontend(path: str = "index.html"):
            target = Path(static_dir) / path
            name = path if target.is_file() else "index.html"
            response = send_from_directory(static_dir, name)
            if name == "index.html":
                # Assets are content-hashed; the page that names them must not
                # be cached, or a rebuild keeps serving the old bundle.
                response.headers["Cache-Control"] = "no-cache"
            return response

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--port", type=int, default=49010)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--static", default="web/dist")
    args = parser.parse_args(argv)
    from agent import at_project_root

    config = at_project_root(args.config)
    app = create_app(config, args.static)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
