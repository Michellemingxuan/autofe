"""One direction run: its parameters, its state, its events, its approvals.

The agent never touches this directly - it calls tools (``agent.tools``), and
each tool is a function over the session. Keeping the logic out of the LLM
layer is what lets it be tested with no model at all, and lets the server
drive the same session the CLI does.

Every step is an event (:class:`agent.events.EventLog`); every script the
agent writes goes out as a ``code_status`` event, so all code is visible.

A run's **parameters** are the analyst's choices for this direction - K, the
model, the engine, the gates a feature must clear (Gini gain, capture-rate
gain), which levels (L1, L2, L3) and which sources the agent may use.
They default to the config's ``agent`` section and are recorded in
``run_started``.

Two steps wait for the analyst, through an :class:`Approver`: confirming a
linkage, and an L3 data pull. Everything else runs on its own.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from agent.events import EventLog, run_logged
from agent.workspace import Workspace

__all__ = ["Decision", "Approver", "AutoApprover", "Session", "RunParams", "SKILLS_DIR",
           "LEVELS", "FEATURE_LEVELS", "load_skills"]

SKILLS_DIR = Path(__file__).parent / "skills"   # appended to every brief in full
FEATURE_LEVELS = ("L1", "L2")
LEVELS = (*FEATURE_LEVELS, "L3")
# Runs saved before the levels were renamed still open.
_RENAMED = {"L1a": "L1", "L1b": "L1", "L2c": "L2"}
# The skills each kind of run is briefed with.
SKILLS_FOR = {"direction": ("data_sourcing", "feature", "evaluate"),
              "linkage": ("data_sourcing",)}


@dataclass
class Decision:
    approved: bool
    note: str = ""


class Approver(Protocol):
    def request(self, kind: str, payload: dict[str, Any]) -> Decision: ...


class AutoApprover:
    """Approves everything. For tests and fully unattended synthetic runs."""

    def __init__(self, approve: bool = True, note: str = ""):
        self.approve, self.note = approve, note

    def request(self, kind: str, payload: dict[str, Any]) -> Decision:
        return Decision(self.approve, self.note)


def load_skills() -> dict[str, dict[str, str]]:
    """Skill files: YAML-ish frontmatter (name, description) then the body."""
    skills = {}
    for path in sorted(SKILLS_DIR.glob("*.md")):
        text = path.read_text()
        meta, body = {}, text
        if text.startswith("---"):
            head, body = text[3:].split("---", 1)
            for line in head.strip().splitlines():
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        skills[meta.get("name", path.stem)] = {
            "description": meta.get("description", ""), "body": body.strip()}
    return skills


@dataclass
class RunParams:
    """The analyst's choices for one direction. None = the config's default."""
    K: int | None = None
    model: str | None = None
    engine: str | None = None
    min_gini_gain: float | None = None
    min_capture_gain: float | None = None   # None here and in the config = not gated
    levels: list[str] | None = None        # subset of LEVELS
    sources: list[str] | None = None       # source names the agent may use

    def resolve(self, ws: Workspace) -> "RunParams":
        agent = ws.cfg.agent
        levels = list(self.levels) if self.levels is not None else list(LEVELS)
        levels = [_RENAMED.get(lv, lv) for lv in levels]
        bad = sorted(set(levels) - set(LEVELS))
        if bad:
            raise ValueError(f"unknown levels {bad}; choose from {LEVELS}")
        l3_only = [lv for lv in LEVELS if lv in levels] == ["L3"]
        if self.engine not in (None, "pandas", "spark", "sql"):
            raise ValueError(f"engine must be pandas, spark or sql, got {self.engine!r}")
        if self.engine == "sql" and not l3_only:
            raise ValueError("the sql engine is for data-request runs (L3 only): features are "
                             "built in pandas or spark")
        known = list(ws.sources())
        sources = list(self.sources) if self.sources is not None else known
        capture = self.min_capture_gain if self.min_capture_gain is not None \
            else agent.min_capture_gain
        return RunParams(
            K=int(self.K or agent.max_intents), model=self.model or agent.llm.model,
            # A data-request run's output is BigQuery SQL; its local checks still
            # run on the configured engine (Session.local_engine).
            engine="sql" if l3_only else (self.engine or agent.engine),
            min_gini_gain=float(agent.min_gini_gain if self.min_gini_gain is None
                                else self.min_gini_gain),
            min_capture_gain=None if capture is None else float(capture),
            levels=[lv for lv in LEVELS if lv in levels],
            sources=[s for s in sources if s in known])


# A mixed run's levels in priority order: L2 builds on additional data, so it
# leads when a linked source is allowed; then L1; then L3, the costliest.
LEVEL_PRIORITY = ("L2", "L1", "L3")


def level_quota(ws: Workspace, params: RunParams, seed: str) -> dict[str, int]:
    """How a run's K intents split across its levels.

    Each of the K intents draws a level at random, with the config's
    probabilities p1 > p2 > p3 taken in priority order over the levels the
    run allows - L2 only when a linked source is among its sources, L3 only
    while an unused_raw CAS column is left that no run has asked for. Seeded by
    the run id, so the split is fixed for the run and on record.
    """
    from agent.memory import scope_left

    linked = any(ws.linkage_path(s).exists() for s in params.sources or [])
    room = "L3" not in params.levels or bool(scope_left(ws))
    order = [lv for lv in LEVEL_PRIORITY if lv in params.levels
             and (lv != "L2" or linked) and (lv != "L3" or room)]
    order = order or list(params.levels)
    if len(order) == 1:
        return {order[0]: int(params.K)}
    weights = list(ws.cfg.agent.level_weights)[: len(order)]
    draws = random.Random(seed).choices(order, weights=weights, k=int(params.K))
    return {lv: draws.count(lv) for lv in order}


@dataclass
class Session:
    ws: Workspace
    direction: str
    approver: Approver = field(default_factory=AutoApprover)
    params: RunParams = field(default_factory=RunParams)
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S_")
                        + uuid.uuid4().hex[:4])
    listeners: list[Callable[[dict[str, Any]], None]] = field(default_factory=list)
    # "direction" - discover features; "linkage" - the setup step where the
    # agent writes one source's linkage for the analyst to confirm.
    kind: str = "direction"
    source: str | None = None

    @staticmethod
    def folder_for(ws: Workspace, run_id: str, kind: str = "direction") -> Path:
        root = Path(ws.cfg.agent.run_dir)
        return (root / "linkage" / run_id) if kind == "linkage" else (root / run_id)

    @classmethod
    def resume(cls, ws: Workspace, run_id: str, kind: str = "direction",
               **kwargs: Any) -> "Session":
        """Reopen a past run - its events and ledger - to show or evaluate it."""
        folder = cls.folder_for(ws, run_id, kind)
        if not (folder / "events.jsonl").exists():
            raise FileNotFoundError(f"no {kind} run {run_id} under {ws.cfg.agent.run_dir}")
        log = EventLog(folder, run_id)
        started = next((e for e in log.events if e["event"] == "run_started"), {})
        params = RunParams(**{k: v for k, v in (started.get("params") or {}).items()
                              if k in RunParams.__dataclass_fields__})
        params.K = params.K or started.get("K")
        params.sources = [s for s in (params.sources or []) if s in ws.sources()] or None
        session = cls(ws, started.get("direction", ""), run_id=run_id, params=params,
                      kind=kind, source=started.get("source"), **kwargs)
        ledger = folder / "ledger.json"
        session.ledger = json.loads(ledger.read_text()) if ledger.exists() else []
        requests = folder / "data_requests.json"
        session.data_requests = json.loads(requests.read_text()) if requests.exists() else []
        session.quota = started.get("quota") or session.quota
        session.intents_used = len(session.ledger) + sum(session.request_spent(r)
                                                         for r in session.data_requests)
        session.finished = any(e["event"] == "run_done" for e in log.events)
        return session

    def __post_init__(self):
        self.params = self.params.resolve(self.ws)
        self.K = self.params.K
        self.run_dir = self.folder_for(self.ws, self.run_id, self.kind)
        self.log = EventLog(self.run_dir, self.run_id)
        self.log.listeners.extend(self.listeners)
        self.listeners = self.log.listeners
        self.events = self.log.events
        self.base_path = self.run_dir / "base.parquet"
        if not self.base_path.exists():
            self.ws.write_base(self.base_path)
        everything = load_skills()
        wanted = ("data_sourcing",) if self.l3_only else SKILLS_FOR[self.kind]
        self.skills = {n: everything[n] for n in wanted if n in everything}
        self.ledger: list[dict[str, Any]] = []
        self.data_requests: list[dict[str, Any]] = []
        self.intents_used = 0
        self.finished = False
        self.cancelled = False
        self.stopped_because = ""
        self.linked: dict[str, str] = {}      # source -> its linked rows this run
        self.shot_round: int | None = None    # taken when the agent first reads shots
        # Ideas first (agent.tools.ideas): the runner requires them before any
        # proposal; tools called directly - tests, scripts - do not.
        self.ideas: list[dict[str, Any]] = []
        self.quota: dict[str, int] = (level_quota(self.ws, self.params, self.run_id)
                                      if self.kind == "direction" else {})
        self.ideas_required = False
        self.focus: list[str] = []
        self.report_refusals = 0              # early reports sent back (agent.tools.report)

    @property
    def local_engine(self) -> str:
        """Where agent-written code runs: the run's engine, or for an sql run the
        configured one - a challenge's construction runs on the local data."""
        return self.ws.cfg.agent.engine if self.params.engine == "sql" else self.params.engine

    @property
    def l3_only(self) -> bool:
        """A data-request run: no features screened; the agent proposes data pulls
        - a rationale and BigQuery SQL each - and every proposal spends an intent."""
        return self.kind == "direction" and self.params.levels == ["L3"]

    # ----------------------------------------------------------------- events
    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        return self.log.emit(event, **payload)

    # ---------------------------------------------------------------- levels
    def request_spent(self, r: dict[str, Any]) -> bool:
        """Whether a data request holds an intent - a dropped one is refunded."""
        return bool(r.get("spent", self.l3_only and not r.get("refunded")))

    def used_at(self, level: str) -> int:
        if level == "L3":
            return sum(self.request_spent(r) for r in self.data_requests)
        return sum(1 for e in self.ledger if e.get("level") == level)

    def level_full(self, level: str) -> str | None:
        """Why a level's share is spent, or None while it has room."""
        if len(self.quota) < 2:
            return None
        left = self.quota.get(level, 0) - self.used_at(level)
        if left > 0:
            return None
        return (f"this run's {level} share is used ({self.used_at(level)} of "
                f"{self.quota.get(level, 0)}); what is left: {self.shares_left()}. "
                "Nothing was spent.")

    def shares_left(self) -> str:
        return ", ".join(f"{lv} {max(n - self.used_at(lv), 0)}" for lv, n in self.quota.items())

    def budget(self) -> str:
        text = f"{self.intents_used}/{self.K} intents used"
        return f"{text}; left by level: {self.shares_left()}" if len(self.quota) > 1 else text

    def start(self) -> None:
        p = self.params
        self.emit("run_started", direction=self.direction, K=self.K, engine=p.engine,
                  quota=self.quota,
                  kind=self.kind, source=self.source, skills=list(self.skills),
                  params={"K": p.K, "model": p.model, "engine": p.engine,
                          "min_gini_gain": p.min_gini_gain,
                          "min_capture_gain": p.min_capture_gain,
                          "levels": p.levels, "sources": p.sources},
                  sources=[s.summary() for n, s in self.ws.sources().items()
                           if n in p.sources])

    def ask(self, kind: str, payload: dict[str, Any]) -> Decision:
        """Wait for the analyst, with the request and the answer both on record."""
        req_id = f"r{len(self.events):04d}"
        self.emit("approval_required", req_id=req_id, kind=kind, **payload)
        decision = self.approver.request(kind, {"req_id": req_id, "kind": kind, **payload})
        self.emit("approval_resolved", req_id=req_id, kind=kind,
                  approved=decision.approved, note=decision.note)
        return decision

    # -------------------------------------------------------------- run code
    def allowed(self, source: str) -> str | None:
        if source not in self.params.sources:
            return (f"{source!r} is not available in this run; the analyst allowed "
                    f"{self.params.sources or 'no sources'}")
        return None

    def raw_paths(self) -> dict[str, str]:
        return {n: str(s.data_path) for n, s in self.ws.sources().items()
                if s.usable and n in self.params.sources}

    def run(self, code: str, mode: str, *, intent: str, level: str | None = None,
            title: str = "", base_path: Path | None = None, **kwargs: Any):
        return run_logged(self.log, code, mode, engine=self.local_engine,
                          id_col=self.ws.id_col, base_path=base_path or self.base_path,
                          timeout_s=self.ws.cfg.agent.code_timeout_s, intent=intent,
                          level=level, title=title, **kwargs)

    # ---------------------------------------------------------------- ledger
    def save_ledger(self) -> None:
        (self.run_dir / "ledger.json").write_text(json.dumps(self.ledger, indent=2, default=str))

    def verified(self) -> list[dict[str, Any]]:
        return [e for e in self.ledger if e["verified"] and not e.get("deleted")]

    def delete_intent(self, name: str) -> bool:
        """Drop one intent from the pool. The trace keeps it, marked deleted."""
        entry = next((e for e in self.ledger if e["name"] == name and not e.get("deleted")), None)
        if entry is None:
            return False
        entry["deleted"] = True
        self.save_ledger()
        self.emit("intent_deleted", name=name, intent=entry["intent"])
        return True

    # ------------------------------------------------------------------- end
    def save_requests(self) -> None:
        from agent.tools.data_pull import save_requests

        save_requests(self)

    def end(self, summary: str, stopped_because: str = "agent reported") -> dict[str, Any]:
        if not self.finished:
            self.finished = True
            self.stopped_because = stopped_because
            self.emit("run_done", summary=summary, stopped_because=stopped_because,
                      intents_used=self.intents_used, K=self.K,
                      verified=[e["name"] for e in self.ledger if e["verified"]])
        return {"ok": True}
