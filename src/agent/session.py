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
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from agent.events import EventLog, run_logged
from agent.execution import spark_available
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
    ideas_per_round: int | None = None     # ideas asked for at a time

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
        if (self.engine or agent.engine) == "spark" and not spark_available():
            raise ValueError("pyspark is not installed here: use the pandas engine")
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
            sources=[s for s in sources if s in known],
            ideas_per_round=max(3, int(self.ideas_per_round or agent.ideas_per_round)))


# A mixed run's levels in priority order: L2 builds on additional data, so it
# leads when a linked source is allowed; then L1; then L3, the costliest.
LEVEL_PRIORITY = ("L2", "L1", "L3")

# A run may make this many attempts per result it aims for (K).
ATTEMPTS_PER_RESULT = 3


def level_quota(ws: Workspace, params: RunParams, seed: str) -> dict[str, int]:
    """How a run's target of K results splits across its levels.

    Each of the K results draws a level at random, with the config's
    probabilities p1 > p2 > p3 taken in priority order over the levels the
    run allows - L2 only when a linked source is among its sources. L3 always has
    room: when the scopes are spent, a request looks beyond scope. Seeded by the
    run id, so the split is fixed for the run and on record.
    """
    linked = any(ws.linkage_path(s).exists() for s in params.sources or [])
    order = [lv for lv in LEVEL_PRIORITY if lv in params.levels and (lv != "L2" or linked)]
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
    # No direction from the user: an open exploration. The agent draws a theme
    # from the pool each round (agent.themes.draw_theme).
    explore: bool = False

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
                      explore=bool(started.get("explore")),
                      kind=kind, source=started.get("source"), **kwargs)
        ledger = folder / "ledger.json"
        session.ledger = json.loads(ledger.read_text()) if ledger.exists() else []
        requests = folder / "data_requests.json"
        session.data_requests = json.loads(requests.read_text()) if requests.exists() else []
        session.quota = started.get("quota") or session.quota
        session.attempts = len(session.ledger) + len(session.data_requests)
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
        self.attempts = 0                     # proposals made: screens and requests
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
        self.gated = False                    # the runner's checks on reports (agent.tools.report)
        # Ideas in rounds (agent.tools.ideas): this round's ideas, the attempts made
        # on them, and whether the round is spent - the runner then asks for more.
        self.round = 0
        self.themes_drawn: dict[int, str] = {}   # an exploration's theme, per round
        self.round_ideas: list[dict[str, Any]] = []
        self.round_attempts = 0
        self.round_spent = False
        self.failed_streak = 0                # feature scripts failed in a row (agent.tools.screen)

    @property
    def local_engine(self) -> str:
        """Where agent-written code runs: a linkage job's on the linkage engine; an sql
        run's (a challenge's construction, on the screen rows) on pandas; a direction's
        on the run's engine."""
        if self.kind == "linkage":
            return self.ws.cfg.agent.linkage_engine
        return "pandas" if self.params.engine == "sql" else self.params.engine

    @property
    def l3_only(self) -> bool:
        """A data-request run: no features screened; the agent proposes data requests -
        within a scope (SQL) or beyond scope - and aims for K the challenge keeps."""
        return self.kind == "direction" and self.params.levels == ["L3"]

    # ----------------------------------------------------------------- events
    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        return self.log.emit(event, **payload)

    # ------------------------------------------------------- target, attempts
    # K is a target: the results a run aims for - verified features (L1/L2) and
    # kept requests (L3). A failed attempt does not count against it; attempts
    # are capped at ATTEMPTS_PER_RESULT x K, so a hard direction still ends.
    @property
    def max_attempts(self) -> int:
        return ATTEMPTS_PER_RESULT * self.K

    def results_at(self, level: str) -> int:
        """Results so far at a level: verified features, or requests the challenge
        kept (an approved one, from runs before requests were challenged everywhere)."""
        if level == "L3":
            return sum(1 for r in self.data_requests
                       if r.get("status") == "kept" or r.get("approved") is True)
        return sum(1 for e in self.ledger
                   if e.get("level") == level and e.get("verified") and not e.get("deleted"))

    def pending_at(self, level: str) -> int:
        """Proposals that may still become results: requests awaiting the challenge."""
        if level != "L3":
            return 0
        return sum(1 for r in self.data_requests if r.get("status") == "proposed")

    def take_attempt(self) -> None:
        self.attempts += 1
        self.round_attempts += 1

    def give_back_attempt(self) -> None:
        self.attempts -= 1
        self.round_attempts -= 1

    def round_over(self) -> str | None:
        """When this round's ideas are used - while results are still wanted and
        attempts remain, and no request awaits its challenge - the round is spent and
        the next one is asked for. Returns the word to the agent, or None."""
        if (not self.round_ideas or self.round_attempts < len(self.round_ideas)
                or self.wanted() == 0 or self.attempts >= self.max_attempts
                or self.pending_at("L3")):
            return None
        self.round_spent = True
        return (f"round {self.round}'s ideas are used ({self.budget()}). Stop here - the next "
                "round of ideas comes next.")

    def results(self) -> int:
        return sum(self.results_at(lv) for lv in ("L1", "L2", "L3"))

    def wanted(self, level: str | None = None) -> int:
        """Results still wanted - at a level of a mixed run's target mix, or in all."""
        if len(self.quota) > 1:
            levels = [level] if level else list(self.quota)
            return sum(max(self.quota.get(lv, 0) - self.results_at(lv) - self.pending_at(lv), 0)
                       for lv in levels)
        pending = sum(self.pending_at(lv) for lv in ("L1", "L2", "L3"))
        return max(self.K - self.results() - pending, 0)

    def target_met(self, level: str) -> str | None:
        """Why no more proposals are taken at a level, or None while results are wanted."""
        if self.attempts >= self.max_attempts:
            return (f"all {self.max_attempts} attempts are used ({self.budget()}); "
                    "call report_findings")
        if self.wanted(level) > 0:
            return None
        if len(self.quota) > 1 and self.wanted() > 0:
            return (f"this run's {level} target is met ({self.results_at(level)} of "
                    f"{self.quota.get(level, 0)}); still wanted: {self.targets_left()}. "
                    "Nothing was spent.")
        waiting = sum(self.pending_at(lv) for lv in ("L1", "L2", "L3"))
        return (f"the target of {self.K} is met ({self.budget()})"
                + (f" once {waiting} proposal(s) are challenged - challenge them" if waiting
                   else "; call report_findings") + ". Nothing was spent.")

    def targets_left(self) -> str:
        return ", ".join(f"{lv} {self.wanted(lv)}" for lv in self.quota)

    def budget(self) -> str:
        noun = "kept requests" if self.l3_only else "results"
        text = (f"{self.results()}/{self.K} {noun}, {self.attempts}/{self.max_attempts} "
                "attempts used")
        return f"{text}; still wanted by level: {self.targets_left()}" if len(self.quota) > 1 else text

    def start(self) -> None:
        p = self.params
        self.emit("run_started", direction=self.direction, explore=self.explore, K=self.K,
                  engine=p.engine,
                  max_attempts=self.max_attempts,
                  quota=self.quota,
                  kind=self.kind, source=self.source, skills=list(self.skills),
                  params={"K": p.K, "model": p.model, "engine": p.engine,
                          "ideas_per_round": p.ideas_per_round,
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
            title: str = "", base_path: Path | None = None, engine: str | None = None,
            **kwargs: Any):
        """Run agent-written code - on the run's engine, or the one given (a linkage
        runs on its own engine, whatever the run's)."""
        agent = self.ws.cfg.agent
        # A linkage reads a whole source; a feature or probe works on the screen rows.
        timeout = agent.timeouts.linkage_s if mode == "linkage" else agent.timeouts.screen_s
        return run_logged(self.log, code, mode, engine=engine or self.local_engine,
                          spark_conf=agent.spark_conf,
                          id_col=self.ws.id_col, base_path=base_path or self.base_path,
                          timeout_s=timeout, intent=intent, level=level, title=title,
                          should_stop=lambda: self.cancelled, **kwargs)

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

    def delete_request(self, intent: str) -> bool:
        """Take one data request off the request set. The trace keeps it, marked
        deleted; later runs no longer count its data as asked for."""
        entry = next((r for r in self.data_requests
                      if r.get("intent") == intent and not r.get("deleted")), None)
        if entry is None:
            return False
        entry["deleted"] = True
        self.save_requests()
        self.emit("request_deleted", intent=intent, source_name=entry.get("source_name"))
        return True

    # ------------------------------------------------------------------- end
    def save_requests(self) -> None:
        from agent.tools.data_pull import save_requests

        save_requests(self)

    def _write_process_log(self) -> None:
        """process.md for the run, and its attempts in the cross-run log - a record
        must never be able to break the run it records."""
        from agent.process_log import write_process_log

        try:
            write_process_log(self.run_dir, self.events, self.ledger,
                              Path(self.ws.cfg.agent.run_dir))
        except Exception as error:  # noqa: BLE001
            logging.getLogger(__name__).warning("process log not written for %s: %s",
                                                self.run_id, error)

    def end(self, summary: str, stopped_because: str = "agent reported") -> dict[str, Any]:
        if not self.finished:
            self.finished = True
            self.stopped_because = stopped_because
            self.emit("run_done", summary=summary, stopped_because=stopped_because,
                      results=self.results(), attempts=self.attempts, K=self.K,
                      verified=[e["name"] for e in self.ledger if e["verified"]])
            if self.kind == "direction":
                self._write_process_log()
        return {"ok": True}
