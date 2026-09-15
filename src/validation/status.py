"""A small, dependency-free terminal run board for the validation pipeline.

Every transition goes through logging, so it appears both live in the terminal
and persistently in ``run.log``. The JSON file keeps the same state structured
for failed-run diagnostics and downstream tooling.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    enabled: bool = True


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"  # error | warning | info


@dataclass
class StageState:
    key: str
    label: str
    status: str = "pending"  # pending | running | passed | warning | skipped | failed
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    detail: str = ""
    checks: List[Check] = field(default_factory=list)
    error: Optional[str] = None
    _started_clock: Optional[float] = field(default=None, repr=False)

    def check(self, name: str, passed: bool, detail: str = "", severity: str = "error") -> None:
        self.checks.append(Check(name, bool(passed), detail, severity))

    def skip(self, detail: str = "disabled") -> None:
        self.status = "skipped"
        self.detail = detail


def stage_specs(cfg: Any) -> List[StageSpec]:
    """The canonical pipeline shape, including disabled stages."""
    return [
        StageSpec("data", "Data"),
        StageSpec("discovery", "Discover", bool(cfg.discovery.enabled)),
        StageSpec("data_quality", "Quality", bool(cfg.data_quality.enabled)),
        StageSpec("feature_selection", "Select", bool(cfg.feature_selection.enabled)),
        StageSpec("modeling", "Model"),
        StageSpec("analysis", "Analyze"),
        StageSpec("verdict", "Verdict", bool(cfg.verdict.enabled)),
    ]


def render_plan(cfg: Any) -> str:
    """Return the pipeline as a compact terminal-friendly graph."""
    nodes = []
    for spec in stage_specs(cfg):
        suffix = " (off)" if not spec.enabled else ""
        nodes.append(f"[{spec.label}{suffix}]")
    return " -> ".join(nodes)


class RunStatus:
    """Persist stage transitions and render them as JSON, HTML, and text."""

    def __init__(self, output_dir: Path, run_name: str, specs: List[StageSpec]):
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.started_at = _now()
        self.finished_at: Optional[str] = None
        self.status = "running"
        self.stages: Dict[str, StageState] = {
            spec.key: StageState(
                spec.key,
                spec.label,
                status="pending" if spec.enabled else "skipped",
                detail="" if spec.enabled else "disabled in config",
            )
            for spec in specs
        }
        self._persist()

    @contextmanager
    def stage(self, key: str) -> Iterator[StageState]:
        state = self.stages[key]
        state.status = "running"
        state.started_at = _now()
        state._started_clock = time.perf_counter()
        self._persist()
        self._print_board()
        try:
            yield state
        except Exception as exc:
            state.status = "failed"
            state.error = f"{type(exc).__name__}: {exc}"
            self.status = "failed"
            self._finish_stage(state)
            reached_failure = False
            for later in self.stages.values():
                if later.key == key:
                    reached_failure = True
                elif reached_failure and later.status == "pending":
                    later.status = "skipped"
                    later.detail = f"not reached after {state.label} failed"
            self.finished_at = _now()
            self._persist()
            self._print_board()
            raise
        else:
            if state.status == "running":
                failed_warnings = any(not c.passed for c in state.checks)
                state.status = "warning" if failed_warnings else "passed"
            self._finish_stage(state)
            self._persist()
            self._print_board()

    def finish(self) -> None:
        self.status = ("succeeded_with_warnings"
                       if any(stage.status == "warning" for stage in self.stages.values())
                       else "succeeded")
        self.finished_at = _now()
        self._persist()
        self._print_board()

    def fail(self, exc: Exception) -> None:
        """Record an error that occurred outside an individual stage."""
        self.status = "failed"
        self.finished_at = _now()
        pending = next((s for s in self.stages.values() if s.status == "running"), None)
        if pending is not None:
            pending.status = "failed"
            pending.error = f"{type(exc).__name__}: {exc}"
            self._finish_stage(pending)
        self._persist()
        self._print_board()

    def payload(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "run": self.run_name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stages": [
                {k: v for k, v in asdict(state).items() if not k.startswith("_")}
                for state in self.stages.values()
            ],
        }

    def _finish_stage(self, state: StageState) -> None:
        state.finished_at = _now()
        if state._started_clock is not None:
            state.elapsed_seconds = round(time.perf_counter() - state._started_clock, 3)

    def _persist(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write("pipeline_status.json", json.dumps(self.payload(), indent=2))

    def _atomic_write(self, name: str, content: str) -> None:
        destination = self.output_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)

    def _print_board(self) -> None:
        marks = {
            "pending": "·", "running": ">", "passed": "OK", "warning": "!",
            "skipped": "-", "failed": "X",
        }
        nodes = [f"[{marks[s.status]} {s.label}]" for s in self.stages.values()]
        logger.info("pipeline | %s", " -> ".join(nodes))


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
