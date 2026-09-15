"""Drive rounds of propose -> screen -> record -> feed back.

Each round asks the proposer for a *batch* of features, screens them one at a
time, and folds every outcome - kept, weak, or rejected - into the history the
next round sees. Batching rather than proposing one feature per call is what
makes a round cheap: one request buys several independent attempts, and a
proposer told to vary them explores more of the space than a single ask does.

Feedback stays per feature. The proposer sees each block with its own delta, not
a batch average, because "these four ideas were worth +0.02 between them" is not
something anyone can act on.

Stopping is a policy object rather than a loop condition, so the rule that ends a
run is visible in the record and can be changed without touching the driver. The
default is one round: propose a batch, screen it, stop. Raise ``max_rounds`` to
let the proposer react to what it learned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from discovery.prompt import parse_candidate
from discovery.screen import ScreenResult, Screener

__all__ = [
    "Candidate",
    "RoundRecord",
    "DiscoveryRun",
    "StoppingRule",
    "Proposer",
    "run_discovery",
]


class Proposer(Protocol):
    """A strategy: given the history so far, write some feature code.

    The framework owns context, screening and bookkeeping; a strategy owns only
    how it asks. That is the single axis the five methods differ on.
    """

    name: str

    def propose(
        self,
        *,
        history: Sequence[dict[str, Any]],
        proposed_names: Sequence[str],
        n_features: int,
        round_index: int,
    ) -> tuple[list[str], dict[str, Any]]:
        """Return (code blocks, call metadata)."""


@dataclass
class Candidate:
    """One proposed feature and everything learned about it."""

    round_index: int
    code: str
    feature_name: str | None = None
    display_name: str | None = None
    description: str | None = None
    rationale: str | None = None
    input_columns: list[str] = field(default_factory=list)
    expression: str | None = None
    screen: ScreenResult | None = None

    @property
    def ok(self) -> bool:
        return bool(self.screen and self.screen.ok)

    @property
    def delta(self) -> float | None:
        return self.screen.delta if self.screen else None

    def as_record(self) -> dict[str, Any]:
        """Flat row: rationale next to numbers, which is how it gets read."""
        screen = self.screen.as_dict() if self.screen else {}
        return {
            "round": self.round_index,
            "feature_name": self.feature_name,
            "display_name": self.display_name,
            "description": self.description,
            "rationale": self.rationale,
            "input_columns": ", ".join(self.input_columns),
            "expression": self.expression,
            "base_score": screen.get("base_score"),
            "candidate_score": screen.get("candidate_score"),
            "delta": screen.get("delta"),
            "error": screen.get("error"),
            "screen_seconds": screen.get("elapsed_seconds"),
            "code": self.code,
        }


@dataclass
class RoundRecord:
    """What one round asked for and what came back."""

    index: int
    requested: int
    returned: int
    kept: int
    rejected: int
    call: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DiscoveryRun:
    """The outcome of a discovery run, ready to hand to the validation stages."""

    candidates: list[Candidate] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    stopped_because: str = ""
    base_score: float | None = None

    @property
    def kept(self) -> list[Candidate]:
        """Candidates that ran cleanly and cleared the keep threshold."""
        return [c for c in self.candidates if c.ok and c.screen.extras.get("kept")]

    def records(self) -> list[dict[str, Any]]:
        return [c.as_record() for c in self.candidates]


@dataclass
class StoppingRule:
    """
    When to stop proposing.

    ``max_rounds`` is the control that always applies; the others end a run early
    when there is no point continuing. Defaults to a single round, so a run does
    exactly one batch unless asked for more.
    """

    max_rounds: int = 1
    target_features: int | None = None      # stop once this many are worth keeping
    patience: int | None = None             # stop after N rounds that kept nothing
    max_candidates: int | None = None       # hard cap on total proposals screened

    def should_stop(self, run: DiscoveryRun) -> str | None:
        """Return the reason to stop, or None to continue."""
        completed = len(run.rounds)
        if completed >= self.max_rounds:
            return f"max_rounds reached ({self.max_rounds})"
        if self.target_features is not None and len(run.kept) >= self.target_features:
            return f"target_features reached ({self.target_features})"
        if self.max_candidates is not None and len(run.candidates) >= self.max_candidates:
            return f"max_candidates reached ({self.max_candidates})"
        if self.patience is not None and completed >= self.patience:
            recent = run.rounds[-self.patience:]
            if all(r.kept == 0 for r in recent):
                return f"no features kept in the last {self.patience} round(s)"
        return None


def run_discovery(
    proposer: Proposer,
    screener: Screener,
    *,
    batch_size: int = 4,
    stopping: StoppingRule | None = None,
    min_delta: float | None = None,
    on_candidate: Callable[[Candidate], None] | None = None,
    logger: Any = None,
    prior_history: Sequence[dict[str, Any]] = (),
    round_offset: int = 0,
) -> DiscoveryRun:
    """
    Run the loop and return everything it learned.

    ``prior_history`` is what earlier runs proposed, each with its final verdict
    and the reason. The proposer reads it ahead of this run's own records, and
    its names are refused as duplicates, so an idea already judged is not simply
    proposed again. Round numbers continue after ``round_offset``, so a strategy
    that rotates its example rows by round keeps rotating across runs instead of
    restarting at the first batch.

    ``min_delta`` decides what is carried forward to the expensive stages. It is
    not an acceptance decision - that belongs to the verdict gates, over full
    data, later.

    None forwards everything that ran, and is the default for a reason: the
    screen's delta comes from one small held-out sample, so for a feature the
    model can already approximate it is noise-dominated and negative about half
    the time. Filtering on it would drop good features by coin flip before the
    stages that exist to judge them. A number pre-filters anyway, which is worth
    it only when a batch is too large to validate whole.
    """
    stopping = stopping or StoppingRule()
    run = DiscoveryRun(base_score=screener.base_score)

    if logger:
        limits = [f"max_rounds={stopping.max_rounds}"]
        for name in ("target_features", "patience", "max_candidates"):
            value = getattr(stopping, name)
            if value is not None:
                limits.append(f"{name}={value}")
        logger.info(
            "discovery loop: %s proposing %d feature(s) per round, stopping on %s; "
            "screen baseline %.4f on %d sample rows",
            getattr(proposer, "name", type(proposer).__name__),
            batch_size,
            ", ".join(limits),
            run.base_score if run.base_score is not None else float("nan"),
            len(getattr(screener, "sample", [])),
        )

    prior_names = [str(r["feature_name"]) for r in prior_history if r.get("feature_name")]

    while True:
        reason = stopping.should_stop(run)
        if reason:
            run.stopped_because = reason
            if logger:
                logger.info("discovery loop stopped: %s", reason)
            break

        round_index = round_offset + len(run.rounds) + 1
        record = RoundRecord(index=round_index, requested=batch_size,
                             returned=0, kept=0, rejected=0)

        if logger:
            # The LLM call dominates a round's wall time, so say what is being
            # waited on before waiting on it - a silent minute reads as a hang.
            logger.info(
                "round %d (%d of %s this run): asking for %d feature(s) "
                "(%d proposed so far, %d forwarded)",
                round_index,
                len(run.rounds) + 1,
                stopping.max_rounds,
                batch_size,
                len(run.candidates),
                len(run.kept),
            )

        started = time.perf_counter()
        try:
            blocks, call = proposer.propose(
                history=[*prior_history, *run.records()],
                proposed_names=[*prior_names,
                                *(c.feature_name for c in run.candidates if c.feature_name)],
                n_features=batch_size,
                round_index=round_index,
            )
            record.call = call
        except Exception as error:  # noqa: BLE001 - a failed round ends the run
            record.error = f"{type(error).__name__}: {error}"
            run.rounds.append(record)
            run.stopped_because = f"proposer failed in round {round_index}: {record.error}"
            break

        record.returned = len(blocks)
        if logger:
            elapsed = time.perf_counter() - started
            logger.info(
                "round %d: proposer returned %d block(s) in %.1fs; screening",
                round_index, len(blocks), elapsed,
            )

        for position, code in enumerate(blocks, start=1):
            # Screened one at a time, against names already *taken*, so a batch
            # cannot smuggle the same feature through twice. A block that failed
            # does not reserve its name: nothing was recorded under it, and the
            # idea may well be right with the code fixed.
            reserved = [*prior_names,
                        *(c.feature_name for c in run.candidates if c.ok and c.feature_name)]
            result = screener.evaluate(code, reserved_names=reserved)
            parsed = parse_candidate(code)

            candidate = Candidate(
                round_index=round_index,
                code=code,
                feature_name=result.feature_name,
                screen=result,
                **parsed.as_dict(),
            )
            if result.ok:
                keep = min_delta is None or (
                    result.delta is not None and result.delta > min_delta
                )
                result.extras["kept"] = keep
                record.kept += int(keep)
            else:
                result.extras["kept"] = False
                record.rejected += 1

            run.candidates.append(candidate)
            if on_candidate:
                on_candidate(candidate)
            if logger:
                verdict = (
                    f"rejected: {result.error}" if result.error
                    else f"delta {result.delta:+.4f}"
                         f"{'' if result.extras.get('kept') else ' (not forwarded)'}"
                )
                logger.info(
                    "  [%d/%d] %s -> %s",
                    position, len(blocks),
                    candidate.feature_name or "<unnamed>",
                    verdict,
                )

        run.rounds.append(record)
        if logger:
            logger.info(
                "round %d complete: %d returned, %d forwarded, %d rejected "
                "(%d forwarded in total)",
                round_index, record.returned, record.kept, record.rejected,
                len(run.kept),
            )

    return run
