"""Stage 0.5: propose features, screen them, hand the survivors to validation.

This is the seam between the two halves. Discovery produces *code*; the
validation stages consume *columns already in the table*. So the stage's real
work, beyond running the loop, is materialising the surviving code onto every
split and returning the column names - after which nothing downstream knows or
cares that a language model was involved.

Two properties worth protecting:

* The same code is applied to every split. A feature computed one way on train
  and another on test is not a feature, it is a bug, so the blocks run through
  one function over all frames rather than being recomputed per split.
* Screening only ever sees train. Validation's whole job is to judge these
  columns on data that had no hand in proposing them, which is void if the
  proposer was shown test rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from discovery.guards import check_finite, check_scale
from discovery.loop import Candidate, DiscoveryRun, StoppingRule, run_discovery
from discovery.prompt import describe_columns, low_variation_columns
from discovery.sandbox import CandidateError, apply_code, validate_single_column
from discovery.screen import Screener
from discovery.strategies import REGISTRY as STRATEGIES, LLMSettings, PromptContext
from validation.data import Dataset, clean_missing, read_frame
from validation.logging_utils import get_logger
from validation.metrics import calc_adj_gini, capture_rate

logger = get_logger(__name__)

__all__ = ["DiscoveryResult", "run_discovery_stage", "STRATEGIES"]

# Strategies are registered in discovery.strategies.REGISTRY; the framework
# around them does not change when one is added.


@dataclass
class DiscoveryResult:
    """What discovery contributed, and everything it tried on the way."""

    dataset: Optional[Dataset] = None
    kept_features: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    stopped_because: str = ""
    base_score: Optional[float] = None
    enabled: bool = True

    def to_frame(self) -> pd.DataFrame:
        """One row per proposal: rationale beside the numbers."""
        return pd.DataFrame(self.records)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "proposed": len(self.records),
            "kept": len(self.kept_features),
            "kept_features": list(self.kept_features),
            "rounds": len(self.rounds),
            "stopped_because": self.stopped_because,
            "screen_base_score": self.base_score,
        }


def _column_descriptions(discovery_cfg: Any) -> dict[str, str]:
    """Inline descriptions, optionally merged with a JSON file."""
    descriptions = dict(discovery_cfg.column_descriptions or {})
    path = discovery_cfg.column_descriptions_path
    if path:
        import json

        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(
                f"{path} must hold a JSON object of {{column: description}}, "
                f"got {type(loaded).__name__}"
            )
        # Inline entries win, so a config can correct one line of a mapping file.
        descriptions = {**{str(k): str(v) for k, v in loaded.items()}, **descriptions}
    return descriptions


def _task_context(discovery_cfg: Any) -> str:
    """The domain block's text: inline, or read from the file it names."""
    inline = (discovery_cfg.task_context or "").strip()
    path = discovery_cfg.task_context_path
    if not path:
        return inline
    text = Path(path).read_text(encoding="utf-8").strip()
    # Inline wins, so a config can override a file it otherwise shares.
    return inline or text


def _categorical_columns(discovery_cfg: Any, base_features: list[str],
                         known: list[str] | None = None) -> list[str]:
    """
    Which columns the prompt should present as coded categories.

    A config declares whichever list is shorter. Naming the continuous columns
    means everything else is categorical, which is the natural way round for a
    clinical table where 99 of 111 columns are coded categories and only a dozen
    are real measurements - and getting it right matters: a column shown as a
    range invites arithmetic on it, so a coded category presented as continuous
    is an invitation to compute the mean of an ICD code.
    """
    if discovery_cfg.continuous_columns is not None:
        continuous = set(discovery_cfg.continuous_columns)
        # Checked against every column in the table, not just the incumbents: a
        # config rightly declares the type of a candidate column too, and only a
        # name matching nothing at all is a mistake worth reporting.
        unknown = continuous - set(known if known is not None else base_features)
        if unknown:
            logger.warning(
                "discovery.continuous_columns names %d column(s) that are not in "
                "the table at all; check for a typo: %s",
                len(unknown), ", ".join(sorted(unknown)),
            )
        return [c for c in base_features if c not in continuous]
    return [c for c in discovery_cfg.categorical_columns if c in base_features]


def _metric(name: str, percent: float):
    """Resolve the screen's score function from a metric name."""
    if name == "adj_gini":
        return calc_adj_gini, "adjusted Gini", ""
    if name.startswith("capture_rate"):
        explanation = (
            f" Capture rate is the share of all positive cases falling in the "
            f"highest-scoring {percent * 100:g}% of rows once ranked by predicted "
            "score, so only the ranking at the very top matters."
        )
        return (lambda df, y, p: capture_rate(df, y, p, percent)), \
               f"capture rate at the top {percent * 100:g}%", explanation
    raise ValueError(f"Unsupported discovery metric {name!r}; use adj_gini or capture_rate")


def _read_rows(cfg: Any, dataset: Dataset, path: str | Path,
               columns: list[str]) -> pd.DataFrame:
    """
    Rows the prepare step wrote for discovery, in the same format as the splits.

    Read and cleaned exactly as the validation tables are - sentinel codes and
    infinities become NaN - so the screen and the prompt see what the models
    see. A column the file lacks but the dataset carries - a missing indicator the
    pipeline derived, say - is joined in by id.
    Rows from the test split are refused: test never reaches discovery.
    """
    frame = read_frame(cfg.data, str(path))
    id_col = cfg.data.id_cols[0] if cfg.data.id_cols else None
    has_ids = bool(id_col) and id_col in frame.columns

    if has_ids and "test" in dataset.frames:
        leaked = sorted(set(frame[id_col]) & set(dataset.split("test")[id_col]))
        if leaked:
            raise ValueError(
                f"{len(leaked)} row(s) in {path} are test rows (e.g. {leaked[:3]}); "
                "test never reaches discovery - rebuild the file with the dataset's "
                "prepare step")

    absent = [c for c in columns if c not in frame.columns]
    if absent:
        if not has_ids:
            raise ValueError(f"{path} lacks column(s) {absent[:5]} and has no "
                             f"{id_col or 'id'!r} column to join them by")
        pool = pd.concat([dataset.frames[name] for name in ("train", "valid")
                          if name in dataset.frames]).set_index(id_col)
        unknown = sorted(set(frame[id_col]) - set(pool.index))
        if unknown or not pool.index.is_unique:
            raise ValueError(
                f"{path} lacks column(s) {absent[:5]}, and {len(unknown)} of its rows "
                f"are not in train or valid to join them from (e.g. {unknown[:3]}); "
                "rebuild it with the dataset's prepare step")
        frame = frame.assign(**{c: pool.loc[frame[id_col], c].to_numpy() for c in absent})

    features = [c for c in columns if c != dataset.target]
    return clean_missing(frame, features, cfg.data.missing_values)


def _load_shot_batches(cfg: Any, dataset: Dataset) -> list[pd.DataFrame]:
    """
    The precomputed example rows, one frame per round.

    The file holds the rows themselves, in the same format as the splits, plus a
    ``batch`` column: round r shows batch r, so a later round reasons from rows
    the earlier ones did not show.
    """
    path = cfg.discovery.few_shot_path
    shots = _read_rows(cfg, dataset, path, [dataset.target, *dataset.base_features])
    if "batch" not in shots.columns:
        raise ValueError(f"{path} must have a 'batch' column saying which round shows "
                         "each row")

    # Every batch, not just this run's rounds: round numbers continue across runs
    # when a history is kept, so the rotation picks up where the last run stopped.
    batches = [group.reset_index(drop=True)
               for _, group in shots.groupby("batch", sort=True)]
    logger.info("discovery: %d batch(es) of %d example row(s) from %s; round r "
                "shows batch r, cycling", len(batches), len(batches[0]), path)
    return batches


def _screen_frames(cfg: Any, dataset: Dataset) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    The rows the screen fits on and the rows it scores on, and where they came from.

    ``splits`` takes the train and valid splits as they are. ``files`` takes the
    samples the prepare step wrote, in the same format as the splits and read the
    same way (see :func:`_read_rows`). Test never reaches the screen. Scoring on
    valid rows means the proposer's feedback comes from valid, so valid stops
    being an independent check on what it proposes; test still is.
    """
    columns = [dataset.target, *dataset.base_features]
    if cfg.discovery.screen_data == "splits":
        if "valid" not in dataset.frames:
            raise ValueError("discovery.screen_data=splits scores on the valid split, "
                             "and this dataset has none")
        return (dataset.split("train")[columns], dataset.split("valid")[columns],
                "the train and valid splits")

    paths = cfg.discovery.screen_paths
    picked = {part: _read_rows(cfg, dataset, paths[part], columns)
              for part in ("train", "valid")}

    id_col = cfg.data.id_cols[0] if cfg.data.id_cols else None
    if id_col and all(id_col in frame.columns for frame in picked.values()):
        shared = set(picked["train"][id_col]) & set(picked["valid"][id_col])
        if shared:
            raise ValueError(
                f"{len(shared)} row(s) are in both screen files (e.g. "
                f"{sorted(shared)[:3]}): the screen would score proposals on rows it "
                "fit them on")
    return (picked["train"][columns].reset_index(drop=True),
            picked["valid"][columns].reset_index(drop=True),
            f"{paths['train']} and {paths['valid']}")


# --------------------------------------------------------------------------- #
# The history: every proposal across runs, with its final verdict
# --------------------------------------------------------------------------- #
HISTORY_COLUMNS = [
    "run", "round", "feature_name", "display_name", "description", "rationale",
    "input_columns", "expression", "code", "base_score", "candidate_score", "delta",
    "error", "outcome", "failed_at", "reason",
]


def _read_history(path: str | Path | None) -> list[dict[str, Any]]:
    """Every proposal earlier runs made, with its verdict, oldest first."""
    if not path or not Path(path).exists():
        return []
    frame = pd.read_csv(path)
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def append_history(
    path: str | Path,
    discovered: DiscoveryResult,
    verdicts: pd.DataFrame,
    run: str,
) -> int:
    """
    Add this run's proposals to the history, each with its final outcome.

    ``accepted`` means the feature cleared every gate: it stays a candidate,
    judged as base vs base plus that one feature, and never joins the incumbent
    set - there are still steps before it can. ``rejected`` carries the gate it
    fell at - ``screen`` when it never reached validation - and the reason, which
    the next run's prompt repeats so the idea is not simply tried again.
    """
    table = (verdicts.set_index("feature")
             if len(verdicts) and "feature" in verdicts.columns else pd.DataFrame())
    rows = []
    for record in discovered.records:
        row = {"run": run, **{k: record.get(k) for k in HISTORY_COLUMNS if k in record}}
        name, error = record.get("feature_name"), record.get("error")
        if error:
            gate = "full splits" if str(error).startswith("Dropped on the full splits") else "screen"
            row.update(outcome="rejected", failed_at=gate, reason=error)
        elif name in table.index:
            verdict = table.loc[name]
            passed = verdict.get("verdict") in ("PASS", "IN")
            row.update(
                outcome="accepted" if passed else "rejected",
                failed_at="" if passed else (verdict.get("failed_at")
                                             or verdict.get("decided_by") or ""),
                reason=verdict.get("reason") or "",
            )
        else:
            row.update(outcome="rejected", failed_at="screen",
                       reason=f"its screen change {record.get('delta'):+.4f} is below "
                              "discovery.min_delta")
        rows.append(row)
    if not rows:
        return 0

    frame = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return len(rows)


def run_discovery_stage(
    cfg: Any,
    dataset: Dataset,
    output_dir: str | Path | None = None,
) -> DiscoveryResult:
    """
    Run discovery and return a dataset carrying the surviving columns.

    The returned dataset's ``new_features`` are the proposals worth forwarding;
    ``base_features`` is untouched. Stages 1-5 then treat them exactly as they
    would a hand-written candidate set - screened for quality, modelled one at a
    time by leave_one_in, and judged by the verdict gates.
    """
    discovery_cfg = cfg.discovery
    if not discovery_cfg.enabled:
        return DiscoveryResult(dataset=dataset, enabled=False,
                               stopped_because="discovery disabled")

    if discovery_cfg.strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown discovery strategy {discovery_cfg.strategy!r}; "
            f"available: {sorted(STRATEGIES)}"
        )

    score_fn, metric_name, metric_explanation = _metric(
        discovery_cfg.metric, discovery_cfg.capture_percent
    )

    # The screen enforces the threshold the selection gate will use, so a
    # candidate cannot pass here and be rejected there for redundancy - which is
    # exactly what happened before: 6 of 7 forwarded features died at that gate
    # after the expensive stages had already run on them.
    redundancy_max_abs = discovery_cfg.redundancy_max_abs
    if redundancy_max_abs is None and cfg.feature_selection.enabled:
        spearman = cfg.feature_selection.spearman
        if getattr(spearman, "enabled", False):
            redundancy_max_abs = spearman.redundancy_max_abs


    # --- the screen's rows: fit on one set, score on another -----------------
    base_features = list(dataset.base_features)
    train, valid, source = _screen_frames(cfg, dataset)
    logger.info(
        "discovery: redundancy limit |rho|<=%s (from %s)",
        f"{redundancy_max_abs:.2f}" if redundancy_max_abs is not None else "off",
        "discovery.redundancy_max_abs" if discovery_cfg.redundancy_max_abs is not None
        else "feature_selection.spearman",
    )
    logger.info(
        "discovery: strategy=%s backend=%s model=%s metric=%s",
        discovery_cfg.strategy, discovery_cfg.llm.backend,
        discovery_cfg.llm.model, metric_name,
    )
    logger.info(
        "discovery: the screen fits on %d row(s) (%d positive) and scores on %d "
        "(%d positive), from %s; %d base features",
        len(train), int(train[dataset.target].sum()),
        len(valid), int(valid[dataset.target].sum()), source, len(base_features),
    )

    subsampled = [k for k in ("colsample_bytree", "colsample_bylevel", "colsample_bynode")
                  if (cfg.model.params or {}).get(k, 1.0) != 1.0]
    if subsampled:
        # The screen strips these for its own measurement, but the validation
        # variants cannot: base and leave_one_in differ in column count, so their
        # gini gain carries the same content-independent offset. Left alone rather
        # than overridden, because changing the model is the user's call.
        logger.warning(
            "model.params sets %s < 1: base and leave_one_in variants differ in "
            "column count, so their gini gain includes an offset unrelated to any "
            "feature (a constant column measured -0.0055 on this data). The screen "
            "strips it for its own delta; set these to 1.0 to remove it from the "
            "verdict too.",
            ", ".join(subsampled),
        )

    screener = Screener(
        train,
        valid,
        dataset.target,
        base_features,
        cfg.model.params or {},
        score=score_fn,
        num_boost_round=discovery_cfg.screen_boost_rounds,
        nthread=1,
        spike_factor=discovery_cfg.spike_factor,
        redundancy_max_abs=redundancy_max_abs,
        # Description -> identifier, so indexing by meaning is answered with the
        # key to use rather than a KeyError.
        column_aliases={v: k for k, v in _column_descriptions(discovery_cfg).items()},
    )

    # --- the proposer --------------------------------------------------------
    descriptions = _column_descriptions(discovery_cfg)
    flat_columns = low_variation_columns(screener.sample, base_features)
    if flat_columns:
        logger.info(
            "discovery: %d/%d columns barely vary across rows and are flagged in the "
            "prompt; a ratio pairing one with a varying column reproduces that "
            "column and is rejected as redundant",
            len(flat_columns), len(base_features),
        )
    described = sum(1 for c in base_features if descriptions.get(c))
    if described < len(base_features):
        logger.warning(
            "discovery: %d/%d columns have no description; a proposer cannot use "
            "real-world knowledge about a column it only knows by name",
            len(base_features) - described, len(base_features),
        )
    categorical_columns = _categorical_columns(
        discovery_cfg, base_features,
        known=[*base_features, *dataset.new_features],
    )
    if categorical_columns:
        logger.info(
            "discovery: %d/%d columns presented as coded categories (levels, not ranges)",
            len(categorical_columns), len(base_features),
        )
    # One batch of example rows per round, precomputed by the dataset's prepare
    # step, so a later round reasons from new evidence instead of re-reading the
    # same rows.
    shot_batches = _load_shot_batches(cfg, dataset)
    context = PromptContext(
        task_description=discovery_cfg.task_description,
        task_context=_task_context(discovery_cfg),
        column_contexts=[
            describe_columns(
                batch,
                base_features,
                descriptions=descriptions,
                categorical=categorical_columns,
                # Computed on the whole sample, not the handful of rows shown,
                # since a few values cannot reveal that a column barely moves.
                low_variation=flat_columns,
                label=dataset.target,
            )
            for batch in shot_batches
        ],
        metric_name=metric_name,
        metric_explanation=metric_explanation,
        n_rows=len(dataset.split("train")),
        redundancy_max_abs=redundancy_max_abs,
    )
    proposer = STRATEGIES[discovery_cfg.strategy](
        context,
        LLMSettings.from_config(discovery_cfg.llm),
        output_dir=Path(output_dir) / "discovery" if output_dir else None,
    )

    # Every proposal earlier runs made, with its verdict: read into the prompt so
    # an idea already judged is not proposed again, and counted so the round
    # numbers - and with them the rotation through the few-shot batches - carry on.
    prior = _read_history(discovery_cfg.history_path)
    round_offset = max((int(r["round"]) for r in prior if r.get("round") is not None),
                       default=0)
    if prior:
        logger.info("discovery: %d earlier proposal(s) read from %s; rounds continue "
                    "from %d", len(prior), discovery_cfg.history_path, round_offset + 1)

    run: DiscoveryRun = run_discovery(
        proposer,
        screener,
        batch_size=discovery_cfg.batch_size,
        stopping=StoppingRule(
            max_rounds=discovery_cfg.max_rounds,
            target_features=discovery_cfg.target_features,
            patience=discovery_cfg.patience,
            max_candidates=discovery_cfg.max_candidates,
        ),
        min_delta=discovery_cfg.min_delta,
        logger=logger,
        prior_history=prior,
        round_offset=round_offset,
    )

    widened, dropped = _materialise(dataset, run.kept, discovery_cfg.spike_factor)
    for candidate, reason in dropped:
        # Recorded as a rejection like any the screen makes, so the run's kept
        # list, its records and its round counts all agree with what validation
        # actually receives.
        logger.warning(
            "discovery: dropping '%s' - it passed the screen but fails on the full "
            "splits: %s", candidate.feature_name, reason,
        )
        candidate.screen.ok = False
        candidate.screen.extras["kept"] = False
        candidate.screen.error = f"Dropped on the full splits: {reason}"
        round_record = run.rounds[candidate.round_index - 1 - round_offset]
        round_record.kept -= 1
        round_record.rejected += 1

    kept = run.kept
    failed = [c for c in run.candidates if not c.ok]
    logger.info(
        "discovery: %d proposed over %d round(s) - %d forwarded, %d rejected",
        len(run.candidates), len(run.rounds), len(kept), len(failed),
    )
    if failed:
        # Surfaced rather than buried: a run where most blocks would not execute
        # is a prompt problem, and the reasons are the evidence for fixing it.
        reasons: dict[str, int] = {}
        for candidate in failed:
            head = (candidate.screen.error or "unknown").split(":")[0]
            reasons[head] = reasons.get(head, 0) + 1
        logger.info("discovery: rejection reasons: %s",
                    ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items())))
    logger.info(
        "discovery: handing %d feature(s) to validation: %s",
        len(kept), ", ".join(c.feature_name for c in kept) or "none",
    )

    return DiscoveryResult(
        dataset=widened,
        kept_features=[c.feature_name for c in kept],
        records=run.records(),
        rounds=[vars(r) for r in run.rounds],
        stopped_because=run.stopped_because,
        base_score=run.base_score,
    )


def _materialise(
    dataset: Dataset, kept: list[Candidate], spike_factor: float,
) -> tuple[Dataset, list[tuple[Candidate, str]]]:
    """
    Compute the kept features on every split and return the widened dataset.

    Applied per split through the same sandbox that screened them, so the column
    in test is computed by exactly the code that was screened on train. Each
    column then goes through the screen's value guards again, now on every split:
    the screen saw a few thousand train rows, and a blow-up on a row it never
    drew - in the split used for refitting, say - validates healthy and deploys
    broken. A candidate that fails anywhere is dropped rather than allowed to
    half-exist across splits, and returned with the reason.
    """
    if not kept:
        return dataset, []

    # Computed from the base columns alone, as the screen did: a block may only
    # read those, and nothing else in the frame should be able to change it.
    columns: dict[str, dict[str, pd.Series]] = {}
    dropped: list[tuple[Candidate, str]] = []
    for candidate in kept:
        name = candidate.feature_name
        try:
            extended = {}
            for split, frame in dataset.frames.items():
                # The screen checked the name against base columns only; the full
                # frame also holds ids, the target and hand-written candidates.
                validate_single_column(candidate.code, frame.columns)
                extended[split] = apply_code(frame[dataset.base_features], [candidate.code])
                check_finite(extended[split], name, split)
            check_scale(extended, name, spike_factor)
        except CandidateError as error:
            dropped.append((candidate, str(error)))
            continue
        except Exception as error:  # noqa: BLE001 - any failure drops the candidate
            dropped.append((candidate, f"{type(error).__name__}: {error}"))
            continue
        columns[name] = {split: frame[name] for split, frame in extended.items()}

    names = list(columns)
    widened = {
        split: frame.assign(**{name: columns[name][split] for name in names})
        for split, frame in dataset.frames.items()
    }

    # Appended, not replaced: a config may already list hand-written candidates,
    # and validating those beside the discovered ones is the useful behaviour.
    combined = [*dataset.new_features, *(n for n in names if n not in dataset.new_features)]

    return Dataset(
        frames=widened,
        target=dataset.target,
        base_features=list(dataset.base_features),
        new_features=combined,
        weight_col=dataset.weight_col,
        meta={**dataset.meta, "discovered_features": names},
    ), dropped
