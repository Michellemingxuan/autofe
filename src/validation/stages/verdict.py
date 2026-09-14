"""Stage 5 - the decision: does each proposed feature earn a place, and if none
of them do, is it time to propose a different batch?

Four gates, applied in order. A feature that fails one never reaches the next,
because it genuinely cannot be measured there - a candidate dropped at selection
is never in a model, so it has no Gini gain and no SHAP rank to read.

    1. data quality      the column itself is unusable
    2. feature selection no signal, or signal the incumbents already carry
    3. gini gain         it entered a model and the model got no better
    4. shap rank         the model kept it but leans on it barely

Then one batch-level call: if nothing cleared all four, the family is exhausted
and the next move is a different batch, not a lower threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from validation.config import Config
from validation.logging_utils import get_logger
from validation.stages.analysis import AnalysisResult
from validation.stages.data_quality import DataQualityResult
from validation.stages.feature_selection import FeatureSelectionResult

logger = get_logger(__name__)

PASS = "PASS"
FAIL = "FAIL"
NOT_REACHED = "not reached"      # an earlier gate already decided (enforce mode only)
NOT_EVALUABLE = "not evaluable"  # the run did not produce what this gate needs

GATES = ["data quality", "feature selection", "gini gain", "shap rank"]


@dataclass
class BatchVerdict:
    """The call on the batch as a whole."""
    verdict: str = "TRY A NEW BATCH"
    n_candidates: int = 0
    n_passed: int = 0
    passed: List[str] = field(default_factory=list)
    failed_at: Dict[str, int] = field(default_factory=dict)
    batch_gini_gain: float = float("nan")
    note: str = ""

    def summary(self) -> Dict[str, object]:
        return {
            "verdict": self.verdict,
            "n_candidates": self.n_candidates,
            "n_passed": self.n_passed,
            "passed": self.passed,
            "failed_at": self.failed_at,
            "batch_gini_gain": None if np.isnan(self.batch_gini_gain) else round(self.batch_gini_gain, 5),
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Per-feature cascade
# --------------------------------------------------------------------------- #
def _gini_lookup(analysis: Optional[AnalysisResult], split: str) -> Dict[str, Dict[str, float]]:
    """Per-feature Gini gain, from the leave-one-in variants when they were trained."""
    if analysis is None or analysis.comparison.empty:
        return {}
    frame = analysis.comparison.set_index("variant")
    out: Dict[str, Dict[str, float]] = {}
    for variant in frame.index:
        if not str(variant).startswith("loi__"):
            continue
        feature = str(variant)[len("loi__"):]
        out[feature] = {
            s: float(frame.loc[variant, f"gini_gain_{s}"])
            for s in (split, "valid") if f"gini_gain_{s}" in frame.columns
        }
    return out


def _shap_lookup(analysis: Optional[AnalysisResult], variant: str) -> pd.DataFrame:
    if analysis is None or analysis.shap_ranking.empty:
        return pd.DataFrame()
    frame = analysis.shap_ranking
    frame = frame[frame["variant"] == variant]
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["shap_rank_pct"] = frame["shap_rank"] / len(frame)
    return frame.set_index("feature")


def build_verdicts(
    candidates: List[str],
    cfg: Config,
    dq: Optional[DataQualityResult] = None,
    fs: Optional[FeatureSelectionResult] = None,
    analysis: Optional[AnalysisResult] = None,
    enforce: bool = True,
) -> pd.DataFrame:
    """One row per proposed feature, carrying every gate's outcome.

    With ``enforce`` false (``run.gates: open``) no gate stops the cascade: every
    candidate is measured at all four, so a feature that would have been cut at
    selection still shows its Gini gain and SHAP rank. That is the diagnostic
    view - what each variable did at each stage - rather than the filter.
    """
    vc = cfg.verdict
    dq_failed = set(dq.failed) if (dq and not dq.skipped) else set()
    selected = set(fs.selected) if fs else set(candidates)
    dropped = dict(fs.dropped) if fs else {}
    stage_of = dict(fs.dropped_stage) if fs else {}

    stats = (fs.target_stats.set_index("feature")
             if fs is not None and not fs.target_stats.empty else pd.DataFrame())
    gini = _gini_lookup(analysis, vc.gini_split)
    shap = _shap_lookup(analysis, vc.shap_variant)

    rows = []
    for feature in candidates:
        row: Dict[str, object] = {"feature": feature}
        for gate in GATES:
            row[gate] = NOT_REACHED
        verdict, failed_at, reason = PASS, "", ""

        reasons: List[str] = []
        stopped = False   # in enforce mode, an earlier gate ends the cascade

        # --- gate 1: data quality ---
        if feature in dq_failed:
            row["data quality"] = FAIL
            verdict, failed_at = FAIL, failed_at or "data quality"
            reasons.append("failed data quality checks")
            stopped = enforce
        else:
            row["data quality"] = PASS if (dq and not dq.skipped) else NOT_EVALUABLE

        # --- gate 2: feature selection ---
        if not stopped:
            if feature in dropped:
                row["feature selection"] = FAIL
                verdict, failed_at = FAIL, failed_at or "feature selection"
                reasons.append(f"{stage_of.get(feature, 'selection')}: {dropped[feature]}")
                stopped = enforce
            elif feature in selected:
                row["feature selection"] = PASS
            else:
                row["feature selection"] = FAIL
                verdict, failed_at = FAIL, failed_at or "feature selection"
                reasons.append("not carried forward")
                stopped = enforce

        # --- gate 3: gini gain ---
        if not stopped:
            gains = gini.get(feature)
            if not gains or np.isnan(gains.get(vc.gini_split, np.nan)):
                row["gini gain"] = NOT_EVALUABLE
            else:
                primary = gains[vc.gini_split]
                ok = primary >= vc.min_gini_gain
                if ok and vc.require_valid_too and not np.isnan(gains.get("valid", np.nan)):
                    ok = gains["valid"] >= vc.min_gini_gain
                row["gini gain"] = PASS if ok else FAIL
                if not ok:
                    verdict, failed_at = FAIL, failed_at or "gini gain"
                    reasons.append(f"gini gain {primary:+.4f} on {vc.gini_split}"
                                   f" (valid {gains.get('valid', float('nan')):+.4f}),"
                                   f" below {vc.min_gini_gain:+.4f}")
                    stopped = enforce

        # --- gate 4: shap rank ---
        if not stopped:
            if shap.empty or feature not in shap.index:
                row["shap rank"] = NOT_EVALUABLE
            else:
                pct = float(shap.loc[feature, "shap_rank_pct"])
                ok = pct <= vc.max_shap_rank_pct
                row["shap rank"] = PASS if ok else FAIL
                if not ok:
                    verdict, failed_at = FAIL, failed_at or "shap rank"
                    reasons.append(f"SHAP rank {int(shap.loc[feature, 'shap_rank'])}"
                                   f" of {len(shap)} ({pct:.0%}), outside the top"
                                   f" {vc.max_shap_rank_pct:.0%}")

        reason = " | ".join(reasons)

        row["verdict"] = verdict
        row["failed_at"] = failed_at
        row["n_gates_failed"] = sum(1 for gate in GATES if row[gate] == FAIL)
        row["reason"] = reason
        for column in ("spearman_target", "spearman_redundancy_base", "mrmr_score"):
            row[column] = (float(stats.loc[feature, column])
                           if len(stats) and feature in stats.index and column in stats.columns
                           else float("nan"))
        gains = gini.get(feature, {})
        row[f"gini_gain_{vc.gini_split}"] = gains.get(vc.gini_split, float("nan"))
        row["shap_rank_pct"] = (float(shap.loc[feature, "shap_rank_pct"])
                                if len(shap) and feature in shap.index else float("nan"))
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    order = ["feature", "verdict", "failed_at", *GATES, "n_gates_failed", "reason",
             "spearman_target", "spearman_redundancy_base",
             f"gini_gain_{vc.gini_split}", "shap_rank_pct", "mrmr_score"]
    table = table[[c for c in order if c in table.columns]]
    table["_rank"] = (table["verdict"] != PASS).astype(int)
    return (table.sort_values(["_rank", f"gini_gain_{vc.gini_split}"], ascending=[True, False])
                 .drop(columns="_rank").reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Batch call
# --------------------------------------------------------------------------- #
def decide_batch(verdicts: pd.DataFrame, cfg: Config, analysis: Optional[AnalysisResult] = None) -> BatchVerdict:
    """Keep what passed, or - if nothing did - go get a different batch."""
    vc = cfg.verdict
    batch_gain = float("nan")
    if analysis is not None and not analysis.comparison.empty:
        frame = analysis.comparison.set_index("variant")
        column = f"gini_gain_{vc.gini_split}"
        if "base_plus_new" in frame.index and column in frame.columns:
            batch_gain = float(frame.loc["base_plus_new", column])

    if verdicts.empty:
        return BatchVerdict(verdict="TRY A NEW BATCH", note="no candidates were proposed")

    passed = verdicts.loc[verdicts["verdict"] == PASS, "feature"].tolist()
    failed_at = (verdicts.loc[verdicts["verdict"] == FAIL, "failed_at"]
                 .value_counts().to_dict())

    if passed:
        note = (f"{len(passed)} candidate(s) cleared every gate; "
                f"the batch moves {batch_gain:+.4f} Gini on {vc.gini_split}.")
        verdict = "KEEP"
    else:
        worst = max(failed_at, key=failed_at.get) if failed_at else "unknown"
        note = (f"No candidate cleared all four gates - most fell at the {worst}. "
                "Propose a different batch rather than relaxing the thresholds.")
        verdict = "TRY A NEW BATCH"
    if not cfg.gates_enforced:
        note += (" Gates are OPEN (run.gates: open): this is advisory only - nothing was"
                 " removed, and every candidate was measured at all four gates.")

    return BatchVerdict(
        verdict=verdict,
        n_candidates=len(verdicts),
        n_passed=len(passed),
        passed=passed,
        failed_at=failed_at,
        batch_gini_gain=batch_gain,
        note=note,
    )


def run_verdict(
    candidates: List[str],
    cfg: Config,
    dq: Optional[DataQualityResult] = None,
    fs: Optional[FeatureSelectionResult] = None,
    analysis: Optional[AnalysisResult] = None,
) -> tuple[pd.DataFrame, BatchVerdict]:
    verdicts = build_verdicts(candidates, cfg, dq=dq, fs=fs, analysis=analysis,
                              enforce=cfg.gates_enforced)
    batch = decide_batch(verdicts, cfg, analysis=analysis)
    logger.info("batch verdict: %s (%d/%d passed)", batch.verdict, batch.n_passed, batch.n_candidates)
    return verdicts, batch
