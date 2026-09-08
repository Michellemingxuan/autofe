"""Stage 2 - feature selection for newly proposed variables.

Two screens, both computed on a sample of the training split:

1. **Signal**   - Spearman rho and mutual information between each *new* feature
                  and the outcome. Weak features are dropped.
2. **Redundancy** - Spearman rho and normalized mutual information between each
                  *new* feature and (a) the incumbent features and (b) the new
                  features already kept. Greedy: candidates are considered
                  strongest-first, and a candidate that duplicates something
                  already in the set is dropped.

Both screens are matrix kernels (BLAS for rank correlation, joint bincount for
MI) fanned out over chunks of new features. Large arrays are handed to workers as
raw numpy so joblib memory-maps them instead of pickling a copy per worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mllite.config import Config
from mllite.data import Dataset
from mllite.logging_utils import get_logger, timed
from mllite.parallel import chunked, parallel_map

logger = get_logger(__name__)


@dataclass
class FeatureSelectionResult:
    target_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    spearman_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    mi_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    redundancy_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected: List[str] = field(default_factory=list)
    dropped: Dict[str, str] = field(default_factory=dict)
    dropped_stage: Dict[str, str] = field(default_factory=dict)   # feature -> which gate
    skipped: bool = False

    def summary(self) -> Dict[str, object]:
        return {
            "skipped": self.skipped,
            "n_candidates": len(self.selected) + len(self.dropped),
            "n_selected": len(self.selected),
            "n_dropped": len(self.dropped),
            "selected": self.selected,
        }


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def pairwise_complete_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of ``x`` against every column of ``y``.

    NaNs are handled by pairwise deletion, expressed as matrix products so the
    whole block is one BLAS call rather than a Python loop over pairs. Feed it
    ranks to get Spearman.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mx = (~np.isnan(x)).astype(np.float64)
    my = (~np.isnan(y)).astype(np.float64)
    zx = np.nan_to_num(x)
    zy = np.nan_to_num(y)

    n = mx.T @ my
    sx = zx.T @ my
    sy = mx.T @ zy
    sxy = zx.T @ zy
    sxx = (zx * zx).T @ my
    syy = mx.T @ (zy * zy)

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sy / n
        varx = sxx - sx * sx / n
        vary = syy - sy * sy / n
        corr = cov / np.sqrt(varx * vary)
    corr[n < 3] = np.nan
    return np.clip(corr, -1.0, 1.0)


def rank_columns(values: np.ndarray) -> np.ndarray:
    """Average-rank transform per column, NaNs preserved."""
    return pd.DataFrame(values).rank(method="average", na_option="keep").to_numpy(dtype=np.float64)


def quantile_codes(values: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Discretize each column into <= ``bins`` quantile buckets.

    NaN gets its own bucket, so "missing" carries information rather than
    silently dropping rows. Returns the integer codes and the per-column bucket
    counts.
    """
    n_rows, n_cols = values.shape
    codes = np.zeros((n_rows, n_cols), dtype=np.int32)
    sizes = np.ones(n_cols, dtype=np.int32)
    for j in range(n_cols):
        col = values[:, j]
        finite = col[~np.isnan(col)]
        if finite.size == 0:
            continue
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1]))
        code = np.searchsorted(edges, col, side="right").astype(np.int32)
        n_levels = len(edges) + 1
        code[np.isnan(col)] = n_levels          # missing bucket
        codes[:, j] = code
        sizes[j] = n_levels + 1
    return codes, sizes


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def mutual_info_codes(a: np.ndarray, b: np.ndarray, na: int, nb: int, normalize: bool = True) -> float:
    """MI (nats) between two already-discretized columns, via a joint histogram.

    With ``normalize`` the result is divided by min(H(a), H(b)), giving a 0-1
    redundancy score comparable across features with different cardinality.
    """
    joint = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(np.float64)
    total = joint.sum()
    if total <= 0:
        return float("nan")
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    ha, hb = _entropy(pa), _entropy(pb)
    nz = joint > 0
    p_joint = joint[nz] / total
    outer = np.outer(pa, pb)[nz] / (total * total)
    mi = float((p_joint * np.log(p_joint / outer)).sum())
    mi = max(mi, 0.0)
    if not normalize:
        return mi
    denom = min(ha, hb)
    return mi / denom if denom > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Parallel workers (module level so they pickle cleanly)
# --------------------------------------------------------------------------- #
def _corr_chunk(idx: List[int], ranks: np.ndarray, target_ranks: np.ndarray) -> Tuple[List[int], np.ndarray, np.ndarray]:
    block = ranks[:, idx]
    return idx, pairwise_complete_corr(block, ranks), pairwise_complete_corr(block, target_ranks).ravel()


def _mi_chunk(
    idx: List[int],
    codes: np.ndarray,
    sizes: np.ndarray,
    target_codes: np.ndarray,
    target_size: int,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """Redundancy row and both flavours of target MI for a chunk of candidates.

    Raw MI (nats) is what the ``target_min`` threshold is expressed in; the
    normalized version is what can be compared against redundancy, which is
    normalized too. Mixing the two makes the mRMR difference meaningless -
    for a 3% event H(y) is about 0.14 nats, so raw relevance can never offset a
    redundancy term that lives on 0-1.
    """
    n_all = codes.shape[1]
    red = np.empty((len(idx), n_all), dtype=np.float64)
    sig = np.empty(len(idx), dtype=np.float64)
    sig_norm = np.empty(len(idx), dtype=np.float64)
    for r, i in enumerate(idx):
        ai, na = codes[:, i], int(sizes[i])
        for j in range(n_all):
            red[r, j] = 1.0 if i == j else mutual_info_codes(ai, codes[:, j], na, int(sizes[j]))
        sig[r] = mutual_info_codes(ai, target_codes, na, target_size, normalize=False)
        sig_norm[r] = mutual_info_codes(ai, target_codes, na, target_size, normalize=True)
    return idx, red, sig, sig_norm


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def _sample_frame(dataset: Dataset, cfg: Config) -> pd.DataFrame:
    fs = cfg.feature_selection
    df = dataset.split(fs.on_split)
    if fs.sample_size and len(df) > fs.sample_size:
        df = df.sample(n=fs.sample_size, random_state=cfg.run.seed)
    return df


def _target_mi_sklearn(frame: pd.DataFrame, features: List[str], target: str, task: str, seed: int) -> np.ndarray:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    x = frame[features].to_numpy(dtype=np.float64)
    x = np.where(np.isnan(x), np.nanmedian(x, axis=0), x)
    y = frame[target].to_numpy()
    fn = mutual_info_classif if task == "binary" else mutual_info_regression
    return fn(x, y, random_state=seed)


def run_feature_selection(dataset: Dataset, cfg: Config) -> FeatureSelectionResult:
    fs = cfg.feature_selection
    new_features = list(dataset.new_features)
    if not fs.enabled or not new_features:
        logger.info("feature selection disabled or no new features; keeping all %d", len(new_features))
        return FeatureSelectionResult(selected=new_features, skipped=True)

    all_features = dataset.all_features
    index = {name: i for i, name in enumerate(all_features)}
    new_idx = [index[f] for f in new_features]

    frame = _sample_frame(dataset, cfg)
    logger.info("feature selection on %d row(s) x %d feature(s)", len(frame), len(all_features))
    values = frame[all_features].to_numpy(dtype=np.float64)
    target = frame[dataset.target].to_numpy(dtype=np.float64)

    chunks = list(chunked(new_idx, fs.chunk_size))
    stats = pd.DataFrame({"feature": new_features}).set_index("feature")
    spearman_matrix = pd.DataFrame(index=new_features, columns=all_features, dtype=float)
    mi_matrix = pd.DataFrame(index=new_features, columns=all_features, dtype=float)

    if fs.spearman.enabled:
        with timed(logger, "spearman screen"):
            ranks = rank_columns(values)
            target_ranks = rank_columns(target.reshape(-1, 1))
            results = parallel_map(
                _corr_chunk, chunks,
                n_jobs=cfg.run.n_jobs, backend=cfg.run.backend, desc="spearman chunk",
                ranks=ranks, target_ranks=target_ranks,
            )
            for idx, red, sig in results:
                names = [all_features[i] for i in idx]
                spearman_matrix.loc[names, :] = red
                stats.loc[names, "spearman_target"] = sig

    if fs.mutual_info.enabled:
        with timed(logger, "mutual information screen"):
            codes, sizes = quantile_codes(values, fs.mutual_info.bins)
            if cfg.model.task == "binary":
                target_codes = target.astype(np.int32)
                target_size = int(target_codes.max()) + 1
            else:
                target_codes, target_sizes = quantile_codes(target.reshape(-1, 1), fs.mutual_info.bins)
                target_codes, target_size = target_codes.ravel(), int(target_sizes[0])
            results = parallel_map(
                _mi_chunk, chunks,
                n_jobs=cfg.run.n_jobs, backend=cfg.run.backend, desc="mi chunk",
                codes=codes, sizes=sizes, target_codes=target_codes, target_size=target_size,
            )
            for idx, red, sig, sig_norm in results:
                names = [all_features[i] for i in idx]
                mi_matrix.loc[names, :] = red
                stats.loc[names, "mi_target"] = sig
                stats.loc[names, "nmi_target"] = sig_norm
            if fs.mutual_info.target_method == "sklearn":
                stats.loc[new_features, "mi_target"] = _target_mi_sklearn(
                    frame, new_features, dataset.target, cfg.model.task, cfg.run.seed
                )

    stats = stats.reset_index()
    stats = _augment_with_redundancy(stats, spearman_matrix, mi_matrix, dataset, cfg)
    selected, dropped, stages, redundancy_rows = _apply_screens(
        stats, spearman_matrix, mi_matrix, dataset, cfg
    )
    logger.info("feature selection kept %d/%d new feature(s)", len(selected), len(new_features))

    return FeatureSelectionResult(
        target_stats=stats,
        spearman_matrix=spearman_matrix,
        mi_matrix=mi_matrix,
        redundancy_summary=pd.DataFrame(redundancy_rows),
        selected=selected,
        dropped=dropped,
        dropped_stage=stages,
    )


def _aggregate(values: pd.Series, stat: str) -> float:
    values = values.dropna()
    if values.empty:
        return float("nan")
    return float(values.max() if stat == "max" else values.mean())


def _augment_with_redundancy(
    stats: pd.DataFrame,
    spearman_matrix: pd.DataFrame,
    mi_matrix: pd.DataFrame,
    dataset: Dataset,
    cfg: Config,
) -> pd.DataFrame:
    """Put relevance and redundancy side by side as sortable columns.

    Both quantities are already computed - this makes the trade-off between them
    directly readable instead of leaving it implicit in two separate thresholds:

        mi_target            raw MI with the outcome, in nats
        nmi_target           the same, normalized to 0-1   higher is better
        mi_redundancy_base   normalized MI with the incumbent set   lower is better
        mrmr_score           nmi_target minus mi_redundancy_base

    The score uses the *normalized* relevance so both halves live on the same
    0-1 scale; subtracting raw nats from a normalized redundancy would let the
    redundancy term dominate every comparison.

    Redundancy is measured against the incumbent features only, so these columns
    are static and order-independent - unlike the greedy screen, which also has to
    account for candidates kept earlier in the same run.
    """
    if stats.empty or not dataset.base_features:
        return stats

    stat = cfg.feature_selection.mutual_info.redundancy_stat
    out = stats.set_index("feature")
    base = [c for c in dataset.base_features if c in spearman_matrix.columns or c in mi_matrix.columns]
    if not base:
        return stats

    if not spearman_matrix.empty:
        rho = spearman_matrix.loc[out.index, base].astype(float).abs()
        out["spearman_redundancy_base"] = [_aggregate(rho.loc[f], stat) for f in out.index]

    if not mi_matrix.empty:
        nmi = mi_matrix.loc[out.index, base].astype(float)
        out["mi_redundancy_base"] = [_aggregate(nmi.loc[f], stat) for f in out.index]
        if "nmi_target" in out.columns:
            out["mrmr_score"] = out["nmi_target"] - out["mi_redundancy_base"]

    return out.reset_index()


def _passes_signal(feature: str, row: pd.Series, cfg: Config) -> Optional[str]:
    """Return the rejection reason, or None when the candidate clears both gates."""
    sp, mi = cfg.feature_selection.spearman, cfg.feature_selection.mutual_info
    if sp.enabled and "spearman_target" in row.index:
        rho = row.get("spearman_target", np.nan)
        if not np.isnan(rho) and abs(rho) < sp.target_min_abs:
            return f"weak spearman vs target ({rho:.4f})"
    if mi.enabled and "mi_target" in row.index:
        value = row.get("mi_target", np.nan)
        if not np.isnan(value) and value < mi.target_min:
            return f"weak mutual information vs target ({value:.5f})"
    return None


def _check_redundancy(
    feature: str,
    compare_to: List[str],
    spearman_matrix: pd.DataFrame,
    mi_matrix: pd.DataFrame,
    cfg: Config,
) -> Tuple[Dict[str, object], Optional[str]]:
    """Measure a candidate against everything already in the set."""
    sp, mi = cfg.feature_selection.spearman, cfg.feature_selection.mutual_info
    record: Dict[str, object] = {"feature": feature}
    reason: Optional[str] = None

    if compare_to and sp.enabled and not spearman_matrix.empty:
        rho = spearman_matrix.loc[feature, compare_to].astype(float).abs()
        if len(rho.dropna()):
            record["max_abs_spearman"] = float(rho.max())
            record["closest_by_spearman"] = str(rho.idxmax())
            if rho.max() > sp.redundancy_max_abs:
                reason = f"redundant with {rho.idxmax()} (|rho|={rho.max():.3f})"

    if compare_to and mi.enabled and not mi_matrix.empty:
        nmi = mi_matrix.loc[feature, compare_to].astype(float)
        if len(nmi.dropna()):
            record["max_nmi"] = float(nmi.max())
            record["closest_by_nmi"] = str(nmi.idxmax())
            if reason is None and nmi.max() > mi.redundancy_max:
                reason = f"redundant with {nmi.idxmax()} (nmi={nmi.max():.3f})"

    return record, reason


def _static_order(stats: pd.DataFrame, cfg: Config) -> List[str]:
    """Candidate order for the non-adaptive rankings."""
    fs = cfg.feature_selection
    strength = stats.set_index("feature")
    key = {"spearman": "spearman_target", "mi": "mi_target"}.get(fs.ranking, "spearman_target")
    if key not in strength.columns:
        key = "mi_target" if "mi_target" in strength.columns else None
    if key is None:
        return list(strength.index)
    values = strength[key].abs() if key == "spearman_target" else strength[key]
    return list(values.sort_values(ascending=False).index)


def _apply_screens(
    stats: pd.DataFrame,
    spearman_matrix: pd.DataFrame,
    mi_matrix: pd.DataFrame,
    dataset: Dataset,
    cfg: Config,
) -> Tuple[List[str], Dict[str, str], Dict[str, str], List[Dict[str, object]]]:
    """Signal screen, then a greedy redundancy screen against everything kept.

    The greedy screen is order-dependent by construction - whichever member of a
    correlated cluster is considered first is the one that survives - so the order
    is a configured choice, not an accident:

        spearman  strongest |rho| with the outcome first (default)
        mi        highest mutual information with the outcome first
        mrmr      re-scored after every pick: relevance to the outcome minus
                  redundancy with the incumbents *and* the candidates kept so far
    """
    fs = cfg.feature_selection
    mi = fs.mutual_info
    strength = stats.set_index("feature")

    against_base = mi.pairwise_against in ("all", "base")
    against_kept = mi.pairwise_against in ("all", "new")
    pool = list(dataset.base_features) if against_base else []

    # Signal gate first, so a weak candidate never influences the greedy ordering.
    survivors, dropped, stages = [], {}, {}
    for feature in _static_order(stats, cfg):
        reason = _passes_signal(feature, strength.loc[feature], cfg)
        if reason:
            dropped[feature] = reason
            stages[feature] = "signal screen"
        else:
            survivors.append(feature)

    use_mrmr = (fs.ranking == "mrmr" and mi.enabled and not mi_matrix.empty
                and "nmi_target" in strength.columns)
    if fs.ranking == "mrmr" and not use_mrmr:
        logger.warning("ranking=mrmr needs mutual information enabled; falling back to a static order")

    kept: List[str] = []
    # `reached_screen` records how far a candidate got, not what rejected it -
    # `kept` and `reason` say that. Candidates cut by the signal screen never reach
    # the redundancy loop, so without a row here they would be absent from the
    # summary entirely and its row count would not reconcile with the candidate
    # count. Their redundancy columns stay empty: those questions were never asked.
    rows: List[Dict[str, object]] = [
        {"feature": feature, "reached_screen": "signal", "kept": False,
         "reason": dropped[feature]}
        for feature in dropped
    ]
    remaining = list(survivors)

    while remaining:
        if use_mrmr:
            feature = _next_by_mrmr(remaining, kept, pool, strength, mi_matrix, mi.redundancy_stat)
        else:
            feature = remaining[0]
        remaining.remove(feature)

        compare_to = [c for c in (pool + (kept if against_kept else [])) if c != feature]
        record, reason = _check_redundancy(feature, compare_to, spearman_matrix, mi_matrix, cfg)
        record["reached_screen"] = "redundancy"   # got past the signal gate
        if use_mrmr:
            record["mrmr_score_at_pick"] = _mrmr_score(
                feature, kept, pool, strength, mi_matrix, mi.redundancy_stat)
        record["kept"] = reason is None
        record["reason"] = reason or ""
        rows.append(record)

        if reason is None:
            kept.append(feature)
        else:
            dropped[feature] = reason
            stages[feature] = "redundancy screen"

    # Preserve the caller's original ordering for reproducibility downstream.
    kept_set = set(kept)
    return [f for f in dataset.new_features if f in kept_set], dropped, stages, rows


def _mrmr_score(
    feature: str,
    kept: List[str],
    pool: List[str],
    strength: pd.DataFrame,
    mi_matrix: pd.DataFrame,
    stat: str,
) -> float:
    """Relevance to the outcome minus redundancy with what is already in the set."""
    relevance = float(strength.loc[feature, "nmi_target"])
    against = [c for c in pool + kept if c != feature and c in mi_matrix.columns]
    if not against:
        return relevance
    redundancy = _aggregate(mi_matrix.loc[feature, against].astype(float), stat)
    return relevance if np.isnan(redundancy) else relevance - redundancy


def _next_by_mrmr(
    remaining: List[str],
    kept: List[str],
    pool: List[str],
    strength: pd.DataFrame,
    mi_matrix: pd.DataFrame,
    stat: str,
) -> str:
    """Pick the candidate with the best relevance/redundancy trade-off right now."""
    scores = {f: _mrmr_score(f, kept, pool, strength, mi_matrix, stat) for f in remaining}
    return max(remaining, key=lambda f: (-np.inf if np.isnan(scores[f]) else scores[f]))


def build_verdict_table(
    candidates: List[str],
    result: "FeatureSelectionResult",
    dq_failed: Optional[List[str]] = None,
) -> pd.DataFrame:
    """One row per proposed feature: did it get in, and what decided it.

    This is the answer the selection stage exists to produce - whether each newly
    proposed feature survives screening alongside the incumbent set. It spans
    stages on purpose: a candidate rejected by data quality never reaches the
    screens, so without this it would simply disappear from the report rather
    than showing up as excluded.

    Note the asymmetry it records. Incumbent features are never screened, so a
    candidate can be dropped for duplicating an incumbent but never the reverse.
    That is deliberate - the incumbent set is the status quo being challenged -
    but it does mean the process is biased toward rejecting candidates, and a
    "close call" here is not the same as the feature being worthless.
    """
    dq_failed = set(dq_failed or [])
    stats = result.target_stats.set_index("feature") if not result.target_stats.empty else pd.DataFrame()
    closest = (result.redundancy_summary.set_index("feature")
               if not result.redundancy_summary.empty else pd.DataFrame())
    selected = set(result.selected)

    rows = []
    for feature in candidates:
        if feature in dq_failed:
            verdict, stage, reason = "OUT", "data quality", "failed data quality checks"
        elif feature in selected:
            verdict, stage, reason = "IN", "-", ""
        elif feature in result.dropped:
            verdict = "OUT"
            stage = result.dropped_stage.get(feature, "selection")
            reason = result.dropped[feature]
        else:
            verdict, stage, reason = "OUT", "not evaluated", "absent from the selection stage"

        row = {"feature": feature, "verdict": verdict, "decided_by": stage, "reason": reason}
        for column in ("spearman_target", "nmi_target",
                       "spearman_redundancy_base", "mi_redundancy_base", "mrmr_score"):
            row[column] = float(stats.loc[feature, column]) if (
                len(stats) and feature in stats.index and column in stats.columns) else float("nan")
        row["closest_existing"] = str(closest.loc[feature, "closest_by_spearman"]) if (
            len(closest) and feature in closest.index
            and "closest_by_spearman" in closest.columns
            and pd.notna(closest.loc[feature, "closest_by_spearman"])) else ""
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["_order"] = (table["verdict"] != "IN").astype(int)
    table["_strength"] = table["spearman_target"].abs().fillna(-1)
    return (table.sort_values(["_order", "_strength"], ascending=[True, False])
                 .drop(columns=["_order", "_strength"])
                 .reset_index(drop=True))
