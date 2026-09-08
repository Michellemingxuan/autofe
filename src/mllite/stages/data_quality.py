"""Stage 1 - variable data quality & stability.

PLACEHOLDER STAGE. The intended implementation is a port of
``https://github.aexp.com/sssheno/AIME_DataStability`` (internal). The plumbing
here - config, parallel fan-out over feature chunks, report shape, and the
downstream contract - is finished, so porting means filling in the two ``TODO``
functions below and nothing else.

Contract expected by the rest of the pipeline:
    * ``report``  - one row per feature, arbitrary metric columns, plus a boolean
                    ``passed`` column.
    * ``failed``  - feature names that fail the configured thresholds. When
                    ``data_quality.drop_failed`` is true these are removed from
                    the feature lists before feature selection runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from mllite.config import Config
from mllite.data import Dataset
from mllite.logging_utils import get_logger, timed
from mllite.parallel import chunked, parallel_map, resolve_n_jobs

logger = get_logger(__name__)


@dataclass
class DataQualityResult:
    report: pd.DataFrame = field(default_factory=pd.DataFrame)
    failed: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    skipped: bool = True

    def summary(self) -> Dict[str, object]:
        return {
            "skipped": self.skipped,
            "n_features_checked": int(len(self.report)),
            "n_failed": len(self.failed),
            "thresholds": self.thresholds,
            "failed_by_check": self.failed_by_check(),
        }

    def failed_by_check(self) -> Dict[str, int]:
        """How many features each threshold rejected. A feature can fail several."""
        if self.report.empty or "failed_checks" not in self.report.columns:
            return {}
        counts: Dict[str, int] = {}
        for entry in self.report["failed_checks"]:
            for check in str(entry).split(", "):
                if check:
                    counts[check] = counts.get(check, 0) + 1
        return counts


def _profile_chunk(features: List[str], df: pd.DataFrame) -> pd.DataFrame:
    """TODO(AIME_DataStability): replace with the real quality checks.

    Currently reports only the two things that are unambiguous and cheap:
    missing rate and cardinality.
    """
    rows = []
    n = max(len(df), 1)
    for col in features:
        series = df[col]
        rows.append({
            "feature": col,
            "missing_rate": float(series.isna().mean()),
            "n_unique": int(series.nunique(dropna=True)),
            "n_rows": n,
        })
    return pd.DataFrame(rows)


def population_stability_index(reference: np.ndarray, comparison: np.ndarray,
                               bins: int = 10) -> float:
    """PSI of one variable between two samples.

    Bin edges are the reference's quantiles, so the reference is uniform across
    buckets by construction and the index measures how far the comparison sample
    departs from it. Missing values get their own bucket - a variable that is 2%
    null in train and 40% null in test has genuinely shifted, and binning NaN away
    would hide exactly that.

        < 0.10  no meaningful shift
        < 0.25  moderate shift, worth a look
        >= 0.25 the variable is not the same thing in the two samples

    Empty buckets are smoothed rather than dropped, so a bucket present in one
    sample and absent in the other contributes a large finite term instead of inf.
    """
    reference = np.asarray(reference, dtype=np.float64)
    comparison = np.asarray(comparison, dtype=np.float64)
    if reference.size == 0 or comparison.size == 0:
        return float("nan")

    finite = reference[~np.isnan(reference)]
    if finite.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1]))
    n_levels = len(edges) + 1

    def buckets(values: np.ndarray) -> np.ndarray:
        codes = np.searchsorted(edges, values, side="right").astype(np.int64)
        codes[np.isnan(values)] = n_levels          # NaN is its own bucket
        return np.bincount(codes, minlength=n_levels + 1).astype(np.float64)

    a, b = buckets(reference), buckets(comparison)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")

    # smoothing floor: half an observation, so an empty bucket stays finite
    pa = np.maximum(a / a.sum(), 0.5 / a.sum())
    pb = np.maximum(b / b.sum(), 0.5 / b.sum())
    return float(((pa - pb) * np.log(pa / pb)).sum())


def _distribution_chunk(features: List[str], matrices: Dict[str, np.ndarray],
                        column_index: Dict[str, int], reference: str,
                        bins: int) -> pd.DataFrame:
    """PSI of every feature in the chunk, reference split vs each other split."""
    others = [s for s in matrices if s != reference]
    rows = []
    for name in features:
        column = column_index[name]
        row: Dict[str, object] = {"feature": name}
        values = {}
        for split in others:
            value = population_stability_index(
                matrices[reference][:, column], matrices[split][:, column], bins)
            row[f"psi_{split}"] = value
            values[split] = value
        finite = {k: v for k, v in values.items() if not np.isnan(v)}
        row["psi_max"] = max(finite.values()) if finite else float("nan")
        row["psi_worst_split"] = max(finite, key=finite.get) if finite else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _stability_chunk(features: List[str], df: pd.DataFrame, by_col: str, reference: str) -> pd.DataFrame:
    """TODO(AIME_DataStability): population stability across ``by_col`` periods.

    Should return one row per feature with at least a ``psi`` column measured
    against ``reference``. Returns an empty frame until ported.
    """
    return pd.DataFrame(columns=["feature", "psi"])


def run_data_quality(dataset: Dataset, cfg: Config) -> DataQualityResult:
    dq = cfg.data_quality
    if not dq.enabled:
        logger.info("data quality stage disabled; skipping")
        return DataQualityResult(skipped=True)

    features = dataset.all_features
    df = dataset.split("train")
    workers = resolve_n_jobs(cfg.run.n_jobs)
    chunks = list(chunked(features, max(1, -(-len(features) // workers))))

    profiles = parallel_map(
        _profile_chunk, chunks,
        n_jobs=cfg.run.n_jobs, backend=cfg.run.backend, desc="dq chunk",
        df=df[features],
    )
    report = pd.concat([p for p in profiles if len(p)], ignore_index=True) if profiles else pd.DataFrame()

    if dq.distribution_check and len(dataset.available_splits()) > 1:
        splits = dataset.available_splits()
        reference = dq.distribution_reference
        if reference not in splits:
            logger.warning("distribution_reference %r is not among the splits %s; "
                           "skipping the distribution check", reference, splits)
        else:
            column_index = {name: i for i, name in enumerate(features)}
            matrices = {s: dataset.split(s)[features].to_numpy(dtype=np.float64) for s in splits}
            with timed(logger, f"distribution check vs {reference}"):
                distribution = parallel_map(
                    _distribution_chunk, chunks,
                    n_jobs=cfg.run.n_jobs, backend=cfg.run.backend, desc="distribution chunk",
                    matrices=matrices, column_index=column_index,
                    reference=reference, bins=dq.distribution_bins,
                )
            distribution = [d for d in distribution if len(d)]
            if distribution:
                report = report.merge(pd.concat(distribution, ignore_index=True),
                                      on="feature", how="left")
                shifted = int((report["psi_max"] > dq.max_psi).sum())
                logger.info("distribution: %d/%d feature(s) shift beyond psi %.2f vs %s",
                            shifted, len(report), dq.max_psi, reference)

    if dq.by_col and dq.by_col in df.columns:
        stability = parallel_map(
            _stability_chunk, chunks,
            n_jobs=cfg.run.n_jobs, backend=cfg.run.backend, desc="stability chunk",
            df=df, by_col=dq.by_col, reference=dq.reference_period,
        )
        stability = [s for s in stability if len(s)]
        if stability:
            report = report.merge(pd.concat(stability, ignore_index=True), on="feature", how="left")
        else:
            logger.warning("stability checks not implemented yet; port AIME_DataStability into _stability_chunk")

    if report.empty:
        return DataQualityResult(report=report, failed=[], skipped=False)

    # Each check is applied separately so the report can say *which* threshold a
    # feature fell foul of, rather than only that it failed something.
    checks = {
        f"missing_rate > {dq.max_missing_rate:g}": report["missing_rate"] > dq.max_missing_rate,
        f"n_unique < {dq.min_unique:g}": report["n_unique"] < dq.min_unique,
    }
    thresholds = {"max_missing_rate": dq.max_missing_rate, "min_unique": dq.min_unique}
    for column in ("psi_max", "psi"):
        if column in report.columns:
            checks[f"{column} > {dq.max_psi:g}"] = report[column].fillna(0.0) > dq.max_psi
            thresholds["max_psi"] = dq.max_psi

    failing = pd.DataFrame(checks)
    report["failed_checks"] = [", ".join(failing.columns[row]) for row in failing.to_numpy()]
    report["passed"] = ~failing.any(axis=1)

    failed = report.loc[~report["passed"], "feature"].tolist()
    result = DataQualityResult(report=report, failed=failed, thresholds=thresholds, skipped=False)
    logger.info("data quality: %d/%d feature(s) failed %s",
                len(failed), len(report), result.failed_by_check() or "")
    return result
