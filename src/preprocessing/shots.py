"""Choose the example rows the proposer is shown - once, in the prepare step.

Each dataset's prepare notebook runs :func:`build_shot_batches` on its train
split and saves those rows with a batch column; a discovery run reads that file back
through ``discovery.few_shot_path`` and shows batch r in round r.

The rows picked here fill the ``Samples [...]`` line under every column of the
prompt, so they are the only concrete data a language model ever sees of the
table. A uniform draw is a bad way to choose them twice over:

* **Class coverage.** At a 3% positive rate a 32-row draw usually contains zero
  positives, and the model is then asked to invent features that separate two
  classes while looking at one. So the batch is built per class - 32 shots over
  2 classes means 16 clusters of each - which guarantees a 16/16 split.

* **Coverage.** Rows drawn at random over-sample wherever the data is dense and
  miss whole regions of it. Taking one representative per KMeans cluster puts
  each shown row in a different region, so a fixed budget of 32 describes more
  of the table. Measured as the mean distance from a table row to the nearest
  shown row, this beats both ``head()`` and the best of 20 random draws on all
  three demo datasets (malware: 1.58 clustered vs 2.35 for head, 1.83 for the
  best random draw).

  Note that the representative is the row *closest to its cluster centre* - the
  most typical member of its region, not an extreme one. So the shown rows are
  more spread out in the sense that matters (they sit in different regions)
  while being individually less extreme than a random draw's outliers.

Cluster sizes are equalised (:func:`balanced_assignment`) only when more than
one batch is wanted, and that condition is the whole story. Since every cluster
contributes exactly one row, a cluster's size does not affect how much of the
shown data comes from it - so equalising sizes buys nothing for a single batch
and costs coverage, because forcing equal counts makes cluster boundaries cut
across genuinely uneven groups. Measured on the three demo datasets, plain
KMeans covers the data better every time (bankruptcy 5.20 vs 5.58, myocardial
4.78 vs 4.83, malware 1.54 vs 1.58).

What equal sizes do buy is rotation depth. A cluster of n rows can supply n
distinct batches before it repeats, and plain KMeans leaves clusters of one row
on two of those datasets - so round two would re-show round one's example.
Balancing lifts the smallest cluster to 13-14 rows there. Hence: one batch uses
plain KMeans for the best coverage, several batches balance to stay disjoint.

Successive batches take the 1st, 2nd, 3rd closest row of each cluster, so
consecutive rounds of a discovery run see disjoint examples drawn from the same
regions of the table. That matters at ``max_rounds > 1``: a second round shown
the identical 32 rows has no new evidence to reason from.

Imputation here is for the clustering only. The values shown in the prompt are
always the real ones, missing included - a NaN in a lab result is information
about the patient, and hiding it from the proposer would misrepresent the table.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = ["build_shot_batches", "balanced_assignment"]


def _encode(
    frame: pd.DataFrame,
    continuous: Sequence[str],
    categorical: Sequence[str],
) -> np.ndarray:
    """
    One numeric matrix for clustering: standardised numbers, one-hot codes.

    Written out rather than delegated to a ColumnTransformer because the input
    can be missing anything: KMeans cannot take NaN, and autofe - unlike the
    benchmark this came from - does not impute its tables. Continuous columns
    are filled with their median and categorical ones get an explicit "missing"
    level, so a column that is 99% absent contributes "absent" as a real
    category instead of dropping its rows out of the clustering.
    """
    blocks: list[np.ndarray] = []

    if len(continuous):
        values = frame[list(continuous)].apply(pd.to_numeric, errors="coerce")
        values = values.fillna(values.median())
        # A column with no observed value at all medians to NaN; it carries no
        # information for clustering, so it becomes a constant zero column.
        matrix = values.to_numpy(dtype=float)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        centre = matrix.mean(axis=0)
        spread = matrix.std(axis=0)
        spread[spread == 0] = 1.0            # a constant column scales to zero
        blocks.append((matrix - centre) / spread)

    for column in categorical:
        codes = frame[column]
        # NaN becomes its own level rather than being dropped or filled.
        levels = pd.Index(sorted(codes.dropna().unique().tolist(), key=str))
        indicators = np.zeros((len(frame), len(levels) + 1), dtype=float)
        position = {level: i for i, level in enumerate(levels)}
        for row, value in enumerate(codes.tolist()):
            indicators[row, position.get(value, len(levels))] = 1.0
        blocks.append(indicators)

    if not blocks:
        raise ValueError("no columns to cluster on")
    return np.hstack(blocks)


def balanced_assignment(distances: np.ndarray) -> np.ndarray:
    """
    Assign rows to clusters keeping cluster sizes as equal as possible.

    Repeatedly commits the row with the most to lose: the one whose gap between
    its best and second-best still-open cluster ("regret") is largest, breaking
    ties by smaller distance and then by lower index.

    Vectorised over the unassigned set; equivalent to the row-by-row loop this
    was ported from but without its quadratic Python overhead.
    """
    n_samples, n_clusters = distances.shape
    base_size, remainder = divmod(n_samples, n_clusters)
    capacities = np.full(n_clusters, base_size, dtype=int)
    capacities[:remainder] += 1

    labels = np.full(n_samples, -1, dtype=int)
    remaining = capacities.copy()
    unassigned = np.arange(n_samples)

    while unassigned.size:
        available = np.flatnonzero(remaining > 0)
        block = distances[np.ix_(unassigned, available)]

        if available.size > 1:
            # The two smallest per row, in order.
            part = np.argpartition(block, 1, axis=1)[:, :2]
            rows = np.arange(block.shape[0])[:, None]
            two = block[rows, part]
            swap = two[:, 0] > two[:, 1]
            part[swap] = part[swap][:, ::-1]
            best_local = part[:, 0]
            preferred = block[rows[:, 0], best_local]
            second = block[rows[:, 0], part[:, 1]]
            regret = second - preferred
        else:
            best_local = np.zeros(block.shape[0], dtype=int)
            preferred = block[:, 0]
            regret = np.full(block.shape[0], np.inf)

        # max regret, then min distance, then min row index
        winner = np.lexsort((unassigned, preferred, -regret))[0]
        labels[unassigned[winner]] = available[best_local[winner]]
        remaining[available[best_local[winner]]] -= 1
        unassigned = np.delete(unassigned, winner)

    return labels


def _cluster_one_class(
    matrix: np.ndarray,
    n_clusters: int,
    seed: int,
    balance: bool,
) -> list[list[int]]:
    """Cluster one class and order each cluster's rows from its centre outward."""
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    distances = kmeans.fit(matrix).transform(matrix)
    # Equal sizes trade coverage for rotation depth; only worth it when there
    # is more than one batch to keep disjoint.
    labels = balanced_assignment(distances) if balance else distances.argmin(axis=1)

    orders: list[list[int]] = []
    for cluster in range(n_clusters):
        members = np.flatnonzero(labels == cluster)
        if members.size == 0:
            continue
        centre = matrix[members].mean(axis=0)
        distance = np.linalg.norm(matrix[members] - centre, axis=1)
        orders.append(members[np.argsort(distance)].tolist())
    return orders


def build_shot_batches(
    sample: pd.DataFrame,
    target: str,
    *,
    columns: Sequence[str],
    categorical: Sequence[str] = (),
    shots: int = 32,
    batches: int = 1,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> list[pd.DataFrame]:
    """
    Class-aware, cluster-representative example rows, one frame per batch.

    Degrades rather than raising: a class with fewer rows than the cluster
    budget gets as many clusters as it has rows, and a request for more batches
    than the smallest cluster can fill starts reusing rows. Both are logged:
    a shot budget that does not divide neatly should cost a warning, not the
    build.
    """
    log = logger or logging.getLogger(__name__)
    columns = [c for c in columns if c in sample.columns]
    categorical = [c for c in categorical if c in columns]
    continuous = [c for c in columns if c not in set(categorical)]

    classes = sorted(sample[target].dropna().unique().tolist(), key=str)
    if not classes or shots < 1:
        return [sample.head(shots).reset_index(drop=True)] * max(1, batches)

    per_class_budget = max(1, shots // len(classes))
    matrix = _encode(sample[columns], continuous, categorical)

    # Positions into `matrix`, which is row-aligned with `sample`.
    orders_by_class: list[tuple[np.ndarray, list[list[int]]]] = []
    for value in classes:
        rows = np.flatnonzero((sample[target] == value).to_numpy())
        n_clusters = min(per_class_budget, len(rows))
        if n_clusters < 1:
            continue
        if n_clusters < per_class_budget:
            log.warning(
                "shots: class %r has only %d row(s) in the sample, so it "
                "contributes %d example(s) instead of %d",
                value, len(rows), n_clusters, per_class_budget,
            )
        orders = _cluster_one_class(matrix[rows], n_clusters, seed,
                                    balance=batches > 1)
        orders_by_class.append((rows, orders))

    smallest = min((len(o) for _, orders in orders_by_class for o in orders),
                   default=0)
    if 0 < smallest < batches:
        log.info(
            "shots: the smallest cluster holds %d row(s) but %d batch(es) were "
            "requested, so some rows repeat across rounds", smallest, batches,
        )

    built: list[pd.DataFrame] = []
    for batch in range(max(1, batches)):
        picked: list[int] = []
        for rows, orders in orders_by_class:
            for order in orders:
                # Batch b takes the b-th closest row of each cluster, so
                # batches stay disjoint until a cluster has to cycle.
                picked.append(int(rows[order[batch % len(order)]]))
        built.append(sample.iloc[picked].reset_index(drop=True))

    total = len(built[0]) if built else 0
    log.info(
        "shots: %d example row(s) per batch x %d batch(es), one per cluster "
        "across %d class(es) (%d requested)",
        total, len(built), len(orders_by_class), shots,
    )
    return built
