"""Choose the example rows the proposer is shown - once, in the prepare step.

Each dataset's prepare notebook runs :func:`build_shot_batches` on its train
split and saves those rows with a batch column; a discovery run reads that file back
through ``discovery.few_shot_path`` and shows batch r in round r.

The rows picked here are printed as the example rows of the prompt, so they are the only concrete data a language model ever sees of the
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
from typing import Sequence

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import pairwise_distances


def _encode_fast(
    frame: pd.DataFrame,
    continuous: Sequence[str],
    categorical: Sequence[str],
):
    """Efficient clustering encoding."""

    blocks = []

    # -----------------------------
    # Numeric features
    # -----------------------------
    if continuous:
        X_num = (
            frame[list(continuous)]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float32)
        )

        # Median imputation.
        imputer = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )
        X_num = imputer.fit_transform(X_num).astype(np.float32)

        # Protect against extreme values.
        low = np.nanpercentile(X_num, 0.5, axis=0)
        high = np.nanpercentile(X_num, 99.5, axis=0)

        X_num = np.clip(X_num, low, high)

        # Standardize.
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num).astype(np.float32)

        # Final safety check.
        X_num = np.nan_to_num(
            X_num,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        blocks.append(X_num)

    # -----------------------------
    # Categorical features
    # -----------------------------
    if categorical:
        X_cat = frame[list(categorical)].copy()

        # IMPORTANT:
        # Mixed float/string values were causing your
        # OneHotEncoder TypeError.
        X_cat = X_cat.astype("string")
        X_cat = X_cat.fillna("__MISSING__")

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )

        X_cat = encoder.fit_transform(X_cat)

        blocks.append(X_cat)

    if not blocks:
        raise ValueError("No columns to cluster on")

    # -----------------------------
    # Combine
    # -----------------------------
    if len(blocks) == 1:
        return blocks[0]

    from scipy.sparse import csr_matrix, hstack

    sparse_blocks = []

    for block in blocks:
        if isinstance(block, np.ndarray):
            sparse_blocks.append(csr_matrix(block))
        else:
            sparse_blocks.append(block)

    return hstack(sparse_blocks, format="csr")


def _get_cluster_orders(
    matrix,
    n_clusters: int,
    seed: int,
    batches: int,
    *,
    kmeans_batch_size: int = 4096,
    distance_chunk_size: int = 50_000,
):
    """
    Cluster the rows and return up to `batches` representative
    rows per cluster, ordered from closest to furthest from
    the KMeans cluster center.
    """

    n_rows = matrix.shape[0]

    if n_rows == 0:
        return []

    if n_rows <= n_clusters:
        return [
            [i]
            for i in range(n_rows)
        ]

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=3,
        batch_size=min(kmeans_batch_size, n_rows),
        max_iter=100,
        max_no_improvement=10,
        reassignment_ratio=0.01,
    )

    kmeans.fit(matrix)

    centers = kmeans.cluster_centers_

    # For each cluster we retain only the closest `batches`
    # rows. We NEVER build a full N x K matrix.
    candidates = [
        []
        for _ in range(n_clusters)
    ]

    for start in range(0, n_rows, distance_chunk_size):
        stop = min(
            start + distance_chunk_size,
            n_rows,
        )

        chunk = matrix[start:stop]

        distances = pairwise_distances(
            chunk,
            centers,
            metric="euclidean",
            squared=True,
        )

        labels = distances.argmin(axis=1)

        for cluster in range(n_clusters):
            local_rows = np.flatnonzero(
                labels == cluster
            )

            if local_rows.size == 0:
                continue

            local_distances = distances[
                local_rows,
                cluster,
            ]

            keep = min(
                batches,
                local_rows.size,
            )

            if local_rows.size > keep:
                part = np.argpartition(
                    local_distances,
                    keep - 1,
                )[:keep]

                local_rows = local_rows[part]
                local_distances = local_distances[part]

            candidates[cluster].extend(
                (
                    float(distance),
                    int(start + row),
                )
                for row, distance
                in zip(local_rows, local_distances)
            )

    # Keep the globally closest rows for every cluster.
    orders = []

    for cluster_candidates in candidates:

        cluster_candidates.sort(
            key=lambda x: (x[0], x[1])
        )

        cluster_candidates = cluster_candidates[
            :batches
        ]

        orders.append(
            [
                row_index
                for _, row_index
                in cluster_candidates
            ]
        )

    return orders


def build_shot_batches_fast(
    sample: pd.DataFrame,
    target: str,
    *,
    columns: Sequence[str],
    categorical: Sequence[str] = (),
    shots: int = 32,
    batches: int = 1,
    seed: int = 42,
    logger: logging.Logger | None = None,
    max_cluster_rows_per_class: int = 50_000,
):
    """
    Fast class-aware representative-shot selection.

    For 2 classes and shots=32:
        16 clusters per class
        32 rows per batch

    Only up to `max_cluster_rows_per_class` rows from each
    class are used for clustering, which avoids running KMeans
    over millions of rows.
    """

    log = logger or logging.getLogger(__name__)

    columns = [
        c for c in columns
        if c in sample.columns
    ]

    categorical = [
        c for c in categorical
        if c in columns
    ]

    continuous = [
        c for c in columns
        if c not in set(categorical)
    ]

    classes = sorted(
        sample[target]
        .dropna()
        .unique()
        .tolist(),
        key=str,
    )

    if not classes or shots < 1:
        return [
            sample.head(shots)
            .reset_index(drop=True)
        ] * max(1, batches)

    # --------------------------------------
    # Number of clusters per class
    # --------------------------------------
    per_class_budget = max(
        1,
        shots // len(classes),
    )

    log.info(
        "shots=%d batches=%d classes=%d -> %d clusters/class",
        shots,
        batches,
        len(classes),
        per_class_budget,
    )

    # --------------------------------------
    # IMPORTANT:
    # Don't encode all 2M rows.
    #
    # Take a balanced clustering sample.
    # --------------------------------------
    cluster_parts = []

    for class_number, value in enumerate(classes):

        class_rows = sample[
            sample[target] == value
        ]

        n_take = min(
            len(class_rows),
            max_cluster_rows_per_class,
        )

        if n_take == 0:
            continue

        # Deterministic class-specific sampling.
        class_sample = class_rows.sample(
            n=n_take,
            random_state=seed + class_number,
        )

        cluster_parts.append(class_sample)

        log.info(
            "class=%r: %d rows available, %d used for clustering",
            value,
            len(class_rows),
            n_take,
        )

    if not cluster_parts:
        return [
            sample.head(shots)
            .reset_index(drop=True)
        ] * max(1, batches)

    cluster_sample = pd.concat(
        cluster_parts,
        ignore_index=True,
    )

    log.info(
        "clustering sample: %d rows",
        len(cluster_sample),
    )

    # --------------------------------------
    # Encode ONLY the smaller clustering set
    # --------------------------------------
    matrix = _encode_fast(
        cluster_sample[columns],
        continuous,
        categorical,
    )

    # Safety check
    if hasattr(matrix, "data"):
        if not np.isfinite(matrix.data).all():
            raise ValueError(
                "Clustering matrix contains non-finite sparse values"
            )
    else:
        if not np.isfinite(matrix).all():
            raise ValueError(
                "Clustering matrix contains non-finite values"
            )

    # --------------------------------------
    # Cluster each class independently
    # --------------------------------------
    orders_by_class = []

    for class_number, value in enumerate(classes):

        rows = np.flatnonzero(
            (
                cluster_sample[target] == value
            ).to_numpy()
        )

        n_clusters = min(
            per_class_budget,
            len(rows),
        )

        if n_clusters == 0:
            continue

        log.info(
            "class=%r: clustering %d rows into %d clusters",
            value,
            len(rows),
            n_clusters,
        )

        class_matrix = matrix[rows]

        orders = _get_cluster_orders(
            class_matrix,
            n_clusters=n_clusters,
            seed=seed + 1000 + class_number,
            batches=batches,
        )

        orders_by_class.append(
            (rows, orders)
        )

    # --------------------------------------
    # Build batches
    # --------------------------------------
    built = []

    for batch in range(max(1, batches)):

        picked = []

        for rows, orders in orders_by_class:

            for order in orders:

                if not order:
                    continue

                # Batch 0 = closest
                # Batch 1 = second closest
                # ...
                # Batch 9 = tenth closest
                picked.append(
                    rows[
                        order[
                            batch % len(order)
                        ]
                    ]
                )

        batch_frame = (
            cluster_sample
            .iloc[picked]
            .reset_index(drop=True)
        )

        built.append(batch_frame)

        log.info(
            "batch %d: %d rows",
            batch,
            len(batch_frame),
        )

    return built