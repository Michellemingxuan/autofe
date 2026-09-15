"""Choosing the example rows the proposer is shown.

These rows are the only concrete data a language model sees of the table, and
two properties decide whether they are usable: every class must appear, and the
rows must describe different regions of the data rather than crowding into
wherever it happens to be dense.

The coverage test is the one that would catch a broken port. Class balance can
be satisfied by a stratified random draw, so it alone does not show the
clustering is doing anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.shots import _encode, balanced_assignment, build_shot_batches


@pytest.fixture
def clustered_frame():
    """Three well-separated blobs per class, so coverage is checkable by eye."""
    rng = np.random.default_rng(0)
    rows = []
    for label, centres in [(0, [(0, 0), (10, 0), (0, 10)]),
                           (1, [(20, 20), (30, 20), (20, 30)])]:
        for cx, cy in centres:
            # Deliberately lopsided: 60 rows in the first blob, 6 in the last.
            n = 60 if (cx, cy) in [(0, 0), (20, 20)] else 6
            for _ in range(n):
                rows.append({"y": label,
                             "a": cx + rng.normal(scale=0.3),
                             "b": cy + rng.normal(scale=0.3)})
    return pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)


def _coverage(matrix, chosen):
    """Mean distance from every row to its nearest chosen row. Lower is better."""
    d = np.linalg.norm(matrix[:, None, :] - matrix[np.asarray(chosen)][None, :, :],
                       axis=-1)
    return float(d.min(axis=1).mean())


# --------------------------------------------------------------------------- #
# class coverage
# --------------------------------------------------------------------------- #
def test_every_class_appears_even_at_a_rare_base_rate():
    """
    The failure this exists for: a uniform draw sees one class only.

    At a 2% positive rate a 32-row random sample usually contains zero
    positives, and the proposer is then asked to invent discriminating features
    having seen no contrast at all.
    """
    frame = pd.DataFrame({
        "y": [1] * 20 + [0] * 980,
        "x": np.arange(1000, dtype=float),
    })
    batch = build_shot_batches(frame, "y", columns=["x"], shots=32)[0]
    assert batch["y"].value_counts().to_dict() == {0: 16, 1: 16}


def test_the_shots_budget_is_split_evenly_across_classes(clustered_frame):
    batch = build_shot_batches(clustered_frame, "y", columns=["a", "b"], shots=12)[0]
    assert batch["y"].value_counts().to_dict() == {0: 6, 1: 6}


def test_a_class_smaller_than_its_budget_degrades_instead_of_raising(caplog):
    """A pipeline should not die because a shot budget divided awkwardly."""
    frame = pd.DataFrame({"y": [1, 1] + [0] * 50,
                          "x": np.arange(52, dtype=float)})
    with caplog.at_level("WARNING"):
        batch = build_shot_batches(frame, "y", columns=["x"], shots=20)[0]
    assert batch["y"].value_counts().to_dict() == {0: 10, 1: 2}
    assert any("only 2 row" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# spatial coverage - the part random sampling cannot do
# --------------------------------------------------------------------------- #
def test_clustered_rows_cover_the_data_better_than_random_draws(clustered_frame):
    columns = ["a", "b"]
    matrix = _encode(clustered_frame[columns], columns, [])
    batch = build_shot_batches(clustered_frame, "y", columns=columns, shots=6)[0]

    chosen = [clustered_frame.index[(clustered_frame[columns] == row).all(axis=1)][0]
              for _, row in batch[columns].iterrows()]
    clustered = _coverage(matrix, chosen)

    rng = np.random.default_rng(7)
    random_draws = [_coverage(matrix, rng.choice(len(clustered_frame), size=6,
                                                 replace=False))
                    for _ in range(30)]
    # Beats not just the average random draw but the best of thirty.
    assert clustered < min(random_draws)


def test_a_small_blob_is_represented_as_well_as_a_large_one(clustered_frame):
    """
    A sparse region gets a representative, not just the dense one.

    The fixture is deliberately lopsided - 60-row and 6-row blobs. This is the
    case that forcing equal cluster sizes gets wrong: equal counts would put 24
    rows in each cluster, cutting across the blobs so the dense one takes three
    representatives and the two sparse ones get none.
    """
    batch = build_shot_batches(clustered_frame, "y", columns=["a", "b"], shots=6)[0]
    # One row near each of the six blob centres.
    centres = [(0, 0), (10, 0), (0, 10), (20, 20), (30, 20), (20, 30)]
    for cx, cy in centres:
        near = ((batch["a"] - cx).abs() < 2) & ((batch["b"] - cy).abs() < 2)
        assert near.sum() == 1, f"blob at {(cx, cy)} got {near.sum()} rows"


# --------------------------------------------------------------------------- #
# rotation across rounds
# --------------------------------------------------------------------------- #
def test_successive_batches_are_disjoint(clustered_frame):
    """A second round shown the same rows has no new evidence to reason from.

    This is what equal cluster sizes are for: asking for several batches
    switches the balanced assignment on so no cluster is too small to supply a
    distinct row per round.
    """
    batches = build_shot_batches(clustered_frame, "y", columns=["a", "b"],
                                 shots=6, batches=3)
    seen = [set(map(tuple, b[["a", "b"]].to_numpy())) for b in batches]
    assert not (seen[0] & seen[1])
    assert not (seen[1] & seen[2])


def test_batches_reuse_rows_only_once_a_cluster_runs_out(caplog):
    frame = pd.DataFrame({"y": [0, 0, 1, 1], "x": [0.0, 0.1, 5.0, 5.1]})
    with caplog.at_level("INFO"):
        batches = build_shot_batches(frame, "y", columns=["x"], shots=4, batches=5)
    assert len(batches) == 5
    assert any("repeat across rounds" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# missing values - autofe keeps them, unlike the benchmark this came from
# --------------------------------------------------------------------------- #
def test_missing_values_do_not_break_the_clustering():
    """The ported original raised on any NaN; myocardial is 8.5% missing."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "y": [0] * 50 + [1] * 50,
        "measured": rng.normal(size=100),
        "mostly_absent": [np.nan] * 98 + [1.0, 2.0],
        "coded": rng.integers(0, 3, size=100).astype(float),
    })
    frame.loc[frame.index[:20], "measured"] = np.nan
    frame.loc[frame.index[:10], "coded"] = np.nan

    batch = build_shot_batches(frame, "y", columns=["measured", "mostly_absent", "coded"],
                               categorical=["coded"], shots=10)[0]
    assert len(batch) == 10
    assert batch["y"].value_counts().to_dict() == {0: 5, 1: 5}


def test_the_rows_shown_keep_their_real_values_including_missing():
    """
    Imputation is for the clustering only.

    A NaN in a lab result is information about the patient. Filling it in the
    shown rows would tell the proposer the table is complete when it is not.
    """
    frame = pd.DataFrame({"y": [0] * 10 + [1] * 10,
                          "x": [np.nan] * 5 + list(np.arange(15.0))})
    batch = build_shot_batches(frame, "y", columns=["x"], shots=20)[0]
    assert batch["x"].isna().any()


def test_an_all_missing_column_does_not_poison_the_matrix():
    frame = pd.DataFrame({"y": [0] * 10 + [1] * 10,
                          "good": np.arange(20.0),
                          "empty": [np.nan] * 20})
    matrix = _encode(frame[["good", "empty"]], ["good", "empty"], [])
    assert np.isfinite(matrix).all()


# --------------------------------------------------------------------------- #
# the balanced assignment itself
# --------------------------------------------------------------------------- #
def test_balanced_assignment_gives_every_cluster_the_same_size():
    rng = np.random.default_rng(0)
    labels = balanced_assignment(rng.random((30, 3)))
    assert sorted(np.bincount(labels).tolist()) == [10, 10, 10]


def test_balanced_assignment_spreads_a_remainder_by_one():
    rng = np.random.default_rng(0)
    labels = balanced_assignment(rng.random((10, 3)))
    assert sorted(np.bincount(labels).tolist()) == [3, 3, 4]


def test_balanced_assignment_prefers_the_nearest_cluster_when_it_can():
    """Two rows, two clusters, each row clearly closest to a different one."""
    distances = np.array([[0.0, 9.0], [9.0, 0.0]])
    assert balanced_assignment(distances).tolist() == [0, 1]
