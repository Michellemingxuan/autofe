"""A synthetic table with a known ground truth, for end-to-end tests.

    old_0 .. old_11               incumbent features
    new_signal_a / new_signal_b   genuinely add signal beyond the incumbents
    new_dup_old0                  a near-copy of old_0 (should fail redundancy)
    new_noise                     pure noise (should fail the signal screen)
    y                             regression target, clipped at 0
    row_id                        id
    split                         train / valid / test label, about 60 / 20 / 20

About 8% of ``new_signal_b`` is coded -9999, to exercise sentinel cleaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_frame(n_rows: int = 60_000, n_old: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    old = rng.normal(size=(n_rows, n_old))
    signal_a = rng.normal(size=n_rows)
    signal_b = rng.gamma(2.0, 1.0, size=n_rows) - 2.0

    latent = (
        0.9 * old[:, 0] + 0.6 * old[:, 1] - 0.4 * old[:, 2]
        + 0.5 * np.sin(old[:, 3]) + 0.35 * old[:, 4] * old[:, 5]
        + 0.8 * signal_a + 0.5 * signal_b + 0.4 * signal_a * old[:, 0]
    )
    y = np.clip(latent + rng.normal(scale=1.0, size=n_rows), 0, None)

    frame = pd.DataFrame({f"old_{i}": old[:, i] for i in range(n_old)})
    frame["new_signal_a"] = signal_a
    frame["new_signal_b"] = signal_b
    frame["new_dup_old0"] = old[:, 0] + rng.normal(scale=0.05, size=n_rows)
    frame["new_noise"] = rng.normal(size=n_rows)

    # Sentinel-coded missings, to exercise the cleaning path.
    mask = rng.random(n_rows) < 0.08
    frame.loc[mask, "new_signal_b"] = -9999

    frame["y"] = y
    frame["row_id"] = np.arange(n_rows)
    draw = rng.random(n_rows)
    frame["split"] = np.where(draw < 0.2, "test", np.where(draw < 0.4, "valid", "train"))
    return frame


def split_by_column(frame: pd.DataFrame, column: str = "split") -> dict[str, pd.DataFrame]:
    """The train / valid / test frames a prepare step would have written."""
    return {name: frame[frame[column] == name].drop(columns=column).reset_index(drop=True)
            for name in ("train", "valid", "test")}
