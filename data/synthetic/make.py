"""Generate a synthetic table for end-to-end smoke runs.

Also imported by the test suite for ``make_frame``. ``.gitignore`` keeps ``*.py``
under ``data/`` tracked precisely so this survives a fresh clone.

Builds incumbent features (``old_*``) and candidate features (``new_*``) with a
known ground truth:

    new_signal_a / new_signal_b   genuinely add signal beyond the incumbents
    new_dup_old0                  a near-copy of old_0 (should fail redundancy)
    new_noise                     pure noise (should fail the signal screen)
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out",
                        default=str(Path(__file__).resolve().parent / "modeling.parquet"))
    args = parser.parse_args()

    frame = make_frame(args.rows, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        frame.to_parquet(out, index=False)
    else:
        frame.to_csv(out, index=False)
    print(f"wrote {len(frame):,} rows x {frame.shape[1]} cols -> {out}")


if __name__ == "__main__":
    main()
