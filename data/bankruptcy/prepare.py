"""Prepare the UCI Taiwanese Bankruptcy Prediction dataset as a demo table.

Source: https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
6,819 companies (1999-2009), 95 financial ratios, 3.2% bankruptcy rate.

The demo scenario: a team already models with the profitability / leverage /
growth ratios (the incumbent set) and proposes adding the **cash-flow family**.
Does it earn its place?

Two controls are planted among the candidates so the selection screens have
something known to catch:

    cand_dup_roa_c   a near-copy of an incumbent feature -> redundancy screen
    cand_noise       pure noise                          -> signal screen

``configs/bankruptcy.yaml`` is the source of truth for everything the pipeline
consumes - where the table goes, what the target is called, the id column and the
list of candidate features. This script owns only the part a config cannot express:
how the raw archive becomes a table, including the recipe for the two planted
controls. It then checks that what it built matches what the config declares.

    python data/bankruptcy/prepare.py                        # uses configs/bankruptcy.yaml
    python data/bankruptcy/prepare.py -c other_config.yaml

Column names are sanitized to snake_case; ``column_mapping.csv`` beside the output
records the original names.
"""

from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
URL = "https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip"
RAW_TARGET = "bankrupt"          # what "Bankrupt?" sanitizes to
ANCHOR = "roa_c_before_interest_and_depreciation_before_interest"

# The candidate family under evaluation: everything cash-flow related.
CASH_FLOW_FEATURES = [
    "cash_flow_rate",
    "cash_flow_per_share",
    "cash_reinvestment_pct",
    "cash_total_assets",
    "cash_current_liability",
    "cash_turnover_rate",
    "cash_flow_to_sales",
    "cash_flow_to_total_assets",
    "cash_flow_to_liability",
    "cfo_to_assets",
    "cash_flow_to_equity",
]
PLANTED = ["cand_dup_roa_c", "cand_noise"]
NEW_FEATURES = CASH_FLOW_FEATURES + PLANTED


def sanitize(name: str) -> str:
    name = name.strip().lower()
    name = name.replace("%", " pct ").replace("&", " and ")
    name = re.sub(r"[^\w]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def download(dest: Path) -> Path:
    """Fetch and unzip the archive unless the CSV is already on disk."""
    csv_path = dest / "raw" / "uci572" / "data.csv"
    if csv_path.exists():
        return csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(csv_path.parent)
    return csv_path


def build(csv_path: Path, target: str, id_col: str, seed: int = 0
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw archive -> modeling table, named as the config expects."""
    raw = pd.read_csv(csv_path)
    mapping = pd.DataFrame({"original": raw.columns, "column": [sanitize(c) for c in raw.columns]})
    frame = raw.copy()
    frame.columns = mapping["column"].tolist()
    frame = frame.rename(columns={RAW_TARGET: target})

    missing = [c for c in CASH_FLOW_FEATURES if c not in frame.columns]
    if missing:
        raise KeyError(f"expected cash-flow columns not found after sanitizing: {missing}")

    # The two planted controls are constructed, not read - that recipe is code,
    # not configuration, so it stays here.
    rng = np.random.default_rng(seed)
    frame["cand_dup_roa_c"] = frame[ANCHOR] * (1 + rng.normal(scale=0.01, size=len(frame)))
    frame["cand_noise"] = rng.normal(size=len(frame))

    frame.insert(0, id_col, np.arange(len(frame)))
    return frame, mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", default=str(ROOT / "configs" / "bankruptcy.yaml"),
                        help="the config this table is built for")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from mllite import load_config

    cfg = load_config(args.config)
    if not cfg.data.path:
        print(f"  ! {args.config} has no data.path to write to", file=sys.stderr)
        return 1

    out = Path(cfg.data.path)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    id_col = cfg.data.id_cols[0] if cfg.data.id_cols else "row_id"
    frame, mapping = build(download(out.parent), cfg.data.target, id_col, args.seed)

    # what the config asks the pipeline to evaluate must actually be in the table
    declared = list(cfg.features.new)
    if cfg.features.new_prefix:
        declared += [c for c in frame.columns
                     if c.startswith(cfg.features.new_prefix) and c not in declared]
    absent = [c for c in declared if c not in frame.columns]
    if absent:
        print(f"  ! features.new names column(s) this script does not build: {absent}",
              file=sys.stderr)
        return 1
    if cfg.data.target not in frame.columns:
        print(f"  ! target {cfg.data.target!r} is not in the built table", file=sys.stderr)
        return 1

    frame.to_parquet(out, index=False)
    mapping.to_csv(out.parent / "column_mapping.csv", index=False)

    reserved = set(declared) | {cfg.data.target, id_col}
    base = [c for c in frame.columns if c not in reserved]
    print(f"wrote {len(frame):,} rows x {frame.shape[1]} cols -> {out}")
    print(f"  target {cfg.data.target!r}: {frame[cfg.data.target].mean():.2%} positive "
          f"({int(frame[cfg.data.target].sum())} of {len(frame):,})")
    print(f"  id column: {id_col!r}")
    print(f"  incumbent features: {len(base)}")
    print(f"  candidate features: {len(declared)} (declared in {Path(args.config).name})")
    print(f"\nready:  mllite -c {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
