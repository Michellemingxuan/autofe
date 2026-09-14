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

One table serves both ways of running the pipeline. With
``discovery.enabled: false`` it evaluates exactly the candidates the config
declares; with discovery on, an LLM proposes further candidates over the same
incumbent set and they are appended to that list, so both are judged at the same
gates in one run. The planted controls then double as a reference for what
"redundant" and "uninformative" look like on this table.

``configs/bankruptcy.yaml`` is the source of truth for everything the pipeline
consumes. This script owns only the part a config cannot express: how the raw
archive becomes a table, including the recipe for the two planted controls.

    python data/bankruptcy/prepare.py
    mllite -c configs/bankruptcy.yaml
    mllite -c configs/bankruptcy.yaml --set discovery.enabled=true
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import (  # noqa: E402
    fetch_archive,
    load_config_from,
    print_report,
    resolve_path,
    sanitize_columns,
    write_modeling_table,
)

URL = "https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip"
MEMBER = "data.csv"
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
PLANTED = {
    "cand_dup_roa_c": "planted control: a near-copy of an incumbent ratio",
    "cand_noise": "planted control: pure noise, unrelated to the target",
}


def build(csv_path: Path, target: str, id_col: str, seed: int = 0
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw archive -> modeling table with readable names, as the config expects."""
    frame, mapping = sanitize_columns(pd.read_csv(csv_path))
    frame = frame.rename(columns={RAW_TARGET: target})

    missing = [c for c in CASH_FLOW_FEATURES if c not in frame.columns]
    if missing:
        raise KeyError(f"expected cash-flow columns not found after sanitizing: {missing}")

    # The two planted controls are constructed, not read - that recipe is code,
    # not configuration, so it stays here. They serve both scenarios: the
    # declared-batch run needs something known for its screens to catch, and a
    # discovery run needs a reference for what "redundant" and "uninformative"
    # look like on this table.
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
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the two planted controls")
    args = parser.parse_args()

    cfg = load_config_from(ROOT, args.config)
    if not cfg.data.path:
        print(f"  ! {args.config} has no data.path to write to", file=sys.stderr)
        return 1

    raw_dir = resolve_path(cfg.data.path, ROOT).parent / "raw" / "uci572"
    id_col = cfg.data.id_cols[0] if cfg.data.id_cols else "row_id"
    frame, mapping = build(fetch_archive(URL, raw_dir, MEMBER),
                           cfg.data.target, id_col, args.seed)

    try:
        report = write_modeling_table(
            cfg, frame, root=ROOT, id_col=id_col, mapping=mapping,
            # The archive's headers describe every real column; the planted two
            # are ours, so they need descriptions of their own.
            extra_descriptions=PLANTED,
        )
    except (KeyError, ValueError) as error:
        print(f"  ! {error}", file=sys.stderr)
        return 1

    print_report(report, cfg, args.config, ROOT, id_col)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
