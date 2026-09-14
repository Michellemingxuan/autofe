"""Prepare the UCI Myocardial Infarction Complications dataset as a demo table.

Source: https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications
1,700 patients admitted with myocardial infarction, 111 clinical features,
predicting chronic heart failure (ZSN) as a complication - 23.2% of patients.

Three things make this the hardest of the three demo tables, and each one exists
here on purpose:

* **Missingness is real.** 7.6% of all cells are missing, and it is wildly
  uneven: the serum CPK result is absent for 99.8% of patients and the
  emergency-team blood pressures for 63%. Nothing is imputed here - the table
  carries NaN and XGBoost handles it natively - so the pipeline's missingness
  gate and its optional missing-indicator features both do real work, unlike on
  bankruptcy where nothing is ever missing.

* **The file has no header.** ``MI.data`` is 124 bare columns, so the column
  names *are* the schema and getting them wrong is silent: every value stays a
  valid number under the wrong name. The names come from the archive's own
  published variable list, which is downloaded and cached beside the data, and
  the structure is checked against it rather than assumed.

* **Most columns are coded categories, not measurements.** 98 of the 110
  features are ordinal or binary codes - angina class, ECG rhythm findings,
  which drugs were given. Only 12 are real measurements. A proposer shown a
  range for an ECG finding will try to do arithmetic on it, so the config
  declares the 12 continuous columns and everything else is presented as levels.

The demo scenario: the incumbent model uses the history, examination, ECG and
treatment columns - everything recorded as part of routine admission. The
proposed candidate family is the **serum laboratory panel**: potassium, sodium,
AlAT, AsAT, CPK, white cell count and ESR, plus the two threshold flags derived
from them. A lab panel has to be ordered, costs money and comes back late, so
whether it adds anything over what admission already records is exactly the kind
of question this pipeline exists to answer.

That family also ships its own redundancy control, with nothing planted:
``gipo_k`` is hypokalemia, defined as ``k_blood < 4 mmol/L``, and ``giper_na``
is ``na_blood > 150 mmol/L``. Each is a thresholded copy of a continuous column
sitting next to it in the same batch, which is precisely what the redundancy
screen is meant to catch.

**Leakage.** The archive carries 12 target columns, all complications of the
same infarction, recorded concurrently. Eleven of them are dropped: predicting
heart failure from whether the patient also went into cardiogenic shock is not a
prediction. ``ZSN_A`` is dropped too - it is chronic heart failure *in the
anamnesis*, so a patient with ``ZSN_A > 0`` largely has the outcome already.

``configs/myocardial.yaml`` is the source of truth for everything the pipeline
consumes. This script owns only the part a config cannot express: how the raw
archive becomes a table, which columns are outcomes rather than features, and
where the column names come from.

    python data/myocardial/prepare.py
    mllite -c configs/myocardial.yaml
    mllite -c configs/myocardial.yaml --set discovery.enabled=true
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import (  # noqa: E402
    fetch_archive,
    load_config_from,
    print_report,
    resolve_path,
    sanitize,
    sanitize_columns,
    write_modeling_table,
)

DATA_URL = ("https://archive.ics.uci.edu/static/public/579/"
            "myocardial+infarction+complications.zip")
DATA_MEMBER = "MI.data"

# The published variable list: names, roles, types and descriptions, in file
# order. Cached beside the data, so a re-run does no network I/O at all.
VARS_URL = "https://archive.ics.uci.edu/api/dataset?id=579"
VARS_MEMBER = "variables.json"

SOURCE_TARGET = "ZSN"            # chronic heart failure, the one outcome modelled

# All 12 outcome columns. Every one is a complication of the same infarction,
# recorded at the same time, so the 11 not being modelled are not features.
TARGET_COLUMNS = [
    "FIBR_PREDS", "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", "A_V_BLOK",
    "OTEK_LANC", "RAZRIV", "DRESSLER", "ZSN", "REC_IM", "P_IM_STEN", "LET_IS",
]

# Chronic heart failure in the anamnesis: a patient with this already has the
# outcome, so it is dropped rather than being allowed to dominate the model.
LEAKY_FEATURES = ["ZSN_A"]

# "?" is this archive's missing marker. Nothing is imputed: the table keeps NaN
# and the pipeline decides what to do about it.
MISSING_MARKER = "?"


def read_variables(path: Path) -> list[dict]:
    """The published variable list, checked hard enough to trust as a schema."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    variables = (payload.get("data") or payload).get("variables") or []

    if len(variables) != 124:
        raise ValueError(f"expected 124 published variables, got {len(variables)}")
    if variables[0]["role"] != "ID":
        raise ValueError(f"expected the first variable to be the record id, "
                         f"got {variables[0]['name']!r}")
    tail = [v["name"] for v in variables[-12:]]
    if tail != TARGET_COLUMNS:
        # A reordering here would rename every column silently, so it stops the
        # build rather than producing a plausible-looking wrong table.
        raise ValueError(
            "the published variable order no longer ends with the 12 outcome "
            f"columns in the expected order.\n  expected: {TARGET_COLUMNS}\n"
            f"  got:      {tail}"
        )
    return variables


def clean_description(text: str) -> str:
    """One line, since a description is shown to a proposer on one line.

    The published descriptions embed the coding scheme over many lines
    ("0: none", "1: I FC", ...). That content is the most useful part - it is
    what tells a proposer a column is a grade and not a count - so it is kept
    and the line breaks are collapsed rather than the text being truncated.
    """
    return re.sub(r"\s+", " ", str(text)).strip()


def build(data_path: Path, vars_path: Path, target: str
          ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Raw archive -> modeling table with readable names, as the config expects."""
    variables = read_variables(vars_path)
    header = [v["name"] for v in variables]

    raw = pd.read_csv(data_path, header=None, na_values=MISSING_MARKER)
    if raw.shape[1] != len(header):
        raise ValueError(f"{data_path.name} has {raw.shape[1]} columns but the "
                         f"published list describes {len(header)}")
    raw.columns = header

    # Spot-check two columns whose real-world range is unmistakable. This is
    # what caught an off-by-one in the header during development: every value
    # stayed a valid number, just under the wrong name.
    for column, (low, high) in {"AGE": (18, 120), "NA_BLOOD": (100, 200)}.items():
        observed = raw[column].dropna()
        if len(observed) and not (observed.min() >= low and observed.max() <= high):
            raise ValueError(
                f"{column} holds values outside its clinical range "
                f"[{low}, {high}]: [{observed.min()}, {observed.max()}]. "
                "The column names are probably misaligned with the data."
            )

    dropped = [c for c in TARGET_COLUMNS if c != SOURCE_TARGET] + LEAKY_FEATURES
    frame = raw.drop(columns=dropped)
    frame, mapping = sanitize_columns(frame)
    frame = frame.rename(columns={sanitize(SOURCE_TARGET): target})
    mapping.loc[mapping["column"] == sanitize(SOURCE_TARGET), "column"] = target

    descriptions = {
        sanitize(v["name"]): clean_description(v["description"])
        + (f" Units: {v['units']}." if v.get("units") else "")
        for v in variables
    }
    descriptions = {k: v for k, v in descriptions.items() if k in frame.columns}
    return frame, mapping, descriptions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", default=str(ROOT / "configs" / "myocardial.yaml"),
                        help="the config this table is built for")
    args = parser.parse_args()

    cfg = load_config_from(ROOT, args.config)
    if not cfg.data.path:
        print(f"  ! {args.config} has no data.path to write to", file=sys.stderr)
        return 1

    raw_dir = resolve_path(cfg.data.path, ROOT).parent / "raw" / "uci579"
    id_col = cfg.data.id_cols[0] if cfg.data.id_cols else "id"

    try:
        frame, mapping, descriptions = build(
            fetch_archive(DATA_URL, raw_dir, DATA_MEMBER),
            fetch_archive(VARS_URL, raw_dir, VARS_MEMBER),
            cfg.data.target,
        )
        report = write_modeling_table(cfg, frame, root=ROOT, id_col=id_col,
                                      mapping=mapping, descriptions=descriptions)
    except (KeyError, ValueError) as error:
        print(f"  ! {error}", file=sys.stderr)
        return 1

    print_report(report, cfg, args.config, ROOT, id_col)

    # Missingness is the defining property of this table, so it is reported
    # rather than left for the pipeline's logs to mention first.
    features = [c for c in frame.columns if c not in {cfg.data.target, id_col}]
    rates = frame[features].isna().mean().sort_values(ascending=False)
    print(f"  missing: {frame[features].isna().to_numpy().mean():.1%} of feature cells, "
          f"{int((rates > 0).sum())} of {len(features)} columns affected")
    print("  most incomplete: " + ", ".join(
        f"{c} {r:.0%}" for c, r in rates.head(4).items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
