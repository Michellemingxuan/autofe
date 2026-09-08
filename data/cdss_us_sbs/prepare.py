"""Shape the cdss_us_sbs train / valid / test extracts into modeling tables.

The split is done upstream, so this never splits anything: three files in, three
files out. What it does is shape them and *verify the split it was handed*, which
is the only place that verification happens - the pipeline itself cannot tell a
leaked split from a clean one.

``configs/cdss_us_sbs.yaml`` is the single source of truth. The target, id columns,
sentinel codes, candidate features and the output paths are all read from it, so
they are declared once and never repeated on the command line. The only arguments
are the raw inputs, because those are upstream of the pipeline and the config has
no opinion about them.

    # 1. first look - no config needed, writes nothing
    python data/cdss_us_sbs/prepare.py --inspect \
        --train raw_train.parquet --valid raw_valid.parquet --test raw_test.parquet

    # 2. fill in configs/cdss_us_sbs.yaml from what that prints, then
    python data/cdss_us_sbs/prepare.py -c configs/cdss_us_sbs.yaml \
        --train raw_train.parquet --valid raw_valid.parquet --test raw_test.parquet

Checks, each covering a failure that surfaces late or not at all:

  * an id in two splits            -> leakage; nothing downstream detects it
  * columns differing across files -> the pipeline raises during stage 0
  * target missing or not binary   -> silently wrong metrics
  * sentinel codes                 -> must be declared in data.missing_values
  * non-numeric feature columns    -> silently excluded from the model
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "valid", "test")


# --------------------------------------------------------------------------- #
# Column names
# --------------------------------------------------------------------------- #
def sanitize(name: str) -> str:
    """Lower-case snake_case, so names survive YAML, CSV and XGBoost alike."""
    name = name.strip().lower()
    name = name.replace("%", " pct ").replace("&", " and ")
    name = re.sub(r"[^\w]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def sanitize_frames(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    reference = frames["train"].columns
    mapping = pd.DataFrame({"original": reference, "column": [sanitize(c) for c in reference]})
    duplicated = mapping["column"][mapping["column"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"sanitizing collapses distinct columns onto one name: {duplicated}")
    for frame in frames.values():
        frame.columns = [sanitize(c) for c in frame.columns]
    return mapping


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_columns_align(frames: Dict[str, pd.DataFrame]) -> None:
    reference = list(frames["train"].columns)
    for name, frame in frames.items():
        missing = [c for c in reference if c not in frame.columns]
        if missing:
            raise ValueError(f"{name!r} is missing column(s) present in train: {missing[:10]}")
        extra = [c for c in frame.columns if c not in reference]
        if extra:
            print(f"  ! {name!r} has {len(extra)} column(s) not in train; dropping {extra[:5]}")
            frames[name] = frame[reference]


def check_no_id_overlap(frames: Dict[str, pd.DataFrame], id_cols: List[str]) -> None:
    """The same unit in two splits leaks the outcome. Nothing downstream catches it."""
    if not id_cols:
        print("  ! data.id_cols is empty, so the split could not be checked for leakage")
        return
    keys = {name: set(map(tuple, frame[id_cols].astype(str).to_numpy()))
            for name, frame in frames.items()}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        shared = keys[left] & keys[right]
        if shared:
            raise ValueError(
                f"{len(shared):,} id(s) appear in both {left!r} and {right!r} "
                f"(e.g. {sorted(shared)[:3]}). That is leakage - fix the split upstream.")
    print(f"  id overlap across splits: none (checked on {id_cols})")


def describe_target(frames: Dict[str, pd.DataFrame], target: str) -> None:
    print(f"  target {target!r}")
    for name, frame in frames.items():
        values = frame[target]
        if values.isna().any():
            print(f"    ! {name}: {int(values.isna().sum()):,} null target rows - drop or impute")
        unique = np.unique(values.dropna())
        shape = "binary" if set(unique) <= {0, 1} else f"{len(unique)} distinct values"
        print(f"    {name:<6} n={len(frame):>9,}  mean={values.mean():.4f}  ({shape})")


def report_columns(frames: Dict[str, pd.DataFrame], reserved: List[str],
                   sentinels: List[float]) -> List[str]:
    train = frames["train"]
    features = [c for c in train.columns if c not in reserved]
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
    if non_numeric:
        print(f"  ! {len(non_numeric)} non-numeric column(s) - the pipeline will not model these; "
              f"encode them or list them under features.exclude: {non_numeric[:8]}")
    numeric = [c for c in features if c not in non_numeric]

    hits = {int(s): int((train[numeric] == s).sum().sum()) for s in sentinels}
    hits = {k: v for k, v in hits.items() if v}
    if hits:
        print(f"  sentinel codes in train: {hits} -> keep them in data.missing_values")
    print(f"  {len(numeric)} numeric feature column(s), "
          f"{int(train[numeric].isna().sum().sum()):,} NaN cells in train")
    return numeric


# --------------------------------------------------------------------------- #
def read_raw(paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    frames = {}
    print("reading")
    for name in SPLITS:
        frame = pd.read_parquet(paths[name])
        print(f"  {name:<6} {frame.shape[0]:>9,} rows x {frame.shape[1]:>4} cols   {paths[name]}")
        frames[name] = frame
    return frames


def print_config_block(frames: Dict[str, pd.DataFrame], numeric: List[str]) -> None:
    """What to paste into configs/cdss_us_sbs.yaml after a first look."""
    print("\nfill these into configs/cdss_us_sbs.yaml:\n")
    print("data:")
    print("  target: <one of the columns below>")
    print("  id_cols: []          # the account/date keys - needed for the leakage check")
    print("  missing_values: [-9999]")
    print("\nfeatures:")
    print("  base: []             # empty = every other numeric column")
    print("  new: []              # the candidates, or set new_prefix")
    print(f"\ncolumns seen ({len(numeric)} numeric):")
    for i in range(0, min(len(numeric), 40), 4):
        print("  " + "  ".join(f"{c:<28}" for c in numeric[i:i + 4]))
    if len(numeric) > 40:
        print(f"  ... and {len(numeric) - 40} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", help="pipeline config; everything but the raw paths comes from it")
    parser.add_argument("--train", required=True, help="raw training extract")
    parser.add_argument("--valid", required=True, help="raw validation extract")
    parser.add_argument("--test", required=True, help="raw test extract")
    parser.add_argument("--inspect", action="store_true",
                        help="report what is in the extracts and write nothing")
    parser.add_argument("--no-sanitize", action="store_true", help="keep column names as they arrive")
    args = parser.parse_args()

    if not args.inspect and not args.config:
        parser.error("-c/--config is required unless --inspect is given")

    frames = read_raw({"train": args.train, "valid": args.valid, "test": args.test})

    if not args.no_sanitize:
        try:
            mapping = sanitize_frames(frames)
        except ValueError as exc:
            print(f"  ! {exc}", file=sys.stderr)
            return 1
        changed = mapping[mapping["original"] != mapping["column"]]
    else:
        mapping, changed = None, []

    print("\nchecking")
    try:
        check_columns_align(frames)
    except ValueError as exc:
        print(f"  ! {exc}", file=sys.stderr)
        return 1

    if args.inspect:
        numeric = report_columns(frames, [], [-9999])
        print_config_block(frames, numeric)
        return 0

    sys.path.insert(0, str(ROOT / "src"))
    from mllite import load_config

    cfg = load_config(args.config)
    if not cfg.data.paths:
        print("  ! this config has no data.paths; add the three output paths", file=sys.stderr)
        return 1
    missing_splits = [s for s in SPLITS if s not in cfg.data.paths]
    if missing_splits:
        print(f"  ! data.paths is missing {missing_splits}", file=sys.stderr)
        return 1

    target, id_cols = cfg.data.target, list(cfg.data.id_cols)
    declared = [target] + id_cols
    absent = [c for c in declared if c not in frames["train"].columns]
    if absent:
        print(f"  ! column(s) named in the config are not in the extracts: {absent}\n"
              f"    run with --inspect to see the real column names", file=sys.stderr)
        return 1

    try:
        check_no_id_overlap(frames, id_cols)
    except ValueError as exc:
        print(f"  ! {exc}", file=sys.stderr)
        return 1

    describe_target(frames, target)
    numeric = report_columns(frames, declared, cfg.data.missing_values)

    candidates = list(cfg.features.new)
    if cfg.features.new_prefix:
        candidates += [c for c in numeric if c.startswith(cfg.features.new_prefix)
                       and c not in candidates]
    unknown = [c for c in candidates if c not in frames["train"].columns]
    if unknown:
        print(f"  ! features.new names column(s) not in the extracts: {unknown}", file=sys.stderr)
        return 1
    if not candidates:
        print("  ! features.new / new_prefix is empty - there is nothing to evaluate yet")
    else:
        print(f"  {len(candidates)} candidate feature(s), "
              f"{len(numeric) - len(candidates)} incumbent(s)")

    print("\nwriting")
    for name in SPLITS:
        path = Path(cfg.data.paths[name])
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[name].to_parquet(path, index=False)
        print(f"  {path}")
    if len(changed):
        mapping.to_csv(Path(cfg.data.paths["train"]).parent / "column_mapping.csv", index=False)
        print(f"  sanitized {len(changed)} column name(s) -> column_mapping.csv")

    print(f"\nready:  mllite -c {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
