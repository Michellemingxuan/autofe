"""Command line entry point: ``autofe -c configs/bankruptcy.yaml``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from validation.logging_utils import setup_logging
from validation.data import build_dataset
from validation.pipeline import Pipeline
from validation.config import load_config
from validation.preflight import run_preflight
from validation.status import render_plan


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got {text!r}")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autofe", description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="path to the YAML run config")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="dotted override, e.g. --set run.n_jobs=8 (repeatable)")
    parser.add_argument("--log-level", default=None, help="override run.log_level")
    parser.add_argument("--plan", action="store_true",
                        help="show the resolved stage graph without reading data")
    parser.add_argument("--check", action="store_true",
                        help="validate data and stage contracts without training")
    return parser


def build_init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autofe init",
        description="Create a small, reusable run config for a new use case.",
    )
    parser.add_argument("config", help="YAML file to create, e.g. configs/churn.yaml")
    parser.add_argument("--train", required=True, help="fixed training table")
    parser.add_argument("--valid", required=True, help="fixed validation table")
    parser.add_argument("--test", required=True, help="fixed test table")
    parser.add_argument("--target", required=True, help="target column")
    parser.add_argument("--task", required=True, choices=("binary", "regression"))
    parser.add_argument("--id", dest="id_cols", action="append", default=[],
                        help="ID column used to check split leakage (repeatable)")
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--new-prefix", help="prefix shared by candidate columns")
    candidates.add_argument("--new", dest="new_features", action="append",
                            help="candidate column (repeatable)")
    parser.add_argument("--force", action="store_true", help="replace an existing config")
    return parser


def _init_config(argv: List[str]) -> int:
    args = build_init_parser().parse_args(argv)
    path = Path(args.config)
    if path.exists() and not args.force:
        raise FileExistsError(f"{path} already exists; pass --force to replace it")
    payload = {
        "run": {"name": path.stem, "output_dir": "outputs", "gates": "open"},
        "data": {
            "paths": {"train": args.train, "valid": args.valid, "test": args.test},
            "target": args.target,
            "id_cols": args.id_cols,
        },
        "features": {
            # An empty base list deliberately activates numeric-column inference.
            "base": [],
            **({"new_prefix": args.new_prefix} if args.new_prefix else
               {"new": args.new_features}),
        },
        "data_quality": {"enabled": True, "drop_failed": True},
        "feature_selection": {"enabled": True},
        "model": {
            "task": args.task,
            "variants": ["base", "base_plus_new", "leave_one_in"],
        },
        "analysis": {"metrics_on": ["train", "valid", "test"]},
        "verdict": {"enabled": True},
    }
    # Round-trip through the typed loader now, so init cannot emit an invalid config.
    from validation.config import Config
    Config.from_dict(payload).validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"created {path}")
    print(f"next: autofe -c {path} --check")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    if raw and raw[0] == "init":
        return _init_config(raw[1:])

    args = build_parser().parse_args(raw)
    overrides: Dict[str, Any] = dict(_parse_override(o) for o in args.overrides)
    if args.log_level:
        overrides["run.log_level"] = args.log_level

    cfg = load_config(args.config, overrides)
    setup_logging(cfg.run.log_level)
    if args.plan:
        print(render_plan(cfg))
        return 0
    if args.check:
        print(render_plan(cfg))
        report = run_preflight(cfg, build_dataset(cfg))
        print(report.render())
        return 0 if report.ok else 2
    result = Pipeline(cfg).run()
    print(json.dumps(result.summary(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
