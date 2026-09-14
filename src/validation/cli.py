"""Command line entry point: ``mllite -c configs/bankruptcy.yaml``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import yaml

from validation.logging_utils import setup_logging
from validation.pipeline import Pipeline
from validation.config import load_config


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got {text!r}")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mllite", description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="path to the YAML run config")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="dotted override, e.g. --set run.n_jobs=8 (repeatable)")
    parser.add_argument("--log-level", default=None, help="override run.log_level")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: Dict[str, Any] = dict(_parse_override(o) for o in args.overrides)
    if args.log_level:
        overrides["run.log_level"] = args.log_level

    cfg = load_config(args.config, overrides)
    setup_logging(cfg.run.log_level)
    result = Pipeline(cfg).run()
    print(json.dumps(result.summary(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
