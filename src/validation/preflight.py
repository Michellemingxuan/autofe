"""Fast checks that answer "is this run wired correctly?" before model fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from validation.config import Config
from validation.data import Dataset


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class PreflightReport:
    checks: List[PreflightCheck]

    @property
    def ok(self) -> bool:
        return not any(check.status == "FAIL" for check in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "checks": [check.to_dict() for check in self.checks]}

    def render(self) -> str:
        marks = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}
        return "\n".join(
            f"[{marks[check.status]}] {check.name}: {check.detail}" for check in self.checks
        )


def run_preflight(cfg: Config, dataset: Dataset) -> PreflightReport:
    """Validate data/model contracts without running any expensive stage."""
    checks: List[PreflightCheck] = []

    def add(name: str, condition: bool, detail: str, *, warning: bool = False) -> None:
        checks.append(PreflightCheck(name, "PASS" if condition else ("WARN" if warning else "FAIL"), detail))

    splits = dataset.available_splits()
    add("training split", "train" in splits, ", ".join(splits) or "no non-empty splits")
    add("resolved feature set", bool(dataset.all_features),
        f"{len(dataset.base_features)} base + {len(dataset.new_features)} candidate")

    train = dataset.split("train")
    non_numeric = [name for name in dataset.all_features
                   if not pd.api.types.is_numeric_dtype(train[name])]
    add("numeric model inputs", not non_numeric,
        "all features numeric" if not non_numeric else "non-numeric: " + ", ".join(non_numeric))

    missing_target = {name: int(frame[dataset.target].isna().sum())
                      for name, frame in dataset.frames.items()}
    add("complete target", not any(missing_target.values()),
        ", ".join(f"{name}={count}" for name, count in missing_target.items()) + " missing")

    if cfg.model.task == "binary":
        values = sorted(pd.unique(train[dataset.target].dropna()).tolist())
        add("binary target", len(values) == 2 and set(values) <= {0, 1},
            f"training values: {values[:10]}")
    else:
        finite = np.isfinite(pd.to_numeric(train[dataset.target], errors="coerce")).all()
        add("numeric regression target", bool(finite), "all training values finite")

    requested_splits = set(cfg.analysis.metrics_on)
    absent_splits = sorted(requested_splits - set(splits))
    add("analysis splits available", not absent_splits,
        "all requested splits present" if not absent_splits else "missing: " + ", ".join(absent_splits),
        warning=True)

    if cfg.data.id_cols:
        missing_ids = [column for column in cfg.data.id_cols if column not in train.columns]
        add("ID columns present", not missing_ids,
            "all present" if not missing_ids else "missing: " + ", ".join(missing_ids))
        if not missing_ids and len(splits) > 1:
            keys = {
                name: set(map(tuple, frame[cfg.data.id_cols].itertuples(index=False, name=None)))
                for name, frame in dataset.frames.items()
            }
            overlaps = []
            names = list(keys)
            for index, left in enumerate(names):
                for right in names[index + 1:]:
                    count = len(keys[left] & keys[right])
                    if count:
                        overlaps.append(f"{left}/{right}={count}")
            add("split IDs are disjoint", not overlaps,
                "no cross-split overlap" if not overlaps else ", ".join(overlaps))

    has_candidates = bool(dataset.new_features) or cfg.discovery.enabled
    add("candidate source", has_candidates,
        (f"{len(dataset.new_features)} declared candidate(s)"
         if dataset.new_features else
         ("discovery will propose candidates" if cfg.discovery.enabled else "no candidates configured")),
        warning=True)

    variants = set(cfg.model.variants)
    add("comparison baseline requested", cfg.analysis.baseline_variant in variants,
        f"baseline={cfg.analysis.baseline_variant}; variants={', '.join(cfg.model.variants)}")
    if cfg.verdict.enabled:
        add("per-candidate verdict evidence", "leave_one_in" in variants,
            "leave_one_in enabled" if "leave_one_in" in variants else
            "add leave_one_in to model.variants for per-candidate Gini", warning=True)
        add("batch verdict evidence", "base_plus_new" in variants,
            "base_plus_new enabled" if "base_plus_new" in variants else
            "add base_plus_new to model.variants for batch gain", warning=True)
        shap_ready = (not cfg.analysis.shap.enabled or
                      (cfg.model.save_models and cfg.verdict.shap_variant in variants))
        add("SHAP verdict evidence", shap_ready,
            ("available" if shap_ready else
             "SHAP needs model.save_models and verdict.shap_variant among model.variants"),
            warning=True)

    return PreflightReport(checks)
