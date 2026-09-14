"""Orchestration: wire the stages together and persist every artifact.

    build_dataset -> data quality -> feature selection -> model builds -> analysis

Each stage is independently callable; ``Pipeline`` only sequences them, applies
the stage-to-stage contracts (e.g. dropping features the screens rejected), and
writes results under ``<output_dir>/<run name>/<timestamp>/``.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from validation.config import Config, load_config
from validation.data import Dataset, build_dataset, prepare_dataset, prepare_dataset_from_frames
from validation.logging_utils import get_logger, setup_logging, timed
from validation.stages.analysis import AnalysisResult, run_analysis
from validation.stages.data_quality import DataQualityResult, run_data_quality
from validation.stages.feature_selection import (
    FeatureSelectionResult,
    build_verdict_table,
    run_feature_selection,
)
from validation.stages.modeling import ModelResult, run_modeling
from validation.stages.verdict import BatchVerdict, run_verdict

logger = get_logger(__name__)

REPORT_TOP_N = 25   # rows of the feature-ranking table; new features are never truncated away


@dataclass
class PipelineResult:
    config: Config
    output_dir: Path
    dataset: Optional[Dataset] = None
    data_quality: Optional[DataQualityResult] = None
    feature_selection: Optional[FeatureSelectionResult] = None
    models: List[ModelResult] = field(default_factory=list)
    analysis: Optional[AnalysisResult] = None
    verdicts: pd.DataFrame = field(default_factory=pd.DataFrame)
    batch: Optional[BatchVerdict] = None
    elapsed_seconds: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "run": self.config.run.name,
            "gates": self.config.run.gates,
            "output_dir": str(self.output_dir),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "dataset": self.dataset.describe() if self.dataset else {},
            "data_quality": self.data_quality.summary() if self.data_quality else {},
            "feature_selection": self.feature_selection.summary() if self.feature_selection else {},
            "candidate_verdicts": (
                self.verdicts.set_index("feature")["verdict"].to_dict()
                if not self.verdicts.empty else {}
            ),
            "batch": self.batch.summary() if self.batch else {},
            "models": [
                {"variant": m.name, "n_features": len(m.features),
                 "best_iteration": m.best_iteration, "tuned": m.tuned,
                 "params": m.params, "error": m.error}
                for m in self.models
            ],
            "analysis": self.analysis.summary() if self.analysis else {},
        }


class Pipeline:
    def __init__(self, config: Config):
        config.validate()
        self.cfg = config
        self.output_dir = self._make_output_dir()

    # ------------------------------------------------------------------ #
    def _make_output_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.cfg.run.output_dir) / self.cfg.run.name / stamp
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run(
        self,
        frame: Optional[pd.DataFrame] = None,
        frames: Optional[Dict[str, pd.DataFrame]] = None,
        dataset: Optional[Dataset] = None,
    ) -> PipelineResult:
        """Run every stage.

        Pass at most one input override:

        ``frame``    one in-memory table, split per ``data.split``;
        ``frames``   already-split tables, ``{"train": df, "valid": df, "test": df}``;
        ``dataset``  an already-built ``Dataset``, skipping loading entirely.

        With none of them, the config decides: ``data.paths`` if set, else ``data.path``.
        """
        setup_logging(self.cfg.run.log_level, self.output_dir / "run.log")
        start = time.perf_counter()
        logger.info("run %r -> %s", self.cfg.run.name, self.output_dir)
        self._write_yaml("config.resolved.yaml", self.cfg.to_dict())

        result = PipelineResult(config=self.cfg, output_dir=self.output_dir)

        with timed(logger, "stage 0: data"):
            given = [name for name, value in
                     (("frame", frame), ("frames", frames), ("dataset", dataset)) if value is not None]
            if len(given) > 1:
                raise ValueError(f"pass at most one of frame/frames/dataset, not both: {given}")
            if dataset is None:
                if frames is not None:
                    dataset = prepare_dataset_from_frames(frames, self.cfg)
                elif frame is not None:
                    dataset = prepare_dataset(frame, self.cfg)
                else:
                    dataset = build_dataset(self.cfg)

        candidates = list(dataset.new_features)   # before any stage narrows them

        with timed(logger, "stage 1: data quality & stability"):
            dq = run_data_quality(dataset, self.cfg)
            if not dq.report.empty:
                self._write_csv("data_quality_report.csv", dq.report)
            if dq.failed and self.cfg.data_quality.drop_failed and self.cfg.gates_enforced:
                failed = set(dq.failed)
                dataset = dataset.with_features(
                    [f for f in dataset.base_features if f not in failed],
                    [f for f in dataset.new_features if f not in failed],
                )
                logger.info("dropped %d feature(s) failing data quality", len(failed))
            elif dq.failed:
                logger.info("data quality flagged %d feature(s); keeping them (gates open)",
                            len(dq.failed))
        result.data_quality = dq

        with timed(logger, "stage 2: feature selection"):
            fs = run_feature_selection(dataset, self.cfg)
            self._write_feature_selection(fs)
            if self.cfg.gates_enforced:
                carried = fs.selected
            else:
                carried = list(dataset.new_features)   # measure, but remove nothing
                if fs.dropped:
                    logger.info("selection flagged %d candidate(s); carrying all %d forward "
                                "(gates open)", len(fs.dropped), len(carried))
            dataset = dataset.with_features(dataset.base_features, carried)
        result.feature_selection = fs
        result.dataset = dataset

        selection_verdicts = build_verdict_table(candidates, fs, dq_failed=dq.failed)
        self._write_csv("feature_selection_verdicts.csv", selection_verdicts)
        logger.info("selection: %d of %d candidate(s) carried forward",
                    int((selection_verdicts["verdict"] == "IN").sum()) if len(selection_verdicts) else 0,
                    len(candidates))

        if not dataset.new_features:
            logger.warning("no new features survived selection; only the baseline will be informative")

        with timed(logger, "stage 3: model builds"):
            models, tuning_log = run_modeling(dataset, self.cfg, model_dir=self.output_dir / "models")
            self._write_csv("tuning_trials.csv", tuning_log)
        result.models = models

        with timed(logger, "stage 4: outcome analysis"):
            analysis = run_analysis(dataset, self.cfg, models)
            self._write_analysis(analysis)
        result.analysis = analysis

        with timed(logger, "stage 5: verdict"):
            if self.cfg.verdict.enabled:
                result.verdicts, result.batch = run_verdict(
                    candidates, self.cfg, dq=dq, fs=fs, analysis=analysis)
                self._write_csv("candidate_verdicts.csv", result.verdicts)
                self._write_json("batch_verdict.json", result.batch.summary())
            else:
                result.verdicts = selection_verdicts

        result.elapsed_seconds = time.perf_counter() - start
        self._write_json("summary.json", {**result.summary(), "environment": _environment()})
        self._write_report(result)
        logger.info("run finished in %.1fs; artifacts in %s", result.elapsed_seconds, self.output_dir)
        return result

    # ------------------------------------------------------------------ #
    # Artifacts
    # ------------------------------------------------------------------ #
    def _write_csv(self, name: str, frame: pd.DataFrame, index: bool = False) -> None:
        if frame is None or frame.empty:
            return
        frame.to_csv(self.output_dir / name, index=index)

    def _write_json(self, name: str, payload: Dict[str, Any]) -> None:
        with open(self.output_dir / name, "w") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)

    def _write_yaml(self, name: str, payload: Dict[str, Any]) -> None:
        with open(self.output_dir / name, "w") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)

    def _write_feature_selection(self, fs: FeatureSelectionResult) -> None:
        self._write_csv("feature_selection_target_stats.csv", fs.target_stats)
        self._write_csv("feature_selection_redundancy.csv", fs.redundancy_summary)
        self._write_csv("feature_selection_spearman_matrix.csv", fs.spearman_matrix, index=True)
        self._write_csv("feature_selection_mi_matrix.csv", fs.mi_matrix, index=True)
        self._write_json("feature_selection_decisions.json", {
            "selected": fs.selected,
            "dropped": fs.dropped,
        })

    def _write_analysis(self, analysis: AnalysisResult) -> None:
        self._write_csv("metrics_by_variant_split.csv", analysis.metrics)
        self._write_csv("variant_comparison.csv", analysis.comparison)
        self._write_csv("shap_ranking.csv", analysis.shap_ranking)
        self._write_csv("xgb_importance.csv", analysis.importance)
        self._write_csv("feature_ranking.csv", analysis.feature_ranking)

    def _write_report(self, result: PipelineResult) -> None:
        lines = [f"# {self.cfg.run.name}", ""]
        lines.append(f"_generated {datetime.now():%Y-%m-%d %H:%M:%S}, "
                     f"{result.elapsed_seconds:.1f}s_")
        lines.append("")

        dataset = result.dataset
        if dataset:
            lines += ["## Data", "", "| split | rows |", "| --- | ---: |"]
            lines += [f"| {k} | {len(v):,} |" for k, v in dataset.frames.items()]
            lines += ["", f"- incumbent features: **{len(dataset.base_features)}**",
                      f"- new features carried into modeling: **{len(dataset.new_features)}**", ""]

        dq = result.data_quality
        if dq is not None and not dq.skipped and not dq.report.empty:
            report_frame = dq.report
            lines += ["## Data quality", "",
                      f"{len(report_frame)} feature(s) checked, **{len(dq.failed)} failed**"
                      + ("" if self.cfg.gates_enforced else
                         " (gates open, so nothing was removed)") + ".", ""]
            if dq.thresholds:
                by_check = dq.failed_by_check()
                lines += ["| threshold | value | features rejected |", "| --- | ---: | ---: |"]
                labels = {"max_missing_rate": ("missing rate must be <=", "missing_rate"),
                          "min_unique": ("distinct values must be >=", "n_unique"),
                          "max_psi": ("distribution shift (PSI) must be <=", "psi_max")}
                for key, value in dq.thresholds.items():
                    label, prefix = labels.get(key, (key, key))
                    hits = sum(n for check, n in by_check.items() if check.startswith(prefix))
                    lines.append(f"| {label} `{key}` | {value:g} | {hits} |")
                lines += ["", "Adjust these under `data_quality:` in the config.", ""]
            if "psi_max" in report_frame.columns:
                reference = self.cfg.data_quality.distribution_reference
                shifted = report_frame[report_frame["psi_max"] > self.cfg.data_quality.max_psi]
                lines += [
                    f"Distribution consistency is measured as PSI against the `{reference}` split "
                    f"(< 0.10 no meaningful shift, >= {self.cfg.data_quality.max_psi:.2f} flagged). "
                    f"**{len(shifted)} of {len(report_frame)}** feature(s) exceed the threshold.", ""]
                worst = report_frame.nlargest(10, "psi_max")
                keep = [c for c in ("feature", "missing_rate", "n_unique", "psi_valid",
                                    "psi_test", "psi_max", "psi_worst_split",
                                    "passed", "failed_checks")
                        if c in worst.columns]
                lines += ["Largest shifts:", "", _md_table(worst[keep]), ""]
            if dq.failed:
                failed_rows = report_frame[~report_frame["passed"]]
                keep = [c for c in ("feature", "missing_rate", "n_unique", "psi_max",
                                    "failed_checks") if c in failed_rows.columns]
                lines += ["Failed:", "", _md_table(failed_rows[keep].head(30)), ""]

        fs = result.feature_selection

        if result.batch is not None:
            batch = result.batch
            lines += [
                f"## Verdict: {batch.verdict}", "",
                f"**{batch.n_passed} of {batch.n_candidates} proposed features cleared all four gates.**", "",
                batch.note, "",
            ]
            if batch.failed_at:
                lines += ["| fell at | count |", "| --- | ---: |"]
                lines += [f"| {gate} | {n} |" for gate, n in batch.failed_at.items()]
                lines.append("")
            if self.cfg.gates_enforced:
                lines += [
                    "`run.gates: enforce` - gates act. Applied in order, and a feature that",
                    "fails one is never measured at the next, so a candidate cut at selection",
                    "has no Gini gain or SHAP rank to read.", ""]
            else:
                lines += [
                    "`run.gates: open` - **nothing was removed**. Every candidate was carried",
                    "through all four stages and measured at each, so the columns below show",
                    "where each variable actually stands rather than where it stopped. Switch",
                    "to `enforce` to have the gates filter.", ""]
            lines += [
                _md_table(result.verdicts[[c for c in (
                    "feature", "verdict", "failed_at", "data quality", "feature selection",
                    "gini gain", "shap rank", "n_gates_failed", "reason")
                    if c in result.verdicts.columns]]),
                "",
            ]

        if not result.verdicts.empty and result.batch is None:
            verdicts = result.verdicts
            n_in = int((verdicts["verdict"] == "IN").sum())
            lines += [
                "## Did the proposed features get in?", "",
                f"**{n_in} of {len(verdicts)} candidates survived screening.**", "",
                "Incumbent features are never screened, so each row asks one question: does "
                "this candidate hold up *alongside* the existing set? A candidate can be cut "
                "for duplicating an incumbent, never the reverse - the incumbent set is the "
                "status quo being challenged, which does mean the process leans toward "
                "rejection.", "",
                _md_table(verdicts[[c for c in (
                    "feature", "verdict", "decided_by", "spearman_target",
                    "spearman_redundancy_base", "closest_existing", "reason")
                    if c in verdicts.columns]]),
                "",
            ]
            by_stage = verdicts.loc[verdicts["verdict"] == "OUT", "decided_by"].value_counts()
            if len(by_stage):
                lines += ["Excluded by: " + ", ".join(f"{n} at the {stage}" for stage, n in by_stage.items()), ""]

        if fs and not fs.skipped and not fs.target_stats.empty:
            top = fs.target_stats.reindex(
                fs.target_stats.get("spearman_target", pd.Series(dtype=float)).abs().sort_values(ascending=False).index
            ).head(20)
            lines += ["### Relevance and redundancy per candidate", "", _md_table(top), ""]

        analysis = result.analysis
        tuning = self.cfg.model.tuning
        if tuning.enabled and result.models:
            distinct = {json.dumps(m.params, sort_keys=True, default=str) for m in result.models}
            lines += [
                "## Hyperparameters", "",
                (f"Tuned by random search, {tuning.n_trials} trial(s), scored as "
                 f"`{tuning.metric}` on valid."), "",
            ]
            if tuning.mode == "shared":
                lines += [f"Mode `shared`: tuned once on `{tuning.tune_on}` and applied to every "
                          "variant, so the Gini differences below are attributable to the feature "
                          "sets rather than to tuning.", ""]
            else:
                lines += ["Mode `per_variant`: every variant was tuned separately, so each feature "
                          "set got its best shot - at the cost of mixing tuning variance into the "
                          "gaps below.", ""]
            lines += [f"Distinct configurations in use: **{len(distinct)}**", "",
                      "| variant | params |", "| --- | --- |"]
            lines += [f"| {m.name} | `{json.dumps(m.params, sort_keys=True, default=str)}` |"
                      for m in result.models[:12]]
            lines.append("")

        if analysis and not analysis.comparison.empty:
            cols = ["variant", "n_features"] + [
                c for c in analysis.comparison.columns
                if c.startswith(("adj_gini_", "gini_gain_", "capture_", "accuracy_"))]
            table = analysis.comparison[cols].copy()
            sort_col = next((c for c in table.columns if c.startswith("gini_gain_")), None)
            if sort_col:
                table = table.sort_values(sort_col, ascending=False)
            lines += ["## Model comparison", "",
                      f"Baseline variant: `{self.cfg.analysis.baseline_variant}`", "",
                      _md_table(table.head(50)), ""]

        if analysis and analysis.new_feature_share:
            lines += ["### Share of SHAP attribution on new features", "",
                      "| variant | share |", "| --- | ---: |"]
            lines += [f"| {k} | {v:.1%} |" for k, v in sorted(
                analysis.new_feature_share.items(), key=lambda kv: -kv[1])[:20]]
            lines.append("")

        if analysis is not None and not analysis.feature_ranking.empty:
            focus = "base_plus_new" if "base_plus_new" in set(analysis.feature_ranking["variant"]) else \
                analysis.feature_ranking["variant"].iloc[0]
            ranked = analysis.feature_ranking[analysis.feature_ranking["variant"] == focus]
            new_features = set(dataset.new_features) if dataset else set()

            # The table is truncated, but a new feature must never be the row that
            # falls off the end - this report exists to judge exactly those.
            top = ranked.head(REPORT_TOP_N)
            below = ranked[ranked["feature"].isin(new_features)
                           & ~ranked["feature"].isin(set(top["feature"]))]
            shown = pd.concat([top, below]) if len(below) else top
            shown = shown.copy()
            shown.insert(1, "is_new", ["new" if f in new_features else ""
                                       for f in shown["feature"]])

            keep = [c for c in ["feature", "is_new", "mean_abs_shap", "shap_share", "shap_rank",
                                "imp_total_gain", "imp_total_gain_pct"] if c in shown.columns]
            caption = f"Top {min(REPORT_TOP_N, len(ranked))} of {len(ranked)} by mean |SHAP|"
            if len(below):
                caption += (f", plus {len(below)} new feature(s) ranking below the cut"
                            " - listed here because they are the point of the run")
            caption += ". Full ranking in `feature_ranking.csv`."
            lines += [f"## Feature ranking - `{focus}`", "", caption, "",
                      _md_table(shown[keep]), ""]

        with open(self.output_dir / "report.md", "w") as fh:
            fh.write("\n".join(lines))


def _md_table(frame: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if frame is None or frame.empty:
        return "_no rows_"
    def fmt(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if pd.isna(value) else floatfmt.format(value)
        return str(value)
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.ndarray, pd.Series)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _environment() -> Dict[str, str]:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "shap"):
        try:
            env[module] = __import__(module).__version__
        except Exception:
            env[module] = "not installed"
    return env


def run_pipeline(
    config_path: str | Path,
    overrides: Optional[Dict[str, Any]] = None,
    frame: Optional[pd.DataFrame] = None,
) -> PipelineResult:
    """Convenience entry point: load a YAML config and run every stage."""
    return Pipeline(load_config(config_path, overrides)).run(frame=frame)
