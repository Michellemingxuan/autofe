"""Evaluate verified features - from any number of directions - on the full splits.

Discovery saw screen rows only. An evaluation takes features from the pool of
every run's verified features (:func:`feature_pool`), runs each one's script
once over every model row - train, valid and test - through the same confirmed
linkage, and has the validation pipeline judge them: ``leave_one_in`` against
base for each, plus ``combo__<name>`` (base + the group) for each combination
the user builds. Test is the out-of-time hold-out; this is the first time
anything touches it.

Features are named ``<run_id>:<name>``. Two runs can both have found a
``spend_90d``; when both are evaluated together the columns become
``spend_90d__<last 4 of run id>`` so neither overwrites the other.

The run's gates come from the config (the Setup page's Evaluation block):
``open`` measures every chosen feature and the verdict table says which would
have passed; ``enforce`` drops failures before the models train.

    python -m agent.evaluate -c configs/synthetic_agent.yaml \\
        --run 20261003_223829_f528 \\
        --feature 20261003_225018_2585:balance_6m_volatility \\
        --combo pay_vol=20261003_223829_f528:payment_to_spend_90d,20261003_225018_2585:balance_6m_volatility
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from agent.events import EventLog, read_events, run_logged
from agent.tools.linkage import engine_of
from agent.workspace import Workspace
from validation.data import read_frame
from validation.pipeline import Pipeline
from validation.stages.analysis import _pct_label

__all__ = ["Evaluation", "feature_pool", "list_evaluations", "evaluation_results", "removed_variants",
           "run_linkage",
           "CAPTURE_PERCENTS"]

# Capture rate at the top 10%, 5% and 1% of scores, reported beside Gini, when
# the config names none (the Setup page's Evaluation block sets them).
CAPTURE_PERCENTS = (0.10, 0.05, 0.01)


def run_linkage(events: list[dict[str, Any]]) -> dict[str, str]:
    """The linkage each source was used under in a run, read from its events.

    Either a proposal the user approved in that run, or a confirmed linkage
    the run reused. Evaluating with this rather than whatever the linkage folder
    holds today reproduces the feature exactly as it was verified.
    """
    used: dict[str, str] = {}
    proposed: dict[str, tuple[str, str]] = {}
    reused: dict[str, tuple[str, str]] = {}
    for e in events:
        if e["event"] == "approval_required" and e.get("kind") == "linkage":
            # The engine travels with the code, as in a confirmed linkage file.
            header = f"# engine: {e['engine']}\n" if e.get("engine") else ""
            proposed[e["req_id"]] = (e["source"], header + e["code"])
        elif e["event"] == "approval_resolved" and e.get("approved") and e["req_id"] in proposed:
            source, code = proposed[e["req_id"]]
            used[source] = code
        elif (e["event"] == "code_status" and e.get("mode") == "linkage"
              and str(e.get("intent", "")).startswith("linkage:")
              and str(e.get("title", "")).startswith("reuse")):
            if e["state"] == "running":
                reused[e["code_id"]] = (e["intent"].split(":", 1)[1], e["code"])
            elif e["state"] == "ok" and e["code_id"] in reused:
                source, code = reused[e["code_id"]]
                used[source] = code
    return used


def feature_pool(ws: Workspace) -> list[dict[str, Any]]:
    """Every verified feature of every run, newest run first.

    Each carries the linkage code for its sources; ``missing_linkage`` lists
    sources with none recorded in the run and none confirmed now.
    """
    root = Path(ws.cfg.agent.run_dir)
    pool = []
    for folder in sorted(root.glob("*"), reverse=True):
        ledger = folder / "ledger.json"
        if not ledger.exists():
            continue
        events = read_events(folder / "events.jsonl")
        started = next((e for e in events if e["event"] == "run_started"), {})
        linkage = run_linkage(events)
        for entry in json.loads(ledger.read_text()):
            if not entry.get("verified") or entry.get("deleted"):
                continue
            codes, missing = {}, []
            for source in entry.get("sources", []):
                current = ws.linkage_path(source)
                code = linkage.get(source) or (current.read_text() if current.exists() else None)
                if code is None:
                    missing.append(source)
                else:
                    codes[source] = code
            pool.append({**entry, "key": f"{folder.name}:{entry['name']}",
                         "run_id": folder.name, "direction": started.get("direction", ""),
                         # the feature's script reruns on the engine it was written for
                         "engine": started.get("engine") or ws.cfg.agent.engine,
                         "linkage": codes, "missing_linkage": missing})
    return pool


def list_evaluations(ws: Workspace) -> list[dict[str, Any]]:
    root = Path(ws.cfg.agent.run_dir) / "evaluations"
    out = []
    for folder in sorted(root.glob("*"), reverse=True) if root.exists() else []:
        events = read_events(folder / "events.jsonl")
        started = next((e for e in events if e["event"] == "eval_started"), None)
        if started is None:
            continue
        ended = {e["event"] for e in events} & {"eval_done", "eval_error"}
        out.append({"eval_id": folder.name, "started": started["ts"],
                    "features": started["features"], "combinations": started["combinations"],
                    "status": "done" if "eval_done" in ended else "error" if ended else "running"})
    return out


def removed_variants(events: list[dict[str, Any]]) -> set[str]:
    """The variants the user took off the results - the evaluation keeps its record."""
    return {e["variant"] for e in events if e["event"] == "variant_removed"}


def evaluation_results(ws: Workspace) -> list[dict[str, Any]]:
    """Every variant of every finished evaluation, one row each, newest first.

    A row is a single feature (base + it) or a combination (base + the group),
    with its test gains against base, where it landed in that model's SHAP
    ranking, and its verdict.
    """
    rows = []
    for summary in list_evaluations(ws):
        if summary["status"] != "done":
            continue
        events = read_events(Path(ws.cfg.agent.run_dir) / "evaluations" / summary["eval_id"]
                             / "events.jsonl")
        done = next(e for e in reversed(events) if e["event"] == "eval_done")
        removed = removed_variants(events)
        by_column = {f["column"]: f for f in summary["features"]}
        verdicts = {v["feature"]: v for v in done.get("verdicts") or []}
        # Older evaluations name none; their columns say which they measured.
        percents = done.get("capture_percents") or sorted(
            {m.group(1) for r in done.get("comparison") or [] for k in r
             if (m := re.fullmatch(r"capture_gain_(top\d+)_test", k))},
            key=lambda p: -int(p[3:]))
        for r in done.get("comparison") or []:
            variant = r["variant"]
            if variant == "base" or variant in removed:
                continue
            combo = variant.startswith("combo__")
            column = variant.split("__", 1)[1] if "__" in variant else variant
            feature = {} if combo else by_column.get(column, {})
            members = summary["combinations"].get(column, {}).get("columns", []) if combo else []
            verdict = verdicts.get(column, {})
            rows.append({
                "eval_id": summary["eval_id"], "evaluated": summary["started"],
                "variant": variant, "kind": "combination" if combo else "feature",
                "name": column, "members": members,
                "key": feature.get("key"), "run_id": feature.get("run_id"),
                "direction": feature.get("direction", ""), "level": feature.get("level"),
                "description": feature.get("description", ""),
                "screen_gain": feature.get("delta"),
                "gini_gain": r.get("gini_gain_test"),
                "capture_gain": {p: r.get(f"capture_gain_{p}_test") for p in percents},
                "shap": (done.get("shap_ranks") or {}).get(variant, []),
                "verdict": verdict.get("verdict"), "reason": verdict.get("reason", ""),
                **({} if combo else (done.get("feature_stats") or {}).get(column, {})),
            })
    return rows


@dataclass
class Evaluation:
    ws: Workspace
    features: list[str]                                  # pool keys, run_id:name
    combinations: dict[str, list[str]] = field(default_factory=dict)   # name -> keys
    eval_id: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S_")
                         + uuid.uuid4().hex[:4])

    def __post_init__(self):
        self.folder = Path(self.ws.cfg.agent.run_dir) / "evaluations" / self.eval_id
        self.log = EventLog(self.folder, self.eval_id)
        self.events = self.log.events
        self.listeners = self.log.listeners
        pool = {f["key"]: f for f in feature_pool(self.ws)}
        keys = list(dict.fromkeys([*self.features,
                                   *(k for ks in self.combinations.values() for k in ks)]))
        missing = [k for k in keys if k not in pool]
        if missing:
            raise KeyError(f"not verified features of any run: {missing}")
        if not keys:
            raise ValueError("choose at least one feature to evaluate")
        self.chosen = [pool[k] for k in keys]
        unlinked = {f["key"]: f["missing_linkage"] for f in self.chosen if f["missing_linkage"]}
        if unlinked:
            raise ValueError(f"no linkage recorded for {unlinked}; confirm it in a run first")
        names = [f["name"] for f in self.chosen]
        for f in self.chosen:
            clash = names.count(f["name"]) > 1
            f["column"] = f"{f['name']}__{f['run_id'][-4:]}" if clash else f["name"]
        self.column = {f["key"]: f["column"] for f in self.chosen}

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        return self.log.emit(event, **payload)

    # ------------------------------------------------------------ the work
    def _run(self, code: str, mode: str, base_path: Path, *, engine: str, **kwargs: Any):
        agent = self.ws.cfg.agent
        return run_logged(self.log, code, mode, engine=engine, id_col=self.ws.id_col,
                          base_path=base_path, timeout_s=agent.code_timeout_s,
                          spark_conf=agent.spark_conf, **kwargs)

    def materialise(self) -> dict[str, pd.DataFrame]:
        """The three splits with the chosen features added, each by its own script."""
        ws, cfg = self.ws, self.ws.cfg
        frames = {split: read_frame(cfg.data, path) for split, path in cfg.data.paths.items()}
        base = pd.concat([f[[ws.id_col, *ws.base_features]] for f in frames.values()],
                         ignore_index=True)
        base_path = self.folder / "base_full.parquet"
        base.to_parquet(base_path, index=False)

        raw = {n: str(s.data_path) for n, s in ws.sources().items() if s.usable}
        # One join per distinct (source, linkage code): features verified under
        # the same linkage share it, and two runs that linked a source
        # differently each get their own.
        linked: dict[tuple[str, str], str] = {}
        for source, code in sorted({(s, f["linkage"][s]) for f in self.chosen
                                    for s in f["sources"]}):
            version = len([k for k in linked if k[0] == source]) + 1
            _, result = self._run(code, "linkage", base_path, engine=engine_of(code),
                                  intent="evaluate",
                                  title=f"linkage for {source} on the full splits"
                                        + (f" (v{version})" if version > 1 else ""),
                                  raw=raw, source=source)
            if not result.ok:
                raise RuntimeError(f"linkage for {source} failed on the full splits: "
                                   f"{result.error}")
            linked[(source, code)] = result.out_path

        for f in self.chosen:
            _, result = self._run(f["code"], "feature", base_path, engine=f["engine"],
                                  intent="evaluate",
                                  level=f["level"], title=f"{f['column']} on the full splits",
                                  sources={s: linked[(s, f["linkage"][s])] for s in f["sources"]})
            if not result.ok:
                raise RuntimeError(f"{f['key']} failed on the full splits: {result.error}")
            values = pd.read_parquet(result.out_path).set_index(ws.id_col)[f["name"]]
            for frame in frames.values():
                frame[f["column"]] = frame[ws.id_col].map(values)
        return frames

    def run(self) -> dict[str, Any]:
        """Materialise, then run the pipeline - narrating both as events.

        On real data this takes a while, so the user sees it move: each script
        as it computes a feature on the full splits, each pipeline stage as it
        starts and ends (``eval_status``), and the pipeline's own log lines
        (``eval_log``) - which variant is training, SHAP progress, the verdict.
        """
        self.emit("eval_started",
                  features=[{k: f[k] for k in ("key", "column", "name", "run_id", "direction",
                                               "level", "delta", "description")}
                            for f in self.chosen],
                  combinations={n: {"keys": ks, "columns": [self.column[k] for k in ks]}
                                for n, ks in self.combinations.items()})
        handler = _EventLogHandler(self)
        logging.getLogger("validation").addHandler(handler)
        try:
            self.emit("eval_log", message="computing the features on every split")
            frames = self.materialise()
            stats = self._feature_stats(frames)
            cfg = copy.deepcopy(self.ws.cfg)
            cfg.discovery.enabled = False
            cfg.features.base = list(self.ws.base_features)
            cfg.features.new = [f["column"] for f in self.chosen]
            cfg.features.new_prefix = None
            cfg.run.name = "evaluation"
            cfg.run.output_dir = str(self.folder)
            cfg.model.variants = ["base", "leave_one_in"]
            # The SHAP gate reads each feature's own model: there is no all-together one.
            cfg.verdict.shap_variant = "leave_one_in"
            cfg.model.save_models = True
            cfg.model.combinations = {n: [self.column[k] for k in ks]
                                      for n, ks in self.combinations.items()}
            percents = sorted(set(cfg.analysis.capture_rate_percents or CAPTURE_PERCENTS),
                              reverse=True)
            cfg.analysis.capture_rate_percents = percents
            cfg.analysis.comparison_capture_percents = percents
            if "test" not in cfg.analysis.metrics_on:
                cfg.analysis.metrics_on = [*cfg.analysis.metrics_on, "test"]
            result = Pipeline(cfg, on_status=self._on_status).run(frames=frames)
        except Exception as error:  # noqa: BLE001 - shown to the user, then re-raised
            self.emit("eval_error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            logging.getLogger("validation").removeHandler(handler)

        comparison = result.analysis.comparison if result.analysis else pd.DataFrame()
        shap = result.analysis.shap_ranking if result.analysis else pd.DataFrame()
        payload = {
            "output_dir": str(result.output_dir),
            "comparison": comparison.reset_index().to_dict("records"),
            "verdicts": result.verdicts.to_dict("records") if not result.verdicts.empty else [],
            "capture_percents": [_pct_label(p) for p in percents],
            "gates": cfg.run.gates,
            "shap_ranks": self._shap_ranks(shap),
            "feature_stats": stats,
        }
        self.emit("eval_done", **payload)
        return payload

    def _feature_stats(self, frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
        """Per chosen feature: its missing rate over every row, and the base feature it
        moves with most - the largest |Spearman| on train (a sample of it, when train is
        large). Against base only, so the number does not depend on what else was chosen."""
        ws = self.ws
        everything = pd.concat(frames.values(), ignore_index=True)
        train = frames.get("train", everything)
        if len(train) > STATS_ROWS:
            train = train.sample(STATS_ROWS, random_state=0)
        base = train[list(ws.base_features)].apply(pd.to_numeric, errors="coerce").rank()
        out = {}
        for column in (f["column"] for f in self.chosen):
            rho = base.corrwith(train[column].rank()).abs().dropna()
            out[column] = {
                "missing_rate": float(everything[column].isna().mean()),
                "max_corr": float(rho.max()) if len(rho) else None,
                "max_corr_with": str(rho.idxmax()) if len(rho) else None,
            }
        return out

    def _on_status(self, board: dict[str, Any]) -> None:
        stages = []
        for st in board["stages"]:
            failed = [c for c in st.get("checks") or [] if not c.get("passed")]
            # Checks that fall short in every evaluation by design are not warnings.
            real = [c for c in failed if c.get("name") not in _BY_DESIGN]
            status = st.get("status")
            if status == "warning" and not real:
                status = "passed"
            stages.append({**{k: st.get(k) for k in ("key", "label", "detail",
                                                    "elapsed_seconds", "error")},
                           "status": status,
                           "warnings": [_plain(c) for c in real]})
        self.emit("eval_status", status=board["status"], stages=stages)

    def _shap_ranks(self, shap: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        """Per variant, where each new feature landed in that model's SHAP ranking."""
        if shap.empty:
            return {}
        new = {f["column"] for f in self.chosen}
        out: dict[str, list[dict[str, Any]]] = {}
        for variant, rows in shap.groupby("variant"):
            n = len(rows)
            mine = rows[rows["feature"].isin(new)]
            out[str(variant)] = [
                {"feature": r.feature, "rank": int(r.shap_rank), "of": n,
                 "share": float(r.shap_share)}
                for r in mine.sort_values("shap_rank").itertuples(index=False)]
        return out

    def delete(self) -> None:
        shutil.rmtree(self.folder)


# The pipeline checks that fall short in every evaluation, by design: an evaluation
# judges each feature in its own model (base + it), so there is no model with every
# chosen feature together and no batch gain; and the data quality stage is off, so
# its gate has no evidence. The verdict says which gates were not applied.
_BY_DESIGN = {"batch verdict evidence", "all enabled gates were evaluable"}

# Rows the correlation is measured on, at most - it ranks every base feature.
STATS_ROWS = 50_000


def _plain(check: dict[str, Any]) -> str:
    """A failed pipeline check, as a sentence."""
    return f"{check.get('name')}: {check.get('detail')}".strip(": ")


class _EventLogHandler(logging.Handler):
    """The pipeline's INFO lines, as eval_log events - minus the redrawn board,
    which eval_status already carries in structured form."""

    def __init__(self, evaluation: "Evaluation"):
        super().__init__(level=logging.INFO)
        self.evaluation = evaluation

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("pipeline |"):
            return
        self.evaluation.emit("eval_log", message=message[:400],
                             level=record.levelname.lower(), source=record.name)


def main(argv: list[str] | None = None) -> None:
    from validation.config import load_config

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--run", action="append", default=[],
                        help="add every verified feature of this run; repeatable")
    parser.add_argument("--feature", action="append", default=[],
                        help="run_id:name; repeatable")
    parser.add_argument("--combo", action="append", default=[],
                        help="name=run_id:f1,run_id:f2 - base + them together; repeatable")
    args = parser.parse_args(argv)
    from agent import at_project_root

    ws = Workspace.from_config(load_config(at_project_root(args.config)))
    keys = [f["key"] for f in feature_pool(ws) if f["run_id"] in args.run] + args.feature
    combos = {}
    for spec in args.combo:
        name, _, members = spec.partition("=")
        combos[name] = [m for m in members.split(",") if m]
    out = Evaluation(ws, keys, combos).run()
    table = pd.DataFrame(out["comparison"])
    cols = [c for c in table.columns if c in ("variant", "note") or c.startswith("gini_gain")]
    print(table[cols].to_string(index=False))
    print(f"\nfull report: {out['output_dir']}")


if __name__ == "__main__":
    main()
