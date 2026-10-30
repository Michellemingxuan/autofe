"""Evaluate verified features - from any number of directions - on the full splits.

Discovery saw screen rows only. An evaluation takes features from the pool of
every run's verified features (:func:`feature_pool`), runs each one's script
once over every model row - train, valid and test - through the same confirmed
linkage, and has the validation pipeline judge them: ``leave_one_in`` against
base for each, plus ``combo__<name>`` (base + the group) for each combination
the user builds. Test is the out-of-time hold-out; this is the first time
anything touches it. The base model is the same in every evaluation on the same
data, so it is trained once and kept (``<run_dir>/base_model``); a later
evaluation reuses it - its tuning and SHAP too - and trains only the challengers.

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
import multiprocessing
import os
import re
import shutil
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from agent.aggregates import materialise_aggregate, spec_key
from agent.events import EventLog, finite, read_events, run_logged
from agent.tools.linkage import engine_of
from agent.workspace import Workspace
from validation.data import read_frame
from validation.pipeline import Pipeline
from validation.stages.analysis import _pct_label

__all__ = ["Evaluation", "EvaluationProcess", "base_model_card", "feature_pool", "list_evaluations",
           "evaluation_results", "evaluation_status", "removed_variants", "combinations_of", "TOGETHER", "score_cutoffs", "untrained", "FLAG_SHARES",
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
            # the sources it reads, and those its aggregates summarise
            for source in dict.fromkeys([*entry.get("sources", []),
                                         *(a["source"] for a in (entry.get("aggregates") or {}).values())]):
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
        ended = {e["event"] for e in events} & {"eval_done", "eval_error", "eval_stopped"}
        status = ("done" if "eval_done" in ended else "stopped" if "eval_stopped" in ended
                  else "error" if ended else "running")
        out.append({"eval_id": folder.name, "started": started["ts"],
                    "features": started["features"], "combinations": combinations_of(events),
                    "together": started.get("together"),
                    "together_note": next((e.get("together_note") for e in events
                                           if e["event"] == "eval_done"), None),
                    "status": status})
    return out


def evaluation_status(ws: Workspace) -> dict[str, dict[str, Any]]:
    """Per pool feature, what its evaluations found: the latest of it alone - its
    verdict and gain, or why it was not trained - and the combinations it was in,
    each with its members' names."""
    out: dict[str, dict[str, Any]] = {}
    for r in evaluation_results(ws):                  # newest first
        if r["kind"] == "feature" and r["key"]:
            out.setdefault(r["key"], {"single": None, "combinations": []})
            if out[r["key"]]["single"] is None:
                out[r["key"]]["single"] = {k: r.get(k) for k in (
                    "eval_id", "evaluated", "verdict", "reason", "gini_gain", "trained",
                    "not_trained")}
        elif r["kind"] == "combination":
            for key in r.get("member_keys") or []:
                entry = out.setdefault(key, {"single": None, "combinations": []})
                if r["name"] not in [c["name"] for c in entry["combinations"]]:
                    entry["combinations"].append({"name": r["name"], "members": r.get("members") or []})
    return out


def combinations_of(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """An evaluation's combinations: the chosen ones, and the one it added itself
    (base + every passing feature)."""
    started = next((e for e in events if e["event"] == "eval_started"), {})
    done = next((e for e in reversed(events) if e["event"] == "eval_done"), {})
    return {**(started.get("combinations") or {}),
            **{n: {**c, "auto": True} for n, c in (done.get("added_combinations") or {}).items()}}


def removed_variants(events: list[dict[str, Any]]) -> set[str]:
    """The variants the user took off the results - the evaluation keeps its record."""
    return {e["variant"] for e in events if e["event"] == "variant_removed"}


def base_model_card(ws: Workspace, top: int = 100) -> dict[str, Any] | None:
    """The base model as the latest finished evaluation trained it - every evaluation
    trains the same one (validation.stages.base_cache): its Gini and capture rates
    per split, its rows and default rate, its features and what drives it."""
    for summary in list_evaluations(ws):
        if summary["status"] != "done":
            continue
        events = read_events(Path(ws.cfg.agent.run_dir) / "evaluations" / summary["eval_id"]
                             / "events.jsonl")
        done = next(e for e in reversed(events) if e["event"] == "eval_done")
        base = next((r for r in done.get("comparison") or [] if r["variant"] == "base"), None)
        if base is None:
            continue
        percents = done.get("capture_percents") or []
        splits = [s for s in ("train", "valid", "test") if f"adj_gini_{s}" in base]
        out_dir = Path(done.get("output_dir") or "")
        rows: dict[str, Any] = {}
        if (out_dir / "metrics_by_variant_split.csv").exists():
            metrics = pd.read_csv(out_dir / "metrics_by_variant_split.csv")
            for r in metrics[metrics["variant"] == "base"].to_dict("records"):
                rows[r["split"]] = {"rows": int(r["n_rows"]), "default_rate": float(r["actual_mean"])}
        drivers: list[dict[str, Any]] = []
        if (out_dir / "shap_ranking.csv").exists():
            shap = pd.read_csv(out_dir / "shap_ranking.csv")
            shap = shap[shap["variant"] == "base"].sort_values("shap_rank").head(top)
            drivers = [{"feature": r["feature"], "share": float(r["shap_share"])}
                       for r in shap.to_dict("records")]
        return finite({
            "eval_id": summary["eval_id"], "evaluated": summary["started"],
            "target": ws.target, "n_features": base.get("n_features"),
            "best_iteration": base.get("best_iteration"), "splits": splits,
            "gini": {s: base.get(f"adj_gini_{s}") for s in splits},
            "capture": {p: {s: base.get(f"capture_{p}_{s}") for s in splits} for p in percents},
            "data": rows, "drivers": drivers,
            "params": {"max_depth": ws.cfg.model.params.get("max_depth"),
                       "eta": ws.cfg.model.params.get("eta"),
                       "num_boost_round": ws.cfg.model.num_boost_round},
        })
    return None


def evaluation_results(ws: Workspace) -> list[dict[str, Any]]:
    """Every variant of every finished evaluation, one row each, newest first.

    A row is a single feature (base + it) or a combination (base + the group),
    with its test gains against base, where it landed in that model's SHAP
    ranking, and its verdict.
    """
    rows = []
    # C2 on the screen rows, for evaluations that did not record it.
    screen_gini = {f["key"]: (f.get("checks") or {}).get("variable_gini") for f in feature_pool(ws)}
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
        trained = {r["variant"] for r in done.get("comparison") or []}
        means = done.get("mean_scores") or {}
        shap_ranks = _shap_table(done)
        for r in [*(done.get("comparison") or []), *untrained(summary, done, trained)]:
            variant = r["variant"]
            if variant == "base" or variant in removed:
                continue
            combo = variant.startswith("combo__")
            column = variant.split("__", 1)[1] if "__" in variant else variant
            feature = {} if combo else by_column.get(column, {})
            members = summary["combinations"].get(column, {}).get("columns", []) if combo else []
            member_keys = summary["combinations"].get(column, {}).get("keys", []) if combo else []
            verdict = verdicts.get(column, {})
            stats = {} if combo else (done.get("feature_stats") or {}).get(column, {})
            nearest = stats.get("max_corr_with")
            rows.append({
                "eval_id": summary["eval_id"], "evaluated": summary["started"],
                "variant": variant, "kind": "combination" if combo else "feature",
                "name": column, "members": members, "member_keys": member_keys,
                "auto": bool(combo and summary["combinations"].get(column, {}).get("auto")),
                "key": feature.get("key"), "run_id": feature.get("run_id"),
                "direction": feature.get("direction", ""), "level": feature.get("level"),
                "description": feature.get("description", ""),
                "screen_gain": feature.get("delta"),
                # The feature alone as a score: on the screen rows (C2) and on test,
                # out of time (variable_gini, in stats) - a gap says it does not hold up.
                "screen_variable_gini": (None if combo else feature.get("screen_variable_gini")
                                         if feature.get("screen_variable_gini") is not None
                                         else screen_gini.get(feature.get("key"))),
                "gini_gain": r.get("gini_gain_test"),
                "capture_gain": {p: r.get(f"capture_gain_{p}_test") for p in percents},
                "shap": (done.get("shap_ranks") or {}).get(variant, []),
                "verdict": verdict.get("verdict") or r.get("verdict"),
                "reason": verdict.get("reason") or r.get("reason", ""),
                "trained": variant in trained,
                "not_trained": r.get("not_trained"),
                # The score level: the mean predicted probability on test, base's and
                # this model's - a new feature should not move the level much.
                "score_base": means.get("base"), "score_mean": means.get(variant),
                "score_change": (means[variant] / means["base"] - 1
                                 if means.get(variant) is not None and means.get("base") else None),
                # The base feature it moves with most: its SHAP rank in base, and
                # in base + this feature - how much of its role the feature took.
                "nearest_shap": ({"feature": nearest,
                                  "base": shap_ranks.get(("base", nearest)),
                                  "variant": shap_ranks.get((variant, nearest))}
                                 if nearest and shap_ranks else None),
                **stats,
            })
    return rows


def _shap_table(done: dict[str, Any]) -> dict[tuple[str, str], int]:
    """(variant, feature) -> SHAP rank, from the evaluation's report folder."""
    path = Path(done.get("output_dir") or "") / "shap_ranking.csv"
    if not path.is_file():
        return {}
    shap = pd.read_csv(path, usecols=["variant", "feature", "shap_rank"])
    return {(v, f): int(r) for v, f, r in shap.itertuples(index=False)}


def untrained(summary: dict[str, Any], done: dict[str, Any], trained: set[str]) -> list[dict[str, Any]]:
    """The chosen features and combinations with no model, and why - a feature the
    enforced gates dropped before modelling, a combination with a member dropped,
    or a variant whose training failed. Without these, an evaluation where only
    base trained would show nothing at all."""
    verdicts = {v["feature"]: v for v in done.get("verdicts") or []}
    errors = done.get("model_errors") or {}
    gates = done.get("gates") or "enforce"
    out = []
    for f in summary["features"]:
        if f.get("single") is False:                    # only in a combination
            continue
        variant = f"loi__{f['column']}"
        if variant in trained:
            continue
        v = verdicts.get(f["column"], {})
        why = (f"training failed: {errors[variant]}" if variant in errors else
               f"not trained: dropped at {v.get('failed_at') or 'selection'} before modelling "
               f"(run.gates: {gates})" + (f" - {v['reason']}" if v.get("reason") else ""))
        out.append({"variant": variant, "not_trained": why,
                    "verdict": v.get("verdict") or "FAIL", "reason": v.get("reason") or why})
    for name, combo in summary["combinations"].items():
        variant = f"combo__{name}"
        if variant in trained:
            continue
        dropped = [c for c in combo["columns"] if f"loi__{c}" not in trained]
        why = (f"training failed: {errors[variant]}" if variant in errors else
               f"not trained: {', '.join(dropped) or 'a member'} dropped before modelling "
               f"(run.gates: {gates})")
        out.append({"variant": variant, "not_trained": why})
    return out


@dataclass
class Evaluation:
    ws: Workspace
    features: list[str]                                  # pool keys, run_id:name
    combinations: dict[str, list[str]] = field(default_factory=dict)   # name -> keys
    eval_id: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S_")
                         + uuid.uuid4().hex[:4])
    # Then one more model, base + the single features together: "all" of them (each
    # that reached modelling) or only the "passing" ones; None: no such model.
    together: str | None = None

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
                          base_path=base_path, timeout_s=agent.timeouts.eval_s,
                          spark_conf=agent.spark_conf, **kwargs)

    def materialise(self) -> dict[str, pd.DataFrame]:
        """The three splits with the chosen features added, each by its own script."""
        ws, cfg = self.ws, self.ws.cfg
        frames = {split: read_frame(cfg.data, path) for split, path in cfg.data.paths.items()}
        base = pd.concat([f[[ws.id_col, *ws.base_features]] for f in frames.values()],
                         ignore_index=True)
        # The full splits can repeat an id the screen sample never does. Scripts get
        # each id once - a repeat would multiply its linked rows and double every sum
        # - and its value goes back to every row with that id.
        repeats = int(base[ws.id_col].duplicated().sum())
        if repeats:
            self.emit("eval_log", message=f"{repeats:,} rows repeat an id; the scripts see each id "
                                          "once (its first row), and its value goes to every row")
            base = base.drop_duplicates(subset=[ws.id_col], ignore_index=True)
        base_path = self.folder / "base_full.parquet"
        base.to_parquet(base_path, index=False)

        raw = {n: str(s.data_path) for n, s in ws.sources().items() if s.usable}
        # One join per distinct (source, linkage code): features verified under
        # the same linkage share it, and two runs that linked a source
        # differently each get their own.
        # On pandas the source is streamed in batches and the joined rows go to disk
        # by chunk of ids - only the columns the features read - and each feature
        # is built a chunk at a time (agent._child): neither is ever held whole.
        linked: dict[tuple[str, str], str] = {}
        for source, code in sorted({(s, f["linkage"][s]) for f in self.chosen
                                    for s in f["sources"]}):
            version = len([k for k in linked if k[0] == source]) + 1
            keep = "\n".join(f["code"] for f in self.chosen
                             if source in f["sources"] and f["linkage"][source] == code)
            _, result = self._run(code, "linkage", base_path, engine=engine_of(code),
                                  intent="evaluate",
                                  title=f"linkage for {source} on the full splits"
                                        + (f" (v{version})" if version > 1 else ""),
                                  raw=raw, source=source,
                                  **({} if engine_of(code) == "spark" else
                                     {"options": {"partition": True, "keep": keep}}))
            if not result.ok:
                raise RuntimeError(f"linkage for {source} failed on the full splits: "
                                   f"{result.error}")
            linked[(source, code)] = result.out_path

        summaries = self._aggregates(base_path, raw)
        for f in self.chosen:
            _, result = self._run(f["code"], "feature", base_path, engine=f["engine"],
                                  intent="evaluate",
                                  level=f["level"], title=f"{f['column']} on the full splits",
                                  sources={**{s: linked[(s, f["linkage"][s])] for s in f["sources"]},
                                           **{a: summaries[spec_key(spec, f["linkage"][spec["source"]])]
                                              for a, spec in (f.get("aggregates") or {}).items()}})
            if not result.ok:
                raise RuntimeError(f"{f['key']} failed on the full splits: {result.error}")
            values = pd.read_parquet(result.out_path).set_index(ws.id_col)[f["name"]]
            for frame in frames.values():
                frame[f["column"]] = frame[ws.id_col].map(values)
        return frames

    def _aggregates(self, base_path: Path, raw: dict[str, str]) -> dict[tuple, str]:
        """Each distinct aggregate the chosen features read, summarised on the whole
        source and joined to every id - once, however many features share it."""
        out: dict[tuple, str] = {}
        for f in self.chosen:
            for name, spec in (f.get("aggregates") or {}).items():
                linkage = f["linkage"][spec["source"]]
                key = spec_key(spec, linkage)
                if key in out:
                    continue
                if spec["source"] not in raw:
                    raise RuntimeError(f"{spec['source']} has no data for the aggregate {name}")

                def run(code: str, mode: str, **kwargs: Any):
                    return self._run(code, mode, base_path, intent="evaluate", **kwargs)

                summary, linked = materialise_aggregate(
                    run, name, spec, linkage, raw[spec["source"]], engine=f["engine"],
                    linkage_engine=engine_of(linkage), where="on the full splits",
                    partition=engine_of(linkage) != "spark")
                failed = summary if not summary.ok else linked
                if not failed.ok:
                    raise RuntimeError(f"the aggregate {name} failed on the full splits: "
                                       f"{failed.error}")
                out[key] = linked.out_path
        return out

    def run(self) -> dict[str, Any]:
        """Materialise, then run the pipeline - narrating both as events.

        On real data this takes a while, so the user sees it move: each script
        as it computes a feature on the full splits, each pipeline stage as it
        starts and ends (``eval_status``), and the pipeline's own log lines
        (``eval_log``) - which variant is training, SHAP progress, the verdict.
        """
        if self.together is True:                       # older callers: the passing ones
            self.together = "passing"
        self.emit("eval_started", together=self.together or None,
                  features=[{**{k: f[k] for k in ("key", "column", "name", "run_id", "direction",
                                                  "level", "delta", "description")},
                             # C2 on the screen rows, as discovery measured it
                             "screen_variable_gini": (f.get("checks") or {}).get("variable_gini"),
                             "single": f["key"] in self.features}
                            for f in self.chosen],
                  combinations={n: {"keys": ks, "columns": [self.column[k] for k in ks]}
                                for n, ks in self.combinations.items()})
        # Each feature as it was evaluated - its script, sources, linkage and engine -
        # so inference reruns exactly these, though the run behind one is deleted.
        (self.folder / "features.json").write_text(json.dumps(
            [{**{k: f[k] for k in ("key", "column", "name", "run_id", "level", "code", "sources",
                                   "engine", "linkage")},
              "aggregates": f.get("aggregates") or {}} for f in self.chosen], indent=2))
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
            # Base is the same in every evaluation on the same data: trained, tuned and
            # explained once, then reused (validation.stages.base_cache).
            cfg.model.base_cache_dir = str(Path(self.ws.cfg.agent.run_dir) / "base_model")
            cfg.model.combinations = {n: [self.column[k] for k in ks]
                                      for n, ks in self.combinations.items()}
            # A feature chosen only inside a combination gets no model of its own.
            cfg.model.single = [self.column[k] for k in self.features]
            # Base + all the singles is known before anything trains: it trains in
            # this same run, beside them. Only the passing ones need the verdicts
            # first, so only they take a second run (_together).
            together_now = None
            if self.together == "all" and len(self.features) >= 2:
                together_now = {"name": TOGETHER["all"], "keys": list(self.features),
                                "columns": [self.column[k] for k in self.features], "note": None}
                cfg.model.combinations[TOGETHER["all"]] = together_now["columns"]
            percents = sorted(set(cfg.analysis.capture_rate_percents or CAPTURE_PERCENTS),
                              reverse=True)
            cfg.analysis.capture_rate_percents = percents
            cfg.analysis.comparison_capture_percents = percents
            if "test" not in cfg.analysis.metrics_on:
                cfg.analysis.metrics_on = [*cfg.analysis.metrics_on, "test"]
            result = Pipeline(cfg, on_status=self._on_status).run(frames=frames)
            added = (together_now if self.together == "all"
                     else self._together(cfg, frames, result) if self.together else None)
        except Exception as error:  # noqa: BLE001 - shown to the user, then re-raised
            self.emit("eval_error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            logging.getLogger("validation").removeHandler(handler)

        comparison = result.analysis.comparison if result.analysis else pd.DataFrame()
        shap = result.analysis.shap_ranking if result.analysis else pd.DataFrame()
        models = list(result.models)
        if added and added.get("result") is not None:
            # The together model joins the evaluation's own, as one more combination.
            extra, variant = added.pop("result"), f"combo__{added['name']}"
            if extra.analysis is not None:
                rows = extra.analysis.comparison.reset_index()
                comparison = pd.concat([comparison.reset_index(), rows[rows["variant"] == variant]],
                                       ignore_index=True).set_index("variant")
                more = extra.analysis.shap_ranking
                if not more.empty:
                    shap = pd.concat([shap, more[more["variant"] == variant]], ignore_index=True)
            models += [m for m in extra.models if m.name == variant]
            kept = Path(extra.output_dir) / "models" / f"{variant}.json"
            if kept.exists():
                (Path(result.output_dir) / "models").mkdir(parents=True, exist_ok=True)
                shutil.copy(kept, Path(result.output_dir) / "models" / kept.name)
        payload = {
            "output_dir": str(result.output_dir),
            "comparison": comparison.reset_index().to_dict("records"),
            "verdicts": result.verdicts.to_dict("records") if not result.verdicts.empty else [],
            "capture_percents": [_pct_label(p) for p in percents],
            "gates": cfg.run.gates,
            "shap_ranks": self._shap_ranks(shap),
            "feature_stats": stats,
            "model_errors": {m.name: m.error for m in models if m.error},
            "mean_scores": {m.name: float(m.predictions["test"].mean()) for m in models
                            if not m.error and len((m.predictions or {}).get("test", []))},
            "score_cutoffs": score_cutoffs(models),
            # Combinations made by the evaluation itself: the together model.
            "added_combinations": ({added["name"]: {k: added[k] for k in ("keys", "columns")}}
                                   if added and added.get("columns") else {}),
            "together_note": (added or {}).get("note"),
        }
        self.emit("eval_done", **payload)
        return payload

    def _together(self, cfg: Any, frames: dict[str, pd.DataFrame], result: Any) -> dict[str, Any]:
        """Base + the single features that passed, in one model - trained after the
        verdicts, in a second run beside the evaluation's own (base + all of them
        trains in the first). Fewer than two passing: none, and a note says why."""
        singles = {self.column[k]: k for k in self.features}
        name = TOGETHER["passing"]
        verdicts = result.verdicts.to_dict("records") if not result.verdicts.empty else []
        passed = [v["feature"] for v in verdicts
                  if v.get("verdict") == "PASS" and v["feature"] in singles]
        what = "passing feature"
        if len(passed) < 2:
            note = (f"base + every {what}: not trained - {len(passed)} passed, "
                    "a model of them together needs two")
            self.emit("eval_log", message=note)
            return {"note": note}
        self.emit("eval_log", message=f"base + the {len(passed)} {what}s, together")
        again = copy.deepcopy(cfg)
        again.features.new = passed
        again.model.variants = ["base"]
        again.model.combinations = {name: passed}
        again.model.single = []
        again.run.name = "together"
        again.run.gates = "open"                       # they passed already
        try:
            extra = Pipeline(again, on_status=self._on_status).run(frames=frames)
        except Exception as error:  # noqa: BLE001 - the singles stand; say why it is missing
            note = f"base + every {what}: training failed - {type(error).__name__}: {error}"
            self.emit("eval_log", message=note)
            return {"note": note}
        return {"name": name, "keys": [singles[c] for c in passed], "columns": passed,
                "result": extra, "note": None}

    def _feature_stats(self, frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
        """Per chosen feature: its missing rate over every row, and the base feature it
        moves with most - the largest |Spearman| on train (a sample of it, when train is
        large) - and its variable Gini on test: the feature alone as a score for the
        target, |2·AUC - 1|, as check C2 reads it on the screen rows. Against base only,
        so the numbers do not depend on what else was chosen."""
        ws = self.ws
        everything = pd.concat(frames.values(), ignore_index=True)
        train = frames.get("train", everything)
        if len(train) > STATS_ROWS:
            train = train.sample(STATS_ROWS, random_state=0)
        base = train[list(ws.base_features)].apply(pd.to_numeric, errors="coerce").rank()
        test = frames.get("test", everything)
        if len(test) > STATS_ROWS:
            test = test.sample(STATS_ROWS, random_state=0)
        out = {}
        for column in (f["column"] for f in self.chosen):
            rho = base.corrwith(train[column].rank()).abs().dropna()
            out[column] = {
                "missing_rate": float(everything[column].isna().mean()),
                "max_corr": float(rho.max()) if len(rho) else None,
                "max_corr_with": str(rho.idxmax()) if len(rho) else None,
                "variable_gini": (variable_gini(test[column], test[ws.target])
                                  if ws.target in test else None),
            }
            if out[column]["variable_gini"] is None:
                out[column]["variable_gini_note"] = note = _no_signal(column, frames)
                self.emit("eval_log", message=f"warning: {column} is {note}")
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


def _evaluate_in_child(cfg: Any, features: list[str], combinations: dict[str, list[str]],
                       eval_id: str, together: str | None = None) -> None:
    """The evaluation's own process. A session of its own, so a stop ends it and
    every script and worker it started, at once. Once it is done, the target cases
    are scored with every model it trained (agent.infer), if they are set up."""
    from agent.infer import infer_after_evaluation

    os.setsid()
    ws = Workspace.from_config(cfg)
    Evaluation(ws, features, combinations, eval_id=eval_id, together=together).run()
    infer_after_evaluation(ws, eval_id)


class EvaluationProcess:
    """An evaluation as the server runs it: in a process of its own, so the user
    can stop it at any point - mid-script or mid-training - and an evaluation that
    runs out of memory on the full data ends alone, not with the server and a
    discovery run beside it. The server sees its events as the child writes them."""

    def __init__(self, job: Evaluation):
        self.job, self.eval_id, self.folder = job, job.eval_id, job.folder
        self.path = job.log.path
        self.events: list[dict[str, Any]] = list(job.events)
        self.listeners: list[Any] = []
        self.process: Any = None
        self.stopped = False
        self._offset = 0
        self._reading = threading.Lock()

    def run(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.process = ctx.Process(
            target=_evaluate_in_child, name=f"evaluation-{self.eval_id}",
            args=(self.job.ws.cfg, self.job.features, self.job.combinations, self.eval_id,
                  self.job.together))
        self.process.start()
        while self.process.is_alive():
            self._read()
            time.sleep(0.3)
        self.process.join()
        self._read()
        if self.done:
            return
        if self.stopped:
            self.emit("eval_stopped", reason="stopped by the user")
        elif self.process.exitcode in (-9, 137):
            self.emit("eval_error", error="the evaluation was killed by the operating system "
                                          "(exit -9) - almost always out of memory on the full data")
        else:
            self.emit("eval_error",
                      error=f"the evaluation process ended unexpectedly (exit {self.process.exitcode})")

    @property
    def done(self) -> bool:
        """Its log has ended - the process may still be exiting."""
        self._read()
        return any(e["event"] in ("eval_done", "eval_error", "eval_stopped") for e in self.events)

    def stop(self) -> None:
        self.stopped = True
        if self.process is None or self.process.pid is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGKILL)     # its scripts and workers too
        except (ProcessLookupError, PermissionError):
            pass
        if self.process.is_alive():
            self.process.kill()

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        """Once the child is done, the server writes to the log itself."""
        record = EventLog(self.folder, self.eval_id).emit(event, **payload)
        self._read()
        return record

    def _read(self) -> None:
        """New complete lines of the child's log, to the events and the listeners."""
        with self._reading:                  # the hub's thread and a request's, both
            if not self.path.exists():
                return
            with open(self.path, "rb") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
            end = chunk.rfind(b"\n")
            if end < 0:
                return
            self._offset += end + 1
            for line in chunk[: end + 1].decode(errors="replace").splitlines():
                if line.strip():
                    record = finite(json.loads(line))
                    self.events.append(record)
                    for listener in list(self.listeners):
                        listener(record)


# The shares of test a model's flag cutoff may be set at (inference): the score
# above which a model puts that share of the test rows.
FLAG_SHARES = (0.01, 0.02, 0.05, 0.10, 0.20)
# The combination an evaluation adds itself: base + every single feature that passed.
TOGETHER = {"all": "all_new", "passing": "all_passing"}


def share_key(share: float) -> str:
    return f"{share:g}"


def variable_gini(values: pd.Series, target: pd.Series) -> float | None:
    """The feature alone as a score for the target: |2·AUC - 1|, AUC from ranks
    (Mann-Whitney), a missing value ranked in the middle - check C2's measure
    (agent.checks.Sketch.variable_gini). None for a constant feature or one class."""
    f = pd.to_numeric(values, errors="coerce").astype(float).reset_index(drop=True)
    f = f.fillna(f.median()) if f.notna().any() else f
    positive = (pd.to_numeric(target, errors="coerce") == 1).to_numpy()
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if f.isna().all() or f.nunique() < 2 or not n1 or not n0:
        return None
    ranks = f.rank().to_numpy()
    auc = (ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return round(abs(2 * auc - 1), 4)


def _no_signal(column: str, frames: dict[str, pd.DataFrame]) -> str:
    """Why a feature has no variable Gini on test: no value there, or one value -
    with the other out-of-time split's state, as a feature whose data stops before
    them is constant on both."""
    def state(split: str) -> str | None:
        x = frames[split][column] if split in frames else None
        if x is None:
            return None
        if x.notna().sum() == 0:
            return "empty"
        if x.nunique() < 2:
            return f"one value ({x.dropna().iloc[0]:g})"
        return None
    test, valid = state("test"), state("valid")
    train = frames.get("train")
    varied = train is not None and train[column].nunique() > 1
    where = "valid and test" if valid and test else "test"
    hint = (", though it varies on train - its data may not cover the later period"
            if varied else "")
    return f"{test or 'no signal'} on {where}{hint}"


def score_cutoffs(models: list[Any], shares: tuple[float, ...] = FLAG_SHARES) -> dict[str, dict[str, float]]:
    """Per trained variant, the test score at each share's cutoff."""
    import numpy as np

    out = {}
    for m in models:
        scores = None if m.error else (m.predictions or {}).get("test")
        if scores is not None and len(scores):
            out[m.name] = {share_key(s): float(np.quantile(scores, 1 - s)) for s in shares}
    return out


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
