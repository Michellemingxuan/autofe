"""Infer: score the target cases with the base model and base + new features.

An evaluation judges features on test as a whole. The analyst also cares about
particular cases - high-balance defaulters, say - and whether the new features
would have seen them sooner. Inference takes every model one finished
evaluation trained - ``base`` and each variant, ``loi__<feature>`` or
``combo__<name>`` - and scores every row of the target cases with each. It runs
when the evaluation ends, if the cases are set up; one evaluated before can be
scored from the page.

The cases (``discovery.additional_data.target_cases``) are rows shaped like the
model database - a customer's transactions, say - with their histories beside
them in the same format as the sources the features were built on. They carry
no model id: ``id_template`` builds one from their columns, in the model id's
form, so the confirmed linkage and the feature scripts run on them unchanged,
exactly as the evaluation ran them (its ``features.json``). A source's history
is the file Setup names for it (``target_cases.sources``); a source the
features read with no history for the cases stops the inference, and Setup
marks it.

Each model flags a row when its score is in its own top ``flag_top`` of test
scores - the same share for each, so none is favoured by a lower bar. A case is
its rows in time order; what is compared is the first row base and each variant
flag: a variant flagging a case at an earlier row sees its risk sooner.

    python -m agent.infer -c configs/synthetic_agent.yaml --evaluation <eval_id>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent.evaluate import FLAG_SHARES, combinations_of, feature_pool, list_evaluations, removed_variants, share_key
from agent.events import EventLog, finite, read_events, run_logged
from agent.tools.linkage import engine_of
from agent.workspace import Workspace
from validation.data import clean_missing, read_frame

__all__ = ["Inference", "build_ids", "case_materials", "infer_after_evaluation", "inference_result",
           "list_inferences", "scored_variants"]

_FIELD = re.compile(r"\{(\w+)(?::([^}]*))?\}")
_DATA_SUFFIXES = (".parquet", ".csv")


# ------------------------------------------------------------------ the cases
def build_ids(rows: pd.DataFrame, template: str) -> pd.Series:
    """The model id of each row, from its columns: ``{col}`` as it is, ``{col:%Y%m%d}``
    a date formatted. Every row must get one, and no two the same."""
    fields = _FIELD.findall(template)
    if not fields:
        raise ValueError(f"the id template {template!r} names no column; write it like "
                         "{customer_id}_{trans_dt:%Y%m%d}_{marker}")
    absent = [c for c, _ in fields if c not in rows.columns]
    if absent:
        raise KeyError(f"the id template names column(s) the cases do not have: {absent}")
    out = pd.Series("", index=rows.index, dtype="object")
    blank = pd.Series(False, index=rows.index)
    pos = 0
    for match in _FIELD.finditer(template):
        column, fmt = match.groups()
        values = rows[column]
        if fmt:
            values = pd.to_datetime(values, errors="coerce").dt.strftime(fmt)
        elif pd.api.types.is_float_dtype(values) and values.dropna().mod(1).eq(0).all():
            values = values.astype("Int64")             # 100001.0 -> 100001, as in the id
        blank |= values.isna()
        out = out + template[pos:match.start()] + values.astype(str)
        pos = match.end()
    out = out + template[pos:]
    if blank.any():
        raise ValueError(f"{int(blank.sum())} case row(s) have no value (or no readable date) in "
                         f"a column of the id template; e.g. row {int(np.argmax(blank.to_numpy()))}")
    dupes = out[out.duplicated(keep=False)]
    if len(dupes):
        raise ValueError(f"the id template gives {len(dupes)} rows the same id as another "
                         f"(e.g. {dupes.iloc[0]}); add the column that tells them apart")
    return out


def _resolve(path: str, folder: str | None) -> Path:
    """A configured path: as given, or inside the target cases folder when it is a
    bare name there; a name without suffix finds its .parquet or .csv."""
    p = Path(path)
    if not (p.is_absolute() or p.exists()) and folder:
        p = Path(folder) / p
    if not p.exists() and not p.suffix:
        # A name without its suffix: the .parquet or .csv of it.
        p = next((p.with_name(p.name + s) for s in _DATA_SUFFIXES
                  if p.with_name(p.name + s).exists()), p)
    return p


def needed_sources(ws: Workspace) -> dict[str, int]:
    """Each source the verified features read, and how many of them read it - the
    histories the cases must bring."""
    counts: dict[str, int] = {}
    for f in feature_pool(ws):
        for source in f.get("sources") or []:
            counts[source] = counts.get(source, 0) + 1
    return counts


def case_materials(ws: Workspace) -> dict[str, Any]:
    """Where the target cases' rows and histories are, as configured - checked, not
    computed: what the Setup page and the inference page show. Each source of the
    additional data gets the path of the cases' own history of it - given in Setup,
    or the source's own file name in the cases' folder; one the verified features
    read and the cases lack is marked. ``error`` says
    what stops an inference."""
    tc = ws.cfg.discovery.additional_data.target_cases
    out: dict[str, Any] = {"rows_path": None, "id_template": tc.id_template,
                           "case_column": tc.case_column, "time_column": tc.time_column,
                           "flag_top": tc.flag_top, "sources": {}, "error": None}
    needed = needed_sources(ws)
    sources = ws.sources()
    for name in sorted({*sources, *tc.sources}):
        # Given in Setup - or, by default, the source's own file name in the cases' folder.
        given = tc.sources.get(name) or (sources[name].data_path.name
                                         if tc.dir and name in sources and sources[name].usable else None)
        path = _resolve(given, tc.dir) if given else None
        out["sources"][name] = {
            "path": str(path) if path else None,
            "state": "provided" if path and path.exists() else "not_found" if path else "not_set",
            "needed_by": needed.get(name, 0)}
    if not tc.rows:
        out["error"] = "set the file of the cases' rows - Setup, Target cases"
        return out
    rows_path = _resolve(tc.rows, tc.dir)
    out["rows_path"] = str(rows_path)
    if not rows_path.exists():
        out["error"] = f"no cases' rows at {rows_path}"
    return out


def read_cases(ws: Workspace, materials: dict[str, Any] | None = None) -> pd.DataFrame:
    """The cases' rows with the model id built, a case and a time column, in order."""
    m = materials or case_materials(ws)
    if m["error"]:
        raise ValueError(m["error"])
    rows = read_frame(ws.cfg.data, m["rows_path"])
    tc = ws.cfg.discovery.additional_data.target_cases
    id_col = ws.id_col
    if tc.id_template:
        if id_col in rows.columns:
            rows = rows.rename(columns={id_col: f"{id_col}_given"})
        rows.insert(0, id_col, build_ids(rows, tc.id_template).to_numpy())
    elif id_col not in rows.columns:
        raise ValueError(f"the cases have no {id_col!r} column: set the id template, which "
                         "builds it from their columns")
    fields = _FIELD.findall(tc.id_template or "")
    case = tc.case_column or next((c for c, f in fields if not f), None)
    when = tc.time_column or next((c for c, f in fields if f), None)
    for label, column in (("case column", case), ("time column", when)):
        if not column:
            raise ValueError(f"set the {label} (Setup, Target cases)")
        if column not in rows.columns:
            raise KeyError(f"the {label} {column!r} is not a column of the cases")
    rows["_case"] = rows[case].astype(str)
    rows["_time"] = pd.to_datetime(rows[when], errors="coerce")
    if rows["_time"].isna().all():
        rows["_time"] = rows[when]                  # an ordinal, not a date
    rows = rows.sort_values(["_case", "_time", id_col], kind="stable", ignore_index=True)
    rows["_t"] = rows.groupby("_case").cumcount() + 1
    return rows


def preview(ws: Workspace) -> dict[str, Any]:
    """The materials, and what reading the cases finds: rows, cases, an id beside a
    model id, and the base columns they lack."""
    m = case_materials(ws)
    if m["error"]:
        return m
    try:
        rows = read_cases(ws, m)
    except Exception as error:  # noqa: BLE001 - shown on the page
        return {**m, "error": f"{type(error).__name__}: {error}"}
    return {**m, "rows": len(rows), "cases": int(rows["_case"].nunique()),
            "columns": len(rows.columns) - (4 if m["id_template"] else 3),
            "id_example": str(rows[ws.id_col].iloc[0]) if len(rows) else None,
            "model_id_example": str(ws.screen[ws.id_col].iloc[0]),
            "missing_base": [c for c in ws.base_features if c not in rows.columns],
            "per_case": {"min": int(rows.groupby("_case").size().min()),
                         "max": int(rows.groupby("_case").size().max())} if len(rows) else None}


# --------------------------------------------------------- what can be scored
def _done_event(folder: Path) -> dict[str, Any] | None:
    events = read_events(folder / "events.jsonl")
    return next((e for e in reversed(events) if e["event"] == "eval_done"), None)


def scored_variants(ws: Workspace, eval_id: str) -> list[dict[str, Any]]:
    """An evaluation's trained variants with a kept model - every one is scored."""
    folder = Path(ws.cfg.agent.run_dir) / "evaluations" / eval_id
    events = read_events(folder / "events.jsonl")
    done = next((e for e in reversed(events) if e["event"] == "eval_done"), None)
    started = next((e for e in events if e["event"] == "eval_started"), None)
    if done is None or started is None:
        return []
    models = Path(done.get("output_dir") or "") / "models"
    if not (models / "base.json").exists():
        return []
    removed = removed_variants(events)
    out = []
    for r in done.get("comparison") or []:
        v = r["variant"]
        if v == "base" or v in removed or not (models / f"{v}.json").exists():
            continue
        name = v.split("__", 1)[1]
        combo = v.startswith("combo__")
        out.append({"variant": v, "kind": "combination" if combo else "feature", "name": name,
                    "columns": list(combinations_of(events)[name]["columns"]) if combo else [name],
                    "gini_gain": r.get("gini_gain_test")})
    return out


# ------------------------------------------------------------------ the job
@dataclass
class Inference:
    """Score the target cases with the base model and every model one evaluation
    trained. One per evaluation: scoring it again replaces it."""
    ws: Workspace
    eval_id: str
    flag_top: float | None = None

    def __post_init__(self):
        root = Path(self.ws.cfg.agent.run_dir)
        self.eval_folder = root / "evaluations" / self.eval_id
        done = _done_event(self.eval_folder)
        if done is None:
            raise ValueError(f"{self.eval_id} is not a finished evaluation")
        self.done = done
        self.models = Path(done.get("output_dir") or "") / "models"
        self.variants = scored_variants(self.ws, self.eval_id)
        if not self.variants:
            raise ValueError(f"evaluation {self.eval_id} kept no trained model besides base")
        self.started = next(e for e in read_events(self.eval_folder / "events.jsonl")
                            if e["event"] == "eval_started")
        self.share = float(self.flag_top or self.ws.cfg.discovery.additional_data.target_cases.flag_top)
        if not 0 < self.share < 1:
            raise ValueError("the flag share must be between 0 and 1, e.g. 0.05")
        self.infer_id = self.eval_id
        self.folder = root / "inferences" / self.infer_id
        if self.folder.exists():
            shutil.rmtree(self.folder)                  # scored again: the new one replaces it
        self.log = EventLog(self.folder, self.infer_id)
        self.events, self.listeners = self.log.events, self.log.listeners
        self.cancelled = False

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        return self.log.emit(event, **payload)

    @property
    def columns(self) -> list[str]:
        """Every new feature column any of the models has."""
        return list(dict.fromkeys(c for v in self.variants for c in v["columns"]))

    def _specs(self) -> dict[str, dict[str, Any]]:
        """Each new column's script, sources, linkage and engine, as evaluated."""
        path = self.eval_folder / "features.json"
        if path.exists():
            specs = {f["column"]: f for f in json.loads(path.read_text())}
        else:                                   # an evaluation from before it kept them
            pool = {f["key"]: f for f in feature_pool(self.ws)}
            specs = {f["column"]: {**pool[f["key"]], "column": f["column"]}
                     for f in self.started["features"] if f["key"] in pool}
        missing = [c for c in self.columns if c not in specs]
        if missing:
            raise ValueError(f"the scripts of {missing} are no longer kept - the runs behind "
                             "them were deleted; evaluate the features again")
        return {c: specs[c] for c in self.columns}

    def _run(self, code: str, mode: str, base_path: Path, *, engine: str, **kwargs: Any):
        agent = self.ws.cfg.agent
        return run_logged(self.log, code, mode, engine=engine, id_col=self.ws.id_col,
                          base_path=base_path, timeout_s=agent.timeouts.eval_s,
                          spark_conf=agent.spark_conf, should_stop=lambda: self.cancelled,
                          **kwargs)

    def features(self, rows: pd.DataFrame, materials: dict[str, Any]) -> pd.DataFrame:
        """The new columns on the cases' rows: each linkage once, then each script."""
        ws = self.ws
        base_path = self.folder / "cases_base.parquet"
        rows[[ws.id_col, *[c for c in ws.base_features if c in rows.columns]]].to_parquet(
            base_path, index=False)
        specs = self._specs()
        raw = {n: s["path"] for n, s in materials["sources"].items() if s["state"] == "provided"}
        linked: dict[tuple[str, str], str] = {}
        for source, code in sorted({(s, f["linkage"][s]) for f in specs.values() for s in f["sources"]}):
            if source not in raw:
                raise ValueError(f"the cases have no history of {source}, which the features read - "
                                 "give its path in Setup, Target cases")
            _, result = self._run(code, "linkage", base_path, engine=engine_of(code),
                                  intent="infer", title=f"linkage for {source}",
                                  raw=raw, source=source)
            if not result.ok:
                raise RuntimeError(f"linkage for {source} failed on the cases: {result.error}")
            linked[(source, code)] = result.out_path
            n = pd.read_parquet(result.out_path, columns=[ws.id_col])[ws.id_col].nunique()
            self.emit("infer_log", message=f"{source}: history for {n:,} of {len(rows):,} case rows")
        out = rows[[ws.id_col]].copy()
        for column, f in specs.items():
            _, result = self._run(f["code"], "feature", base_path, engine=f["engine"],
                                  intent="infer", level=f.get("level"), title=f"{column} on the cases",
                                  sources={s: linked[(s, f["linkage"][s])] for s in f["sources"]})
            if not result.ok:
                raise RuntimeError(f"{column} failed on the cases: {result.error}")
            values = pd.read_parquet(result.out_path).set_index(ws.id_col)[f["name"]]
            out[column] = out[ws.id_col].map(values)
        return out

    def run(self) -> dict[str, Any]:
        self.emit("infer_started", eval_id=self.eval_id, variants=self.variants,
                  columns=self.columns, flag_top=self.share)
        try:
            materials = case_materials(self.ws)
            rows = read_cases(self.ws, materials)
            absent = [c for c in self.ws.base_features if c not in rows.columns]
            if absent:
                raise ValueError(f"the cases lack {len(absent)} base column(s) the models use: "
                                 f"{absent[:10]}{' ...' if len(absent) > 10 else ''}")
            self.emit("infer_log", message=f"{len(rows):,} rows of {rows['_case'].nunique():,} cases")
            new = self.features(rows, materials)
            frame = rows.drop(columns=[c for c in self.columns if c in rows.columns]).merge(
                new, on=self.ws.id_col, how="left")
            names = ["base", *(v["variant"] for v in self.variants)]
            quantiles = self._cutoffs(names)
            cutoffs = {n: quantiles[n][share_key(self.share)] for n in names}
            for name in names:
                frame[f"score__{name}"] = self._score(name, frame)
            target = self.ws.target if self.ws.target in frame.columns else None
            keep = [self.ws.id_col, "_case", "_time", "_t", *self.columns, *([target] if target else []),
                    *(f"score__{n}" for n in names)]
            frame[keep].to_parquet(self.folder / "scores.parquet", index=False)
            summaries = {v["variant"]: summarise(frame, target, cutoffs["base"], v["variant"],
                                                 cutoffs[v["variant"]])
                         for v in self.variants}
        except Exception as error:  # noqa: BLE001 - shown to the user, then re-raised
            if self.cancelled:
                self.emit("infer_stopped", reason="stopped by the user")
            else:
                self.emit("infer_error", error=f"{type(error).__name__}: {error}")
            raise
        payload = {"cutoffs": cutoffs, "quantiles": quantiles, "summaries": summaries,
                   "sources": {n: s["state"] for n, s in materials["sources"].items()},
                   "rows_path": materials["rows_path"]}
        self.emit("infer_done", **payload)
        return payload

    # ------------------------------------------------------------ the models
    def _score(self, variant: str, frame: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(self.models / f"{variant}.json"))
        names = list(booster.feature_names or [])
        data = self.ws.cfg.data
        suffix = data.missing_indicators.suffix
        matrix = pd.DataFrame(index=frame.index)
        for name in names:
            if name in frame.columns:
                matrix[name] = pd.to_numeric(frame[name], errors="coerce")
            elif name.endswith(suffix) and name[: -len(suffix)] in frame.columns:
                matrix[name] = np.nan                   # filled below, after cleaning
            else:
                raise KeyError(f"the {variant} model uses {name!r}, which the cases lack")
        matrix = clean_missing(matrix, names, data.missing_values)
        for name in names:
            if name not in frame.columns:
                matrix[name] = matrix[name[: -len(suffix)]].isna().astype("int8")
        best = next((r.get("best_iteration") for r in self.done.get("comparison") or []
                     if r["variant"] == variant), None)
        dm = xgb.DMatrix(matrix.to_numpy(dtype=np.float32), feature_names=names, missing=np.nan)
        kwargs = {"iteration_range": (0, int(best) + 1)} if best is not None else {}
        return booster.predict(dm, **kwargs)

    def _cutoffs(self, names: list[str]) -> dict[str, dict[str, float]]:
        """Each model's flag cutoff at every share: the score of its top share of
        test - so the page can flag at any of them, or at base's, without scoring
        again. Kept by the evaluation; for one from before it kept them, scored on
        test again."""
        shares = sorted({*FLAG_SHARES, self.share})
        kept = self.done.get("score_cutoffs") or {}
        if all(share_key(s) in kept.get(n, {}) for n in names for s in shares):
            return {n: {share_key(s): kept[n][share_key(s)] for s in shares} for n in names}
        self.emit("infer_log", message="this evaluation kept no score cutoffs: scoring test "
                                       "again to set them")
        test = self._test_frame()
        out = {}
        for n in names:
            scores = self._score(n, test)
            out[n] = {share_key(s): float(np.quantile(scores, 1 - s)) for s in shares}
        return out

    def _test_frame(self) -> pd.DataFrame:
        """Test, with the new columns as the evaluation computed them."""
        ws = self.ws
        test = read_frame(ws.cfg.data, ws.cfg.data.paths["test"])
        events = read_events(self.eval_folder / "events.jsonl")
        names = {f["column"]: f["name"] for f in self.started["features"]}
        for column in self.columns:
            title = f"{column} on the full splits"
            code_id = next((e["code_id"] for e in events if e["event"] == "code_status"
                            and e.get("title") == title and e["state"] == "ok"), None)
            path = self.eval_folder / "code" / f"{code_id}.parquet"
            if code_id is None or not path.exists():
                raise ValueError(f"evaluation {self.eval_id} kept neither score cutoffs nor "
                                 f"{column}'s values; evaluate it again")
            values = pd.read_parquet(path).set_index(ws.id_col)[names[column]]
            test[column] = test[ws.id_col].map(values)
        return test


def summarise(frame: pd.DataFrame, target: str | None, base_cutoff: float, variant: str,
              cutoff: float) -> dict[str, Any]:
    """Per case, the first row base and the variant each flag; over the cases, how
    often the variant flags earlier, later or alone, and by how many rows and days."""
    flags = {"base": frame["score__base"] >= base_cutoff,
             "variant": frame[f"score__{variant}"] >= cutoff}
    cases = []
    for case, rows in frame.groupby("_case", sort=True):
        first, when = {}, {}
        for m, flagged in flags.items():
            hit = rows.loc[flagged.loc[rows.index], ["_t", "_time"]]
            first[m] = int(hit["_t"].iloc[0]) if len(hit) else None
            when[m] = hit["_time"].iloc[0] if len(hit) else None
        b, v = first["base"], first["variant"]
        outcome = ("neither" if b is None and v is None else "only_variant" if b is None
                   else "only_base" if v is None else "earlier" if v < b
                   else "later" if v > b else "same")
        days = None
        if b and v and isinstance(when["base"], pd.Timestamp) and isinstance(when["variant"], pd.Timestamp):
            days = int((when["base"] - when["variant"]).days)
        cases.append({"case": str(case), "rows": int(len(rows)),
                      "first_base": b, "first_variant": v, "outcome": outcome,
                      "rows_sooner": (b - v) if b and v else None, "days_sooner": days,
                      "defaulted": (int(rows[target].max()) if target and rows[target].notna().any()
                                    else None)})
    counts = pd.Series([c["outcome"] for c in cases]).value_counts().to_dict()
    return finite({
        "cases": len(cases),
        "outcomes": {k: int(counts.get(k, 0)) for k in
                     ("earlier", "same", "later", "only_variant", "only_base", "neither")},
        "per_case": cases,
    })


# ------------------------------------------------------------- reading back
def list_inferences(ws: Workspace) -> list[dict[str, Any]]:
    """Every finished evaluation with a trained variant, and its inference - or
    none yet."""
    root = Path(ws.cfg.agent.run_dir) / "inferences"
    out = []
    for summary in list_evaluations(ws):
        if summary["status"] != "done":
            continue
        variants = scored_variants(ws, summary["eval_id"])
        if not variants:
            continue
        events = read_events(root / summary["eval_id"] / "events.jsonl")
        started = next((e for e in events if e["event"] == "infer_started"), None)
        ended = next((e for e in reversed(events)
                      if e["event"] in ("infer_done", "infer_error", "infer_stopped")), None)
        status = ("none" if started is None else "running" if ended is None else
                  {"infer_done": "done", "infer_error": "error", "infer_stopped": "stopped"}[ended["event"]])
        out.append({"eval_id": summary["eval_id"], "evaluated": summary["started"],
                    "variants": variants, "status": status,
                    "scored": started["ts"] if started else None,
                    "error": (ended or {}).get("error"),
                    "sooner": {v: s["outcomes"]["earlier"] + s["outcomes"]["only_variant"]
                               for v, s in ((ended or {}).get("summaries") or {}).items()}})
    return out


def inference_result(ws: Workspace, infer_id: str) -> dict[str, Any]:
    """One evaluation's inference: how it went, its summaries, and every case's
    rows - time, each model's score, the new columns - for the case's chart."""
    folder = Path(ws.cfg.agent.run_dir) / "inferences" / infer_id
    events = read_events(folder / "events.jsonl")
    started = next((e for e in events if e["event"] == "infer_started"), None)
    if started is None:
        raise FileNotFoundError(f"no inference {infer_id}")
    ended = next((e for e in reversed(events)
                  if e["event"] in ("infer_done", "infer_error", "infer_stopped")), None)
    codes: dict[str, dict[str, Any]] = {}
    for e in events:
        if e["event"] == "code_status":
            codes.setdefault(e["code_id"], {}).update(
                {k: e[k] for k in ("title", "state", "elapsed_s") if k in e})
    out: dict[str, Any] = {
        "infer_id": infer_id, **{k: started[k] for k in ("eval_id", "variants", "columns",
                                                         "flag_top", "ts")},
        "status": ("running" if ended is None else ended["event"].replace("infer_", "")),
        "error": (ended or {}).get("error"),
        "steps": list(codes.values()),
        "logs": [e["message"] for e in events if e["event"] == "infer_log"],
        "done": ended if ended and ended["event"] == "infer_done" else None}
    path = folder / "scores.parquet"
    if out["done"] and path.exists():
        scores = pd.read_parquet(path)
        scores["_time"] = scores["_time"].astype(str)
        names = ["base", *(v["variant"] for v in started["variants"])]
        out["timelines"] = {
            str(case): [finite({"t": int(r["_t"]), "id": str(r[ws.id_col]), "time": r["_time"],
                                "y": r.get(ws.target),
                                "scores": {n: float(r[f"score__{n}"]) for n in names},
                                "values": {c: r[c] for c in started["columns"]}})
                        for r in rows.to_dict("records")]
            for case, rows in scores.groupby("_case", sort=True)}
    return out


def infer_after_evaluation(ws: Workspace, eval_id: str) -> dict[str, Any] | None:
    """Score the target cases with every model an evaluation trained - run when the
    evaluation ends, if the cases are set up. Its failure is the inference's own:
    the evaluation stands."""
    if case_materials(ws)["error"] or not scored_variants(ws, eval_id):
        return None
    try:
        return Inference(ws, eval_id).run()
    except Exception:  # noqa: BLE001 - recorded in the inference's own log
        return None


def main(argv: list[str] | None = None) -> None:
    from agent import at_project_root
    from validation.config import load_config

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--flag-top", type=float, default=None,
                        help=f"share of test each model flags; one of {FLAG_SHARES}")
    args = parser.parse_args(argv)
    ws = Workspace.from_config(load_config(at_project_root(args.config)))
    job = Inference(ws, args.evaluation, args.flag_top)
    out = job.run()
    for variant, s in out["summaries"].items():
        print(variant, json.dumps(s["outcomes"]))
    print(f"\nscores: {job.folder / 'scores.parquet'}")


if __name__ == "__main__":
    main()
