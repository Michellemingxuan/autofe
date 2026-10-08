"""The setup step: where the model database, the context and the extra data live.

The run config (YAML) gives the defaults. What the user changes on the Setup
page is kept as dotted-key overrides in ``<agent.run_dir>/workspace_overrides.json``
and applied on top of the YAML - so the YAML stays the reviewed default, a
restart keeps the user's choices, and "reset" is deleting one file.

The page is in blocks - model database, context, shots, evaluation, additional
data, scope -
each applied on its own. A change is applied only if the workspace it
describes actually loads: the screen files are read, every other file must
exist (the splits are not read - they may be tens of GB).

Small files - the task context, column descriptions, scope notes - can be
uploaded instead of pointed at; they are kept under ``<agent.run_dir>/uploads``
and the setting points there. A Word or PDF note is stored with its text
extracted, since the agent reads text.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml

from agent.tools.shots import parse_spec
from agent.workspace import Workspace
from validation.config import AdditionalDataConfig, Config

__all__ = ["BLOCKS", "Setup", "UPLOAD_LIMIT", "blocks", "fields"]

UPLOAD_LIMIT = 10 * 1024 * 1024
UPLOAD_KINDS = {
    "context": (".md", ".txt", ".json", ".csv"),
    "source_schema": (".json",),
    "scope_file": (".csv",),
    "scope_note": (".md", ".txt", ".docx", ".pdf"),
    "shot_spec": (".md",),
    "shot_table": (".csv",),
}

# kind: path - a big file, pointed at; file - a small file, path or upload;
# files - a list of small files; text / longtext - a value.
BLOCKS: list[dict[str, Any]] = [
    {"key": "model", "title": "Model database", "folder": True,
     "help": "The split files, usually in one folder - set the folder once, then file names.",
     "fields": {
         "data.paths.train": {"label": "Train", "kind": "path",
                              "help": "Rows the final models fit on"},
         "data.paths.valid": {"label": "Valid", "kind": "path",
                              "help": "Rows for early stopping in the final models"},
         "data.paths.test": {"label": "Test (out of time)", "kind": "path",
                             "help": "The hold-out evaluation scores on; the agent never sees it"},
         "discovery.screen_paths.train": {"label": "Screen fit", "kind": "path",
                                          "help": "Sample the screen fits each feature on"},
         "discovery.screen_paths.valid": {"label": "Screen score", "kind": "path",
                                          "help": "Sample the screen scores each feature on"},
         "data.target": {"label": "Target column", "kind": "text",
                         "help": "What the model predicts"},
         "data.id_cols": {"label": "Id column", "kind": "text",
                          "help": "One row key, used to join features back"},
         "data.id_format": {"label": "Id format", "kind": "text",
                             "help": "How the id encodes the join keys and the as-of date"},
     }},
    {"key": "context", "title": "Context for the agent", "folder": False,
     "help": "What the data means and what to improve. Small files: point at them or upload.",
     "fields": {
         "discovery.task_description": {"label": "Task description", "kind": "longtext",
                                        "help": "What the features should improve"},
         "discovery.task_context_path": {"label": "Task context", "kind": "file",
                                         "upload": "context",
                                         "help": "Background on the model and its use (.md)"},
         "discovery.column_descriptions_path": {"label": "Column descriptions", "kind": "file",
                                                "upload": "context",
                                                "help": "JSON of {column: meaning}"},
     }},
    {"key": "shots", "title": "Shots", "folder": False,
     "help": "Labelled example rows the agent reads, by category - the clustering shots "
             "first, then your own - each rotated by batch across discovery runs or shown "
             "whole.",
     "fields": {
         "discovery.few_shot_path": {"label": "Clustering shots", "kind": "file",
                                     "upload": "context",
                                     "help": "Rows picked per class by KMeans, with a batch "
                                             "column (.csv)"},
         "discovery.shot_spec_paths": {"label": "Your shot categories", "kind": "files",
                                   "upload": "shot_spec",
                                   "help": "One markdown spec per category, appended in order"},
     }},
    {"key": "evaluation", "title": "Evaluation", "folder": False,
     "help": "How chosen features are judged on the full splits: the model each variant "
             "trains, what is reported, and the bar a feature must clear.",
     "fields": {
         "model.num_boost_round": {"label": "Boosting rounds", "kind": "number", "int": True,
                                   "section": "Model", "help": "Upper bound; early stopping ends sooner"},
         "model.early_stopping_rounds": {"label": "Early stopping", "kind": "number", "int": True,
                                         "section": "Model", "help": "Rounds without a valid gain"},
         "model.params.eta": {"label": "Learning rate", "kind": "number", "section": "Model",
                              "help": "XGBoost eta"},
         "model.params.max_depth": {"label": "Max depth", "kind": "number", "int": True,
                                    "section": "Model", "help": "XGBoost max_depth"},
         "model.params.min_child_weight": {"label": "Min child weight", "kind": "number",
                                           "section": "Model", "help": "XGBoost min_child_weight"},
         "model.params.subsample": {"label": "Row subsample", "kind": "number", "section": "Model",
                                    "help": "XGBoost subsample"},
         "model.tuning.enabled": {"label": "Tune hyper-parameters", "kind": "bool",
                                  "section": "Model", "help": "Optuna search before training - slower"},
         "analysis.capture_rate_percents": {"label": "Capture rate at", "kind": "numbers",
                                            "section": "Report",
                                            "help": "Top shares of scores, e.g. 0.10, 0.05, 0.01"},
         "analysis.shap.enabled": {"label": "SHAP ranks", "kind": "bool", "section": "Report",
                                   "help": "Where each new feature ranks in its model"},
         "analysis.shap.sample_size": {"label": "SHAP sample rows", "kind": "number", "int": True,
                                       "section": "Report", "help": "Test rows SHAP is computed on"},
         "verdict.min_gini_gain": {"label": "Min Gini gain", "kind": "number", "section": "Verdict",
                                   "help": "On test, against base, for a PASS"},
         "verdict.max_shap_rank_pct": {"label": "Max SHAP rank", "kind": "number",
                                       "section": "Verdict",
                                       "help": "Share of the ranking a feature must land within"},
         "feature_selection.spearman.redundancy_max_abs": {
             "label": "Redundancy limit |rho|", "kind": "number", "section": "Verdict",
             "help": "Above this a feature duplicates a base column"},
         "run.gates": {"label": "Gates", "kind": "select", "options": ["open", "enforce"],
                       "section": "Verdict",
                       "help": "open: measure every feature; enforce: drop failures before modelling"},
     }},
    {"key": "additional", "title": "Additional data", "folder": False,
     "help": "Sources: data + sample JSON in this folder, or registered by path.",
     "fields": {
         "discovery.additional_data.dir": {"label": "Additional data folder", "kind": "path",
                                       "help": "Holds the sources and, by default, the scopes' variable lists"},
     }},
    {"key": "scope", "title": "Scope", "folder": False,
     "help": "The tables the agent may request data from, by scope - CAS, say - each "
             "flagged by whether the model uses its variables, with your notes on how to "
             "use them. A request outside every scope is 'beyond scope'.",
     "fields": {}},                      # one section per configured scope: _scope_fields
]
# Keys that moved out of the agent section: overrides saved before still load.
MOVED = {"agent.id_format": "data.id_format",
         "agent.shot_spec_paths": "discovery.shot_spec_paths",
         "agent.additional_data_dir": "discovery.additional_data.dir",
         "agent.linkage_dir": "discovery.additional_data.linkage_dir",
         "agent.scope_glob": "discovery.additional_data.scopes.CAS.glob",
         "agent.scope_paths": "discovery.additional_data.scopes.CAS.paths",
         "agent.scope_notes_paths": "discovery.additional_data.scopes.CAS.notes_paths",
         "discovery.additional_data.scope_glob": "discovery.additional_data.scopes.CAS.glob",
         "discovery.additional_data.scope_paths": "discovery.additional_data.scopes.CAS.paths",
         "discovery.additional_data.scope_notes_paths":
             "discovery.additional_data.scopes.CAS.notes_paths"}


def _scope_fields(names: list[str]) -> dict[str, dict[str, Any]]:
    """The Scope block's fields: the same three for each scope, under its keyword."""
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        key = f"discovery.additional_data.scopes.{name}"
        out[f"{key}.glob"] = {"label": "Files in the folder", "kind": "text", "section": name,
                              "help": "Pattern of its variable lists in the additional data folder"}
        out[f"{key}.paths"] = {"label": "More variable lists", "kind": "files",
                               "upload": "scope_file", "section": name,
                               "help": "Flagged exports kept elsewhere (.csv)"}
        out[f"{key}.notes_paths"] = {"label": "Notes", "kind": "files", "upload": "scope_note",
                                     "section": name,
                                     "help": "Your guidance on this scope, for the agent "
                                             "(.md, .txt, .docx, .pdf)"}
    return out


def blocks(cfg: Config) -> list[dict[str, Any]]:
    """The Setup page's blocks for this config - the Scope block has its scopes."""
    names = list(cfg.discovery.additional_data.scopes)
    return [{**b, "fields": _scope_fields(names)} if b["key"] == "scope" else b for b in BLOCKS]


def fields(cfg: Config) -> dict[str, dict[str, Any]]:
    return {k: {**meta, "block": b["key"]} for b in blocks(cfg) for k, meta in b["fields"].items()}


def _lists(fields: dict[str, dict[str, Any]]) -> set[str]:
    return {k for k, meta in fields.items() if meta["kind"] == "files"}


def _files(fields: dict[str, dict[str, Any]]) -> set[str]:
    return {k for k, meta in fields.items() if meta["kind"] in ("path", "file", "files")}


def _coerce(meta: dict[str, Any], value: Any) -> Any:
    """A form value as the config wants it: numbers, flags, lists of numbers."""
    kind = meta["kind"]
    if kind == "number":
        if value in ("", None):
            raise ValueError(f"{meta['label']} needs a number")
        number = float(value)
        return int(number) if meta.get("int") else number
    if kind == "bool":
        return value if isinstance(value, bool) else str(value).lower() in ("true", "1", "yes", "on")
    if kind == "numbers":
        items = value if isinstance(value, list) else re.split(r"[\s,;]+", str(value).strip())
        return [float(v) for v in items if str(v).strip()]
    if kind == "select" and value not in meta["options"]:
        raise ValueError(f"{meta['label']} must be one of {meta['options']}")
    return value


def _get(cfg: Config, dotted: str) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if node is None:
            return ""
        node = node.get(key) if isinstance(node, dict) else getattr(node, key)
    if dotted == "data.id_cols":
        return node[0] if node else ""
    if isinstance(node, (list, tuple)):
        return list(node)
    return node if node is not None else ""


def _set(payload: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for key in keys[:-1]:
        payload = payload.setdefault(key, {})
    payload[keys[-1]] = value


def _text_of(path: Path) -> str:
    """Plain text from a note: as is, or pulled out of a .docx / .pdf."""
    if path.suffix == ".docx":
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf8", errors="replace")
        paragraphs = re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)
        return "\n".join("".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p)) for p in paragraphs)
    if path.suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise ValueError("reading a PDF needs pypdf (pip install pypdf); "
                             "or upload the note as .md or .txt") from error
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(errors="replace")


class Setup:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.base = self._config({})
        self.root = Path(self.base.agent.run_dir)
        self.path = self.root / "workspace_overrides.json"
        self.error: str | None = None

    def overrides(self) -> dict[str, Any]:
        saved = json.loads(self.path.read_text()) if self.path.exists() else {}
        return {MOVED.get(k, k): v for k, v in saved.items()}

    def load(self) -> tuple[Config, Workspace]:
        """The workspace with the saved overrides - or the YAML's, if they no
        longer load (a file moved), with the reason kept for the Setup page."""
        overrides = self.overrides()
        if overrides:
            try:
                cfg = self._config(overrides)
                return cfg, Workspace.from_config(cfg)
            except Exception as error:  # noqa: BLE001 - reported, then fall back
                self.error = f"saved setup no longer loads ({error}); using the config file"
        return self.base, Workspace.from_config(self.base)

    def _config(self, overrides: dict[str, Any]) -> Config:
        payload = yaml.safe_load(open(self.config_path)) or {}
        if any(k.startswith("discovery.additional_data.scopes.") for k in overrides):
            # A scope edited on the page, in a config that names none: start from
            # the default one, so the edit adds to it rather than replacing it.
            extra = payload.setdefault("discovery", {}).setdefault("additional_data", {})
            if "scopes" not in extra:
                extra["scopes"] = {name: dataclasses.asdict(scope) for name, scope
                                   in AdditionalDataConfig().scopes.items()}
        for dotted, value in overrides.items():
            _set(payload, dotted, [value] if dotted == "data.id_cols" else value)
        cfg = Config.from_dict(payload)
        cfg.validate()
        return cfg

    def view(self, cfg: Config) -> dict[str, Any]:
        overrides = self.overrides()
        return {
            "config_file": self.config_path,
            "blocks": [{"key": b["key"], "title": b["title"], "help": b["help"],
                        "folder": b["folder"],
                        "fields": [{"key": k, **{m: v for m, v in meta.items()},
                                    "value": _get(cfg, k), "default": _get(self.base, k),
                                    "changed": k in overrides}
                                   for k, meta in b["fields"].items()]}
                       for b in blocks(cfg)],
            "upload_limit": UPLOAD_LIMIT,
            "error": self.error,
        }

    def apply(self, values: dict[str, Any]) -> tuple[Config, Workspace]:
        """Validate the new settings by loading them; save them only if they load."""
        try:
            known = fields(self._config(self.overrides()))
        except Exception:  # noqa: BLE001 - saved overrides that no longer load
            known = fields(self.base)
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"not setup fields: {unknown}")
        for key in _lists(known) & set(values):
            if not isinstance(values[key], list):
                raise ValueError(f"{key} takes a list of paths")
        values = {k: _coerce(known[k], v) for k, v in values.items()}
        merged = {**self.overrides(), **values}
        overrides = {k: v for k, v in merged.items() if v != _get(self.base, k)}
        cfg = self._config(overrides)
        missing = []
        for key in _files(known) & set(values):
            for path in (_get(cfg, key) if key in _lists(known) else [_get(cfg, key)]):
                if path and not Path(str(path)).exists():
                    missing.append(f"{known[key]['label']}: {path}")
        if missing:
            raise ValueError("not found - " + "; ".join(missing))
        for path in cfg.discovery.shot_spec_paths:
            parse_spec(Path(path).read_text(errors="replace"), Path(path).stem)
        ws = Workspace.from_config(cfg)
        if cfg.data.target not in ws.screen.columns:
            raise ValueError(f"target {cfg.data.target!r} is not in the screen files")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(overrides, indent=2))
        self.error = None
        return cfg, ws

    def reset(self) -> tuple[Config, Workspace]:
        self.path.unlink(missing_ok=True)
        self.error = None
        return self.base, Workspace.from_config(self.base)

    # -------------------------------------------------------------- files
    def save_upload(self, kind: str, filename: str, content: bytes) -> dict[str, Any]:
        """Keep an uploaded small file; return the path a setting can point at."""
        if kind not in UPLOAD_KINDS:
            raise ValueError(f"unknown upload kind {kind!r}")
        name = Path(filename or "").name
        suffix = Path(name).suffix.lower()
        if not name or suffix not in UPLOAD_KINDS[kind]:
            raise ValueError(f"{kind} uploads take {', '.join(UPLOAD_KINDS[kind])}; got {name!r}")
        if len(content) > UPLOAD_LIMIT:
            raise ValueError(f"{name} is {len(content) / 1e6:.1f} MB; uploads are for small files "
                             f"(up to {UPLOAD_LIMIT // 2**20} MB) - point at big files by path")
        folder = self.root / "uploads" / kind
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / name
        target.write_bytes(content)
        if kind == "scope_note" and suffix in (".docx", ".pdf"):
            text = _text_of(target)
            if not text.strip():
                raise ValueError(f"no text could be read from {name}")
            target = target.with_suffix(target.suffix + ".txt")
            target.write_text(text)
        if kind == "source_schema":
            json.loads(target.read_text())          # must parse
        if kind == "shot_spec":
            parse_spec(target.read_text(errors="replace"), target.stem)   # must have ids
        return {"path": str(target), "name": name, "bytes": len(content),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    @staticmethod
    def check(paths: list[str]) -> dict[str, dict[str, Any]]:
        """Whether each path exists, and how big it is - for the form's ticks."""
        out = {}
        for p in paths:
            path = Path(str(p))
            out[str(p)] = {"exists": path.exists(), "dir": path.is_dir(),
                           "bytes": path.stat().st_size if path.is_file() else None}
        return out
