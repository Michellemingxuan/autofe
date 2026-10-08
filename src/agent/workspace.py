"""Everything the agent can see about one use case, read from its config.

The model database is seen only through the screen's rows (fit on
screen_train, score on screen_valid) and the few-shot rows; valid and test
never reach the agent. Around it sit the cached sources in
``discovery.additional_data.dir``, the scope files beside them (each scope by its
keyword - CAS, say), and the linkage
confirmed for each source in ``discovery.additional_data.linkage_dir``.

A source is a data file with a sample JSON next to it:

    spends.parquet + spends_data_sample.json     (or spends.json)

The JSON is {column: [description, [sample values]]} - the format the real
extracts already use. A JSON with no data file is listed as *schema only*: the
agent can read about it, and propose an L3 pull for it, but not compute on it.
Sources are rescanned on every call, so a file the user drops in mid-run is
seen from the next step on.

A source can also be *registered* where it lies - a 20 GB extract need not be
copied: ``sources.json`` in the same folder maps a name to its data file and
its sample JSON, and a registered name takes precedence over a scanned one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from validation.metrics import calc_adj_gini, capture_rate
from discovery.screen import Screener
from validation.config import Config
from validation.data import clean_missing

__all__ = ["Source", "Workspace"]

_DATA_SUFFIXES = (".parquet", ".csv")
# What marks a scope column as identifying a customer, account or card.
_IDENTIFIER_NAME = r"^(?:cm1[15]|cm\d{2}|cust(?:omer)?_?id\w*|acct_?(?:id|nbr|num)\w*|card_?(?:nbr|num)\w*)$"
_IDENTIFIER_TEXT = r"card ?member number|customer id|account number|card number"
_SAMPLE_SUFFIXES = ("_data_sample.json", ".json")


@dataclass
class Source:
    name: str
    schema_path: Path
    data_path: Path | None
    columns: dict[str, str]                 # column -> description
    samples: dict[str, list[Any]]

    @property
    def usable(self) -> bool:
        return self.data_path is not None

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "usable": self.usable,
                "data": str(self.data_path) if self.data_path else None,
                "columns": self.columns}


def _count_rows(path: Path) -> int | None:
    """Parquet from its metadata; a CSV by its line breaks, read in blocks."""
    try:
        if path.is_dir() or path.suffix == ".parquet":
            import pyarrow.dataset as ds

            return int(ds.dataset(path, format="parquet").count_rows())
        with open(path, "rb") as fh:
            lines = sum(block.count(b"\n") for block in iter(lambda: fh.read(1 << 24), b""))
        return max(lines - 1, 0)                     # the header
    except Exception:  # noqa: BLE001 - a size is a hint, never a failure
        return None


def _disk_bytes(path: Path) -> int:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size if path.exists() else 0


def _examples(values: list[Any], n: int = 5) -> list[Any]:
    """A few distinct values, in order - what a column holds, at a glance."""
    out: list[Any] = []
    for v in values:
        v = round(v, 4) if isinstance(v, float) else v
        if v not in out:
            out.append(v)
        if len(out) == n:
            break
    return out


def _read_sample_json(path: Path) -> tuple[dict[str, str], dict[str, list[Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns, samples = {}, {}
    for col, entry in payload.items():
        if isinstance(entry, list) and len(entry) == 2 and isinstance(entry[1], list):
            columns[col], samples[col] = str(entry[0]), entry[1]
        else:
            columns[col], samples[col] = str(entry), []
    return columns, samples


@dataclass
class Workspace:
    cfg: Config
    base_features: list[str]
    screen: pd.DataFrame                    # id, target, base; train rows first
    n_screen_train: int
    descriptions: dict[str, str]
    task_context: str
    shots: pd.DataFrame | None
    _screener: Screener | None = field(default=None, repr=False)
    # The scopes as last read, and each table's profile - kept until a scope
    # file changes. A real scope is thousands of variables, asked about per column.
    _scope_cache: tuple[Any, pd.DataFrame] | None = field(default=None, repr=False)
    _size_cache: dict[tuple[str, int], int | None] = field(default_factory=dict, repr=False)
    _profiles: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_config(cls, cfg: Config) -> "Workspace":
        d, data = cfg.discovery, cfg.data
        if not data.id_cols:
            raise ValueError("the agent needs data.id_cols: features are joined back by id")
        if sorted(d.screen_paths) != ["train", "valid"]:
            raise ValueError("the agent screens on discovery.screen_paths {train, valid}")
        id_col = data.id_cols[0]

        parts = [pd.read_csv(d.screen_paths[p]) if d.screen_paths[p].endswith(".csv")
                 else pd.read_parquet(d.screen_paths[p]) for p in ("train", "valid")]
        if set(parts[0][id_col]) & set(parts[1][id_col]):
            raise ValueError("screen_train and screen_valid share rows")
        screen = pd.concat(parts, ignore_index=True)

        base = list(cfg.features.base) or [
            c for c in screen.select_dtypes("number").columns
            if c not in (data.target, *data.id_cols, *cfg.features.exclude)]
        screen = clean_missing(screen[[id_col, data.target, *base]], base, data.missing_values)

        descriptions = dict(d.column_descriptions)
        if d.column_descriptions_path and Path(d.column_descriptions_path).exists():
            descriptions.update(json.loads(Path(d.column_descriptions_path).read_text()))
        context = d.task_context
        if d.task_context_path and Path(d.task_context_path).exists():
            context = "\n\n".join(filter(None, [context, Path(d.task_context_path).read_text()]))
        shots = pd.read_csv(d.few_shot_path) if d.few_shot_path and Path(d.few_shot_path).exists() else None

        return cls(cfg=cfg, base_features=base, screen=screen,
                   n_screen_train=len(parts[0]), descriptions=descriptions,
                   task_context=context, shots=shots)

    @property
    def id_col(self) -> str:
        return self.cfg.data.id_cols[0]

    @property
    def target(self) -> str:
        return self.cfg.data.target

    @property
    def extra_dir(self) -> Path | None:
        d = self.cfg.discovery.additional_data.dir
        return Path(d) if d else None

    @property
    def linkage_dir(self) -> Path:
        return Path(self.cfg.discovery.additional_data.linkage_dir or "linkage")

    # ---------------------------------------------------------------- sizes
    def rows_of(self, path: str | Path) -> int | None:
        """Rows in a data file or parquet folder - counted once per version of it."""
        path = Path(path)
        try:
            key = (str(path), path.stat().st_mtime_ns)
        except OSError:
            return None
        if key not in self._size_cache:
            self._size_cache[key] = _count_rows(path)
        return self._size_cache[key]

    def data_size(self) -> dict[str, Any]:
        """What the agent's code meets outside the screen: the full splits it runs
        on at evaluation, and each usable source as a whole."""
        full = [self.rows_of(p) for p in self.cfg.data.paths.values()]
        return {"full_rows": sum(full) if all(r is not None for r in full) else None,
                "sources": {n: {"rows": self.rows_of(s.data_path),
                                "bytes": _disk_bytes(s.data_path)}
                            for n, s in self.sources().items() if s.usable}}

    # ---------------------------------------------------------------- sources
    def sources(self) -> dict[str, Source]:
        """Rescan the folder: every sample JSON is a source, usable if its data is there."""
        if not self.extra_dir or not self.extra_dir.exists():
            return {}
        found: dict[str, Source] = {}
        for path in sorted(self.extra_dir.glob("*.json")):
            if path.name == "sources.json":            # the registry, not a source
                continue
            name = path.name
            for suffix in _SAMPLE_SUFFIXES:
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            if name in found:
                continue
            data = next((self.extra_dir / f"{name}{s}" for s in _DATA_SUFFIXES
                         if (self.extra_dir / f"{name}{s}").exists()), None)
            columns, samples = _read_sample_json(path)
            found[name] = Source(name, path, data, columns, samples)
        for name, entry in self._registry().items():
            schema = Path(entry["schema"])
            if not schema.exists():
                continue
            data = Path(entry["data"]) if entry.get("data") else None
            columns, samples = _read_sample_json(schema)
            found[name] = Source(name, schema, data if data and data.exists() else None,
                                 columns, samples)
        return found

    # ------------------------------------------------------------ registry
    def _registry_path(self) -> Path:
        if not self.extra_dir:
            raise ValueError("set discovery.additional_data.dir first")
        return self.extra_dir / "sources.json"

    def _registry(self) -> dict[str, dict[str, Any]]:
        if not self.extra_dir:
            return {}
        path = self.extra_dir / "sources.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def register_source(self, name: str, schema: str, data: str | None = None) -> Source:
        """Add a source by path: its sample JSON (the description) and its data."""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
            raise ValueError(f"{name!r} is not a valid source name (letters, digits, _)")
        if not Path(schema).is_file():
            raise ValueError(f"no description file at {schema}")
        _read_sample_json(Path(schema))                     # must parse
        if data and not Path(data).exists():
            raise ValueError(f"no data at {data}")
        if data and Path(data).suffix not in _DATA_SUFFIXES and not Path(data).is_dir():
            raise ValueError(f"data must be parquet or csv, got {data}")
        registry = self._registry()
        registry[name] = {"schema": str(schema), "data": str(data) if data else None}
        self.extra_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path().write_text(json.dumps(registry, indent=2))
        return self.sources()[name]

    def unregister_source(self, name: str) -> bool:
        """Forget a registered source. Never touches its files."""
        registry = self._registry()
        if registry.pop(name, None) is None:
            return False
        self._registry_path().write_text(json.dumps(registry, indent=2))
        return True

    def registered(self) -> set[str]:
        return set(self._registry())

    def linkage_path(self, source: str) -> Path:
        return self.linkage_dir / f"{source}.py"

    def linked(self) -> list[str]:
        return [n for n in self.sources() if self.linkage_path(n).exists()]

    def linkage_code(self, source: str) -> str | None:
        path = self.linkage_path(source)
        return path.read_text() if path.exists() else None

    # ------------------------------------------------------------------ scope
    @property
    def scopes(self) -> dict[str, Any]:
        """The scopes data may be requested from, by keyword."""
        return self.cfg.discovery.additional_data.scopes

    def scope_files(self) -> list[tuple[str, Path]]:
        """Each scope's variable lists, as (scope, file): its pattern in the folder,
        then any listed. A file claimed by two scopes stays with the first."""
        out: dict[Path, str] = {}
        for name, sc in self.scopes.items():
            found = sorted(self.extra_dir.glob(sc.glob)) if self.extra_dir and sc.glob else []
            for path in [*found, *(Path(p) for p in sc.paths if Path(p).is_file())]:
                out.setdefault(path, name)
        return [(name, path) for path, name in out.items()]

    def scope_notes(self) -> list[tuple[str, str]]:
        """The user's guidance documents on each scope, as (scope: file name, text)."""
        return [(f"{name}: {Path(p).name}", Path(p).read_text(errors="replace"))
                for name, sc in self.scopes.items()
                for p in sc.notes_paths if Path(p).is_file()]

    def scope_of(self, table: str) -> str | None:
        """The scope a table belongs to, by keyword."""
        scope = self.scope()
        if not len(scope):
            return None
        rows = scope[scope["table"].astype(str).str.lower() == table.lower()]
        return str(rows["scope"].iloc[0]) if len(rows) else None

    def scope(self) -> pd.DataFrame:
        """Every scope's variables, one row each, with the scope's keyword and where
        each stands against the model. Read once, and again only when a scope file
        is added, removed or changed."""
        files = self.scope_files()
        key = tuple((name, str(p), p.stat().st_mtime_ns) for name, p in files)
        if self._scope_cache is None or self._scope_cache[0] != key:
            self._scope_cache = (key, self._read_scope(files))
            self._profiles = {}
        return self._scope_cache[1]

    def _read_scope(self, files: list[tuple[str, Path]]) -> pd.DataFrame:
        frames = [pd.read_csv(p).assign(_scope=name) for name, p in files]
        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)
        flag_col = next((c for c in raw.columns if c.upper().endswith("_FLAG")), None)
        matched_col = next((c for c in raw.columns if c.upper().endswith("_MATCHED_VARIABLE")), None)
        flag = raw[flag_col].fillna("").str.upper() if flag_col else pd.Series("", index=raw.index)
        status = pd.Series("unused_raw", index=raw.index)
        status[flag.str.startswith("USED")] = "in_model"
        status[flag.str.startswith("UNUSED")] = "in_model_unused"
        def flag(column: str) -> pd.Series:
            if column not in raw:
                return pd.Series(False, index=raw.index)
            return raw[column].fillna("").astype(str).str.strip().str.lower().isin(("yes", "y", "true", "1"))

        return pd.DataFrame({
            "scope": raw["_scope"],
            "variable": raw["NAME"],
            "table": raw.get("TABLE NAME"),
            "description": raw.get("DESCRIPTION"),
            "type": raw.get("TYPE"),
            "status": status,
            "model_variable": raw[matched_col] if matched_col else None,
            "primary": flag("PRIMARY"),
            "partition": flag("PARTITION"),
        })

    def table_profile(self, table: str) -> dict[str, Any]:
        """A scope table's columns, its partition date, and the columns that identify
        a customer, account or card - what a pull must select to be linked later."""
        scope = self.scope()
        if table.lower() in self._profiles:
            return self._profiles[table.lower()]
        self._profiles[table.lower()] = profile = self._profile(scope, table)
        return profile

    def _profile(self, scope: pd.DataFrame, table: str) -> dict[str, Any]:
        rows = scope[scope["table"].astype(str).str.lower() == table.lower()] if len(scope) else scope
        if not len(rows):
            return {"table": table, "columns": [], "partition": [], "identifiers": []}
        ident = rows["variable"].astype(str).str.lower().str.contains(_IDENTIFIER_NAME) | \
            rows["description"].astype(str).str.lower().str.contains(_IDENTIFIER_TEXT)
        return {"table": str(rows["table"].iloc[0]),
                "columns": rows["variable"].astype(str).tolist(),
                "partition": rows.loc[rows["partition"], "variable"].astype(str).tolist(),
                "identifiers": rows.loc[ident & ~rows["partition"], "variable"].astype(str).tolist()}

    # ---------------------------------------------------------------- catalog
    def catalog(self, query: str = "", limit: int = 25) -> dict[str, Any]:
        """Search model columns, sources and the scopes by keywords.

        A model or source column comes with a few example values, so one search
        both finds a column and shows what it holds - a category or a country, a
        flag or an amount - without a second call for rows. Terms are matched
        separately, so one query can look for several things at once.
        """
        sources = self.sources()
        scope = self.scope()
        if not query.strip():
            return {
                "model_database": {"rows_fit": self.n_screen_train,
                                   "rows_scored": len(self.screen) - self.n_screen_train,
                                   "id_column": self.id_col,
                                   "id_format": self.cfg.data.id_format,
                                   "base_features": len(self.base_features)},
                "sources": [{"name": s.name, "usable": s.usable,
                             "linkage_confirmed": self.linkage_path(s.name).exists(),
                             "columns": list(s.columns)} for s in sources.values()],
                "scopes": {
                    "what": "Tables you may request data from, by scope. They have no rows "
                            "here - list their variables with scope(); get their data with "
                            "an L3 request. Data outside every scope is 'beyond scope'.",
                    **{name: {"description": sc.description, "sql": sc.sql_dialect,
                              "tables": ({t: {**g["status"].value_counts().to_dict(),
                                              **{k: v for k, v in self.table_profile(str(t)).items()
                                                 if k in ("partition", "identifiers")}}
                                          for t, g in scope[scope["scope"] == name].groupby("table")}
                                         if len(scope) else {})}
                       for name, sc in self.scopes.items()},
                    "status_meaning": {"in_model": "already used by the model",
                                       "in_model_unused": "in the model's inputs, no importance",
                                       "unused_raw": "not used by the model - the room you have"},
                },
                "scope_notes": [name for name, _ in self.scope_notes()],
            }
        terms = [t for t in re.split(r"\W+", query.lower()) if t]

        def hits(*texts: Any) -> int:
            blob = " ".join(str(t) for t in texts if t is not None).lower()
            return sum(t in blob for t in terms)

        found: list[tuple[int, dict[str, Any]]] = []
        fit = self.screen.iloc[: self.n_screen_train]
        for col in self.base_features:
            found.append((hits(col, self.descriptions.get(col)),
                          {"where": "model_database", "column": col,
                           "description": self.descriptions.get(col, ""),
                           "examples": _examples(fit[col].dropna().head(200).tolist())}))
        for s in sources.values():
            for col, text in s.columns.items():
                found.append((hits(s.name, col, text),
                              {"where": f"source:{s.name}", "column": col, "description": text,
                               "examples": _examples(s.samples.get(col, []))}))
        for row in scope.itertuples(index=False):
            found.append((hits(row.variable, row.table, row.description, row.status),
                          {"where": f"scope:{row.scope}:{row.table}", "column": row.variable,
                           "description": row.description, "status": row.status}))
        found = [f for f in found if f[0] > 0]
        found.sort(key=lambda f: -f[0])
        return {"query": query, "matches": [f[1] for f in found[:limit]],
                "total_matches": len(found)}

    def scope_variables(self, query: str = "", status: str = "", table: str = "",
                        limit: int = 60, scope_name: str = "") -> dict[str, Any]:
        """The scopes' variables, filtered by scope, status, table and keywords."""
        scope = self.scope()
        if not len(scope):
            return {"variables": [], "total": 0, "note": "no scope files are set up"}
        rows = scope
        if scope_name.strip():
            rows = rows[rows["scope"].astype(str).str.lower() == scope_name.strip().lower()]
        if status.strip():
            rows = rows[rows["status"] == status.strip()]
        if table.strip():
            rows = rows[rows["table"].astype(str).str.lower() == table.strip().lower()]
        terms = [t for t in re.split(r"\W+", query.lower()) if t]
        if terms:
            blob = (rows["variable"].astype(str) + " " + rows["description"].astype(str)).str.lower()
            rows = rows[blob.apply(lambda b: any(t in b for t in terms))]
        profiles = {t: self.table_profile(str(t)) for t in set(rows["table"].astype(str))}

        def role(r: Any) -> dict[str, str]:
            p = profiles.get(str(r.table), {})
            if r.variable in p.get("partition", []):
                return {"role": "partition date - filter on it"}
            if r.variable in p.get("identifiers", []):
                return {"role": "identifier - select it to link the result"}
            if r.primary:
                return {"role": "row key"}
            return {}

        out = [{"scope": r.scope, "variable": r.variable, "table": r.table, "type": r.type,
                "status": r.status,
                "description": r.description, **role(r),
                **({"model_variable": r.model_variable} if isinstance(r.model_variable, str) else {})}
               for r in rows.head(limit).itertuples(index=False)]
        tables = sorted(set(rows["table"].astype(str))) if len(rows) else []
        keys = {t: {k: v for k, v in self.table_profile(t).items() if k in ("partition", "identifiers")}
                for t in tables[:5]}
        return {"variables": out, "total": int(len(rows)), "table_keys": keys,
                **({"note": f"showing {limit} of {len(rows)} - narrow with query, status or table"}
                   if len(rows) > limit else {})}

    def sample_rows(self, source: str = "model_database", n: int = 5,
                    columns: list[str] | None = None) -> str:
        """A few rows, as text. The model database's are the screen's fit rows,
        labelled; curated examples are the shots (``agent.tools.shots``)."""
        if source == "model_database":
            frame = self.screen.iloc[: self.n_screen_train].head(n)
        else:
            src = self.sources().get(source)
            if src is None:
                raise KeyError(f"no source {source!r}; known: {sorted(self.sources())}")
            if src.data_path is None:
                frame = pd.DataFrame(src.samples).head(n)
            elif src.data_path.suffix == ".parquet":
                frame = pd.read_parquet(src.data_path).head(n)
            else:
                frame = pd.read_csv(src.data_path, nrows=n)
        if columns:
            frame = frame[[c for c in columns if c in frame.columns]]
        return frame.to_string(index=False, max_colwidth=40)

    # ----------------------------------------------------------------- screen
    @property
    def screener(self) -> Screener:
        """Fit the baseline once, on first use.

        It scores Gini and, off the same fits, capture rate at the top
        ``discovery.capture_percent`` - the two gains the analyst gates on.
        """
        if self._screener is None:
            d = self.cfg.discovery
            pct = d.capture_percent
            redundancy = d.redundancy_max_abs
            if redundancy is None and self.cfg.feature_selection.enabled \
                    and self.cfg.feature_selection.spearman.enabled:
                redundancy = self.cfg.feature_selection.spearman.redundancy_max_abs
            cols = [self.target, *self.base_features]
            self._screener = Screener(
                self.screen.iloc[: self.n_screen_train][cols],
                self.screen.iloc[self.n_screen_train:][cols],
                self.target, self.base_features, self.cfg.model.params,
                score=calc_adj_gini, num_boost_round=d.screen_boost_rounds,
                also_score={"capture": lambda df, y, p: capture_rate(df, y, p, pct)},
                nthread=self.cfg.model.threads_per_model or 1,
                spike_factor=d.spike_factor, redundancy_max_abs=redundancy)
        return self._screener

    def write_base(self, path: Path) -> Path:
        """The model rows a script may read: ids and base columns, never the target."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.screen.drop(columns=[self.target]).to_parquet(path, index=False)
        return path
