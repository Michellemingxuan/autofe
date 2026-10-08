"""Shots: labelled example rows the agent reads, in categories, rotated by batch.

The first category is the **clustering** shots - representative rows picked
per class with KMeans (``preprocessing/shots.py``), split into batches. They
are the prepare step's file (``discovery.few_shot_path``), or generated here
from the screen's fit rows (:func:`generate_clustering`). After them come the
user's own categories, one markdown file each (``discovery.shot_spec_paths``),
appended in order. A category gives its examples one of two ways - as ids
looked up in the model rows, or as a table of example rows of its own:

    # Early cures                         <- the category's name (else the file name)
    ## Context
    Customers who went 30 days past due and cured within two cycles.
    ## IDs
    - 105131_20240301_B
    - 104444_20240701_A, 104990_20240501_C
    ## Same examples each discovery?
    No - rotate, 4 per batch

or, with its own rows instead of ids:

    # Seasonal spenders
    ## Context
    Customers whose spend peaks in Q4.
    ## Table
    seasonal_spenders.csv                 <- relative to this file, or absolute

The sections are found by their headings, numbered or not: one about the
context, one holding the ids (bullets, commas, spaces or a code block) or a
``Table`` section naming a CSV, one on reuse - "yes" / "same" keeps every id in every run, "no" / "rotate" splits the
ids into batches (the first number in the section is the batch size, 8 if none).

Ids are looked up in the screen's fit rows and the train split only - valid and
test never reach the agent - and the rows are cached by the file's content.

Rotation runs across discovery runs: each run that looks at the shots takes the
next round from a counter under ``agent.run_dir``, and round r shows batch
r mod (number of batches) of every rotating category.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from agent.workspace import Workspace
from validation.data import clean_missing

if TYPE_CHECKING:
    from agent.session import Session

__all__ = ["ShotCategory", "parse_spec", "categories", "generate_clustering", "shots",
           "write_spec", "DEFAULT_BATCH"]

DEFAULT_BATCH = 8
CLUSTERING = "clustering"


@dataclass
class ShotCategory:
    key: str
    kind: str                              # clustering | ids | table
    name: str
    context: str
    rotate: bool
    batch_size: int | None
    path: str | None                       # the spec file; None for the clustering shots
    ids: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    rows: pd.DataFrame | None = field(default=None, repr=False)
    batch_of: list[int] = field(default_factory=list, repr=False)   # per row

    @property
    def batches(self) -> int:
        return max(len(set(self.batch_of)), 1)

    def for_round(self, round_: int) -> pd.DataFrame:
        if self.rows is None or not len(self.rows):
            return pd.DataFrame()
        if not self.rotate:
            return self.rows
        order = sorted(set(self.batch_of))
        pick = order[round_ % len(order)]
        return self.rows[[b == pick for b in self.batch_of]]

    def summary(self, target: str | None = None) -> dict[str, Any]:
        rows = self.rows if self.rows is not None else pd.DataFrame()
        classes: dict[str, int] = {}
        if target and target in rows.columns:
            classes = {str(k): int(v) for k, v in
                       rows[target].value_counts(dropna=False).sort_index().items()}
        per_batch = (pd.Series(self.batch_of).value_counts().sort_index().tolist()
                     if self.batch_of else [len(rows)])
        return {"key": self.key, "kind": self.kind, "name": self.name, "context": self.context,
                "rotate": self.rotate, "batch_size": self.batch_size,
                "batches": self.batches, "per_batch": per_batch, "path": self.path,
                "ids": len(self.ids), "found": len(rows), "columns": int(rows.shape[1]),
                "classes": classes,
                "missing": self.missing[:20], "n_missing": len(self.missing)}


# ------------------------------------------------------------------ parsing
def _heading(line: str) -> str:
    """'## 2. IDs' -> 'ids'."""
    return re.sub(r"^[#\s]*(\d+[.)]\s*)?", "", line).strip().lower()


def parse_spec(text: str, fallback_name: str) -> dict[str, Any]:
    """The name, context, ids and rotation a shot spec asks for."""
    name, sections, current = "", {}, None
    for line in text.splitlines():
        if re.match(r"^#\s", line) and not name:
            name = line[2:].strip()
        elif re.match(r"^#{2,}\s", line):
            current = _heading(line)
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    def find(*words: str) -> str | None:
        for heading, body in sections.items():
            if any(re.search(rf"\b{w}", heading) for w in words):
                return "\n".join(body).strip()
        return None

    context = find("context", "about", "what") or ""
    table = None
    table_text = find("table", "examples? file", "csv")
    if table_text:
        paths = re.findall(r"[^\s`'\"]+\.csv", table_text)
        if not paths:
            raise ValueError(f"{fallback_name}: the Table section names no .csv file")
        table = paths[0]
    id_text = "" if table else (find("ids?", "identifiers", "rows") or "")
    id_text = re.sub(r"```\w*", " ", id_text)
    ids = [t for t in re.split(r"[\s,;|`]+", id_text)
           if t and not re.fullmatch(r"[-*+]|\d+[.)]", t)]
    if not ids and not table:
        raise ValueError(f"{fallback_name}: no examples found - list ids under '## IDs' "
                         "(one per line or comma-separated), or name a CSV under '## Table'")
    reuse = (find("same", "reuse", "rotat", "batch", "each discovery") or "yes").lower()
    rotate = "rotat" in reuse or bool(re.match(r"^\W*no\b", reuse))
    size = re.search(r"\d+", reuse)
    return {"name": name or fallback_name, "context": context, "ids": list(dict.fromkeys(ids)),
            "table": table, "rotate": rotate,
            "batch_size": (int(size.group()) if size else DEFAULT_BATCH) if rotate else None}


# ------------------------------------------------------------------- lookup
def _shot_columns(ws: Workspace) -> list[str]:
    return [ws.id_col, ws.target, *ws.base_features]


def _lookup(ws: Workspace, ids: list[str]) -> pd.DataFrame:
    """The rows for these ids from the screen's fit rows, then the train split."""
    want = set(ids)
    fit = ws.screen.iloc[: ws.n_screen_train]
    found = [fit[fit[ws.id_col].astype(str).isin(want)]]
    want -= set(found[0][ws.id_col].astype(str))
    train = ws.cfg.data.paths.get("train")
    if want and train and Path(train).exists():
        cols = _shot_columns(ws)
        if str(train).endswith(".parquet"):
            part = pd.read_parquet(train, columns=cols)
            found.append(part[part[ws.id_col].astype(str).isin(want)])
        else:
            for chunk in pd.read_csv(train, usecols=lambda c: c in cols, chunksize=200_000):
                hit = chunk[chunk[ws.id_col].astype(str).isin(want)]
                if len(hit):
                    found.append(hit)
                    want -= set(hit[ws.id_col].astype(str))
                if not want:
                    break
        found[1:] = [clean_missing(f, ws.base_features, ws.cfg.data.missing_values)
                     for f in found[1:]]
    rows = pd.concat(found, ignore_index=True).drop_duplicates(ws.id_col)
    rows[ws.id_col] = rows[ws.id_col].astype(str)
    order = {i: n for n, i in enumerate(ids)}
    return rows.sort_values(ws.id_col, key=lambda s: s.map(order)).reset_index(drop=True)


def _cache_dir(ws: Workspace) -> Path:
    return Path(ws.cfg.agent.run_dir) / "shots"


def _from_table(ws: Workspace, path: Path, spec: dict[str, Any]) -> ShotCategory:
    """A category that brings its own example rows."""
    # As written (absolute, or from the project root - where uploads land),
    # else beside the spec file.
    given = Path(spec["table"])
    table = next((t for t in (given, path.parent / given) if t.is_file()), None)
    if table is None:
        raise ValueError(f"{path.name}: its table {given} is not there")
    rows = pd.read_csv(table)
    held_out: list[str] = []
    if ws.id_col in rows.columns:
        # Rows the screen scores on are held out from the agent; a table may
        # not bring them in by the back door.
        rows[ws.id_col] = rows[ws.id_col].astype(str)
        scored = set(ws.screen.iloc[ws.n_screen_train:][ws.id_col].astype(str))
        held_out = [i for i in rows[ws.id_col] if i in scored]
        rows = rows[~rows[ws.id_col].isin(scored)].reset_index(drop=True)
    size = spec["batch_size"] or 1
    return ShotCategory(
        key=path.stem, kind="table", name=spec["name"], context=spec["context"],
        rotate=spec["rotate"], batch_size=spec["batch_size"], path=str(path),
        ids=rows[ws.id_col].tolist() if ws.id_col in rows.columns else [], rows=rows,
        missing=[f"{i} (held out)" for i in held_out],
        batch_of=[n // size if spec["rotate"] else 0 for n in range(len(rows))])


def _from_spec(ws: Workspace, path: Path) -> ShotCategory:
    text = path.read_text(errors="replace")
    spec = parse_spec(text, path.stem)
    if spec["table"]:
        return _from_table(ws, path, spec)
    train = str(ws.cfg.data.paths.get("train", ""))
    stamp = Path(train).stat().st_mtime if train and Path(train).exists() else 0
    key = "|".join([text, train, str(stamp), *_shot_columns(ws)])
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    cache = _cache_dir(ws) / f"{path.stem}_{digest}.csv"
    if cache.exists():
        rows = pd.read_csv(cache, dtype={ws.id_col: str})
    else:
        rows = _lookup(ws, spec["ids"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(cache, index=False)
    found = rows[ws.id_col].tolist()
    position = {i: n for n, i in enumerate(i for i in spec["ids"] if i in set(found))}
    size = spec["batch_size"] or 1
    return ShotCategory(
        key=path.stem, kind="ids", name=spec["name"], context=spec["context"], rotate=spec["rotate"],
        batch_size=spec["batch_size"], path=str(path), ids=spec["ids"],
        missing=[i for i in spec["ids"] if i not in set(found)], rows=rows,
        batch_of=[position[i] // size if spec["rotate"] else 0 for i in found])


def _clustering(ws: Workspace) -> ShotCategory | None:
    if ws.shots is None or not len(ws.shots):
        return None
    batch = ws.shots["batch"].tolist() if "batch" in ws.shots else [0] * len(ws.shots)
    rows = ws.shots.drop(columns=["batch"], errors="ignore")
    return ShotCategory(
        key=CLUSTERING, kind="clustering", name="Clustering shots",
        context="Representative rows picked per class by KMeans in the prepare step - "
                "one per cluster, so they cover the table and every class appears.",
        rotate=len(set(batch)) > 1, batch_size=None, path=ws.cfg.discovery.few_shot_path,
        ids=rows[ws.id_col].astype(str).tolist(), rows=rows, batch_of=batch)


def generate_clustering(ws: Workspace, shots: int = 32, batches: int = 4,
                        seed: int = 42) -> Path:
    """Pick representative rows per class from the screen's fit rows, in batches.

    One row per KMeans cluster, the clusters split evenly over the classes, so
    every class appears and the rows cover the table; batch b takes each
    cluster's b-th closest row, so successive runs see different rows from the
    same regions. Written as a CSV with a ``batch`` column - the same format the
    prepare step writes - for ``discovery.few_shot_path`` to point at.
    """
    from preprocessing.shots import build_shot_batches_fast

    if shots < 2 or batches < 1:
        raise ValueError("generate at least 2 shots in at least 1 batch")
    fit = ws.screen.iloc[: ws.n_screen_train]
    parts = build_shot_batches_fast(fit, ws.target, columns=ws.base_features,
                                    shots=shots, batches=batches, seed=seed)
    rows = pd.concat([p.assign(batch=b) for b, p in enumerate(parts)], ignore_index=True)
    path = _cache_dir(ws) / f"clustering_{shots}x{batches}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows[[*_shot_columns(ws), "batch"]].to_csv(path, index=False)
    return path


def write_spec(folder: Path, name: str, context: str, *, ids: list[str] | None = None,
               table: str | None = None, rotate: bool = False,
               batch_size: int | None = None) -> Path:
    """Write a category's markdown spec from a form - ids, or a table of rows."""
    if bool(ids) == bool(table):
        raise ValueError("give the examples as ids or as a table - one of the two")
    slug = re.sub(r"\W+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("the category needs a name")
    lines = [f"# {name.strip()}", "", "## Context", context.strip() or "(none given)", ""]
    if table:
        lines += ["## Table", str(table), ""]
    else:
        lines += ["## IDs", *[f"- {i}" for i in ids or []], ""]
    lines += ["## Same examples each discovery?",
              f"No - rotate, {batch_size or DEFAULT_BATCH} per batch" if rotate else "Yes", ""]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slug}.md"
    path.write_text("\n".join(lines))
    parse_spec(path.read_text(), slug)                  # what the agent will read
    return path


def categories(ws: Workspace) -> list[ShotCategory]:
    """The clustering shots first, then each spec file in order."""
    out = [c for c in [_clustering(ws)] if c is not None]
    for p in ws.cfg.discovery.shot_spec_paths:
        if not Path(p).is_file():
            continue
        try:
            out.append(_from_spec(ws, Path(p)))
        except Exception as error:  # noqa: BLE001 - one bad file must not hide the rest
            out.append(ShotCategory(key=Path(p).stem, kind="error", name=Path(p).stem,
                                    context=str(error), rotate=False, batch_size=None,
                                    path=str(p)))
    return out


# ----------------------------------------------------------------- rotation
def _next_round(ws: Workspace) -> int:
    counter = _cache_dir(ws) / "rotation.json"
    current = json.loads(counter.read_text())["next"] if counter.exists() else 0
    counter.parent.mkdir(parents=True, exist_ok=True)
    counter.write_text(json.dumps({"next": current + 1}))
    return current


def round_of(session: "Session") -> int:
    """This run's round, taken from the counter the first time it is needed."""
    if getattr(session, "shot_round", None) is None:
        session.shot_round = _next_round(session.ws)
        session.emit("shots_round", round=session.shot_round)
    return session.shot_round


# --------------------------------------------------------------------- tool
def shots(session: "Session", category: str = "") -> str:
    """No category: what categories there are. A category: its rows for this run."""
    cats = [c for c in categories(session.ws) if c.kind != "error"]
    if not cats:
        return "no shots are set up; sample_rows('model_database') shows a few screen rows"
    r = round_of(session)
    if not category.strip():
        lines = [f"Shot categories (this run is rotation round {r}):"]
        for c in cats:
            mode = (f"rotating, batch {r % c.batches + 1} of {c.batches}" if c.rotate
                    else "the same rows every run")
            lines.append(f"* `{c.key}` - {c.name}: {len(c.for_round(r))} rows, {mode}. "
                         f"{c.context.splitlines()[0] if c.context else ''}")
        lines.append("Call shots(category) to see one category's rows.")
        return "\n".join(lines)
    wanted = category.strip().lower()
    match = next((c for c in cats if wanted in (c.key.lower(), c.name.lower())), None)
    if match is None:
        return f"no shot category {category!r}; categories: {[c.key for c in cats]}"
    rows = match.for_round(r)
    head = f"## {match.name}\n{match.context}\n\n" if match.context else f"## {match.name}\n\n"
    return head + rows.to_string(index=False, max_colwidth=30)
