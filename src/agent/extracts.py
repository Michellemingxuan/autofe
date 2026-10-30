"""Discovery's sample of a source: the screen customers' rows, cut once and cached.

Discovery builds features only for the screen ids, so it needs only their
customers' events - a small share of a long source. The first time a confirmed
linkage reads the source, the same pass keeps every row whose key the screen ids
link to (``agent._child``), and that cut is saved here. From then on discovery's
linkage, aggregates and features read the cut instead of the whole file; each
customer's history in it is complete, so a feature's screen values are the ones
evaluation computes on the full source.

A cut stays valid while the file it came from, the screen ids, and the
linkage's key are unchanged. Evaluation and inference never read it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

__all__ = ["linkage_key", "fingerprint", "cached_extract", "save_extract"]


def linkage_key(code: str) -> list[str]:
    """The source column(s) a linkage joins on: from its header, else from its merge
    (`on=` / `right_on=`); empty when neither says."""
    header = re.search(r"^# linkage for .*, key=(.+)$", code, flags=re.M)
    if header:
        return [k for k in header.group(1).split(",") if k]
    merge = re.search(r"""\b(?:right_on|on)\s*=\s*\[?\s*['"]([^'"]+)['"]""", code)
    return [merge.group(1)] if merge else []


def _folder(ws: Any) -> Path:
    return Path(ws.cfg.agent.run_dir) / "extracts"


def fingerprint(ws: Any, source_path: str, key: list[str]) -> dict[str, Any]:
    """What a cut depends on: the file it came from, the screen ids, the key."""
    path = Path(source_path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    ids = "\n".join(sorted(map(str, ws.screen[ws.id_col].unique())))
    return {"source_path": str(path.resolve()),
            "source_size": sum(p.stat().st_size for p in files),
            "source_mtime": max((p.stat().st_mtime for p in files), default=0.0),
            "screen": hashlib.sha1(ids.encode()).hexdigest(), "key": list(key)}


def cached_extract(ws: Any, source: str, source_path: str, key: list[str]) -> str | None:
    """The saved cut of a source, when it is still valid."""
    folder = _folder(ws)
    meta, data = folder / f"{source}.json", folder / f"{source}.parquet"
    if not key or not meta.exists() or not data.exists():
        return None
    saved = json.loads(meta.read_text())
    return str(data) if saved.get("fingerprint") == fingerprint(ws, source_path, key) else None


def save_extract(ws: Any, source: str, made: str, source_path: str, key: list[str],
                 stats: dict[str, Any]) -> str:
    """Keep a cut a linkage run made, for every later discovery run."""
    folder = _folder(ws)
    folder.mkdir(parents=True, exist_ok=True)
    data = folder / f"{source}.parquet"
    shutil.copyfile(made, data)
    (folder / f"{source}.json").write_text(json.dumps(
        {"fingerprint": fingerprint(ws, source_path, key), **stats}, indent=2))
    return str(data)
