"""Run an agent-written script in a subprocess, with a timeout and a guard.

The guard is a reading check, not a sandbox: it refuses scripts that reach for
files, the shell or dynamic imports, so the only data a script sees is what the
runner hands it (see ``agent._child``). That is what keeps two rules true -
valid and test never reach the agent, and an extra source only reaches a
feature through its confirmed point-in-time linkage.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["CodeResult", "check_code", "run_code"]

_FORBIDDEN = [
    (r"\bread_(csv|parquet|json|excel|table|pickle|feather|sql)\b", "reads a file directly"),
    (r"\bspark\.read\b", "reads a file directly"),
    (r"\bopen\s*\(", "opens a file"),
    (r"^\s*(import|from)\s+(os|sys|subprocess|shutil|pathlib|glob|socket|requests|urllib)\b",
     "imports a module that reaches outside the script"),
    (r"\b(__import__|eval|exec|compile)\s*\(", "runs dynamic code"),
    (r"\bto_(csv|parquet|pickle)\s*\(|\.write\b", "writes a file"),
]


def check_code(code: str) -> str | None:
    """The first rule the script breaks, in words, or None."""
    for pattern, why in _FORBIDDEN:
        if re.search(pattern, code, flags=re.MULTILINE):
            return (f"the script {why} (`{pattern}`); use the frames it is given - "
                    "base, sources, raw - and nothing else")
    return None


@dataclass
class CodeResult:
    ok: bool
    mode: str
    stdout: str = ""
    error: str | None = None
    result: dict[str, Any] | None = None
    feature: str | None = None
    elapsed_s: float = 0.0
    out_path: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def for_agent(self) -> dict[str, Any]:
        out = {"ok": self.ok, "elapsed_s": self.elapsed_s}
        for key in ("stdout", "error", "result", "feature"):
            value = getattr(self, key)
            if value:
                out[key] = value
        out.update(self.extras)
        return out


def run_code(code: str, mode: str, *, workdir: Path, engine: str, id_col: str,
             base_path: Path, sources: dict[str, str] | None = None,
             raw: dict[str, str] | None = None, source: str | None = None,
             timeout_s: float = 600.0, tag: str = "code") -> CodeResult:
    """Write the script, run it in a child process, and read back what happened."""
    problem = check_code(code)
    if problem:
        return CodeResult(ok=False, mode=mode, error=problem)

    workdir.mkdir(parents=True, exist_ok=True)
    code_path = workdir / f"{tag}.py"
    code_path.write_text(code)
    out_path = workdir / f"{tag}.parquet"
    result_path = workdir / f"{tag}.result.json"
    spec = {"engine": engine, "mode": mode, "id_col": id_col,
            "code_path": str(code_path), "base_path": str(base_path),
            "sources": sources or {}, "raw": raw or {}, "source": source,
            "out_path": str(out_path), "result_path": str(result_path)}
    spec_path = workdir / f"{tag}.spec.json"
    spec_path.write_text(json.dumps(spec))

    started = time.perf_counter()
    src_root = str(Path(__file__).resolve().parents[1])
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "agent._child", str(spec_path)],
            capture_output=True, text=True, timeout=timeout_s,
            env={**_env(), "PYTHONPATH": src_root})
    except subprocess.TimeoutExpired:
        return CodeResult(ok=False, mode=mode, elapsed_s=round(time.perf_counter() - started, 2),
                          error=f"timed out after {timeout_s:.0f}s; work on fewer rows, "
                                "or aggregate before joining")
    if not result_path.exists():
        return CodeResult(ok=False, mode=mode, elapsed_s=round(time.perf_counter() - started, 2),
                          error=f"the runner died (exit {proc.returncode}): "
                                f"{(proc.stderr or proc.stdout)[-3000:]}")
    payload = json.loads(result_path.read_text())
    return CodeResult(ok=payload["ok"], mode=mode, stdout=payload.get("stdout", ""),
                      error=payload.get("error"), result=payload.get("result"),
                      feature=payload.get("feature"), elapsed_s=payload.get("elapsed_s", 0.0),
                      out_path=str(out_path) if out_path.exists() else None)


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)
