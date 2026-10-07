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
from typing import Any, Callable

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


_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+), in ')


def brief_error(error: str | None) -> str | None:
    """A traceback cut to what the agent can act on: the lines of ITS script that
    led to the error, and the error itself. A full pandas traceback is mostly
    library frames - fed back whole, the agent reasons from the noise."""
    if not error or "Traceback" not in error:
        return error
    rows = error.splitlines()
    lines = []
    for i, row in enumerate(rows):
        match = _FRAME.match(row)
        if match and re.search(r"/code/[^/]+\.py$", match.group(1)):   # the agent's own script
            source = rows[i + 1].strip() if i + 1 < len(rows) and not _FRAME.match(rows[i + 1]) else ""
            lines.append(f"line {match.group(2)}: {source}")
    last = [row for row in rows if row.strip()][-1].strip()
    return "\n".join([*lines, last])


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

    @property
    def short_error(self) -> str | None:
        return brief_error(self.error)

    def for_agent(self) -> dict[str, Any]:
        out = {"ok": self.ok, "elapsed_s": self.elapsed_s}
        for key in ("stdout", "short_error", "result", "feature"):
            value = getattr(self, key)
            if value:
                out["error" if key == "short_error" else key] = value
        out.update(self.extras)
        return out


def run_code(code: str, mode: str, *, workdir: Path, engine: str, id_col: str,
             base_path: Path, sources: dict[str, str] | None = None,
             raw: dict[str, str] | None = None, source: str | None = None,
             timeout_s: float = 600.0, tag: str = "code",
             probe_rows: int | None = None,
             spark_conf: dict[str, str] | None = None,
             should_stop: Callable[[], bool] | None = None) -> CodeResult:
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
            "out_path": str(out_path), "result_path": str(result_path),
            **({"probe_rows": probe_rows} if probe_rows else {}),
            **({"spark_conf": spark_conf} if spark_conf and engine == "spark" else {})}
    spec_path = workdir / f"{tag}.spec.json"
    spec_path.write_text(json.dumps(spec))

    started = time.perf_counter()
    src_root = str(Path(__file__).resolve().parents[1])
    # Polled, not waited on: a stop from the user kills the script at once, and the
    # timeout is enforced the same way.
    # Output goes to a file, not a pipe: a pipe nobody reads while polling fills up
    # and stalls the script.
    log_path = workdir / f"{tag}.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "agent._child", str(spec_path)],
                                stdout=log, stderr=subprocess.STDOUT, text=True,
                                env={**_env(), "PYTHONPATH": src_root})
        stopped = None
        while proc.poll() is None:
            if should_stop and should_stop():
                stopped = "stopped by the user"
            elif time.perf_counter() - started > timeout_s:
                stopped = (f"timed out after {timeout_s:.0f}s - vectorise: no loops or apply "
                           "over rows; aggregate with groupby, filter before joining")
            if stopped:
                proc.kill()
                proc.wait()
                return CodeResult(ok=False, mode=mode, error=stopped,
                                  elapsed_s=round(time.perf_counter() - started, 2))
            time.sleep(0.2)
    stdout, stderr = "", log_path.read_text(errors="replace")
    if not result_path.exists():
        elapsed = round(time.perf_counter() - started, 2)
        if proc.returncode in (-9, 137):
            # SIGKILL: the operating system ended it - the stderr tail (a pandas
            # warning, say) is not the reason and would point the agent the wrong way.
            return CodeResult(ok=False, mode=mode, elapsed_s=elapsed, error=(
                "the runner was killed by the operating system (exit -9) - almost always "
                "out of memory: the frames it loaded did not fit. Read less: in a probe, raw "
                "sources are already a sample; otherwise use only the columns you need, "
                "filter rows before joining, aggregate early - or run on the spark engine."))
        return CodeResult(ok=False, mode=mode, elapsed_s=elapsed,
                          error=f"the runner died (exit {proc.returncode}): "
                                f"{(stderr or stdout)[-3000:]}")
    payload = json.loads(result_path.read_text())
    return CodeResult(ok=payload["ok"], mode=mode, stdout=payload.get("stdout", ""),
                      error=payload.get("error"), result=payload.get("result"),
                      feature=payload.get("feature"), elapsed_s=payload.get("elapsed_s", 0.0),
                      out_path=str(out_path) if out_path.exists() else None)


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)
