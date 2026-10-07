"""The process log: what a run tried, what worked, what failed and why.

Written from the run's events when it ends - by code, not by the agent, so it
says what happened rather than what the agent made of it:

* ``<run>/process.md`` - one run, to read: the outcome, the results and how many
  tries each took, the failures grouped by cause, what the checks sent back, and
  the lessons the failures share.
* ``<run_dir>/attempts.jsonl`` - every run, one line per attempt or sent-back
  proposal, each with an outcome category: the failure patterns across directions,
  models and levels, to count.

A deleted run leaves its lines in ``attempts.jsonl``: it is a record of what was
tried, not of what is in the pool. For runs made before the log existed:

    PYTHONPATH=src python -m agent.process_log outputs/agent
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agent.execution import brief_error

__all__ = ["outcome_of_feature", "outcome_of_refusal", "attempts_of", "process_report",
           "write_process_log", "rebuild"]

# What each outcome category means, for the report.
CATEGORIES = {
    "verified": "verified - cleared every gate",
    "kept": "kept - the current data cannot supply it",
    "approved": "approved by the analyst",
    "below_gate": "screened, but the gain did not clear the gates",
    "redundant": "redundant with a base column",
    "spike": "a spike - an epsilon-guarded division blew up",
    "non_finite": "infinite or not-a-number values",
    "script_error": "the script failed",
    "contract": "build() returned the wrong shape",
    "screen_error": "the screen could not score it",
    "dropped": "dropped - the current data already covers it",
    "rejected": "rejected by the analyst",
    # sent back by a check, nothing spent
    "duplicate": "the same as an earlier feature or request",
    "sql": "SQL that does not fit the CAS columns",
    "unknown_source": "a source that does not exist (a requested pull is not data)",
    "no_linkage": "a source with no confirmed linkage",
    "name": "a name already taken, or not a valid name",
    "target_met": "a level whose target was already met",
    "attempts": "no attempts left",
    "probe_gate": "screened again before probing, after failed scripts",
    "ideas_first": "proposed before the ideas were recorded",
    "early_report": "reported short of the target",
    "ideas": "ideas that did not pass the checks",
    "other": "another check",
}


def outcome_of_feature(e: dict[str, Any]) -> str:
    """The category of a screened feature, from its result."""
    if e.get("verified"):
        return "verified"
    reason = str(e.get("reason") or "")
    for pattern, category in ((r"^script failed", "script_error"), (r"^build\(\) returned", "contract"),
                              (r"redundant", "redundant"), (r"spikes", "spike"),
                              (r"not finite|infinite|NaN|inf\b", "non_finite"),
                              (r"is not above", "below_gate")):
        if re.search(pattern, reason):
            return category
    return "screen_error"


def outcome_of_refusal(error: str) -> str:
    """The category of a proposal a check sent back, from its message."""
    for pattern, category in (
            (r"identical to|the same data as", "duplicate"),
            (r"look before the next attempt", "probe_gate"),
            (r"no ideas recorded", "ideas_first"),
            (r"target .*is met|target of \d+ is met", "target_met"),
            (r"attempts are used", "attempts"),
            (r"no source \[|A data request is not data", "unknown_source"),
            (r"no confirmed linkage", "no_linkage"),
            (r"already taken|already proposed|not a valid column name|snake_case", "name"),
            (r"CAS scope|columns not in|partition date|identifier|selects from no table", "sql"),
            (r"more result\(s\) wanted|intent\(s\) left", "early_report")):
        if re.search(pattern, error):
            return category
    return "other"


def _short(text: Any, n: int = 240) -> str:
    text = brief_error(str(text or "")) or ""
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def attempts_of(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every attempt and every sent-back proposal of a run, categorised."""
    started = next((e for e in events if e["event"] == "run_started"), {})
    params = started.get("params") or {}
    common = {"run_id": started.get("run_id"), "direction": started.get("direction", ""),
              "model": params.get("model"), "engine": params.get("engine")}
    requests: dict[str, dict[str, Any]] = {}
    calls: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for e in events:
        k = e["event"]
        if k == "tool_started":
            calls[e.get("call_id", "")] = e.get("args", "")
        elif k == "feature_screened":
            out.append({**common, "ts": e["ts"], "kind": "feature", "id": e.get("intent"),
                        "name": e.get("name"), "level": e.get("level"),
                        "outcome": outcome_of_feature(e), "gini_gain": e.get("delta"),
                        "coverage": e.get("coverage"), "detail": _short(e.get("reason"))})
        elif k == "data_request":
            requests[e["intent"]] = {**common, "ts": e["ts"], "kind": "request", "id": e["intent"],
                                     "name": e.get("source_name"), "level": "L3",
                                     "scope": e.get("scope", "cas"), "outcome": "proposed",
                                     "detail": ""}
        elif k == "request_challenged" and e.get("intent") in requests:
            r = requests[e["intent"]]
            r.update(outcome=e.get("status"), verdict=e.get("verdict"),
                     detail=_short(e.get("reasoning")))
        elif k == "approval_resolved" and e.get("kind") == "data_pull":
            pass                                            # a mixed run's request: see below
        elif k == "tool_completed":
            try:
                reply = json.loads(e.get("output") or "{}")
            except ValueError:
                continue
            if not isinstance(reply, dict):
                continue
            if reply.get("ok") is False and "intent" not in reply and reply.get("error"):
                out.append({**common, "ts": e["ts"], "kind": "sent_back", "id": None,
                            "tool": e.get("tool"), "outcome": outcome_of_refusal(reply["error"]),
                            "detail": _short(reply["error"])})
            elif e.get("tool") in ("screen_request", "propose_new_data") and "approved" in reply:
                args = _args(calls.get(e.get("call_id", ""), ""))
                out.append({**common, "ts": e["ts"], "kind": "request", "id": None,
                            "name": args.get("source_name"), "level": "L3",
                            "scope": "beyond_cas" if e["tool"] == "propose_new_data" else "cas",
                            "outcome": "approved" if reply["approved"] else "rejected",
                            "detail": _short(reply.get("user_note"))})
        elif k == "ideas_sent_back":
            out.append({**common, "ts": e["ts"], "kind": "sent_back", "id": None, "tool": "ideas",
                        "outcome": "ideas", "detail": _short(e.get("error"))})
    out.extend(requests.values())
    return sorted(out, key=lambda r: r["ts"])


def _args(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def process_report(events: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> str:
    """One run as a readable account: outcome, results, failures by cause, mistakes."""
    from agent.tools.screen import lessons

    started = next((e for e in events if e["event"] == "run_started"), {})
    done = next((e for e in reversed(events) if e["event"] == "run_done"), {})
    params = started.get("params") or {}
    rows = attempts_of(events)
    attempts = [r for r in rows if r["kind"] in ("feature", "request")]
    sent_back = [r for r in rows if r["kind"] == "sent_back"]
    good = [r for r in attempts if r["outcome"] in ("verified", "kept", "approved")]
    probes = [e for e in events if e["event"] == "code_status" and e.get("mode") == "probe"
              and e.get("state") in ("ok", "error")]

    lines = [f"# Process log - {started.get('direction', '')}", "",
             f"Run `{started.get('run_id', '')}` · {params.get('model', '')} · {params.get('engine', '')} · "
             f"levels {', '.join(params.get('levels') or [])} · target K={started.get('K')}"
             + (f" ({', '.join(f'{lv} x{n}' for lv, n in (started.get('quota') or {}).items())})"
                if len(started.get("quota") or {}) > 1 else ""), "",
             f"**Outcome:** {len(good)} of {started.get('K')} reached in {len(attempts)} attempts"
             f"{' (cap ' + str(started['max_attempts']) + ')' if started.get('max_attempts') else ''}"
             f" · {len(sent_back)} proposals sent back by the checks · {len(probes)} probes · "
             f"ended: {done.get('stopped_because', 'not ended')}", ""]

    counts = Counter(r["outcome"] for r in attempts)
    lines += ["## Attempts by outcome", ""]
    lines += [f"* {n} × {CATEGORIES.get(cat, cat)}" for cat, n in counts.most_common()] or ["* none"]
    lines.append("")

    lines += ["## What worked", ""]
    if not good:
        lines.append("Nothing reached the target.")
    tries = _tries_before(attempts)
    for r in good:
        extra = (f" · Gini gain {r['gini_gain']:+.4f}" if isinstance(r.get("gini_gain"), (int, float)) else "")
        before = tries.get(r["id"], 0)
        lines.append(f"* **{r['id']} `{r['name']}`** ({r.get('level')}{extra})"
                     + (f" - after {before} failed attempt(s) just before it" if before else ""))
    lines.append("")

    failed = defaultdict(list)
    for r in attempts:
        if r["outcome"] not in ("verified", "kept", "approved"):
            failed[r["outcome"]].append(r)
    lines += ["## What failed, and why", ""]
    if not failed:
        lines.append("No failed attempts.")
    for cat, group in sorted(failed.items(), key=lambda kv: -len(kv[1])):
        lines += [f"### {CATEGORIES.get(cat, cat)} ({len(group)})", ""]
        lines += [f"* {r['id']} `{r['name']}` ({r.get('level')}) - {r['detail']}" for r in group]
        lines.append("")

    lines += ["## Sent back by the checks (nothing spent)", ""]
    if not sent_back:
        lines.append("None.")
    by = defaultdict(list)
    for r in sent_back:
        by[r["outcome"]].append(r)
    for cat, group in sorted(by.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"* {len(group)} × {CATEGORIES.get(cat, cat)} - e.g. {group[0]['detail']}")
    lines.append("")

    learned = lessons(ledger)
    lines += ["## Lessons the failures share", ""]
    lines += [f"* {x}" for x in learned] or ["None repeated."]
    lines.append("")
    if done.get("summary"):
        lines += ["## The agent's own summary", "", done["summary"].strip(), ""]
    return "\n".join(lines)


def _tries_before(attempts: list[dict[str, Any]]) -> dict[str, int]:
    """For each success, the failed attempts that came right before it - how hard it was."""
    out, streak = {}, 0
    for r in attempts:
        if r["outcome"] in ("verified", "kept", "approved"):
            out[r["id"]] = streak
            streak = 0
        else:
            streak += 1
    return out


def write_process_log(run_dir: Path, events: list[dict[str, Any]],
                      ledger: list[dict[str, Any]], log_dir: Path) -> None:
    """Write the run's process.md, and append its attempts to the cross-run log."""
    (run_dir / "process.md").write_text(process_report(events, ledger))
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "attempts.jsonl", "a") as fh:
        for r in attempts_of(events):
            fh.write(json.dumps(r, default=str) + "\n")


def rebuild(run_root: Path) -> int:
    """Rewrite process.md for every run under ``run_root``, and attempts.jsonl from
    them all - for runs made before the log existed, or after a change to it."""
    from agent.events import read_events

    lines, written = [], 0
    for folder in sorted(p for p in run_root.iterdir() if (p / "events.jsonl").exists()):
        events = read_events(folder / "events.jsonl")
        if not any(e["event"] == "run_started" and e.get("kind", "direction") == "direction"
                   for e in events):
            continue
        ledger_path = folder / "ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
        (folder / "process.md").write_text(process_report(events, ledger))
        lines += [json.dumps(r, default=str) for r in attempts_of(events)]
        written += 1
    (run_root / "attempts.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
    return written


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/agent")
    print(f"process logs written for {rebuild(root)} runs; {root / 'attempts.jsonl'}")
