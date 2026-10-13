"""A minimal viewer for the model-call trace (agent.llm_trace): plain HTML, no JS.

Three pages, read-only over the SQLite file:

    /                  the runs - direction, calls, tokens, when
    /run/<run_id>      the run as one conversation: each brief once, then every
                       message once, in order - on the left the user turns (the
                       runner's messages), on the right the assistant's words and
                       tool calls, each call followed by its tool result; a marker
                       per model call
    /prompt/<sha>      a brief (system prompt) in full

    PYTHONPATH=src python -m agent.trace_viewer
    PYTHONPATH=src python -m agent.trace_viewer --db outputs/agent/llm_traces.db --port 49020

It needs nothing of the app: run it beside a live run, or on a copied file.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from flask import Flask, abort

__all__ = ["create_viewer", "main"]

_CSS = """
body { font: 13px/1.45 -apple-system, 'Segoe UI', Helvetica, sans-serif; color: #1f2937; margin: 20px 28px; }
a { color: #0b78d0; text-decoration: none; } a:hover { text-decoration: underline; }
h1 { font-size: 18px; margin: 0 0 4px; } .sub { color: #6b7280; margin-bottom: 12px; }
table { border-collapse: collapse; width: 100%; } th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #eef0f3; vertical-align: top; }
th { font-size: 11px; color: #6b7280; font-weight: 600; } td.n { text-align: right; font-variant-numeric: tabular-nums; }
code { font: 11.5px ui-monospace, Menlo, monospace; background: #f1f5f9; padding: 1px 5px; border-radius: 4px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font: 11.5px/1.45 ui-monospace, Menlo, monospace; margin: 4px 0 0; }
details.more > summary { cursor: pointer; color: #0b78d0; font-size: 11px; }
.bad { color: #dc2626; } .ok { color: #157f3b; font-weight: 600; } .no { color: #dc2626; font-weight: 600; } .na { color: #6b7280; }
.step-k { font-size: 11px; color: #6b7280; margin-bottom: 4px; }
.say { white-space: pre-wrap; overflow-wrap: anywhere; margin: 4px 0; }
.idea { border-left: 3px solid #8b5cf6; background: #faf8ff; padding: 4px 8px; margin: 4px 0; font-size: 12px; }
.idea ul { margin: 2px 0 0 16px; padding: 0; color: #4b5563; }
.legend { font-size: 11px; color: #6b7280; margin: 6px 0 10px; }
.conv { display: flex; flex-direction: column; }
.turn { border-radius: 8px; padding: 7px 11px; margin: 5px 0; font-size: 12.5px; min-width: 0; }
.turn .who { font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: #4b5563; margin-bottom: 2px; }
.turn details > summary { cursor: pointer; }
.t-system { background: #f3f4f6; }
.t-user { align-self: flex-start; width: min(76%, 1200px); background: #eef6fd; border-left: 3px solid #4fa9e8; }
.t-assistant, .t-call, .t-result { align-self: flex-end; width: min(76%, 1200px); }
.t-assistant { background: #f0faf3; border-right: 3px solid #34a853; }
.t-call { background: #f6f3fe; border-right: 3px solid #8b5cf6; }
.t-result { background: #fffaf0; border-right: 3px solid #f59e0b; width: min(72%, 1150px); }
.divider { display: flex; align-items: center; gap: 8px; color: #9ca3af; font-size: 11px; margin: 14px 0 4px; }
.divider::before, .divider::after { content: ""; flex: 1; border-top: 1px dashed #d1d5db; }
@media (max-width: 900px) { .t-user, .t-assistant, .t-call, .t-result { width: 92%; } }
"""


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{escape(title)}</title>"
            f"<style>{_CSS}</style></head><body>{body}</body></html>")


def _when(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""


def _num(v: Any) -> str:
    return f"{int(v):,}" if v is not None else ""


def _secs(ms: Any) -> str:
    return "" if ms is None else f"{ms / 1000:.1f} s"


def _text_of(item: dict[str, Any]) -> tuple[str, str, str]:
    """A message as (css class, label, text)."""
    kind = item.get("type") or ("message" if item.get("role") else "item")
    if kind == "function_call":
        args = item.get("arguments") or ""
        try:
            args = json.dumps(json.loads(args), indent=2)
        except (TypeError, ValueError):
            pass
        return "call_", f"call · {item.get('name', '')}", args
    if kind == "function_call_output":
        out = item.get("output")
        return "tool", "tool result", out if isinstance(out, str) else json.dumps(out, indent=2)
    content = item.get("content")
    if isinstance(content, list):
        content = "\n".join(str(c.get("text") or c.get("content") or json.dumps(c))
                            if isinstance(c, dict) else str(c) for c in content)
    role = item.get("role") or kind
    return role, role, content if isinstance(content, str) else json.dumps(item, indent=2)


def _items(raw: str | None) -> list[Any]:
    try:
        items = json.loads(raw) if raw else []
    except ValueError:
        return []
    return items if isinstance(items, list) else []


def _tool_results(run_dir: Path | None, run_id: str) -> dict[str, dict[str, Any]]:
    """Each tool call's input and result, by call id, from the run's events."""
    events = run_dir / run_id / "events.jsonl" if run_dir else None
    out: dict[str, dict[str, Any]] = {}
    if not events or not events.exists():
        return out
    for line in open(events):
        if '"tool_started"' not in line and '"tool_completed"' not in line:
            continue
        e = json.loads(line)
        slot = out.setdefault(e.get("call_id") or "", {"tool": e.get("tool")})
        if e["event"] == "tool_started":
            slot["args"] = e.get("args") or ""
        else:
            slot["output"] = e.get("output") or ""
    return out


def _verdict(tool: str, output: str) -> tuple[str, str]:
    """A tool result in a few words, and its tone (ok, no, na)."""
    try:
        r = json.loads(output)
    except (TypeError, ValueError):
        return ("ran", "na") if output else ("no result recorded", "na")
    if not isinstance(r, dict):
        return "ran", "na"
    if r.get("ok") is False:
        return "refused - " + str(r.get("error", ""))[:140], "no"
    if tool == "screen_feature":
        if r.get("verified"):
            return f"verified · Gini {r.get('gini_gain', 0):+.4f}", "ok"
        return "failed - " + str(r.get("reason", "")).splitlines()[0][:140] if r.get("reason") else "failed", "no"
    if tool in ("screen_request", "propose_new_data") and r.get("recorded"):
        return f"recorded {r.get('intent', '')}", "ok"
    if tool == "challenge_request":
        return str(r.get("status", "")), "ok" if r.get("status") == "kept" else "no"
    if tool == "report_findings":
        return "run ended", "ok"
    return "ran", "na"


def _said(text: str) -> str:
    """What the model said, as it reads best: a structured ideas answer as a list of
    ideas; long text collapsed after its opening."""
    try:
        answer = json.loads(text)
    except (TypeError, ValueError):
        answer = None
    if isinstance(answer, dict) and isinstance(answer.get("ideas"), list):
        out = [f"<div class='step-k'>answered with {len(answer['ideas'])} ideas</div>"]
        for i in answer["ideas"]:
            if not isinstance(i, dict):
                continue
            measures = "".join(f"<li>{escape(str(m))}</li>" for m in i.get("measures") or [])
            out.append(f"<div class='idea'><code>{escape(str(i.get('name', '')))}</code> · "
                       f"{escape(str(i.get('engine', '')))} · {escape(str(i.get('level', '')))} · "
                       f"{escape(str(i.get('lens', '')))}<div>{escape(str(i.get('description', '')))}</div>"
                       + (f"<ul>{measures}</ul>" if measures else "") + "</div>")
        return "".join(out)
    if len(text) <= 700:
        return f"<div class='say'>{escape(text)}</div>"
    return (f"<div class='say'>{escape(text[:700])}…</div><details class='more'><summary>all "
            f"{len(text):,} chars</summary><div class='say'>{escape(text)}</div></details>")


def _long(text: str, limit: int = 1500) -> str:
    """Text in a <pre>, collapsed after its opening when long."""
    text = text or ""
    if len(text) <= limit:
        return f"<pre>{escape(text)}</pre>"
    return (f"<pre>{escape(text[:limit])}…</pre><details class='more'><summary>all "
            f"{len(text):,} chars</summary><pre>{escape(text)}</pre></details>")


def _pretty(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2)
    except (TypeError, ValueError):
        return text or ""


def _call_args(arguments: str) -> str:
    """A tool call's arguments as they read best: the short ones listed, the code (or
    SQL) as code - not an escaped string."""
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return _long(arguments)
    if not isinstance(args, dict):
        return _long(_pretty(arguments))
    code = {k: args.pop(k) for k in ("code", "sql") if isinstance(args.get(k), str) and args.get(k)}
    rest = "".join(f"<div><span class='na'>{escape(k)}</span> {escape(v if isinstance(v, str) else json.dumps(v))}</div>"
                   for k, v in args.items())
    return rest + "".join(_long(v, 2500) for v in code.values())


def _result_body(output: str) -> str:
    """A tool result as it reads best: printed output and errors as text, the other
    fields as a short JSON, collapsed when long."""
    try:
        r = json.loads(output)
    except (TypeError, ValueError):
        return _long(output or "(no result recorded)", 800)
    if not isinstance(r, dict):
        return _long(_pretty(output), 800)
    texts = {k: r.pop(k) for k in ("stdout", "error", "guide") if isinstance(r.get(k), str) and r.get(k)}
    shown = "".join(f"<div class='na'>{escape(k)}</div>{_long(v, 1200)}" for k, v in texts.items())
    return shown + (_long(json.dumps(r, indent=2), 800) if r else "")


def _conversation(rows: list[Any], prompts: dict[str, str],
                  results: dict[str, dict[str, Any]]) -> str:
    """The run as one transcript: each brief once, then every message once, in order -
    the runner's turns, a marker per model call, the agent's words and tool calls,
    each call followed by its result. A call's input is the conversation so far, so
    only what is new in it (past the last call's input and output) is shown."""
    # Every tool result by call id, in its first - full - form: later calls may have
    # seen it trimmed. The events fill in the ones no later call saw (the last).
    outputs: dict[str, str] = {k: v.get("output", "") for k, v in results.items() if v.get("output")}
    for r in rows:
        for i in _items(r["input_json"]):
            if isinstance(i, dict) and i.get("type") == "function_call_output":
                cid = i.get("call_id") or ""
                if cid and (cid not in outputs or len(str(i.get("output"))) > len(outputs[cid])):
                    outputs[cid] = i.get("output") if isinstance(i.get("output"), str) else json.dumps(i.get("output"))
    html, seen, brief = [], 0, None
    for r in rows:
        inputs, out = _items(r["input_json"]), _items(r["output_json"])
        if r["system_sha"] != brief:
            brief = r["system_sha"]
            text = prompts.get(brief or "", "")
            html.append(f"<div class='turn t-system'><details><summary class='who'>system · the "
                        f"{escape(r['stage'])} brief · {len(text):,} chars</summary>"
                        f"<pre>{escape(text)}</pre></details></div>")
        new = inputs[seen:] if len(inputs) >= seen else inputs
        seen = len(inputs) + len(out)
        for i in new:
            if not isinstance(i, dict) or i.get("type") == "function_call_output":
                continue                          # shown beside its call
            if i.get("role") == "user":
                html.append(f"<div class='turn t-user'><div class='who'>Runner (user message)</div>"
                            f"{_long(_text_of(i)[2])}</div>")
        html.append(f"<div class='divider'>model call {r['call']} · {escape(r['stage'])}"
                    f"{' · round ' + str(r['round']) if r['round'] else ''} · {_when(r['started_at'])} · "
                    f"{_secs(r['duration_ms'])} · {_num(r['input_tokens'])} → {_num(r['output_tokens'])} "
                    f"tokens{' (est.)' if r['tokens_estimated'] else ''}</div>")
        if r["outcome"] != "ok":
            html.append("<div class='turn t-assistant'><div class='who bad'>LLM model - no answer: the call "
                        "never returned</div></div>")
        for i in out:
            if not isinstance(i, dict):
                continue
            if i.get("type") == "function_call":
                cid, name = i.get("call_id") or "", i.get("name", "")
                got = outputs.get(cid, "")
                verdict, tone = _verdict(name, got)
                html.append(f"<div class='turn t-call'><div class='who'>LLM model (assistant message) → tool call · {escape(name)}</div>"
                            f"{_call_args(i.get('arguments') or '')}</div>"
                            f"<div class='turn t-result'><div class='who'>Tool (run by the runner) · {escape(name)} → "
                            f"<span class='{tone}'>{escape(verdict)}</span></div>"
                            f"{_result_body(got)}</div>")
            else:
                html.append(f"<div class='turn t-assistant'><div class='who'>LLM model (assistant message)</div>"
                            f"{_said(_text_of(i)[2] or '')}</div>")
    return "<div class='conv'>" + "".join(html) + "</div>"


def _direction(run_dir: Path | None, run_id: str) -> str:
    events = run_dir / run_id / "events.jsonl" if run_dir else None
    if events and events.exists():
        for line in open(events):
            if '"run_started"' in line:
                return json.loads(line).get("direction", "")
    return ""


def create_viewer(db: str | Path, run_dir: str | Path | None = None) -> Flask:
    db = Path(db)
    runs_dir = Path(run_dir) if run_dir else db.parent
    app = Flask(__name__)

    def conn() -> sqlite3.Connection:
        if not db.exists():
            abort(404, f"no trace file at {db}")
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        return c

    @app.get("/")
    def index():
        rows = conn().execute(
            "SELECT run_id, COUNT(*) n, SUM(input_tokens) tin, SUM(output_tokens) tout, "
            "SUM(duration_ms) ms, MIN(started_at) first, MAX(started_at) last, "
            "MAX(tokens_estimated) est, SUM(outcome != 'ok') bad, GROUP_CONCAT(DISTINCT stage) stages "
            "FROM llm_call GROUP BY run_id ORDER BY first DESC").fetchall()
        body = ["<h1>Model calls</h1>", f"<div class='sub'>{escape(str(db))} · {len(rows)} runs</div>",
                "<table><tr><th>Run</th><th>Direction</th><th>Stages</th><th class='n'>Calls</th>"
                "<th class='n'>Input tokens</th><th class='n'>Output tokens</th>"
                "<th class='n'>In the model</th><th>Started</th></tr>"]
        for r in rows:
            bad = f" <span class='bad'>({r['bad']} unfinished)</span>" if r["bad"] else ""
            body.append(
                f"<tr><td><a href='/run/{escape(r['run_id'])}'><code>{escape(r['run_id'])}</code></a></td>"
                f"<td>{escape(_direction(runs_dir, r['run_id'])[:90])}</td><td>{escape(r['stages'] or '')}</td>"
                f"<td class='n'>{r['n']}{bad}</td><td class='n'>{_num(r['tin'])}</td>"
                f"<td class='n'>{_num(r['tout'])}{' est.' if r['est'] else ''}</td>"
                f"<td class='n'>{_secs(r['ms'])}</td><td>{_when(r['first'])}</td></tr>")
        body.append("</table>")
        return _page("Model calls", "".join(body))

    def _header(run_id: str, rows: list[Any]) -> list[str]:
        tin = sum(r["input_tokens"] or 0 for r in rows)
        tout = sum(r["output_tokens"] or 0 for r in rows)
        return ["<a href='/'>← runs</a>", f"<h1><code>{escape(run_id)}</code></h1>",
                f"<div class='sub'>{escape(_direction(runs_dir, run_id))}<br>{len(rows)} model calls · "
                f"{tin:,} input · {tout:,} output tokens · "
                f"{_secs(sum(r['duration_ms'] or 0 for r in rows))} in the model</div>"]

    @app.get("/run/<run_id>")
    def conversation(run_id: str):
        c = conn()
        rows = c.execute("SELECT * FROM llm_call WHERE run_id = ? ORDER BY call", (run_id,)).fetchall()
        if not rows:
            abort(404, f"no calls for {run_id}")
        prompts = {r["sha"]: r["text"] for r in c.execute("SELECT sha, text FROM prompts")}
        body = _header(run_id, rows)
        body.append("<div class='legend'>left: the <b>Runner</b> - the code driving the run, writing "
                    "user messages (your direction is the first) · right: the <b>LLM model</b> - its "
                    "assistant messages, words and tool calls - and each <b>Tool</b> it called, run by "
                    "the runner, whose result goes back to the model as a message of its own</div>")
        body.append(_conversation(rows, prompts, _tool_results(runs_dir, run_id)))
        return _page(run_id, "".join(body))

    @app.get("/prompt/<sha>")
    def prompt(sha: str):
        row = conn().execute("SELECT text FROM prompts WHERE sha = ?", (sha,)).fetchone()
        if not row:
            abort(404, f"no prompt {sha}")
        return _page(f"prompt {sha}", f"<a href='javascript:history.back()'>← back</a>"
                                      f"<h1>Instructions <code>{escape(sha)}</code></h1>"
                                      f"<div class='sub'>{len(row['text']):,} chars</div>"
                                      f"<pre>{escape(row['text'])}</pre>")

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="outputs/agent/llm_traces.db")
    parser.add_argument("--runs", default=None, help="the run folder, for directions; "
                                                     "default: beside the trace file")
    parser.add_argument("--port", type=int, default=49020)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    print(f"trace viewer on http://{args.host}:{args.port}  ({args.db})")
    create_viewer(args.db, args.runs).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
