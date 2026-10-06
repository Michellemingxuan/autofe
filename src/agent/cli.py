"""Run one direction from the terminal.

    python -m agent.cli -c configs/synthetic_agent.yaml \\
        -d "use payment behaviour relative to spend" -k 5

Events print as they happen; approvals are asked for at the prompt
(``--yes`` approves them all, for unattended synthetic runs). The run's
events, ledger, scripts and verified features land in ``agent.run_dir/<run_id>``.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from typing import Any

from agent.session import AutoApprover, Decision, RunParams, Session
from agent.workspace import Workspace
from validation.config import load_config


class ConsoleApprover:
    def request(self, kind: str, payload: dict[str, Any]) -> Decision:
        print(f"\n=== approval needed: {kind} ===")
        for key, value in payload.items():
            if key in ("req_id", "kind"):
                continue
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            print(f"--- {key}\n{text}")
        answer = input("approve? [y/N] ").strip().lower()
        note = input("note for the agent (optional): ").strip()
        return Decision(answer in ("y", "yes"), note)


def _short(text: Any, width: int = 400) -> str:
    return textwrap.shorten(str(text), width=width, placeholder=" ...")


def print_event(e: dict[str, Any]) -> None:
    kind = e["event"]
    if kind == "agent_message":
        print(f"\n[agent] {e['text']}")
    elif kind == "tool_started":
        print(f"  -> {e['tool']}({_short(e['args'], 160)})")
    elif kind == "code_status":
        if e["state"] == "running":
            print(f"  [code {e['code_id']} {e['mode']} {e.get('level') or ''}] {e.get('title', '')}")
            print(textwrap.indent(e["code"].rstrip(), "      | "))
        else:
            print(f"  [code {e['code_id']}] {e['state']} in {e['elapsed_s']}s")
            if e.get("error"):
                print(textwrap.indent(_short(e["error"], 800), "      ! "))
    elif kind == "feature_screened":
        mark = "VERIFIED" if e["verified"] else "rejected"
        gini = f"{e['delta']:+.4f}" if e["delta"] is not None else "n/a"
        capture = f"{e['capture_gain']:+.4f}" if e.get("capture_gain") is not None else "n/a"
        print(f"  [{e['intent']} {e['intents_used']}/{e['K']}] {e['name']} ({e['level']}) "
              f"gini={gini} capture={capture} {mark} {e['reason'] or ''}")
    elif kind in ("run_started", "run_done", "approval_resolved", "run_error"):
        print(f"[{kind}] " + _short({k: v for k, v in e.items()
                                     if k not in ("seq", "run_id", "ts", "event", "sources")}))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-d", "--direction", required=True)
    parser.add_argument("-k", "--max-intents", type=int, default=None)
    parser.add_argument("--min-gini-gain", type=float, default=None,
                        help="verify only features whose Gini gain is above this")
    parser.add_argument("--min-capture-gain", type=float, default=None,
                        help="...and whose capture-rate gain is above this")
    parser.add_argument("--yes", action="store_true", help="approve every request")
    args = parser.parse_args(argv)
    from agent import at_project_root

    args.config = at_project_root(args.config)

    from agent.agent import run_direction

    ws = Workspace.from_config(load_config(args.config))
    session = Session(ws, args.direction,
                      approver=AutoApprover() if args.yes else ConsoleApprover(),
                      params=RunParams(K=args.max_intents, min_gini_gain=args.min_gini_gain,
                                       min_capture_gain=args.min_capture_gain), listeners=[print_event])
    print(f"run {session.run_id} -> {session.run_dir}")
    run_direction(session)
    print("\nverified:")
    for e in session.verified():
        print(f"  {e['name']:30s} {e['level']}  gini={e['delta']:+.4f}  {e['description']}")


if __name__ == "__main__":
    main()
