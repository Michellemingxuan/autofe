"""Print the prompt a run would get - its brief and its first messages - without running.

    PYTHONPATH=src python -m agent.composer -c configs/synthetic_agent.yaml
    PYTHONPATH=src python -m agent.composer -c configs/synthetic_agent.yaml --levels L3 -k 5
    PYTHONPATH=src python -m agent.composer -c configs/synthetic_agent.yaml --linkage spends

The run is a dry one: its folder is made under a temporary directory, so nothing is
written to the agent's run folder (the rotation counters for shots and focus lenses
included).
"""

from __future__ import annotations

import argparse
import tempfile


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-d", "--direction", default="(the direction)")
    parser.add_argument("--levels", nargs="+", default=None, help="e.g. L1 L2, or L3")
    parser.add_argument("-k", "--max-intents", type=int, default=None)
    parser.add_argument("--linkage", metavar="SOURCE", help="the brief of a linkage job instead")
    args = parser.parse_args(argv)

    from agent import at_project_root
    from agent.composer import compose, message, template_for
    from agent.session import RunParams, Session
    from agent.tools.ideas import lenses_for_turn, next_turn
    from agent.workspace import Workspace
    from validation.config import load_config

    cfg = load_config(at_project_root(args.config))
    real_runs = cfg.agent.run_dir
    ws = Workspace.from_config(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        cfg.agent.run_dir = tmp                       # the dry run's own folder
        session = Session(ws, args.direction, kind="linkage" if args.linkage else "direction",
                          source=args.linkage,
                          params=RunParams(K=args.max_intents, levels=args.levels))
        cfg.agent.run_dir = real_runs                 # earlier runs, for the memory section
        session.focus = lenses_for_turn(next_turn(real_runs))   # peeked, not taken
        brief = compose(session)
    first = (message("linkage", source=args.linkage) if args.linkage
             else message("direction", direction=args.direction) + "\n\n" + message("ideas"))
    print(f"# brief (templates/{template_for(session)}.md) - the system instructions\n")
    print(brief)
    print("\n# first message - the user turn\n")
    print(first)


if __name__ == "__main__":
    main()
