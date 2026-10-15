"""How many new ideas each round proposed, across runs.

Input:  the agent's run folders (``agent.run_dir``, default ``outputs/agent``) -
        each run's ideas (``ideas.json``, or the ``ideas_recorded`` events of an
        older run).
Output: ``docs/results/idea_novelty.png`` and ``docs/results/idea_novelty.csv`` -
        one row per (run, round): ideas proposed, how many were new, how many
        repeated an earlier one.

An idea is **new** when it is not as close in words as ``--threshold`` (the
config's ``agent.checks.same_meaning``, 0.6) to any idea proposed before it - in
an earlier run, or an earlier round of its own run. The same measure as the
idea check (``agent.checks.similarity``), applied afresh to every run - so older
runs, recorded before ideas were linked (``same_as``), count the same way, and a
repeat of an earlier round of the same run counts too, which ``same_as`` misses.

    python scripts/idea_novelty.py [--run-dir outputs/agent] [--threshold 0.6]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.board import run_ideas  # noqa: E402
from agent.checks import similarity  # noqa: E402
from agent.memory import _NOT_RUNS, _direction  # noqa: E402


def novelty(run_dir: Path, threshold: float) -> pd.DataFrame:
    """One row per idea, oldest run first: its run, round, and the earlier idea it
    repeats (empty when it is new)."""
    runs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name not in _NOT_RUNS)
    rows, seen = [], []                     # seen: (run, round, name, text) proposed before
    for folder in runs:
        ideas = run_ideas(folder)
        direction = _direction(folder)
        for rnd in sorted({i.get("round") or 1 for i in ideas}):
            this = [i for i in ideas if (i.get("round") or 1) == rnd]
            for idea in this:
                text = f"{idea['name']} {idea.get('description') or idea.get('hypothesis') or ''}"
                scores = similarity(text, [s[3] for s in seen])
                best = max(range(len(scores)), key=scores.__getitem__) if scores else None
                repeats = (best is not None and scores[best] >= threshold)
                rows.append({"run_id": folder.name, "direction": direction, "round": rnd,
                             "idea": idea["name"], "engine": idea.get("engine") or "",
                             "repeats": f"{seen[best][0]}:r{seen[best][1]}:{seen[best][2]}"
                                        if repeats else "",
                             "overlap": round(scores[best], 3) if best is not None else None,
                             "same_as": idea.get("same_as") or ""})
            # The round's ideas count as earlier only for the next round: within a
            # round the idea check already refuses a repeat.
            seen += [(folder.name, rnd, i["name"],
                      f"{i['name']} {i.get('description') or i.get('hypothesis') or ''}")
                     for i in this]
    return pd.DataFrame(rows)


def per_round(ideas: pd.DataFrame) -> pd.DataFrame:
    out = (ideas.assign(new=ideas["repeats"] == "")
           .groupby(["run_id", "round"], sort=False)
           .agg(direction=("direction", "first"), proposed=("idea", "size"), new=("new", "sum"))
           .reset_index())
    out["repeated"] = out["proposed"] - out["new"]
    out["cumulative_new"] = out["new"].cumsum()
    return out


def plot(rounds: pd.DataFrame, threshold: float, path: Path) -> None:
    x = range(len(rounds))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(max(8, 0.45 * len(rounds) + 3), 7),
                                      sharex=True, height_ratios=[3, 2])
    top.bar(x, rounds["new"], color="#2a7ab0", label="new")
    top.bar(x, rounds["repeated"], bottom=rounds["new"], color="#c9c9c9",
            label=f"repeats an earlier idea (overlap ≥ {threshold})")
    top.set_ylabel("ideas proposed")
    top.set_ylim(0, rounds["proposed"].max() * 1.2)           # room for the run ids
    top.legend(loc="lower right", bbox_to_anchor=(1, 1), frameon=False, ncols=2)
    top.set_title("New ideas per round, across runs", loc="left")
    bottom.plot(x, rounds["cumulative_new"], marker="o", color="#2a7ab0")
    bottom.set_ylabel("distinct ideas so far")
    bottom.set_xticks(list(x), [f"r{r}" for r in rounds["round"]], fontsize=8)
    # A line between runs, the run's id above its first round.
    starts = rounds.index[rounds["run_id"] != rounds["run_id"].shift()].tolist()
    for k in starts:
        for ax in (top, bottom):
            ax.axvline(k - 0.5, color="#888", lw=0.8, ls="--")
        top.annotate(rounds.loc[k, "run_id"], (k - 0.4, 0.98), xycoords=("data", "axes fraction"),
                     va="top", fontsize=7, color="#555")
    for ax in (top, bottom):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", default=str(ROOT / "outputs" / "agent"))
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--out", default=str(ROOT / "docs" / "results"))
    args = parser.parse_args()

    ideas = novelty(Path(args.run_dir), args.threshold)
    if ideas.empty:
        sys.exit(f"no ideas under {args.run_dir}")
    rounds = per_round(ideas)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rounds.to_csv(out / "idea_novelty.csv", index=False)
    ideas.to_csv(out / "idea_novelty_ideas.csv", index=False)
    plot(rounds, args.threshold, out / "idea_novelty.png")
    print(rounds.to_string(index=False))
    print(f"\n{int(rounds['new'].sum())} new of {int(rounds['proposed'].sum())} ideas -> {out}")


if __name__ == "__main__":
    main()
