"""PromptFE: search a fixed operator grammar, guided by scored examples.

Different in kind from the other four. Rather than asking for features in prose,
PromptFE constrains the model to a small algebra and turns generation into a
guided search: the prompt lists the operators, shows the best expressions tried
so far *with their scores*, shows the ones that errored, and asks for more.

    unary:  log, sqrt_abs, min_max, reciprocal
    binary: +, -, *, /

The score-ranked examples are the mechanism. Where CAAFE feeds back prose
feedback, PromptFE feeds back a leaderboard, so the model is doing hill-climbing
over a grammar rather than reasoning about the domain. That makes it the only
strategy whose prompt gets *more* informative as a run goes on regardless of how
articulate the model is.

ADAPTATION. The released implementation carries expressions as RPN trees and
mutates them structurally. Here the model returns pandas expressions built from
the same operator set, and the leaderboard is assembled from the framework's own
history. The constrained grammar and the scored feedback - the parts that make
the method what it is - are preserved; the tree machinery is not ported, so this
explores by prompting rather than by structural mutation.
"""

from __future__ import annotations

from typing import Any, Sequence

from discovery.prompt import extract_blocks, verdict_line
from discovery.strategies.base import StrategyBase

__all__ = ["PromptFeProposer", "OPERATORS"]

OPERATORS = """unary operators, applied to one column:
  log(x)          element-wise logarithm of the absolute value: np.log(np.abs(x) + 1)
  sqrt_abs(x)     element-wise square root of the absolute value: np.sqrt(np.abs(x))
  min_max(x)      element-wise min-max normalisation
  reciprocal(x)   element-wise reciprocal, guarded away from zero

binary operators, applied to two columns:
  +  -  *  /      element-wise, with division guarded away from zero"""


class PromptFeProposer(StrategyBase):
    """Asks for expressions over a fixed operator set, ranked by what scored well."""

    name = "promptfe"
    system_prompt = "You are an expert data scientist assistant"

    def propose(
        self,
        *,
        history: Sequence[dict[str, Any]],
        proposed_names: Sequence[str],
        n_features: int,
        round_index: int,
    ) -> tuple[list[str], dict[str, Any]]:
        self.round_index = round_index
        prompt = self._prompt(history, n_features)
        reply, call = self._ask(prompt)
        self._save(round_index, "prompt.txt", prompt)
        self._save(round_index, "reply.txt", reply)

        blocks = extract_blocks(reply)
        return blocks[:n_features], {
            "strategy": self.name,
            "round": round_index,
            "requested": n_features,
            "returned": len(blocks[:n_features]),
            "llm": call,
        }

    def _prompt(self, history: Sequence[dict[str, Any]], n_features: int) -> str:
        redundancy = ""
        if self.context.redundancy_max_abs is not None:
            redundancy = (
                f"\n\nAn expression correlating above |rho| = "
                f"{self.context.redundancy_max_abs:.2f} with any existing column is "
                "discarded before it is scored, so a single operator applied to one "
                "column rarely survives - it preserves that column's ordering."
            )
        return f"""Dataset description:
{self.context.task_description}

The table contains the following columns. Each line begins with the exact expression used to index it:
{self.context.context_for(self.round_index)}

You may build new features using only these operators:
{OPERATORS}

{self._leaderboard(history)}

Propose {n_features} new features for predicting {self.context.target}, each as a
single pandas expression built only from the columns above and the operators
listed. Score is {self.context.metric_name}; higher is better.{self.context.metric_explanation}

Each expression must produce no infinity and no value far above its own bulk
(NaN is fine; it is read as missing) - guard every division and reciprocal
away from zero rather than adding a
tiny epsilon, which turns one zero denominator into a value of order 1e6. Index
`df` only by the identifiers listed above.{redundancy}

Return {n_features} blocks, one per feature:
```python
# ({{feature name}}, {{the expression in words}})
# Usefulness: {{what relationship it captures}}
df["<name>"] = <expression>
```end
"""

    @staticmethod
    def _leaderboard(history: Sequence[dict[str, Any]]) -> str:
        """
        Scored expressions, best first, with the failures listed separately.

        Ranked rather than chronological: the point of this method is that the
        model sees which shapes scored well, and a run ordered by time buries
        that under whatever was tried most recently.
        """
        if not history:
            return (
                "No features have been scored yet. Start with simple relationships "
                "between columns from different families."
            )

        scored = [r for r in history if r.get("delta") is not None]
        failed = [r for r in history if r.get("error")]
        scored.sort(key=lambda r: r["delta"], reverse=True)

        lines = ["Features tried so far, best score first:"]
        for record in scored:
            expression = record.get("expression") or record.get("feature_name") or "?"
            # An earlier run's feature also carries what validation made of it.
            note = verdict_line(record)
            lines.append(f"\nFeature\n{expression}\nScore\n{record['delta']:+.4f}"
                         + (f"\n{note}" if note else ""))

        if failed:
            lines.append("\nThese were rejected before they could be scored:")
            for record in failed:
                expression = record.get("expression") or record.get("feature_name") or "?"
                lines.append(f"\n{expression}\nError: {record['error']}")

        return "\n".join(lines)
