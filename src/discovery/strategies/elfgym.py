"""ELF-Gym: propose features in prose, then write code for each one.

Two phases. The first asks for a bulleted list of feature *ideas* with no code;
the second takes each bullet on its own and asks for the pandas expression. The
released method keeps only lines beginning with "* " and handles every proposed
feature independently, which is why the second phase runs per idea rather than
once for the batch.

NOTE ON FIDELITY. The structure here follows the released run - bulleted
proposals, then independent per-idea code generation. The exact prompt wording
could not be recovered: g-autofe/elfgym builds it into a
``prompts_feature_proposal/`` directory that was never generated, so what
follows is reconstructed from the notebook's own description of the method
rather than copied. Worth checking against the paper before using these runs as
a published-method comparison.
"""

from __future__ import annotations

from typing import Any, Sequence

from discovery.prompt import format_history
from discovery.strategies.base import TwoPhaseProposer

__all__ = ["ElfGymProposer"]


class ElfGymProposer(TwoPhaseProposer):
    name = "elfgym"

    def ideas_prompt(self, history: Sequence[dict[str, Any]], n_features: int) -> str:
        feedback = ""
        if history:
            feedback = (
                "\n\nFeatures already tried, and how they scored:\n"
                + format_history(history, self.context.metric_name)
            )
        return f"""Dataset description:
{self.context.task_description}

Columns available in the table. Each line begins with the exact expression used to index it:
{self.context.context_for(self.round_index)}

Propose {n_features} new features that would help a model predict {self.context.target}.
Write each as one bullet beginning with "* ", stating what to compute and why it
should matter for this problem. Describe the features in words only - do not write
any code in this reply.

Propose features that combine columns in ways none of the existing columns already
expresses. A feature that restates an existing column under a new name adds nothing.{feedback}

Proposals:
"""

    def code_prompt(self, idea: str, index: int) -> str:
        redundancy = ""
        if self.context.redundancy_max_abs is not None:
            redundancy = (
                f"\n- It must not correlate above |rho| = "
                f"{self.context.redundancy_max_abs:.2f} with any existing column."
            )
        return f"""The dataframe `df` is loaded and in memory. Its columns are:
{self.context.context_for(self.round_index)}

Write pandas code implementing exactly this proposed feature:

{idea}

Requirements:
- Exactly one new column, assigned as df["<name>"] = <expression>.
- Index `df` only by the identifiers listed above, never by a description.
- Vectorised pandas/numpy only, using `df`, `pd` and `np`. No imports, no
  functions, no file access.
- No infinity in the result (NaN is fine; it is read as missing), and no value
  far above the column's own bulk: `a / (b + 1e-6)` turns one `b == 0` into a
  value of order 1e6 and is rejected. Mask the undefined rows to NaN, clip, or
  use a denominator that cannot approach zero.{redundancy}

Return only the code block:
```python
# ({{feature name}}, {{short description}})
# Usefulness: {{why it helps}}
df["<name>"] = <expression>
```end
"""
