"""FeRG-LLM: reason about derived variables, then write the code down.

Two phases, and the published wording is quite specific about both. The first
turn asks for "key ideas for creating new derived variables to improve the
performance of the following ML problem", framed with the domain, the task type
and the variables present. The second turn is triggered by the phrase "write
down python code", which the system instruction binds to immediate code output:

    "You are an automated ML engineer who writes Python code based on a given
     domain, present variables in a dataframe, and ML problems.
     Whenever the sentence write down python code appears, immediately write
     Python code."

That system instruction is carried through verbatim, because the trigger phrase
is load-bearing: the second prompt relies on it rather than on a format example.
"""

from __future__ import annotations

from typing import Any, Sequence

from discovery.prompt import format_history
from discovery.strategies.base import TwoPhaseProposer

__all__ = ["FergProposer"]

# From FeRG-LLM Appendix B / Table 6.
FERG_SYSTEM_INSTRUCTION = (
    "You are an automated ML engineer who writes Python code based on a given "
    "domain, present variables in a dataframe, and ML problems.\n"
    "Whenever the sentence write down python code appears, immediately write Python code."
)


class FergProposer(TwoPhaseProposer):
    name = "ferg"
    system_prompt = FERG_SYSTEM_INSTRUCTION

    def ideas_prompt(self, history: Sequence[dict[str, Any]], n_features: int) -> str:
        domain = self.context.domain or self.context.task_description
        feedback = ""
        if history:
            feedback = (
                "\n\nDerived variables already attempted, and their measured effect:\n"
                + format_history(history, self.context.metric_name)
            )
        return f"""domain : {domain}
type: {self.context.task_type}
{self.context.task_context_block}present variable in dataframe:
{self.context.context_for(self.round_index)}

Provide key ideas for creating new derived variables to improve the performance
of the following ML problem using the given dataset.
Machine Learning PROBLEM: predict {self.context.target}

Give {n_features} ideas, one per line, each beginning with "* ". Ideas only in
this reply - no code yet. Each derived variable must express a relationship that
none of the present variables already captures.{feedback}
"""

    def code_prompt(self, idea: str, index: int) -> str:
        guard = ""
        if self.context.redundancy_max_abs is not None:
            guard = (
                f" The result must not correlate above |rho| = "
                f"{self.context.redundancy_max_abs:.2f} with any present variable."
            )
        return f"""present variable in dataframe:
{self.context.context_for(self.round_index)}

Derived variable to create:
{idea}

write down python code that adds this single derived variable to the dataframe
`df`. Assign exactly one new column as df["<name>"] = <expression>, indexing `df`
only by the identifiers listed above. Use vectorised pandas/numpy with `df`, `pd`
and `np` only - no imports, no function definitions, no file access. The result
must contain no infinity (NaN is fine; it is read as missing) and no value far
above the column's own bulk: adding 1e-6 to a zero denominator produces a value
of order 1e6 and is rejected.{guard}

Return the code in one block:
```python
# ({{variable name}}, {{short description}})
# Usefulness: {{why it helps}}
df["<name>"] = <expression>
```end
"""
