"""FeatLLM: extract conditions per class, then turn each into an indicator.

Two phases, following the released wording closely. The first asks the model to
extract *conditions* that separate the classes:

    "You are an expert. Given the task description and the list of features and
     data examples, you are extracting conditions for each answer class to solve
     the task."

    Step 1. Analyze the causal relationship or tendency between each feature and
            task description based on general knowledge and common sense.
    Step 2. Infer N different conditions per answer.

The second phase turns conditions into code. Unlike the other methods, FeatLLM's
features are BINARY INDICATORS - "does this row satisfy the condition" - not
continuous ratios, which is why its prompt insists the code match the feature
type (numerical vs categorical).

ADAPTATION. The released second phase emits one function returning a dataframe
with one column per condition. Here each condition becomes its own single-column
block instead. The framework evaluates one feature at a time, and the validation
stages give each its own leave_one_in build and verdict, so a multi-column block
would collapse several independent rules into one indivisible candidate that
could not be judged or rejected individually.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from discovery.prompt import format_history
from discovery.strategies.base import StrategyBase
from discovery.prompt import extract_blocks

__all__ = ["FeatLlmProposer"]

# Conditions come back grouped under a class heading; both a bullet and a bare
# "column op value" line are accepted, since the format drifts between replies.
_CONDITION = re.compile(r"^\s*(?:[*\-•]|\d+[.)])\s*(.{6,})$")
_CLASS_HEADING = re.compile(r"^\s*(?:answer|class|label)\s*[:=]?\s*(.+?)\s*:?\s*$", re.I)


class FeatLlmProposer(StrategyBase):
    """Conditions per class, then one indicator column per condition."""

    name = "featllm"

    def propose(
        self,
        *,
        history: Sequence[dict[str, Any]],
        proposed_names: Sequence[str],
        n_features: int,
        round_index: int,
    ) -> tuple[list[str], dict[str, Any]]:
        self.round_index = round_index
        prompt = self._conditions_prompt(history, n_features)
        reply, conditions_call = self._ask(prompt)
        self._save(round_index, "phase1_prompt.txt", prompt)
        self._save(round_index, "phase1_reply.txt", reply)

        conditions = self._parse_conditions(reply, limit=n_features)

        blocks: list[str] = []
        code_calls: list[dict[str, Any]] = []
        transcript: list[str] = []
        for index, (label, condition) in enumerate(conditions, start=1):
            code_prompt = self._indicator_prompt(label, condition, index)
            code_reply, call = self._ask(code_prompt)
            code_calls.append(call)
            transcript.append(
                f"### condition {index} (class {label})\n{condition}\n\n"
                f"--- prompt ---\n{code_prompt}\n\n--- reply ---\n{code_reply}"
            )
            found = extract_blocks(code_reply)
            if found:
                blocks.append(found[0])
        self._save(round_index, "phase2.txt", "\n\n".join(transcript))

        return blocks, {
            "strategy": self.name,
            "round": round_index,
            "requested": n_features,
            "conditions": [f"{label}: {text}" for label, text in conditions],
            "returned": len(blocks),
            "llm": conditions_call,
            "llm_code_calls": code_calls,
        }

    # ------------------------------------------------------------------ #
    def _conditions_prompt(self, history: Sequence[dict[str, Any]], n: int) -> str:
        feedback = ""
        if history:
            feedback = (
                "\n\nConditions already turned into features, and their effect:\n"
                + format_history(history, self.context.metric_name)
            )
        return f"""You are an expert. Given the task description and the list of features and data examples, you are extracting conditions for each answer class to solve the task.

Task:
{self.context.task_description}

Features (each line begins with the exact expression used to index the column):
{self.context.context_for(self.round_index)}

Let's first understand the problem and solve the problem step by step.

Step 1. Analyze the causal relationship or tendency between each feature and the task description based on general knowledge and common sense, within a short sentence.

Step 2. Based on Step 1, infer {n} different conditions that separate the classes of {self.context.target}. Group them under a heading per answer class, and write each condition as one bullet beginning with "* ". A condition must make sense, match the data, and match the value type of the column it uses.

Conditions must be discriminative rather than restatements of a single column's value: a condition that simply thresholds one column reproduces information the model already has.{feedback}

Format:
Answer: <class>
* <column identifier> <comparison> <value>
"""

    def _indicator_prompt(self, label: str, condition: str, index: int) -> str:
        redundancy = ""
        if self.context.redundancy_max_abs is not None:
            redundancy = (
                f"\n\nThe indicator must not correlate above |rho| = "
                f"{self.context.redundancy_max_abs:.2f} with any existing column: a "
                "threshold that simply reproduces one column's ordering adds nothing."
            )
        return f"""The dataframe `df` is loaded and in memory. Its columns are:
{self.context.context_for(self.round_index)}

Write pandas code adding one binary indicator column for this condition,
associated with class {label} of {self.context.target}:

{condition}

The column must be 1 where the condition holds and 0 where it does not, with no
missing values. Be sure the code matches the value type of the columns used
(numerical vs categorical). Index `df` only by the identifiers listed above.
Vectorised pandas/numpy only, with `df`, `pd` and `np` - no imports, no
functions, no file access. Cast the result to int.

The indicator must contain no NaN: a condition over a column with missing values
must decide what those rows are. If the condition involves a ratio, guard the
division away from zero rather than adding a tiny epsilon - `a / (b + 1e-6)`
turns one `b == 0` into a value of order 1e6.{redundancy}

Return only the code block:
```python
# ({{indicator name}}, {{the condition in words}})
# Usefulness: {{why this condition separates the classes}}
df["<name>"] = (<boolean expression>).astype(int)
```end
"""

    @staticmethod
    def _parse_conditions(reply: str, limit: int | None = None) -> list[tuple[str, str]]:
        """Conditions paired with the class heading they appeared under."""
        without_code = re.sub(r"```.*?```", "", reply, flags=re.DOTALL)
        label = "unknown"
        found: list[tuple[str, str]] = []
        for line in without_code.splitlines():
            heading = _CLASS_HEADING.match(line)
            if heading and not _CONDITION.match(line):
                label = heading.group(1).strip() or label
                continue
            item = _CONDITION.match(line)
            if item:
                text = item.group(1).strip().rstrip(".")
                if text and (label, text) not in found:
                    found.append((label, text))
        return found[:limit] if limit else found
