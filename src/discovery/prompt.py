"""Build what the proposer sees, and read back what it wrote.

Three jobs, kept separate because they fail differently:

    describe_columns   what the data looks like - types, ranges, real values
    format_history     what previous rounds tried and what it cost them
    build_prompt       the instructions wrapped around both

Plus the inverse: :func:`extract_code` pulls the block out of a reply, and
:func:`parse_candidate` lifts the rationale comments into structured fields so a
feature's reasoning survives next to its numbers instead of only inside a
comment nobody reads.

The sample rows matter more than they look. They are the only concrete data the
model ever sees, and on an imbalanced target a uniform draw is usually all one
class - so the caller is expected to pass a class-balanced sample, and the
context says how many of each are present.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CandidateText",
    "describe_columns",
    "low_variation_columns",
    "format_history",
    "build_prompt",
    "extract_code",
    "parse_candidate",
]

_CODE_FENCE_END = re.compile(r"```python\s*(.*?)```end", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class CandidateText:
    """The rationale a proposer attached to a block, lifted out of its comments."""

    display_name: str | None = None
    description: str | None = None
    rationale: str | None = None
    input_columns: list[str] = field(default_factory=list)
    expression: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "description": self.description,
            "rationale": self.rationale,
            "input_columns": list(self.input_columns),
            "expression": self.expression,
        }


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (float, np.floating)):
        return str(round(float(value), 3))
    return str(value)


def low_variation_columns(
    sample: pd.DataFrame,
    columns: Sequence[str],
    threshold: float = 0.05,
) -> list[str]:
    """
    Columns that barely vary across rows, relative to their own level.

    Worth naming because of a specific, measurable failure: a ratio A/B is a
    monotone function of whichever input actually varies, so pairing a varying
    column with a near-constant one reproduces the varying column's ordering
    exactly - Spearman |rho| around 1.0 - and is rejected as redundant.

    On the bankruptcy table 34 of 95 columns fall below this threshold, and every
    single redundancy rejection in one run was a ratio involving one of them. The
    model cannot see this from descriptions or from a handful of sample values, so
    it has to be told.
    """
    found: list[str] = []
    for column in columns:
        values = pd.to_numeric(sample[column], errors="coerce").dropna()
        if len(values) < 2:
            continue
        level = abs(float(values.mean()))
        spread = float(values.std())
        if level > 0 and spread / level < threshold:
            found.append(column)
    return found


def describe_columns(
    sample: pd.DataFrame,
    columns: Sequence[str],
    descriptions: Mapping[str, str] | None = None,
    categorical: Sequence[str] = (),
    low_variation: Sequence[str] = (),
) -> str:
    """
    One block per column: dtype, range or categories, description, real values.

    Ranges come from the sample actually shown rather than the full table, so the
    numbers quoted are the numbers the model can see - a range it cannot reconcile
    with the samples below it is worse than no range at all.
    """
    descriptions = descriptions or {}
    categorical = set(categorical)
    flat = set(low_variation)
    lines: list[str] = []

    for column in columns:
        values = sample[column]
        description = str(descriptions.get(column, "")).strip()
        if column in categorical:
            seen = sorted(values.dropna().unique().tolist(), key=str)
            detail = f"categorical; values seen={seen}"
        else:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            detail = (
                f"continuous; observed range=[{_format_value(numeric.min())}, "
                f"{_format_value(numeric.max())}]"
                if len(numeric)
                else "continuous; no numeric values in the sample"
            )
        shown = ", ".join(_format_value(v) for v in values.tolist())
        # The identifier leads and is quoted exactly as it must be typed. With the
        # description first, a model reliably writes df["Debt ratio %"] instead of
        # df["X36"] - 10 of 10 proposals failed that way in one run.
        if column in flat:
            # Flagged inline, where the column is actually read, rather than in a
            # list further up that a model scanning 95 columns will skip.
            detail += "; NEARLY CONSTANT across rows"
        head = f'df["{column}"] ({values.dtype}; {detail})'
        lines.append(f"{head} - {description}\nSamples [{shown}]" if description
                     else f"{head}\nSamples [{shown}]")

    return "\n\n".join(lines)


def format_history(records: Sequence[Mapping[str, Any]], metric_name: str) -> str:
    """
    What earlier rounds produced, and what happened to it.

    This is the only learning signal in the loop: no weights change, nothing is
    fine-tuned, the proposer simply reads its own scoreboard. So failures are
    reported as plainly as successes - a block that would not run is more useful
    feedback than one that merely did not help.
    """
    pieces: list[str] = []
    for record in records:
        index = record.get("round")
        code = str(record.get("code", "")).strip()
        if not code:
            continue
        header = f"Previous code block {index}:\n```python\n{code}\n```end"

        if record.get("error"):
            pieces.append(
                f"{header}\nThis block was rejected: {record['error']}\n"
                "It was not added to the dataframe. Fix or avoid this problem."
            )
            continue

        base, cand = record.get("base_score"), record.get("candidate_score")
        delta = record.get("delta")
        pieces.append(
            f"{header}\n"
            f"Score without the feature ({metric_name}): {base:.4f}\n"
            f"Score with the feature ({metric_name}): {cand:.4f}\n"
            f"Change ({metric_name}): {delta:+.4f}\n"
            "This column was recorded as a proposal. It is not present in `df` for "
            "later blocks: every block starts from the same original columns."
        )

    return "\n\n".join(pieces) or "No previous code blocks or feedback are available."


def build_prompt(
    task_description: str,
    column_context: str,
    history: str,
    metric_name: str,
    metric_explanation: str = "",
    already_proposed: Sequence[str] = (),
    n_rows: int | None = None,
    n_features: int = 1,
    redundancy_max_abs: float | None = None,
) -> str:
    """Assemble the instructions. Everything lives in one human message."""
    proposed_block = ""
    if already_proposed:
        names = ", ".join(f'"{name}"' for name in already_proposed)
        proposed_block = (
            f"\n\nThese columns have already been proposed: {names}. They are recorded "
            "as results only and are NOT present in `df`. Be aware of them so you do "
            "not regenerate them: do not propose them again, do not reuse their names, "
            "and do not reference them in your expression. To build on one of those "
            "ideas, re-derive it inline from the original columns."
        )

    size = f" It holds {n_rows:,} rows." if n_rows else ""
    plural = "s" if n_features != 1 else ""

    # Stated up front rather than learned one rejection at a time. Each of these
    # is a rule a proposal is actually checked against, so a column that breaks
    # one is discarded before it is ever scored - the round is spent either way.
    redundancy_rule = ""
    if redundancy_max_abs is not None:
        redundancy_rule = (
            f"\n2. REDUNDANT WITH AN EXISTING COLUMN. A proposal correlating above "
            f"|rho| = {redundancy_max_abs:.2f} (Spearman) with any column listed above is "
            "discarded, however well it scores, because it re-derives information the "
            "table already holds under another name. This is by far the most common "
            "rejection, and it usually happens for a mechanical reason rather than a "
            "conceptual one:\n"
            "   A ratio A/B is a monotone function of whichever input actually varies. "
            "So dividing by a column marked NEARLY CONSTANT above - or dividing a nearly "
            "constant column by a varying one - reproduces the varying column's ordering "
            "exactly and is always rejected. Check that BOTH inputs vary substantially "
            "before proposing a ratio of them.\n"
            "   Safer shapes: combine three or more columns; take a difference of two "
            "ratios; compare a column to a group-level or distribution-level reference; "
            "or express a contrast that no single listed column can be monotone in."
        )

    rejections = f"""A proposal is checked before it is scored, and discarded if it breaks any of these. Read them as hard constraints, not advice:

1. INFINITE OR NON-NUMERIC VALUES. Any infinity, or any value that is not a number, in the new column is a rejection. NaN is allowed: the model reads it as missing, so it is the right value for a row where the feature is undefined.{redundancy_rule}
{'3' if redundancy_max_abs is not None else '2'}. EXTREME VALUES. A single value far above the column's own bulk is a rejection.

   A division needs a fallback for the rows where it is undefined. This fails:
     df["x"] = df["a"] / (df["b"] + 1e-6)                          # one b == 0 becomes ~1e6
   These work:
     df["x"] = df["a"] / df["b"].where(df["b"] != 0)               # undefined rows become NaN
     df["x"] = (df["a"] / df["b"].clip(lower=df["b"][df["b"] > 0].min())).clip(upper=<a sane bound>)
   Marking an undefined row NaN is correct - the model treats it as missing rather than as a real value. Substituting an epsilon is not.
{'4' if redundancy_max_abs is not None else '3'}. WRONG SHAPE. Exactly one new column per block, and no modification of any existing column. Name it descriptively in snake_case for what it measures - `debt_to_equity_ratio`, `cash_coverage_gap` - and never by continuing the table's own identifier scheme. `X96` is not an acceptable name: the name and the comment header are how a reader will understand the feature later, and both are required.
{'5' if redundancy_max_abs is not None else '4'}. UNKNOWN COLUMN. Every column you read must be indexed by the identifier shown in the list above - `df["X36"]`, not `df["Total debt/Total net worth"]`. Descriptions tell you what a column means; they are not keys."""
    diversity = (
        "Make the columns different from one another: several variations on one "
        "idea are worth little more than the idea alone, since each is judged on "
        "what it adds beyond the same baseline."
        if n_features > 1 else ""
    )

    return f"""The dataframe `df` is loaded and in memory.{size}

Description of the dataset in `df`:
{task_description}

Columns in `df`. Each line begins with the exact expression to index it by, followed by its type and what it means; categorical variables may be numerically encoded. Index `df` ONLY by those identifiers - never by a column's description:
{column_context}

This code is written by an expert data scientist working to improve predictions. It is pandas code that adds {n_features} new column{plural} to the dataset, one per code block.

{diversity}

The column should add new semantic information: real-world knowledge about the dataset, expressed as a combination, transformation or aggregation of existing columns. Scale and offset do not matter. Use only columns that exist, and follow the descriptions above closely.

Each proposal is evaluated independently against the same baseline: `df` always contains exactly the original columns listed above. No previously generated column is present, whether it helped or not. The evaluation metric is {metric_name}.{metric_explanation}{proposed_block}

Previously generated code and feedback:
{history}

Do not drop or modify existing columns. Generate exactly {n_features} additive column{plural}, each in its own code block. Use only the columns listed above, and never the target column. Use only vectorised pandas/numpy expressions with `df`, `pd` and `np`; do not import modules, define functions, access files, or use eval/exec.

{rejections}

Format:
```python
# (<feature name>, <short description>)
# Usefulness: <why this adds useful real-world knowledge for this prediction problem>
# Input samples: '<column>': [v1, v2, v3], '<column>': [v1, v2, v3]
df["<new_feature_name>"] = <vectorised pandas/numpy expression>
```end

Each code block adds exactly one column, starts with ```python and ends with ```end. Emit {n_features} such block{plural} and nothing else.

Codeblock{plural}:
"""


def extract_blocks(reply: str) -> list[str]:
    """
    Pull every code block out of a reply, in order.

    A round asks for several features at once, so a reply normally holds several
    fenced blocks. The ```end form is preferred because that is what the prompt
    specifies; a reply that used plain fences is still read rather than discarded,
    since the blocks themselves are usually fine.
    """
    blocks = [m.group(1).strip() for m in _CODE_FENCE_END.finditer(reply)]
    if not blocks:
        blocks = [m.group(1).strip() for m in _CODE_FENCE.finditer(reply)]
    if not blocks and "df[" in reply and "=" in reply:
        # Unfenced but recognisably code: take it. The check matters - without it
        # a refusal ("I cannot write that.") became a candidate that then failed
        # in the sandbox with an error about syntax rather than about the refusal.
        blocks = [reply.strip()]
    return [b for b in blocks if b]


def extract_code(reply: str) -> str:
    """The first block of a reply. Kept for callers that ask for exactly one."""
    blocks = extract_blocks(reply)
    return blocks[0] if blocks else reply.strip()


def parse_candidate(code: str) -> CandidateText:
    """
    Lift the rationale comments into fields.

    ``input_columns`` comes from the AST, not the "Input samples" comment: the
    comment is what the proposer believes it used, the AST is what the code
    actually reads, and they do diverge.
    """
    from discovery.sandbox import referenced_columns

    display_name = description = rationale = None
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        body = stripped.lstrip("#").strip()
        if display_name is None and body.startswith("(") and body.endswith(")"):
            try:
                parsed = ast.literal_eval(body)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, tuple) and len(parsed) >= 2:
                display_name, description = str(parsed[0]), str(parsed[1])
        elif rationale is None and body.lower().startswith("usefulness:"):
            rationale = body.split(":", 1)[1].strip()

    expression = None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            if isinstance(node, ast.Assign):
                expression = ast.unparse(node.value)

    return CandidateText(
        display_name=display_name,
        description=description,
        rationale=rationale,
        input_columns=referenced_columns(code),
        expression=expression,
    )
