"""Build feature-generation prompts and parse generated feature candidates.

The prompt is organized into explicit sections and supports an optional
additional data resource represented as JSON objects whose keys identify the
fields and whose values provide representative sample values. The additional
resource may be used to design new transaction-level or aggregated features,
while temporal leakage requirements are made explicit for any historical
feature.

Public helpers are kept compatible with the previous implementation:
    describe_columns
    format_task_context
    low_variation_columns
    format_history
    build_prompt
    extract_blocks
    extract_code
    parse_candidate
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CandidateText",
    "describe_columns",
    "format_task_context",
    "low_variation_columns",
    "format_history",
    "build_prompt",
    "extract_blocks",
    "extract_code",
    "parse_candidate",
]


_CODE_FENCE_END = re.compile(
    r"```python\s*(.*?)```end",
    re.DOTALL | re.IGNORECASE,
)
_CODE_FENCE = re.compile(
    r"```(?:python)?\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class CandidateText:
    """Structured metadata parsed from a generated feature code block."""

    display_name: str | None = None
    description: str | None = None
    rationale: str | None = None
    evidence: str | None = None
    input_columns: list[str] = field(default_factory=list)
    expression: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "description": self.description,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "input_columns": list(self.input_columns),
            "expression": self.expression,
        }


# ============================================================================
# General formatting helpers
# ============================================================================


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
    """Return columns whose relative variation is below ``threshold``."""
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


def _row_tables(
    sample: pd.DataFrame,
    columns: Sequence[str],
    label: str | None = None,
    max_width: int = 240,
) -> list[str]:
    """Render sample rows as readable markdown tables."""
    labels = [f"r{i}" for i in range(1, len(sample) + 1)]
    cells = {
        column: [_format_value(value) for value in sample[column].tolist()]
        for column in columns
    }

    classes = (
        [_format_value(value) for value in sample[label].tolist()]
        if label is not None and label in sample.columns
        else None
    )

    chunks: list[list[str]] = []
    current: list[str] = []
    width = 0

    for column in columns:
        needed = max(len(column), *(len(value) for value in cells[column])) + 3
        if current and width + needed > max_width:
            chunks.append(current)
            current = []
            width = 0
        current.append(column)
        width += needed

    if current:
        chunks.append(current)

    lead = ["row"] + ([label] if classes is not None else [])
    tables: list[str] = []

    for chunk in chunks:
        header = "| " + " | ".join([*lead, *chunk]) + " |"
        rule = "| --- " * (len(lead) + len(chunk)) + "|"
        body = []

        for index, row_label in enumerate(labels):
            front = [row_label] + (
                [classes[index]] if classes is not None else []
            )
            body.append(
                "| "
                + " | ".join(
                    [*front, *(cells[column][index] for column in chunk)]
                )
                + " |"
            )

        tables.append("\n".join([header, rule, *body]))

    return tables


def describe_columns(
    sample: pd.DataFrame,
    columns: Sequence[str],
    descriptions: Mapping[str, str] | None = None,
    categorical: Sequence[str] = (),
    low_variation: Sequence[str] = (),
    label: str | None = None,
) -> str:
    """Build the current-feature catalogue followed by example rows."""
    descriptions = descriptions or {}
    categorical_set = set(categorical)
    low_variation_set = set(low_variation)
    catalogue: list[str] = []

    for column in columns:
        values = sample[column]
        description = str(descriptions.get(column, "")).strip()

        if column in categorical_set:
            seen = sorted(values.dropna().unique().tolist(), key=str)
            detail = f"categorical; values seen={seen}"
        else:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            if len(numeric):
                detail = (
                    "continuous; observed range=["
                    f"{_format_value(numeric.min())}, "
                    f"{_format_value(numeric.max())}]"
                )
            else:
                detail = "continuous; no numeric values in the sample"

        if column in low_variation_set:
            detail += "; NEARLY CONSTANT across rows"

        head = f'df["{column}"] ({values.dtype}; {detail})'
        catalogue.append(
            f"{head} - {description}" if description else head
        )

    shown_label = (
        label if label is not None and label in sample.columns else None
    )
    tables = _row_tables(sample, list(columns), label=shown_label)

    if not tables:
        return "\n".join(catalogue)

    intro = (
        f"Example rows ({len(sample)} of them). One line is one record: the "
        "values on a line belong together, and r1 is the same record in every "
        "table below."
    )

    if shown_label is not None:
        counts = sample[shown_label].value_counts().sort_index()
        mix = ", ".join(
            f"{count} with {shown_label}={_format_value(value)}"
            for value, count in counts.items()
        )
        intro += (
            f" The first column is the outcome: {mix}. That mix comes from how "
            "these rows were chosen and is not the table's rate. Read "
            f"`{shown_label}` to see what separates the records; it is not a "
            "column of `df` and no proposal may use it."
        )

    return "\n".join(catalogue) + "\n\n" + intro + "\n\n" + "\n\n".join(tables)


def format_task_context(text: str) -> str:
    """Return domain context as a standalone text block for compatibility."""
    return (text or "").strip()


def format_history(
    records: Sequence[Mapping[str, Any]],
    metric_name: str,
) -> str:
    """Format prior generated features and their evaluation outcomes."""
    pieces: list[str] = []

    for record in records:
        index = record.get("round")
        code = str(record.get("code") or "").strip()
        if not code:
            continue

        header = f"Previous code block {index}:\n```python\n{code}\n```end"

        if record.get("error"):
            pieces.append(
                f"{header}\n"
                f"This block was rejected: {record['error']}\n"
                "It was not added to the dataframe. Fix or avoid this problem."
            )
            continue

        base = record.get("base_score")
        candidate = record.get("candidate_score")
        delta = record.get("delta")

        score_lines: list[str] = []
        if base is not None:
            score_lines.append(
                f"Score without the feature ({metric_name}): {base:.4f}"
            )
        if candidate is not None:
            score_lines.append(
                f"Score with the feature ({metric_name}): {candidate:.4f}"
            )
        if delta is not None:
            score_lines.append(
                f"Change ({metric_name}): {delta:+.4f}"
            )

        pieces.append(
            "\n".join(
                [
                    header,
                    *score_lines,
                    verdict_line(record)
                    or "This column was recorded as a proposal.",
                    "It is not present in `df` for later blocks: every block "
                    "starts from the same original columns.",
                ]
            )
        )

    return "\n\n".join(pieces) or "No previous code blocks or feedback are available."


def verdict_line(record: Mapping[str, Any]) -> str:
    """Return the validation outcome for a previous proposal."""
    outcome = record.get("outcome")
    failed_at = record.get("failed_at")

    if outcome == "accepted":
        return (
            "Validated on the full data: ACCEPTED - it cleared every gate and "
            "is kept as a candidate feature. Do not propose it, or a close "
            "variant, again."
        )

    if outcome == "rejected" and failed_at == "screen":
        return f"It was not forwarded to validation: {record.get('reason')}."

    if outcome == "rejected" and failed_at:
        return (
            f"Validated on the full data: REJECTED at the {failed_at} gate - "
            f"{record.get('reason') or 'no reason recorded'}. Do not repeat "
            "this idea; propose something that avoids the reason it failed."
        )

    return ""


# ============================================================================
# Prompt section builders
# ============================================================================


def _extract_example_column(column_context: str) -> str:
    """Return one real df column identifier for examples in the prompt."""
    match = re.search(r'df\["([^"]+)"\]', column_context)
    return match.group(1) if match else "the_column_identifier"


def _build_task_context_section(
    task_description: str,
    task_context: str = "",
    n_rows: int | None = None,
) -> str:
    """Build the task and domain-context section."""
    row_text = (
        f"The current model data contains approximately {n_rows:,} "
        "transaction rows."
        if n_rows
        else ""
    )

    parts = [
        "## 1. Task & Domain Context",
        "",
        "### Task",
        "",
        task_description.strip(),
    ]

    if row_text:
        parts.extend(["", row_text])

    if task_context.strip():
        parts.extend([
            "",
            "### Domain context",
            "",
            task_context.strip(),
        ])

    return "\n".join(parts)


def _build_current_features_section(
    column_context: str,
    example_column: str,
) -> str:
    """Build the current transaction-level feature section."""
    return f"""## 2. Current Model Features

The current model data is transaction-level.

The dataframe `df` contains the features currently available to the model.
The information below provides:

- exact dataframe column identifiers;
- data types;
- observed ranges or categorical values;
- feature descriptions; and
- representative transaction-level sample rows.

The feature catalogue is authoritative for the exact identifiers that can be
used in current-model Python expressions.

When referencing an existing model feature, always use its exact identifier,
for example:

    df["{example_column}"]

Never use a natural-language feature description as a dataframe key.

The sample rows are actual observations from the current model data. They may
be used as empirical evidence when identifying relationships between current
features and the target.

{column_context}"""


def _format_json_schema(
    schema: str | Mapping[str, Any],
) -> str:
    """Format a schema as readable JSON."""
    if isinstance(schema, str):
        return schema.strip()

    return json.dumps(
        schema,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def _build_additional_data_section(
    schema: str | Mapping[str, Any],
    source_name: str = "additional data resource",
    source_description: str = "",
) -> str:
    """Build the optional additional-data section.

    The source is provided as JSON objects where each key identifies an
    available field and the corresponding value contains representative sample
    values. These samples help the model understand field format, plausible
    values, and semantics, but do not establish relationships with the model
    target unless such evidence is explicitly supplied and aligned to the
    current transactions.
    """
    schema_text = _format_json_schema(schema)
    if not schema_text:
        return ""

    parts = [
        "## 3. Optional Additional Data Resource",
        "",
        (
            f"An additional data resource named `{source_name}` is available "
            "for feature discovery."
        ),
        "",
        (
            "The resource is represented as JSON. Each JSON key identifies an "
            "available source field, and its value contains representative "
            "sample values for that field."
        ),
    ]

    if source_description.strip():
        parts.extend([
            "",
            "### Source description",
            "",
            source_description.strip(),
        ])

    parts.extend([
        "",
        "### How to use this resource",
        "",
        (
            "Use the field names, descriptions/schema information, and sample "
            "values to identify useful information that could complement the "
            "current model features."
        ),
        "",
        "A proposed feature may:",
        "",
        "1. use information entirely from this resource;",
        "2. combine information from this resource with current `df` features;",
        "3. aggregate historical information to the customer or another entity;",
        "4. create a transaction-level feature from customer history; or",
        "5. create a customer-level or customer-time feature that can be aligned "
        "to each transaction.",
        "",
        (
            "The feature does not have to be transaction-level. Choose the "
            "natural grain of the signal, provided that it can be aligned to "
            "the transaction-level model without temporal leakage."
        ),
        "",
        "### What the sample values mean",
        "",
        (
            "The supplied values are representative examples of what the source "
            "fields may contain. They are useful for understanding data format, "
            "units, coding, possible value ranges, and whether a field appears "
            "numeric, categorical, textual, or temporal."
        ),
        "",
        "Do NOT treat the sample values as customer-level observations for the "
        "current model population unless the data is explicitly identified as "
        "aligned to the current transactions.",
        "",
        "In particular, do not infer from these samples:",
        "",
        "- correlations with the default target;",
        "- class-specific behavior;",
        "- population distributions or prevalence;",
        "- customer-level relationships; or",
        "- time trends in the model population.",
        "",
        "Do not invent fields that are not present in the supplied JSON.",
        "",
        "### Requirements for downstream data retrieval",
        "",
        (
            "When a feature uses this resource, specify enough information for "
            "another agent to retrieve and prepare the underlying data."
        ),
        "",
        "The feature specification should identify:",
        "",
        "- required source fields;",
        "- entity or join key;",
        "- feature grain;",
        "- historical lookback window, if applicable;",
        "- event timestamp or other temporal field;",
        "- as-of cutoff required to avoid leakage; and",
        "- required aggregation or transformation.",
        "",
        "### Additional data JSON",
        "",
        "```json",
        schema_text,
        "```",
    ])

    return "\n".join(parts)

def _build_feature_discovery_section(
    n_features: int,
    has_additional_data: bool,
    section_number: int,
) -> str:
    """Build the feature-generation objective."""
    plural = "s" if n_features != 1 else ""

    if has_additional_data:
        source_guidance = """
A proposed feature may use:

1. current transaction-level features in `df`;
2. variables from the optional additional data resource; or
3. a combination of current model features and additional-resource variables.

Actively look for information that is not represented by the current model
features alone.

The additional resource can support both transaction-level features and
historical customer-level features, including rolling, cumulative, recency,
frequency, trend, concentration, and behavioral measures.
"""
    else:
        source_guidance = """
A proposed feature should be derived from the current transaction-level
features available in `df`.
"""

    diversity = ""
    if n_features > 1:
        diversity = """
When generating multiple features, make them meaningfully different from one
another. Avoid producing minor variations of the same underlying idea.
"""

    return f"""## {section_number}. Feature Discovery Objective

Generate exactly {n_features} new feature{plural}.

The objective is to identify features that add **incremental predictive
information** rather than simply rename, rescale, or mechanically reproduce
existing variables.

{source_guidance}

Prioritize meaningful constructions involving:

- behavioral patterns;
- spending and payment dynamics;
- temporal deterioration or improvement;
- recency, frequency, and persistence;
- concentration or diversification;
- financial pressure relative to capacity;
- discrepancies between related signals;
- interactions across different risk dimensions;
- structural or operational patterns;
- sudden transitions or escalation; and
- customer history that adds information to an individual transaction.

Straightforward arithmetic combinations of existing variables should only be
proposed when they express a genuinely meaningful relationship that is likely
to add information beyond the incumbent feature set.

Scale and offset alone do not add useful semantic information.
{diversity}"""


def _build_temporal_rules_section(section_number: int) -> str:
    """Build leakage-prevention rules for transaction-level modeling."""
    return f"""## {section_number}. Temporal Leakage & As-of Rules

The underlying model is transaction-level.

For every transaction occurring at time `t`, a feature must use only
information that would have been available by the prediction time.

This is a hard requirement.

### Historical features

For any feature based on historical events:

- use only events at or before the transaction's as-of time;
- exclude future events;
- use a strict `< t` cutoff when same-timestamp events would not have been
  available at prediction time;
- explicitly state the lookback window;
- explicitly state the event timestamp; and
- explicitly state whether the current transaction/event is included.

### Allowed feature grains

A feature may be:

- transaction-level;
- customer-level;
- customer-time-level; or
- another aggregated level,

as long as the value assigned to each transaction is computed only from
information available at that transaction's as-of time.

### Forbidden information

Never use:

- future transactions;
- future payments;
- future delinquency events;
- future bureau information;
- future account state;
- future aggregates;
- the target; or
- any feature derived using information after the prediction timestamp.

When the available schema does not provide enough temporal information to
define a leakage-safe feature, state the missing temporal requirement instead
of inventing one."""


def _build_previous_proposals_block(
    already_proposed: Sequence[str],
) -> str:
    """Build the already-proposed-feature block."""
    if not already_proposed:
        return ""

    names = ", ".join(f"`{name}`" for name in already_proposed)

    return f"""### Already proposed features

The following features have already been proposed:

{names}

These are recorded as results only and are NOT present in `df`.

Do not regenerate them, reuse their names, or reference them as previously
generated columns.

A previous idea may inspire a genuinely different feature, but it must be
re-derived from the original available information."""


def _build_evaluation_section(
    metric_name: str,
    metric_explanation: str,
    history: str,
    already_proposed: Sequence[str],
    section_number: int,
) -> str:
    """Build the evaluation and previous-feedback section."""
    metric_detail = (
        f"\n\n{metric_explanation.strip()}"
        if metric_explanation.strip()
        else ""
    )

    previous = _build_previous_proposals_block(already_proposed)

    return f"""## {section_number}. Evaluation & Previous Feedback

Each proposal is evaluated independently against the same baseline.

The baseline contains the original transaction-level model features.
Previously generated features are not present in `df`, regardless of whether
they improved or worsened the metric.

Evaluation metric: **{metric_name}**.{metric_detail}

{previous}

### Generation history

{history}

Use previous results to avoid repeating rejected ideas and to avoid regenerating
features that have already been explored."""


def _build_redundancy_rule(
    redundancy_max_abs: float | None,
) -> str:
    """Build the optional feature-redundancy rule."""
    if redundancy_max_abs is None:
        return ""

    return f"""### Redundancy

A feature is rejected if its absolute Spearman correlation with an existing
model feature exceeds:

    |rho| = {redundancy_max_abs:.2f}

The feature should introduce new information rather than reproduce an existing
feature under another representation.

Be especially cautious with ratios involving nearly constant variables. Such
ratios can reproduce the ordering of the varying input rather than introducing
genuinely new information."""


def _build_constraints_section(
    example_column: str,
    redundancy_max_abs: float | None,
    has_additional_data: bool,
    section_number: int,
) -> str:
    """Build hard validation and implementation constraints."""
    redundancy = _build_redundancy_rule(redundancy_max_abs)

    external_rules = ""
    if has_additional_data:
        external_rules = """
### Additional-resource references

Any additional-resource field used by a proposal must exist explicitly in the
provided schema.

Do not invent fields, source tables, join keys, timestamps, or measurements.

The supplied sample values describe the source fields but are not evidence
about their relationship with the model target or current customer population
unless explicit alignment is provided.

The proposal must contain enough information for a downstream data-retrieval
agent to obtain the required source data.
"""

    return f"""## {section_number}. Hard Constraints

These are hard constraints, not suggestions. A violation causes rejection.

### 1. Numeric validity

The resulting feature must be numeric.

Infinity and other non-numeric values are not allowed.

NaN is allowed when the feature is genuinely undefined or unavailable.

### 2. Extreme values

Do not create artificial extreme values through unsafe numerical operations.

Avoid using a small epsilon merely to prevent division by zero.

Bad:

    df["x"] = df["a"] / (df["b"] + 1e-6)

Prefer an explicitly undefined value when the denominator is zero:

    df["x"] = df["a"] / df["b"].where(df["b"] != 0)

### 3. Feature shape

Each code block defines exactly one new feature.

Do not drop, rename, or modify existing model columns.

Use a descriptive `snake_case` feature name that explains what the feature
measures.

### 4. Current model feature references

Every current-model feature used in executable Python must be referenced using
the exact identifier in the feature catalogue.

For example:

    df["{example_column}"]

Never use a natural-language description as a dataframe identifier.

Never reference the target variable.

### 5. Implementation

For features that are executable using currently available dataframe columns,
use vectorized pandas/numpy operations only.

Available objects:

- `df`
- `pd`
- `np`

Do not import modules, define functions, access files, use `eval`, or use
`exec`.

{external_rules}

{redundancy}"""


def _build_output_section(
    n_features: int,
    has_additional_data: bool,
    section_number: int,
) -> str:
    """Build the final feature-specification output contract."""
    plural = "s" if n_features != 1 else ""

    if has_additional_data:
        evidence_line = (
            "# Evidence: <observed current-data evidence and/or schema-based "
            "reasoning; clearly distinguish the two>"
        )
        resource_lines = """
# Source fields: <additional-resource fields used, if any>
# Feature grain: <transaction / customer / customer-time / other>
# Join key: <customer or other alignment key, if external data is used>
# Event timestamp: <source timestamp, if applicable>
# Lookback window: <historical window, if applicable>
# As-of rule: <exact leakage-safe cutoff>
# Query requirements: <data that the downstream query agent must retrieve>"""
    else:
        evidence_line = (
            "# Evidence: <the observed rows that motivated the feature, "
            "e.g. r17, r19 against r1, r4>"
        )
        resource_lines = ""

    return f"""## {section_number}. Output Format

Generate exactly {n_features} feature{plural}.

Return one code block per feature and nothing else.

Each block must follow this structure:

```python
# (<feature name>, <short description>)
# Usefulness: <why this feature could add incremental predictive information>
{evidence_line}
{resource_lines}
df["<new_feature_name>"] = <feature calculation>
```end

### Output requirements

- Exactly one feature per code block.
- Exactly {n_features} code block{plural}.
- No explanatory text outside the code blocks.
- Start every block with ```python.
- End every block with ```end.
- Use the exact feature name in the assignment.

For current-data-only features, provide executable vectorized pandas/numpy
code.

For features requiring the optional additional data resource, the code should
serve as a **feature implementation specification**. The comments must clearly
describe the external fields, grain, temporal logic, and retrieval requirements
needed by the downstream query agent.

Codeblock{plural}:"""


# ============================================================================
# Main prompt builder
# ============================================================================


def build_prompt(
    task_description: str,
    column_context: str,
    history: str,
    metric_name: str,
    task_context: str = "",
    metric_explanation: str = "",
    already_proposed: Sequence[str] = (),
    n_rows: int | None = None,
    n_features: int = 1,
    redundancy_max_abs: float | None = None,
    additional_data_schema: str | Mapping[str, Any] | None = None,
    additional_data_name: str = "additional data resource",
    additional_data_description: str = "",
) -> str:
    """Build the feature-generation prompt.

    ``additional_data_schema`` is optional. When it is omitted or empty, the
    generated prompt does not mention an additional resource and follows the
    current transaction-level feature-generation workflow.

    When supplied, the schema-only resource becomes an additional design
    space. The LLM may propose transaction-level or aggregated features using
    the source, but must document temporal/as-of requirements so another agent
    can retrieve the underlying data without leakage.
    """
    if n_features < 1:
        raise ValueError("n_features must be at least 1")

    example_column = _extract_example_column(column_context)

    if isinstance(additional_data_schema, Mapping):
        has_additional_data = bool(additional_data_schema)
    else:
        has_additional_data = bool(
            additional_data_schema
            and str(additional_data_schema).strip()
        )

    sections: list[str] = []

    # 1. Why are we doing this?
    sections.append(
        _build_task_context_section(
            task_description=task_description,
            task_context=task_context,
            n_rows=n_rows,
        )
    )

    # 2. What does the incumbent model already know?
    sections.append(
        _build_current_features_section(
            column_context=column_context,
            example_column=example_column,
        )
    )

    # 3. Optional additional source.
    if has_additional_data:
        sections.append(
            _build_additional_data_section(
                schema=additional_data_schema,
                source_name=additional_data_name,
                source_description=additional_data_description,
            )
        )

    # Remaining section numbering depends on whether section 3 is present.
    discovery_number = 4 if has_additional_data else 3
    sections.append(
        _build_feature_discovery_section(
            n_features=n_features,
            has_additional_data=has_additional_data,
            section_number=discovery_number,
        )
    )

    if has_additional_data:
        sections.append(
            _build_temporal_rules_section(
                section_number=5,
            )
        )
        evaluation_number = 6
    else:
        evaluation_number = 5

    sections.append(
        _build_evaluation_section(
            metric_name=metric_name,
            metric_explanation=metric_explanation,
            history=history,
            already_proposed=already_proposed,
            section_number=evaluation_number,
        )
    )

    constraints_number = evaluation_number + 1
    sections.append(
        _build_constraints_section(
            example_column=example_column,
            redundancy_max_abs=redundancy_max_abs,
            has_additional_data=has_additional_data,
            section_number=constraints_number,
        )
    )

    output_number = constraints_number + 1
    sections.append(
        _build_output_section(
            n_features=n_features,
            has_additional_data=has_additional_data,
            section_number=output_number,
        )
    )

    return "\n\n".join(
        section.strip()
        for section in sections
        if section and section.strip()
    ) + "\n"


# ============================================================================
# Reply extraction and candidate parsing
# ============================================================================


def extract_blocks(reply: str) -> list[str]:
    """Extract all generated Python blocks from an LLM reply, in order."""
    blocks = [
        match.group(1).strip()
        for match in _CODE_FENCE_END.finditer(reply)
    ]

    if not blocks:
        blocks = [
            match.group(1).strip()
            for match in _CODE_FENCE.finditer(reply)
        ]

    if not blocks and "df[" in reply and "=" in reply:
        # Preserve the previous fallback for unfenced but recognisable code.
        blocks = [reply.strip()]

    return [block for block in blocks if block]


def extract_code(reply: str) -> str:
    """Return the first generated code block, or the stripped reply."""
    blocks = extract_blocks(reply)
    return blocks[0] if blocks else reply.strip()


def parse_candidate(code: str) -> CandidateText:
    """Parse rationale metadata and the feature expression from a code block.

    ``input_columns`` is derived from the AST/sandbox helper rather than from
    the human-written comments. This keeps the parsed dependency list tied to
    what the expression actually references.
    """
    from discovery.sandbox import referenced_columns

    display_name: str | None = None
    description: str | None = None
    rationale: str | None = None
    evidence: str | None = None

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
                display_name = str(parsed[0])
                description = str(parsed[1])

        elif rationale is None and body.lower().startswith("usefulness:"):
            rationale = body.split(":", 1)[1].strip()

        elif evidence is None and body.lower().startswith("evidence:"):
            evidence = body.split(":", 1)[1].strip()

    expression: str | None = None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None

    if tree is not None:
        for node in tree.body:
            if isinstance(node, ast.Assign):
                expression = ast.unparse(node.value)
                break

    return CandidateText(
        display_name=display_name,
        description=description,
        rationale=rationale,
        evidence=evidence,
        input_columns=referenced_columns(code),
        expression=expression,
    )
