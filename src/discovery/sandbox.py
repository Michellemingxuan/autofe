"""Execute LLM-written feature code without trusting it.

A discovery strategy hands back Python it did not write and nobody reviewed, so
every block goes through the same three steps before its values reach a model:

    validate_code            reject syntax that has no business in a feature
    validate_single_column   require exactly one new column, named and unused
    apply_code               execute in a scope holding only df, pd, np, math

The restriction is deliberately blunt. A feature is an expression over existing
columns; it never needs to import, define a function, open a file, or reach for
a dunder. Anything that does is far more likely to be a model going off-script
than a legitimate feature, so it is refused rather than sanitised.
"""

from __future__ import annotations

import ast
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CandidateError",
    "validate_code",
    "validate_references",
    "validate_single_column",
    "apply_code",
    "assigned_columns",
    "referenced_columns",
]

# Syntax a feature expression never legitimately needs.
_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
    ast.ClassDef, ast.With, ast.AsyncWith, ast.Try, ast.Raise,
    ast.Global, ast.Nonlocal, ast.Lambda, ast.Delete,
)
_FORBIDDEN_CALLS = frozenset(
    {"eval", "exec", "open", "compile", "globals", "locals", "input", "__import__",
     "getattr", "setattr", "delattr", "vars", "dir", "breakpoint"}
)

# Builtins a feature expression may legitimately need. Emptying __builtins__
# entirely looked safe but rejected ordinary pandas: `np.finfo(float).eps` and
# `.astype(float)` both need `float`, and 2 of 5 real proposals died on it. So
# the scope carries type constructors and pure numeric/sequence helpers, and
# nothing that can reach the filesystem, the import system or an object's
# internals. Anything omitted here is still blocked by name in validate_code.
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs", "bool", "dict", "divmod", "enumerate", "float", "int", "len",
        "list", "max", "min", "pow", "range", "round", "set", "sorted", "str",
        "sum", "tuple", "zip", "True", "False", "None",
    )
    if name not in ("True", "False", "None")
}


class CandidateError(ValueError):
    """A generated block was rejected. The message is fed back to the model."""


def validate_code(code: str) -> None:
    """Reject a block that uses syntax outside the feature-expression subset."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise CandidateError(f"Forbidden syntax: {type(node).__name__}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FORBIDDEN_CALLS
        ):
            raise CandidateError(f"Forbidden call: {node.func.id}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise CandidateError("Dunder names are forbidden.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CandidateError("Dunder attributes are forbidden.")


def _column_subscripts(tree: ast.AST, *, store: bool) -> list[str]:
    """Every df["literal"] in the tree, on the assigned or the read side."""
    wanted = ast.Store if store else ast.Load
    found: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "df"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and isinstance(node.ctx, wanted)
        ):
            found.append(node.slice.value)
    return list(dict.fromkeys(found))


def assigned_columns(code: str) -> list[str]:
    """Columns the block writes. Empty if the block does not parse."""
    try:
        return _column_subscripts(ast.parse(code), store=True)
    except SyntaxError:
        return []


def referenced_columns(code: str) -> list[str]:
    """
    Columns the block reads.

    Taken from the AST rather than any comment the model wrote: the comment says
    what it believes it used, the AST says what it actually used.
    """
    try:
        return _column_subscripts(ast.parse(code), store=False)
    except SyntaxError:
        return []


def validate_single_column(
    code: str,
    existing_columns: Iterable[str],
    reserved_names: Iterable[str] = (),
) -> str:
    """
    Require exactly one new column, and return its name.

    Enforced before execution so a block that would overwrite an input column, or
    quietly add several, is refused rather than silently changing the matrix a
    later comparison depends on.
    """
    # Parse first, so unparseable code is reported as a syntax error rather than
    # as "no assignment found", which sends the proposer looking in the wrong place.
    try:
        ast.parse(code)
    except SyntaxError as error:
        raise CandidateError(f"SyntaxError: {error}") from error

    names = assigned_columns(code)
    if len(names) != 1:
        raise CandidateError(
            f"Expected exactly one df column assignment; found {names or 'none'}."
        )
    name = names[0]
    if name in set(existing_columns):
        raise CandidateError(f"Generated feature '{name}' already exists.")
    if name in set(reserved_names):
        raise CandidateError(
            f"Generated feature '{name}' was already proposed in an earlier round. "
            "Propose a materially different feature under a different name."
        )
    return name


def validate_references(
    code: str,
    available: Iterable[str],
    aliases: Mapping[str, str] | None = None,
) -> None:
    """
    Reject a block that reads a column the frame does not have.

    Caught here so the proposer is told which identifier to use instead of
    receiving a bare KeyError. ``aliases`` maps a column's description back to
    its identifier, because the common failure is indexing by meaning -
    ``df["Debt ratio %"]`` rather than ``df["X36"]`` - and a message that names
    the right key is feedback, where a KeyError is only a symptom.
    """
    available = set(available)
    aliases = aliases or {}
    unknown = [c for c in referenced_columns(code) if c not in available]
    if not unknown:
        return

    hints = []
    for name in unknown:
        identifier = aliases.get(name) or aliases.get(name.strip())
        hints.append(
            f"'{name}' is a description, not a key - use df[\"{identifier}\"]"
            if identifier else f"'{name}' does not exist"
        )
    raise CandidateError(
        "Unknown column(s): " + "; ".join(hints) + ". Index df only by the "
        "identifiers listed in the prompt."
    )


def apply_code(frame: pd.DataFrame, blocks: Sequence[str]) -> pd.DataFrame:
    """
    Run feature blocks against a copy of `frame` and return the result.

    The execution scope holds df, pd, np, math and the safe builtins in
    _SAFE_BUILTINS - enough for real feature code, with nothing that can reach
    the filesystem, the import system or an object's internals even if
    validate_code somehow let it through. The input frame is never mutated.
    """
    df = frame.copy(deep=True)
    scope = {
        "df": df, "pd": pd, "np": np, "math": math,
        "__builtins__": dict(_SAFE_BUILTINS),
    }
    for code in blocks:
        validate_code(code)
        exec(compile(ast.parse(code), "<generated_feature>", "exec"), scope, scope)
        df = scope["df"]
    if not df.columns.is_unique:
        raise CandidateError("Generated code produced duplicate columns.")
    return df
