"""The cheap checks on a feature, before the screen's fit - C0, C1 and C2.

The screen's fit costs seconds and the full verdict minutes; these cost
milliseconds, so every feature gets them and only what passes is fitted:

* **C0 - seen before?** Against every earlier feature, this run's and earlier
  directions', on three levels:
  * *code* - the script with its variable names, comments and its own column
    name taken out (:func:`code_fingerprint`). The same code under a new name is
    refused before it runs.
  * *meaning* - the words of the name and description (:func:`similarity`). A
    feature this close to an earlier one must say what differs (``differs``)
    before it is built; a real variation is still worth screening.
  * *values* - a rank correlation with the earlier feature's values on the
    screen rows, after the build (``agent.memory.same_feature``).
* **C1 - redundancy with the model?** The largest |Spearman rho| between the
  feature and any one model column, and that column (``nearest``). Above
  ``agent.checks.max_model_rho`` the feature is a near-copy of a column the model
  already has - unless C2 says it still tracks what the model misses. Pairwise on
  purpose: it says which column, in a number anyone reads; a feature rebuilt
  from two or three columns together can pass it, and the screen's fit judges
  that one.
* **C2 - new signal?** The rank correlation with what the base model misses -
  the residual, y minus the baseline's prediction on the screen's held-out rows.
  Correlation with y alone can be high for a feature the model already has.
  Information only by default (``agent.checks.min_residual_rho: null``): on a
  few thousand held-out rows its noise is about 1/sqrt(rows), and it sees only
  monotone signal, so as a gate it dropped features the fit would have verified.
  It still decides C1's exception, and it is on the board.

C1 and C2 read one precomputed sketch of the screen rows (:class:`Sketch`),
built once per screener. The thresholds are ``agent.checks`` in the config.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from collections import Counter
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Above this |rho| a model column is the feature, ranked: C2 cannot save it.
REBUILT = 0.99

__all__ = ["code_fingerprint", "words", "similarity", "nearest_meaning", "Sketch", "sketch_for",
           "model_checks"]


# ------------------------------------------------------------------ C0: code
class _Rename(ast.NodeTransformer):
    """Local names in order of first use - `v0`, `v1` ... - so two scripts that
    differ only in what they call things compare equal."""

    KEEP = {"build", "link", "spark", "sources", "base", "raw", "pd", "np", "F", "Window"}

    def __init__(self, feature: str):
        self.names: dict[str, str] = {}
        self.feature = feature

    def _new(self, name: str) -> str:
        if name in self.KEEP:
            return name
        return self.names.setdefault(name, f"v{len(self.names)}")

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._new(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._new(node.arg)
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._new(node.name)
        node.returns = None
        # A docstring is prose, not computation.
        if node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(getattr(node.body[0], "value", None), ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
        self.generic_visit(node)
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        # `assign(<feature>=...)` names the output the same way a string does.
        if node.arg == self.feature:
            node.arg = "_feature_"
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        # The feature's own column name says nothing about what it computes.
        if node.value == self.feature:
            node.value = "<feature>"
        return node


def code_fingerprint(code: str, feature: str = "") -> str | None:
    """A fingerprint of what a script computes: equal for scripts that differ only
    in names, comments, docstrings or layout. None when the code does not parse."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    tree = _Rename(feature).visit(tree)
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(
        getattr(n, "value", None), ast.Constant) and isinstance(n.value.value, str))]
    dumped = ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=False)
    return hashlib.sha1(dumped.encode()).hexdigest()[:16]


# --------------------------------------------------------------- C0: meaning
_STOP = set("""a an and are as at be by for from has have in is it its of on or that the
their them they this to was were which with what how when why who whose per vs over
under into than then there these those each every any all one two our your not
customer customers account accounts feature features measure measures""".split())


def words(text: str) -> list[str]:
    """The content words of a name or description: snake_case split, lower case,
    stop words out, a plural's s cut."""
    out = []
    for w in re.findall(r"[a-z0-9]+", re.sub(r"_", " ", str(text or "").lower())):
        if len(w) < 3 or w in _STOP:
            continue
        out.append(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def _vectors(texts: list[str]) -> list[dict[str, float]]:
    """Word counts weighted by rarity across the texts given (tf-idf)."""
    bags = [Counter(words(t)) for t in texts]
    df = Counter(w for bag in bags for w in bag)
    n = len(bags)
    return [{w: c * (math.log((1 + n) / (1 + df[w])) + 1) for w, c in bag.items()} for bag in bags]


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(v * b.get(w, 0.0) for w, v in a.items())
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def similarity(text: str, others: list[str]) -> list[float]:
    """How close a text is to each of the others in meaning, 0 to 1: the cosine of
    their word weights. Words, not an embedding - it needs no model call, and it is
    what flags "the same idea worded alike"; a check that asks the agent to say
    what differs is the rest."""
    if not others:
        return []
    vectors = _vectors([text, *others])
    return [_cosine(vectors[0], v) for v in vectors[1:]]


def nearest_meaning(text: str, items: Iterable[dict[str, Any]],
                    key: Any = lambda i: f"{i.get('name', '')} {i.get('description', '')}"
                    ) -> tuple[dict[str, Any] | None, float]:
    """The item closest in meaning to `text`, and how close."""
    items = list(items)
    scores = similarity(text, [key(i) for i in items])
    if not scores:
        return None, 0.0
    best = int(np.argmax(scores))
    return items[best], float(scores[best])


# ------------------------------------------------------------- C1 and C2
def _ranked(values: np.ndarray) -> np.ndarray:
    """Centred ranks, a missing value at the centre - it carries no order."""
    s = pd.Series(np.asarray(values, dtype=float))
    r = s.rank().to_numpy()
    r = r - np.nanmean(r) if np.isfinite(r).any() else np.zeros(len(r))
    return np.nan_to_num(r, nan=0.0)


class Sketch:
    """The screen rows, ranked once: the model columns standardised (C1 - every
    column's Spearman rho with a feature - is one matrix product), and the
    baseline's residual on the held-out rows (C2)."""

    def __init__(self, base: pd.DataFrame, eval_rows: np.ndarray, y: np.ndarray,
                 base_predictions: np.ndarray | None):
        ranks = np.column_stack([_ranked(base[c].to_numpy()) for c in base.columns]) \
            if len(base.columns) else np.zeros((len(base), 0))
        spread = ranks.std(axis=0)
        keep = spread > 0
        self.columns = [c for c, k in zip(base.columns, keep) if k]
        ranks = ranks[:, keep]
        self.n = len(base)
        self.standard = ranks / spread[keep] if ranks.shape[1] else ranks
        self.eval_rows = np.asarray(eval_rows)
        self.residual = None
        if base_predictions is not None and len(base_predictions) == len(self.eval_rows):
            self.residual = _ranked(np.asarray(y, dtype=float)[self.eval_rows]
                                    - np.asarray(base_predictions, dtype=float))

    def explained(self, values: np.ndarray) -> dict[str, Any] | None:
        """C1: the largest |Spearman rho| with any one model column, and that column.
        None for a constant feature."""
        f = _ranked(values)
        if float(f @ f) == 0 or not self.columns:
            return None
        rho = self.standard.T @ (f / f.std()) / self.n
        j = int(np.argmax(np.abs(rho)))
        return {"max_model_rho": round(float(abs(rho[j])), 4), "nearest": self.columns[j],
                "nearest_rho": round(float(rho[j]), 4)}

    def signal(self, values: np.ndarray) -> float | None:
        """C2: the rank correlation of the feature with the baseline's residual, on
        the held-out rows. None without a residual or for a feature constant there."""
        if self.residual is None:
            return None
        f = _ranked(np.asarray(values, dtype=float)[self.eval_rows])
        r = self.residual
        denom = math.sqrt(float(f @ f) * float(r @ r))
        return round(float(f @ r) / denom, 4) if denom else None


def sketch_for(screener: Any) -> Sketch:
    """The screener's sketch, built on first use and kept with it."""
    sketch = getattr(screener, "_agent_sketch", None)
    if sketch is None:
        sketch = Sketch(screener.sample[screener.base_features], screener.eval_rows,
                        screener.y, getattr(screener, "base_predictions", None))
        screener._agent_sketch = sketch
    return sketch


def model_checks(screener: Any, values: np.ndarray, checks: Any) -> tuple[dict[str, Any], str]:
    """C1 and C2 on one feature's values (one per screen row, in order): what they
    measured, and the reason it fails - empty when it passes."""
    sketch = sketch_for(screener)
    out: dict[str, Any] = {}
    rho = sketch.signal(values)
    out["residual_rho"] = rho
    reason = ""
    if (c1 := sketch.explained(values)):
        out.update(c1)
        limit, keep = checks.max_model_rho, checks.keep_if_residual_rho
        if limit is not None and c1["max_model_rho"] > limit:
            # A ranked copy of a model column adds nothing, whatever C2 says.
            if keep is not None and rho is not None and abs(rho) >= keep \
                    and c1["max_model_rho"] < REBUILT:
                out["note"] = (f"|ρ| {c1['max_model_rho']:.2f} with `{c1['nearest']}`, but it "
                               f"tracks the residual (ρ {rho:+.3f}): fitted anyway")
            else:
                reason = (f"explained by the model: |ρ| {c1['max_model_rho']:.2f} with the model "
                          f"column `{c1['nearest']}` (above {limit:g})")
    floor = checks.min_residual_rho
    if not reason and floor is not None and rho is not None and abs(rho) < floor:
        reason = (f"no new signal: |ρ| with the base model's residual is {abs(rho):.3f}, "
                  f"below {floor:g} - it does not track what the model misses")
    return out, reason
