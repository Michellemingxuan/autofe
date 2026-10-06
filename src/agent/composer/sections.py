"""The parts of the prompt that are filled in from the workspace and the run.

Each function returns one section, by the name of the template field it fills -
`{memory}` in a brief is :func:`memory`, `{ideas}` is :func:`ideas`, and so on.
The wording around them is in the templates beside this file.

    skills        the run's skills, in full (agent/skills/*.md)
    shot_list     the shot categories, for the agent to read with `shots`
    gates         the analyst's minimum gains, in words
    scope_notes   the analyst's notes on the CAS scope, verbatim
    quota         a mixed run's split of intents across levels
    memory        what earlier directions proposed - not to be repeated
    ideas         the ideas stage: the lenses, this run's focus, the L3 rule and CAS list
    current_data  the model columns and linked sources - what a challenge checks against
    columns       the columns a feature can read - base and the run's sources, described
"""

from __future__ import annotations

from typing import Any

from agent.memory import open_raw_columns, prior_features, prior_requests, requested_columns
from agent.session import Session
from agent.tools.ideas import LENSES, MIN_FOCUS, MIN_LENSES, _wanted, focus_lenses

__all__ = ["skills", "shot_list", "gates", "scope_notes", "quota", "memory", "ideas", "current_data",
           "columns"]

# Base features listed in the brief, at most - catalog(query) finds the rest.
COLUMNS_LISTED = 200


def skills(session: Session) -> str:
    """The run's skills, in full: their rules are binding."""
    return "\n\n".join(f"## Skill: {name}\n{skill['body']}"
                         for name, skill in session.skills.items())


def shot_list(ws: Any) -> str:
    from agent.tools.shots import categories

    cats = categories(ws)
    return ", ".join(f"`{c.key}` ({c.name})" for c in cats) if cats else "none set up"


def gates(session: Session) -> str:
    p = session.params
    gates = [f"Gini gain above {p.min_gini_gain:+.4f}"]
    if p.min_capture_gain is not None:
        pct = session.ws.cfg.discovery.capture_percent * 100
        gates.append(f"capture-rate gain (top {pct:g}%) above {p.min_capture_gain:+.4f}")
    return "; ".join(gates)


def scope_notes(ws: Any, limit: int = 8000) -> str:
    """The user's guidance on the CAS scope, verbatim - it outranks the skills."""
    notes = ws.scope_notes()
    if not notes:
        return ""
    body = "\n\n".join(f"### {name}\n{text.strip()}" for name, text in notes)
    if len(body) > limit:
        body = body[:limit] + "\n[... truncated]"
    return ("## Scope notes from the user\nFollow these when choosing data and writing "
            f"L3 requests.\n\n{body}\n\n")


def quota(session: Session) -> str:
    """A mixed run's split of its intents across levels, and a level left out."""
    out = ""
    if len(session.quota) > 1:
        split = ", ".join(f"{lv} x{n}" for lv, n in session.quota.items())
        out = (f"* Your {session.K} intents are split by level: {split} - drawn at random, "
               "weighted to L2 (features on the additional data) > L1 > L3 (data requests). "
               "Each level's share is its own; one left unused does not move to another.")
    if "L3" in session.params.levels and "L3" not in session.quota:
        out += ("\n* L3 has no share: every unused_raw CAS column is already requested by "
                "earlier directions. Do not give L3 ideas or request data pulls.")
    return out


def memory(session: Session, limit: int = 40) -> str:
    """The brief's section on earlier directions; empty when there are none."""
    feats = prior_features(session.ws, session.run_id)[-limit:]
    reqs = prior_requests(session.ws, session.run_id)[-limit:]
    if not feats and not reqs:
        return ""
    lines = ["## Earlier directions - build on them, do not repeat them",
             "Read this before you give your ideas: it is what is already proposed. Plan "
             "them around it."]
    if feats:
        lines.append("Features already proposed (✓ verified, ✗ not):")
        for f in feats:
            mark = "✓" if f["verified"] else "✗"
            gain = f" {f['delta']:+.4f}" if isinstance(f["delta"], (int, float)) else ""
            lines.append(f"* {mark} `{f['name']}` ({f['level']}{gain}) - {f['description']}")
        lines.append("Do not propose these again, under any name. A variation that measures "
                     "something else - another window (30 vs 90 days), another ratio, another "
                     "source - is fine: say what is new. A ✗ feature is a lead only with a "
                     "real change.")
    if reqs:
        lines.append("Data already requested - asked for, not yet pulled. A request is not a "
                     "source: features cannot read it until the analyst drops its result in "
                     "the additional data folder, where it appears as a source.")
        for r in reqs:
            lines.append(f"* `{r['source_name']}` - {', '.join(r['tables'] or [])}: "
                         f"{', '.join(r['columns'] or [])} ({r['status']}"
                         + (f": \"{r['note']}\"" if r.get("note") else "") + ")")
        requested = requested_columns(session)
        taken = sorted({f"{t}.{c}" for t, c in requested})
        if taken and not open_raw_columns(session, requested):
            lines.append("Every unused_raw CAS column is already asked for - no new raw data is "
                         "left to request. Ask only for what the model and those requests lack "
                         "(a model variable's history, a finer grain), or report that the "
                         "scope is used up for this direction.")
        if taken:
            lines.append("CAS columns already asked for: " + ", ".join(f"`{c}`" for c in taken)
                         + ". scope() marks them `requested`. Build L3 ideas on other columns. "
                         "A taken column belongs in a new request only beside new columns "
                         "that add information. If the direction needs only taken columns, "
                         "say so in your report instead of asking again.")
    return "\n".join(lines) + "\n\n"


def ideas(session: Session) -> str:
    """The brief's section on ideas: the lenses, and this run's focus."""
    focus = focus_lenses(session)
    lenses = "\n".join(f"* `{k}`{' (focus)' if k in focus else ''} - {v}" for k, v in LENSES.items())
    return f"""## Ideas first - diverge, then choose
The run opens with your ideas: look at the data you need, then answer with at
least {_wanted(session)} ideas - more than you can spend - each through one lens
below, at least {MIN_LENSES} different lenses in all. Each idea has a name, a level,
a lens, a description (what it measures, why it should carry risk) and its data.
This run leans on **{", ".join(focus)}**: at least {MIN_FOCUS} ideas use them. The
obvious idea is fine once; look for the angle earlier directions did not take.
Then propose the most promising and most different ones, by their names.
{_l3_idea_rule(session)}
{lenses}

"""


def _l3_idea_rule(session: Session) -> str:
    """How an L3 idea must be written, with a real variable as the example."""
    if "L3" not in session.quota and not session.l3_only:
        return ""
    return f"""
An L3 idea is a data request: in its `data`, write the CAS variables it needs,
spelled exactly as listed below. An idea that names no CAS variable is built
from the data the model already has - that is an L1/L2 idea, and is sent back.
If the direction finds nothing in these variables, take the angle they do allow.

{_cas_list(session)}
"""


# CAS variables listed in the brief, at most - scope(query=...) finds the rest.
CAS_LISTED = 80


def _cas_list(session: Session) -> str:
    """The unused_raw CAS variables an L3 idea can build on, those already requested
    marked - in the brief, so the ideas need no look-up to be grounded."""
    from agent.memory import requested_columns

    scope = session.ws.scope()
    if not len(scope):
        return "The CAS scope is empty: no L3 idea is possible."
    keys = {c for t in set(scope["table"].astype(str))
            for k in ("partition", "identifiers") for c in session.ws.table_profile(t)[k]}
    rows = scope[(scope["status"] == "unused_raw") & ~scope["variable"].isin(keys)
                 & ~scope["variable"].astype(str).str.endswith("pkey")]
    taken = requested_columns(session)
    lines = [f"* `{r.variable}` ({r.table}) - {r.description}"
             + (" - requested already" if (str(r.table).lower(), str(r.variable).lower()) in taken else "")
             for r in rows.head(CAS_LISTED).itertuples(index=False)]
    more = (f"\n... and {len(rows) - CAS_LISTED} more: scope(query=...) finds them."
            if len(rows) > CAS_LISTED else "")
    return "The unused_raw CAS variables:\n" + "\n".join(lines) + more


def current_data(session: Session) -> str:
    """The data that exists now - what a challenge checks a proposal against."""
    ws = session.ws
    base = "\n".join(f"  * `{c}` - {ws.descriptions.get(c, '')}" for c in ws.base_features)
    sources = []
    for name in [s for s in session.params.sources if ws.linkage_path(s).exists()]:
        src = ws.sources()[name]
        cols = "; ".join(f"`{c}` ({d})" for c, d in src.columns.items())
        sources.append(f"  * `{name}` - joined as id, as_of + {cols}")
    return (f"## The data that exists now\nModel database - id `{ws.id_col}`, "
            f"{len(ws.base_features)} columns:\n{base}\n\nLinked sources (event rows "
            f"before each row's as-of date):\n{chr(10).join(sources) or '  (none)'}\n")


def _examples(values: Any, n: int = 3) -> str:
    """A few distinct example values, short."""
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and v != v) or v in out:
            continue
        out.append(round(v, 4) if isinstance(v, float) else v)
        if len(out) == n:
            break
    return ", ".join(str(v)[:24] for v in out)


def columns(session: Session) -> str:
    """The columns a feature script can read, by name, with what each holds - so the
    code names real columns: the base features (``base``), then each of the run's
    sources (``sources[...]``, once linked), with example values."""
    ws = session.ws
    base = ws.base_features[:COLUMNS_LISTED]
    lines = ["## The columns you can use",
             "Use these names exactly - a column not listed here does not exist. Example "
             "values are from the screen rows (base) or the source's sample.",
             "", f"`base` - one row per id: `{ws.id_col}` and {len(ws.base_features)} base features:"]
    for c in base:
        desc = ws.descriptions.get(c, "") or "(no description)"
        lines.append(f"  * `{c}` - {desc}  e.g. {_examples(ws.screen[c].head(200))}")
    if len(ws.base_features) > COLUMNS_LISTED:
        lines.append(f"  ... and {len(ws.base_features) - COLUMNS_LISTED} more: "
                     "catalog(query) finds them.")
    sources = ws.sources()
    run_sources = [s for s in session.params.sources if s in sources]
    if run_sources:
        lines += ["", "`sources[\"<name>\"]` - event rows, once the source's linkage is "
                      f"confirmed: `{ws.id_col}`, `as_of`, and its own columns:"]
        for name in run_sources:
            src = sources[name]
            state = ("linked" if ws.linkage_path(name).exists() else
                     "needs linkage first" if src.usable else "schema only - no data")
            lines.append(f"  * `{name}` ({state}):")
            for c, d in src.columns.items():
                lines.append(f"      * `{c}` - {d or '(no description)'}"
                             f"  e.g. {_examples(src.samples.get(c, []))}")
    return "\n".join(lines) + "\n"
