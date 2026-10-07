"""The parts of the prompt that are filled in from the workspace and the run.

Each function returns one section, by the name of the template field it fills -
`{memory}` in a brief is :func:`memory`, `{ideas}` is :func:`ideas`, and so on.
The wording around them is in the templates beside this file.

    skills        the run's skills, in full (agent/skills/*.md)
    shot_list     the shot categories, for the agent to read with `shots`
    gates         the analyst's minimum gains, in words
    scope_notes   the analyst's notes on the data - the CAS scope and beyond it, verbatim
    quota         a mixed run's target, split across levels
    memory        what earlier directions proposed - not to be repeated
    ideas         the ideas stage: the lenses, this run's focus, the L3 rule and CAS list
    current_data  the model columns and linked sources - what a challenge checks against
    columns       the columns a feature can read - base and the run's sources, described
"""

from __future__ import annotations

from typing import Any

from agent.memory import open_raw_columns, prior_features, prior_requests, requested_columns
from agent.session import Session
from agent.tools.ideas import LENSES, MIN_FOCUS, MIN_LENSES, focus_lenses, ideas_in_round

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
    return ("## Notes from the user on the data\nOn the CAS scope and on data beyond it. "
            f"Follow these when choosing data and writing L3 requests.\n\n{body}\n\n")


def quota(session: Session) -> str:
    """A mixed run's target, split across levels."""
    out = ""
    if len(session.quota) > 1:
        split = ", ".join(f"{lv} x{n}" for lv, n in session.quota.items())
        out = (f"* Your target of {session.K} results is split by level: {split} - drawn at "
               "random, weighted to L2 (features on the additional data) > L1 > L3 (data "
               "requests). A level whose target is met takes no more proposals; the attempts "
               f"({session.max_attempts}) are shared.")
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
            what = (f"{', '.join(r['tables'] or [])}: {', '.join(r['columns'] or [])}"
                    if r.get("tables") else f"beyond CAS: {str(r.get('data') or '')[:160]}")
            lines.append(f"* `{r['source_name']}` - {what} ({r['status']}"
                         + (f": \"{r['note']}\"" if r.get("note") else "") + ")")
        requested = requested_columns(session)
        taken = sorted({f"{t}.{c}" for t, c in requested})
        if taken and not open_raw_columns(session, requested):
            lines.append("Every unused_raw CAS column is already asked for. Within the CAS "
                         "scope, ask only for what the model and those requests lack (a model "
                         "variable's history, a finer grain); the room is beyond it.")
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
    n = ideas_in_round(session)
    return f"""## Ideas first - diverge, then choose
The run works in rounds. Each opens with your ideas: look at the data you need,
then answer with {n} to {session.params.ideas_per_round} ideas, each through one lens below, at least
{min(MIN_LENSES, n)} different lenses in all. Each idea has a name, a level, a lens, a
description (what it measures, why it should carry risk) and its data. This run
leans on **{", ".join(focus)}**: at least {min(MIN_FOCUS, n)} ideas use them. The
obvious idea is fine once; look for the angle earlier directions did not take.
Then propose them, by their names, the most promising first. When a round's
ideas are used and the target is not reached, the next round asks for new ideas -
with what worked and what failed so far in front of you.
{_l3_idea_rule(session)}
{lenses}

"""


def _l3_idea_rule(session: Session) -> str:
    """How an L3 idea must be written, with a real variable as the example."""
    if "L3" not in session.quota and not session.l3_only:
        return ""
    return f"""
An L3 idea is a data request, of one of two kinds:
* **Within the CAS scope** (`beyond_cas` false): in its `data`, the CAS variables
  it needs, spelled exactly as listed below. It is proposed with BigQuery SQL
  (screen_request), which is screened against the CAS columns.
* **Beyond the CAS scope** (`beyond_cas` true): data the bank or the market
  holds outside CAS - external information (bureau triggers, macro, merchant
  or industry data), the strategies applied to an account (RLA, line actions,
  collections treatment), calling and contact history, servicing and complaints,
  ... In its `data`, what it needs and where it would come from. No SQL: it is
  proposed as an idea (propose_new_data), and the idea is what counts - be
  creative, and specific about the signal.
Both are challenged: can the data that exists now already supply it? An idea
the model database and the linked sources already carry is an L1/L2 idea.

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
             "", f"`base` - one row per id: `{ws.id_col}` and {len(ws.base_features)} base features. "
             "It has NO `as_of` column: every `sources[...]` row carries the as-of date of "
             "its id, or parse it from the id (see the id format).",
             "Base features:"]
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
