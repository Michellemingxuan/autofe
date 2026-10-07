"""The agent's tools: plain functions over a :class:`~agent.session.Session`.

    explore    catalog · scope · sample_rows · run_probe   look at the data
    shots      shots                                    labelled examples by category, rotated
    linkage    propose_linkage                          join a source, point in time
    screen     screen_feature                           build + score one feature
    data_pull  screen_request · propose_new_data        L3: data within the CAS scope (SQL) or beyond it
    challenge  challenge_request                        L3: record a verdict, run its construction
    report     report_findings                          end the run with a summary

The ideas are not a tool: the run's first stage (agent.agent) answers with them
as structured output (:class:`Ideas`), checked by agent.tools.ideas.

Each function owns its logic and its checks; :func:`for_agent` wraps them for
the agents SDK, binding the session, so the model only sees the arguments it
fills in. Tests call the functions directly, with no model.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from agent.session import Session
from agent.tools.challenge import challenge_request
from agent.tools.data_pull import propose_new_data, screen_request
from agent.tools.explore import catalog, run_probe, sample_rows, scope
from agent.tools.linkage import propose_linkage
from agent.tools.report import report_findings
from agent.tools.screen import screen_feature
from agent.tools.shots import shots

__all__ = ["catalog", "scope", "sample_rows", "run_probe", "shots", "propose_linkage", "screen_feature",
           "screen_request", "propose_new_data", "challenge_request", "report_findings", "for_agent",
           "Ideas", "Idea", "TOOLSETS", "END_TOOL"]

# Which tools each kind of run gets. A linkage job only explores and links.
TOOLSETS = {
    "direction": ("catalog", "scope", "sample_rows", "shots", "run_probe", "propose_linkage",
                  "screen_feature", "screen_request", "propose_new_data", "challenge_request",
                  "report_findings"),
    "linkage": ("sample_rows", "run_probe", "propose_linkage", "report_findings"),
    # A data-request run (L3 only): explore, then for each pull - propose it (the
    # SQL is validated), challenge it yourself, and it is kept or dropped. No screening.
    "l3": ("catalog", "scope", "sample_rows", "shots", "run_probe", "screen_request",
           "propose_new_data", "challenge_request", "report_findings"),
    # Before any of them: the ideas stage, which looks and answers with ideas.
    "ideas": ("catalog", "scope", "sample_rows", "shots", "run_probe"),
}
END_TOOL = "report_findings"


class Idea(BaseModel):
    """One idea - a feature to build (L1/L2) or data to request (L3)."""
    name: str = Field(description="snake_case name of the feature, or of the data request")
    level: str = Field(description="L1, L2 or L3 - one of the levels this run allows")
    lens: str = Field(description="one of the brief's lenses")
    description: str = Field(description="what it measures, and why it should carry default risk")
    data: str = Field(description="the columns and sources it uses. For L3 within the CAS "
                                  "scope: the CAS variables it needs, spelled exactly as "
                                  "scope() lists them. Beyond it: the data and where it would "
                                  "come from")
    beyond_cas: bool = Field(description="L3 only: true when the data lies outside the CAS "
                                         "scope - external information, strategies applied "
                                         "(RLA), calling or contact history, ...; false otherwise")


class Ideas(BaseModel):
    """The ideas, before any proposal: more than the run can spend, through different lenses."""
    ideas: list[Idea]


def _dump(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def for_agent(session: Session, toolset: str | None = None) -> list[Any]:
    """The run's tools as SDK function tools, bound to the session - or those of one
    toolset, such as the ideas stage's."""
    from agents import function_tool

    @function_tool(name_override="catalog")
    def _catalog(query: str = "") -> str:
        """Search model columns, extra sources and the CAS scope; empty query = overview.

        Args:
            query: keywords, e.g. "payment returned".
        """
        return _dump(catalog(session, query))

    @function_tool(name_override="scope")
    def _scope(query: str = "", status: str = "", table: str = "") -> str:
        """The CAS variables: name, table, description, and whether the model uses them.

        Args:
            query: keywords in a variable's name or description, e.g. "decline".
            status: in_model, in_model_unused, or unused_raw (not used by the model).
            table: one CAS table.
        """
        return _dump(scope(session, query, status, table))

    @function_tool(name_override="sample_rows")
    def _sample_rows(source: str = "model_database", n: int = 5,
                     columns: list[str] | None = None) -> str:
        """A few rows from the model database (labelled examples) or a source.

        Args:
            source: model_database, or a source name from catalog.
            n: rows to show, at most 50.
            columns: optional subset of columns.
        """
        return sample_rows(session, source, n, columns)

    @function_tool(name_override="shots")
    def _shots(category: str = "") -> str:
        """Labelled example rows, by category. Empty category = the list of categories.

        Args:
            category: a category key or name from the list, e.g. "clustering".
        """
        return shots(session, category)

    @function_tool(name_override="run_probe")
    def _run_probe(code: str, purpose: str) -> str:
        """Run exploratory code over base, raw[...] and sources[...]; returns printed output.
        raw[...] here is the first 200,000 rows of each source - for looking, not computing.

        Args:
            code: the script; print results or set `result` to a frame.
            purpose: one line saying what you are checking.
        """
        return _dump(run_probe(session, code, purpose))

    @function_tool(name_override="propose_linkage")
    def _propose_linkage(source: str, code: str, time_column: str,
                         rule: str = "strict") -> str:
        """Propose the point-in-time join of a source to the model ids; the analyst confirms.

        Args:
            source: the source name.
            code: a script defining link(base_ids, source).
            time_column: the source's event-date column, kept in the output.
            rule: strict (event < as_of) or inclusive (event <= as_of).
        """
        return _dump(propose_linkage(session, source, code, time_column, rule))

    @function_tool(name_override="screen_feature")
    def _screen_feature(name: str, description: str, level: str, code: str) -> str:
        """Build one feature on the screen rows and score it against base. One attempt.

        Args:
            name: the new column's name.
            description: one sentence on what it measures and why it matters.
            level: L1 or L2.
            code: a script defining build(spark, sources, base).
        """
        return _dump(screen_feature(session, name, description, level, code))

    @function_tool(name_override="screen_request")
    def _screen_request(gap: str, sql: str, source_name: str, features: str = "") -> str:
        """L3: propose a data pull from the CAS scope - the rationale and the BigQuery SQL.

        Args:
            gap: the rationale - what is missing, and why the direction needs it.
            sql: the BigQuery SQL; select only the key, the event date and the needed columns.
            source_name: snake_case name for the new source.
            features: the features this data would enable, one per line.
        """
        return _dump(screen_request(session, gap, sql, source_name, features))

    @function_tool(name_override="propose_new_data")
    def _propose_new_data(gap: str, source_name: str, data: str, features: str = "") -> str:
        """L3 beyond the CAS scope: propose data the bank or the market may hold outside CAS -
        external information, strategies applied (RLA), calling or contact history ... No SQL:
        the idea is the deliverable. It is challenged like any request.

        Args:
            gap: the rationale - what is missing, and why the direction needs it.
            source_name: snake_case name for the new source.
            data: the data it needs and where it would come from - the system or team that
                holds it, its grain, how far back it should go.
            features: the features this data would enable, one per line.
        """
        return _dump(propose_new_data(session, gap, source_name, data, features))

    @function_tool(name_override="challenge_request")
    def _challenge_request(intent: str, verdict: str, reasoning: str,
                           columns: list[str] | None = None, code: str = "") -> str:
        """Challenge your own proposal: can its information be built from the data that exists
        now? Record the verdict; a construction you give is run on the screen rows, and one
        that runs drops the request.

        Args:
            intent: the proposal, e.g. "R1".
            verdict: constructible, partly or new - can its information be built from current data?
            reasoning: two or three sentences - what exists now, and what does not.
            columns: the existing columns a construction uses.
            code: required for constructible, welcome for partly: a script defining
                build(spark, sources, base) returning the id column and one column, `proxy`.
        """
        return _dump(challenge_request(session, intent, verdict, reasoning, columns, code))

    @function_tool(name_override="report_findings")
    def _report_findings(summary: str) -> str:
        """End the run with your findings. The run stops here.

        Args:
            summary: per feature this run verified, what it measures and why; what failed;
                data requests. Only this run's work - earlier directions' features are not
                yours to report. For a linkage job, one line on how the join works.
        """
        return _dump(report_findings(session, summary))

    tools = {"catalog": _catalog, "scope": _scope, "sample_rows": _sample_rows, "shots": _shots,
             "run_probe": _run_probe,
             "propose_linkage": _propose_linkage, "screen_feature": _screen_feature,
             "screen_request": _screen_request, "propose_new_data": _propose_new_data,
             "challenge_request": _challenge_request,
             "report_findings": _report_findings}
    return [tools[name] for name in TOOLSETS[toolset or ("l3" if session.l3_only else session.kind)]]
