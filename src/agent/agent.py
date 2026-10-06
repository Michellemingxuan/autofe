"""The one agent and its run: build it, drive its stages, stream what it does.

What the agent reads - its brief, the messages that start and move each stage,
its skills - is put together by :mod:`agent.composer`; what it can do is in :mod:`agent.tools`.
This module only wires them to the model and runs the conversation. The run ends
when the agent calls ``report_findings`` (the SDK stops there) or the turn
backstop is hit.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

from agents import (Agent, FunctionToolResult, ItemHelpers, ModelSettings, Runner,
                    ToolsToFinalOutputResult)

from agent import tools
from agent.llm import build_model
from agent.composer import compose, message
from agent.session import Session
from agent.tools.ideas import record_ideas

__all__ = ["build_agent", "run_direction", "IDEAS_TRIES"]


def _stop_on_accepted_report(_context: Any, results: list[FunctionToolResult]) -> ToolsToFinalOutputResult:
    """Stop when report_findings is accepted - a refused report (verdicts still
    missing) goes back to the agent like any other tool error."""
    for result in results:
        if result.tool.name == tools.END_TOOL:
            try:
                accepted = json.loads(result.output).get("ok") is True
            except (TypeError, ValueError):
                accepted = True
            if accepted:
                return ToolsToFinalOutputResult(is_final_output=True, final_output=result.output)
    return ToolsToFinalOutputResult(is_final_output=False)


def build_agent(session: Session, stage: str = "propose") -> Agent:
    """The run's agent. A direction has two stages with one brief: the ideas,
    which may look at the data and answers with its ideas as structured output, then
    the proposals, with every tool."""
    name = "linkage_writer" if session.kind == "linkage" else "feature_engineer"
    instructions = compose(session)
    if stage == "ideas":
        return Agent(
            name=name,
            instructions=instructions,
            tools=tools.for_agent(session, "ideas"),
            model=build_model(replace(session.ws.cfg.agent.llm, model=session.params.model)),
            output_type=tools.Ideas,
        )
    return Agent(
        name=name,
        instructions=instructions,
        tools=tools.for_agent(session),
        model=build_model(replace(session.ws.cfg.agent.llm, model=session.params.model)),
        # A data-request run may send several calls in one response - the requests,
        # then the verdicts, do not depend on each other. The SDK still runs the
        # (synchronous) tools one by one, in order. Feature screening stays one
        # call at a time: each result should shape the next idea.
        model_settings=ModelSettings(parallel_tool_calls=session.l3_only),
        tool_use_behavior=_stop_on_accepted_report,
    )


NUDGES = 3


def _stage_done(session: Session) -> bool:
    """Whether the agent ended the run by reporting - the only way it may end."""
    return session.finished or session.cancelled


async def _run_stage(session: Session, task: str) -> None:
    """One agent stage. A model that answers with text alone has ended its run in
    the SDK's eyes; that is not a report, so the same conversation continues
    with a nudge - up to NUDGES times - instead of the stage ending empty."""
    items: Any = task
    for attempt in range(NUDGES + 1):
        stream = await _run(session, items)
        if _stage_done(session):
            return
        if attempt == NUDGES:
            break
        session.emit("agent_nudged", attempt=attempt + 1,
                     reason="stopped with a message and no report")
        items = stream.to_input_list() + [{"role": "user", "content": message("nudge")}]


IDEAS_TRIES = 3


async def _ideas_stage(session: Session, task: str) -> list[Any]:
    """The ideas stage: the agent may look, then answers with its ideas as
    structured output. The checks (agent.tools.ideas) send a short or one-sided
    answer back, up to IDEAS_TRIES times. Returns the conversation so far."""
    agent = build_agent(session, "ideas")
    items: Any = task
    for attempt in range(1, IDEAS_TRIES + 1):
        stream = await _run(session, items, agent)
        if session.cancelled:
            break
        answer = stream.final_output
        result = (record_ideas(session, [i.model_dump() for i in answer.ideas])
                  if isinstance(answer, tools.Ideas)
                  else {"ok": False, "error": "answer with your ideas in the structured form"})
        if result["ok"]:
            return stream.to_input_list()
        session.emit("ideas_sent_back", attempt=attempt, error=result["error"])
        items = stream.to_input_list() + [{"role": "user",
                                           "content": message("sent_back", error=result["error"])}]
    # No spread that passes: the run goes on rather than ending empty, ungated.
    session.ideas_required = False
    return stream.to_input_list()


async def _direction(session: Session) -> None:
    """A direction: the ideas, then the proposals, in one conversation."""
    items = await _ideas_stage(session, message("direction", direction=session.direction)
                               + "\n\n" + message("ideas"))
    if session.cancelled:
        return
    names = ", ".join(i["name"] for i in session.ideas) or "none passed the checks"
    await _run_stage(session, items + [{"role": "user", "content": message("propose", names=names)}])


async def _run(session: Session, task: Any, agent: Agent | None = None) -> Any:
    agent = agent or build_agent(session)
    stream = Runner.run_streamed(
        agent,
        input=task,
        max_turns=40 if session.kind == "linkage" else session.ws.cfg.agent.max_turns)
    started: dict[str, str] = {}
    # One model response yields its tool calls before its text; hold the calls
    # until the response's other items arrive, so the narration reads first.
    pending: list[Any] = []

    def flush() -> None:
        for raw in pending:
            call_id = getattr(raw, "call_id", None) or getattr(raw, "id", "")
            started[call_id] = getattr(raw, "name", "?")
            session.emit("tool_started", call_id=call_id, tool=started[call_id],
                         args=getattr(raw, "arguments", ""))
        pending.clear()

    async for event in stream.stream_events():
        if session.cancelled:
            stream.cancel()
            session.end("", stopped_because="stopped by the user")
            break
        if event.type != "run_item_stream_event":
            continue
        item = event.item
        if item.type == "tool_call_item":
            pending.append(item.raw_item)
            continue
        if item.type == "message_output_item":
            text = ItemHelpers.text_message_output(item)
            # A structured answer (the ideas) is recorded as itself, not as talk.
            if text.strip() and not (agent.output_type and text.lstrip().startswith("{")):
                session.emit("agent_message", text=text)
        flush()
        if item.type == "tool_call_output_item":
            raw = item.raw_item
            call_id = raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", "")
            session.emit("tool_completed", call_id=call_id, tool=started.get(call_id, "?"),
                         output=str(item.output)[:4000])
    flush()
    return stream


def run_direction(session: Session) -> Session:
    """Run one session - a direction, or a linkage job - to the end."""
    session.ideas_required = session.kind == "direction"
    session.start()
    # The run keeps the brief it was given - the prompt, as the model read it.
    (session.run_dir / "brief.md").write_text(compose(session))
    try:
        if session.kind == "linkage":
            asyncio.run(_run_stage(session, message("linkage", source=session.source)))
        else:
            asyncio.run(_direction(session))
    except Exception as error:  # noqa: BLE001 - recorded, then re-raised
        session.emit("run_error", error=f"{type(error).__name__}: {error}")
        session.end("", stopped_because=f"error: {type(error).__name__}")
        raise
    if not session.finished:
        session.end("", stopped_because="the agent stopped without reporting")
    return session
