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
from pathlib import Path
from typing import Any

from agents import (Agent, AgentOutputSchema, FunctionToolResult, ItemHelpers, ModelSettings,
                    RunConfig, Runner, ToolsToFinalOutputResult)
from agents.run import CallModelData, ModelInputData
from agents.exceptions import ModelBehaviorError

from agent import tools
from agent.llm import build_model
from agent.llm_trace import TraceHooks, store_for
from agent.tools.linkage import ensure_linked
from agent.composer import compose, message
from agent.session import Session
from agent.board import order_ideas, session_ideas
from agent.tools.ideas import MIN_FOCUS, focus_engines, ideas_in_round, record_ideas

__all__ = ["build_agent", "run_direction", "IDEAS_TRIES"]


def _stopper(session: Session):
    """When a proposal stage ends: an accepted report_findings - a refused one goes
    back to the agent like any other tool error - or a round whose ideas are used,
    so the runner can ask for the next round."""
    def stop(_context: Any, results: list[FunctionToolResult]) -> ToolsToFinalOutputResult:
        for result in results:
            if result.tool.name == tools.END_TOOL:
                try:
                    accepted = json.loads(result.output).get("ok") is True
                except (TypeError, ValueError):
                    accepted = True
                if accepted:
                    return ToolsToFinalOutputResult(is_final_output=True,
                                                    final_output=result.output)
        if session.round_spent:
            return ToolsToFinalOutputResult(is_final_output=True, final_output="round spent")
        return ToolsToFinalOutputResult(is_final_output=False)
    return stop


class FirstAnswer(AgentOutputSchema):
    """A structured answer, read from its first JSON value. A model - or its
    transport - sometimes sends the same answer twice (`{...}{...}`), which is not
    JSON; the first copy is the answer, so it is read and the repeat ignored.
    Anything else that does not parse is still an error."""

    def validate_json(self, json_str: str) -> Any:
        try:
            return super().validate_json(json_str)
        except ModelBehaviorError as error:
            text = json_str.strip()
            try:
                _, end = json.JSONDecoder().raw_decode(text)
            except ValueError:
                raise error from None
            rest = text[end:].strip()
            if rest and not rest.startswith("{"):
                raise error from None
            return super().validate_json(text[:end])


def _model(session: Session):
    agent = session.ws.cfg.agent
    return build_model(replace(agent.llm, model=session.params.model),
                       log_dir=Path(agent.run_dir) / "llm_logs")


def build_agent(session: Session, stage: str = "propose") -> Agent:
    """The run's agent. A direction has two stages with one brief: the ideas,
    which may look at the data and answers with its ideas as structured output, then
    the proposals, with every tool."""
    name = "linkage_writer" if session.kind == "linkage" else "feature_engineer"
    instructions = compose(session, "ideas" if stage == "ideas" else "propose")
    if stage == "ideas":
        return Agent(
            name=name,
            instructions=instructions,
            tools=tools.for_agent(session, "ideas"),
            model=_model(session),
            output_type=FirstAnswer(tools.Ideas),
        )
    return Agent(
        name=name,
        instructions=instructions,
        tools=tools.for_agent(session),
        model=_model(session),
        # A data-request run may send several calls in one response - the requests,
        # then the verdicts, do not depend on each other. The SDK still runs the
        # (synchronous) tools one by one, in order. Feature screening stays one
        # call at a time: each result should shape the next idea.
        model_settings=ModelSettings(parallel_tool_calls=session.l3_only),
        tool_use_behavior=_stopper(session),
    )


NUDGES = 3


def _stage_done(session: Session) -> bool:
    """Whether the agent ended the run by reporting - the only way it may end."""
    return session.finished or session.cancelled


async def _run_stage(session: Session, task: Any) -> list[Any]:
    """One proposal stage, until a report or a spent round. A model that answers
    with text alone has ended its run in the SDK's eyes; that is not a report, so
    the same conversation continues with a nudge - up to NUDGES times - instead of
    the stage ending empty. Returns the conversation so far."""
    items: Any = task
    for attempt in range(NUDGES + 1):
        stream = await _run(session, items)
        if _stage_done(session) or session.round_spent:
            break
        if attempt == NUDGES:
            break
        session.emit("agent_nudged", attempt=attempt + 1,
                     reason="stopped with a message and no report")
        items = stream.to_input_list() + [{"role": "user", "content": message("nudge")}]
    return stream.to_input_list()


IDEAS_TRIES = 3


async def _ideas_stage(session: Session, task: str) -> list[Any]:
    """The ideas stage: the agent may look, then answers with its ideas as
    structured output. The checks (agent.tools.ideas) send a short or one-sided
    answer back, up to IDEAS_TRIES times - each time with the ideas it gave, so the
    trace shows them. Returns the conversation so far."""
    agent = build_agent(session, "ideas")
    items: Any = task
    last: list[dict[str, Any]] = []
    for attempt in range(1, IDEAS_TRIES + 1):
        try:
            stream = await _run(session, items, agent)
        except ModelBehaviorError as error:
            # An answer that does not parse is sent back like one that fails the
            # checks - the run does not end on it.
            problem = ("your answer was not one valid JSON object of the ideas schema "
                       f"({str(error)[:160]}...). Answer once, with one JSON object")
            session.emit("ideas_sent_back", attempt=attempt, error=problem, ideas=[])
            items = (items if isinstance(items, list) else [{"role": "user", "content": items}]) \
                + [{"role": "user", "content": message("sent_back", error=problem)}]
            continue
        if session.cancelled:
            return stream.to_input_list()
        answer = stream.final_output
        given = [i.model_dump() for i in answer.ideas] if isinstance(answer, tools.Ideas) else []
        last = given or last
        result = (record_ideas(session, given) if isinstance(answer, tools.Ideas)
                  else {"ok": False, "error": "answer with your ideas in the structured form"})
        if result["ok"]:
            return stream.to_input_list()
        session.emit("ideas_sent_back", attempt=attempt, error=result["error"], ideas=given)
        items = stream.to_input_list() + [{"role": "user",
                                           "content": message("sent_back", error=result["error"])}]
    items = items if isinstance(items, list) else [{"role": "user", "content": items}]
    # Still no spread: the last answer's valid ideas are recorded anyway, with the rules
    # they missed, so the run and the board keep its ideas.
    if last and record_ideas(session, last, lenient=True)["ok"]:
        return items
    # Nothing valid at all: the run goes on rather than ending empty - proposals are no
    # longer held for ideas, and no new round is asked for after this one.
    session.ideas_required = False
    session.round_ideas = []
    return items


async def _direction(session: Session) -> None:
    """A direction, in rounds within one conversation: ideas, then proposals on them
    until the round's ideas are used - then the next round's ideas, knowing what
    worked - until the report."""
    start = (message("explore") if session.explore
             else message("direction", direction=session.direction))
    items = await _ideas_stage(session, start + "\n\n" + message("ideas", **_focus(session)))
    while not session.cancelled:
        items = await _run_stage(session, items + [{"role": "user",
                                                    "content": _propose(session)}])
        if _stage_done(session) or not session.round_spent:
            return
        session.round_spent = False
        items = await _ideas_stage(session, items + [{"role": "user", "content": _next_round(session)}])


def _focus(session: Session) -> dict[str, Any]:
    """The upcoming round's focus engines, for its message."""
    return {"focus": ", ".join(focus_engines(session)),
            "min_focus": min(MIN_FOCUS, ideas_in_round(session))}


def _propose(session: Session) -> str:
    """The proposal message of a round: its ideas in the order to work them, with
    their hypotheses, measures and records - and how each attempt goes."""
    from agent.composer import sections

    return message("propose", round=session.round, budget=session.budget(),
                   targets_left=sections.targets_left(session),
                   round_ideas=sections.round_ideas(session, _work_order(session)),
                   gates=sections.gates(session))


def _work_order(session: Session) -> list[str]:
    """The run's ideas, every round's, in the order to work them: Thompson sampling
    over their records (agent.board.order_ideas), recorded."""
    order = order_ideas(session_ideas(session), seed=f"{session.run_id}:order:{session.round}")
    session.emit("ideas_ordered", round=session.round, order=order)
    return order


def _next_round(session: Session) -> str:
    """The next round's request: where the run stands, what worked, what failed."""

    worked = ([f"{e['name']} ({e['level']})" for e in session.ledger
               if e.get("verified") and not e.get("deleted")]
              + [f"{r['source_name']} (L3)" for r in session.data_requests if r.get("status") == "kept"])
    failed = ([f"{e['name']} - {str(e.get('reason') or '').splitlines()[0][:110]}"
               for e in session.ledger if not e.get("verified")]
              + [f"{r['source_name']} - dropped: the current data covers it"
                 for r in session.data_requests if r.get("status") == "dropped"])
    text = message("next_round", round=session.round + 1, budget=session.budget(),
                   worked=", ".join(worked) or "nothing yet",
                   failed="; ".join(failed[-8:]) or "nothing", n=ideas_in_round(session),
                   most=session.params.ideas_per_round, **_focus(session))
    if "L3" in session.quota or session.l3_only:
        from agent.tools.ideas import _taken_variables

        if taken := sorted(_taken_variables(session)):
            text += ("\nScope variables requested already - an idea on them alone is not a new "
                     "request: " + ", ".join(f"`{v}`" for v in taken) + ".")
    if session.explore:
        text += "\nThis round's theme first: call draw_theme, then give ideas about it."
    return text


# Tool results the model sees in full: the latest ones. Older results are cut to
# their head - an early failure kept verbatim keeps steering the attempts after it.
KEEP_FULL = 4
TRIMMED_TO = 400


def _trim_old_outputs(data: CallModelData[Any]) -> ModelInputData:
    """Before each model call: older tool results cut to their first lines. The
    conversation keeps them whole (the trace, a resumed run); only what this call
    sends is shortened."""
    items = list(data.model_data.input)
    outputs = [i for i, item in enumerate(items)
               if isinstance(item, dict) and item.get("type") == "function_call_output"]
    for i in outputs[:-KEEP_FULL]:
        text = items[i].get("output")
        if isinstance(text, str) and len(text) > TRIMMED_TO:
            items[i] = {**items[i], "output": text[:TRIMMED_TO] + " ... [an earlier result, cut]"}
    return ModelInputData(input=items, instructions=data.model_data.instructions)


def _hooks(session: Session, agent: Agent) -> TraceHooks:
    """The trace hooks for one runner: every model call it makes, recorded with the
    run, the stage and the round (agent.llm_trace)."""
    if session.kind == "linkage":
        stage, rnd = "linkage", None
    elif agent.output_type is not None:            # the ideas stage answers with ideas
        stage, rnd = "ideas", session.round + 1
    else:
        stage, rnd = "propose", session.round
    return TraceHooks(store_for(session.ws.cfg), session.run_id, stage, kind=session.kind, round=rnd)


async def _run(session: Session, task: Any, agent: Agent | None = None) -> Any:
    agent = agent or build_agent(session)
    stream = Runner.run_streamed(
        agent,
        input=task,
        hooks=_hooks(session, agent),
        run_config=RunConfig(call_model_input_filter=_trim_old_outputs),
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

    async def watch() -> None:
        # A stop must not wait for the next event: a model call can take minutes.
        while not stream.is_complete:
            if session.cancelled:
                stream.cancel()
                return
            await asyncio.sleep(0.5)

    watcher = asyncio.create_task(watch())
    async for event in stream.stream_events():
        if session.cancelled:
            stream.cancel()
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
    watcher.cancel()
    flush()
    if session.cancelled and not session.finished:
        session.end("", stopped_because="stopped by the user")
    return stream


def run_direction(session: Session) -> Session:
    """Run one session - a direction, or a linkage job - to the end."""
    session.ideas_required = session.kind == "direction"
    session.gated = session.kind == "direction"
    session.start()
    if session.kind == "direction":
        # Linked first, so the brief lists the columns the linkage really returns.
        for source in session.params.sources:
            if session.ws.linkage_path(source).exists():
                ensure_linked(session, source)
    # The run keeps the brief it was given - the prompt, as the model read it.
    (session.run_dir / "brief.md").write_text(compose(session))
    if session.kind == "direction":
        (session.run_dir / "brief_ideas.md").write_text(compose(session, "ideas"))
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
