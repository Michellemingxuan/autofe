"""SafeChainAsyncOpenAI - an ``openai.AsyncOpenAI`` stand-in that calls SafeChain.

Copied from AgenticSys_v2 (``llm/safechain_client.py``, at 0966d9f), where it is
measured against the private environment, and revised for autofe:

* no node-trace telemetry and no token estimates (tiktoken's first-use download
  is what stalled every round there) - noteworthy events go to the run's LLM log;
* one concurrency gate, not two pools (:mod:`agent.llm.firewall`);
* the stream drops a chunk that re-sends the whole answer so far - a structured
  answer arriving twice (``{...}{...}``) is not valid JSON, and the run failed on it.

Kept as is, because each was a measured fix:

* STALL-AND-RETRY. SafeChain calls do not run slow, they stall: a call still
  running at 40s (``SAFECHAIN_STALL_RETRY_S``) is wedged, and a fresh request has
  a fresh chance. Healthy calls take 2-13s; a stall that is ridden out resolves
  at 126-131s, so the second attempt gets the whole budget
  (``SAFECHAIN_CALL_TIMEOUT_S``, 180s). ``SAFECHAIN_STALL_RETRY_S=0`` turns it off.
* ``ainvoke``, not a thread: it is genuinely cancellable, so a timeout aborts the
  request instead of leaving a worker running. Nothing on the loop may block -
  a blocking call makes requests slow and unkillable at once.
* the model is built with ``await amodel(...)`` (token acquisition), bounded by
  the same timeout; HTTP 401 rebuilds it once; 403 / 400 are firewall rejections,
  retried with guidance.
* native transport: the SDK's own payload goes through ``model.bind(...)`` inside
  ``ValidChatPromptTemplate`` (kept for compliance), tool calls round-trip as
  LangChain tool calls, and the reply converts back to an OpenAI ChatCompletion.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

from agent.llm.firewall import (
    FIREWALL_GUIDANCE,
    Firewall,
    FirewallRejection,
    response_format_for_round,
    sanitize_message,
)

__all__ = ["SafeChainAsyncOpenAI"]

# safechain bridges its sync token call to async with asyncio.run(), which fails
# inside a running loop unless nest_asyncio has patched it.
try:
    import nest_asyncio as _nest_asyncio  # type: ignore[import-not-found]

    _nest_asyncio.apply()
except ImportError:  # pragma: no cover - only the private env needs it
    pass

_SAFECHAIN_CALL_TIMEOUT_S = float(os.environ.get("SAFECHAIN_CALL_TIMEOUT_S", "180"))
_SAFECHAIN_STALL_RETRY_S = float(os.environ.get("SAFECHAIN_STALL_RETRY_S", "40"))


class SafeChainAsyncOpenAI:
    """Drop-in for ``openai.AsyncOpenAI`` that calls SafeChain underneath."""

    # Endpoints the SDK may probe but never routes work through.
    _UNSUPPORTED_ENDPOINTS: frozenset = frozenset({
        "responses", "embeddings", "files", "images", "audio",
        "fine_tuning", "moderations", "completions", "batches",
        "uploads", "vector_stores", "assistants", "threads", "beta",
    })

    def __init__(self, *, model_name: str, firewall: Firewall):
        self._model_name = model_name
        self._firewall = firewall
        self._llm: Any = None                       # built on first use
        self.chat = _SafeChainChat(self)

    def __getattr__(self, name: str):
        if name in type(self)._UNSUPPORTED_ENDPOINTS:
            raise AttributeError(f"SafeChainAsyncOpenAI does not expose '{name}'. Only "
                                 "chat completions are routed through SafeChain.")
        if name.startswith("_"):
            raise AttributeError(f"SafeChainAsyncOpenAI has no internal attribute {name!r}.")
        # Passive attributes the SDK reads for tracing (base_url, api_key, ...).
        return None

    async def _aensure_llm(self) -> Any:
        if self._llm is None:
            await self._arefresh_llm()
        return self._llm

    async def _arefresh_llm(self) -> None:
        """(Re)build the safechain model - the first call, and a 401 token refresh."""
        try:
            from safechain.core.model import amodel  # type: ignore[import-not-found]
        except ImportError as e:
            raise NotImplementedError("safechain is not installed in this environment; "
                                      "the safechain backend is for the private env") from e
        model_id = os.environ.get("SAFECHAIN_MODEL", self._model_name)
        try:
            self._llm = await asyncio.wait_for(amodel(model_id),
                                               timeout=_SAFECHAIN_CALL_TIMEOUT_S)
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"safechain amodel() build did not return within "
                               f"{_SAFECHAIN_CALL_TIMEOUT_S:.0f}s") from e


class _SafeChainChat:
    def __init__(self, parent: SafeChainAsyncOpenAI):
        self.completions = _SafeChainChatCompletions(parent)


class _SafeChainChatCompletions:
    """Mimics ``AsyncOpenAI.chat.completions``: only ``create`` is called."""

    def __init__(self, parent: SafeChainAsyncOpenAI):
        self._parent = parent

    async def create(self, *, model: str, messages: list[dict], tools: list[dict] | None = None,
                     response_format: Any = None, stream: bool = False, **kw: Any) -> Any:
        tool_choice = kw.pop("tool_choice", None)
        # An allow-list: forwarding arbitrary SDK extras risks a 400.
        passthrough = {k: kw.pop(k, None) for k in
                       ("max_tokens", "parallel_tool_calls", "temperature", "top_p", "seed", "stop")}
        firewall = self._parent._firewall
        messages = [_redact_message(m) for m in messages]
        attempt = 0
        while True:
            try:
                async with firewall.gate():
                    return await self._invoke(model=model, messages=messages, tools=tools,
                                              response_format=response_format, stream=stream,
                                              tool_choice=tool_choice, passthrough=passthrough)
            except FirewallRejection as e:
                firewall.logger.log("firewall_rejection", {"code": e.code, "message": e.message,
                                                           "attempt": attempt})
                if attempt >= firewall.max_retries:
                    raise
                attempt += 1
                messages = _inject_guidance(messages)

    async def _invoke(self, *, model: str, messages: list[dict], tools: list[dict] | None,
                      response_format: Any, stream: bool, tool_choice: Any,
                      passthrough: dict | None) -> Any:
        try:
            from langchain_core.prompts import MessagesPlaceholder  # type: ignore[import-not-found]
            from safechain.prompts import ValidChatPromptTemplate  # type: ignore[import-not-found]
        except ImportError as e:
            raise NotImplementedError("safechain is not installed; the safechain backend is "
                                      "for the private env") from e

        llm = await self._parent._aensure_llm()
        lc_messages = _to_lc_messages(messages)
        bind_kwargs = _bind_kwargs(tools, tool_choice, response_format, extra=passthrough)
        log = self._parent._firewall.logger

        def _chain(active_model: Any):
            bound = active_model.bind(**bind_kwargs) if bind_kwargs else active_model
            return ValidChatPromptTemplate.from_messages([MessagesPlaceholder("messages")]) | bound

        async def _run(active_model: Any) -> Any:
            # A short first attempt, then one that may outlast a stall. The first is
            # clamped so a lowered call timeout is never exceeded.
            first_s = min(_SAFECHAIN_STALL_RETRY_S, _SAFECHAIN_CALL_TIMEOUT_S)
            if first_s > 0:
                try:
                    return await asyncio.wait_for(
                        _chain(active_model).ainvoke({"messages": lc_messages}), timeout=first_s)
                except asyncio.TimeoutError:
                    log.log("safechain_call_stalled", {"stalled_after_s": first_s})
            try:
                return await asyncio.wait_for(
                    _chain(active_model).ainvoke({"messages": lc_messages}),
                    timeout=_SAFECHAIN_CALL_TIMEOUT_S)
            except asyncio.TimeoutError:
                if first_s > 0:
                    log.log("safechain_retry_stalled", {"first_attempt_s": first_s,
                                                        "retry_budget_s": _SAFECHAIN_CALL_TIMEOUT_S})
                raise

        async def _run_stream(active_model: Any):
            return _chain(active_model).astream({"messages": lc_messages})

        try:
            reply = await (_run_stream(llm) if stream else _run(llm))
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"safechain LLM call did not return within "
                               f"{_SAFECHAIN_CALL_TIMEOUT_S:.0f}s") from e
        except Exception as e:  # noqa: BLE001 - re-classified below
            es = str(e)
            if "401" in es:                          # token expiry: rebuild, retry once
                await self._parent._arefresh_llm()
                refreshed = await self._parent._aensure_llm()
                try:
                    reply = await (_run_stream(refreshed) if stream else _run(refreshed))
                except asyncio.TimeoutError as te:
                    raise TimeoutError(f"safechain LLM call did not return within "
                                       f"{_SAFECHAIN_CALL_TIMEOUT_S:.0f}s (after token refresh)"
                                       ) from te
            elif "403" in es:
                raise FirewallRejection("403", f"safechain blocked: {es}")
            elif "400" in es:
                raise FirewallRejection("400", f"safechain bad request: {es}")
            else:
                raise

        if stream:
            return _SafeChainStream(agen=reply, model=model, log=log)
        return _completion_from_message(reply, model)


# ---------------------------------------------------------------- helpers
def _redact_message(message: dict) -> dict:
    if not isinstance(message, dict):
        return message
    content = message.get("content")
    if isinstance(content, str):
        return {**message, "content": sanitize_message(content)}
    return message


def _inject_guidance(messages: list[dict]) -> list[dict]:
    """Append the firewall guidance to the first system message, re-redact all."""
    out, appended = [], False
    for m in messages:
        m = _redact_message(m)
        if not appended and m.get("role") == "system":
            m = {**m, "content": (m.get("content") or "") + "\n\n" + FIREWALL_GUIDANCE}
            appended = True
        out.append(m)
    return out


def _to_lc_messages(messages: list[dict]) -> list:
    """OpenAI-wire message dicts -> LangChain messages. An assistant message's
    tool calls must come back with `args` as a dict, and each tool result as a
    ToolMessage bound by its call id, or the provider rejects the next round."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    out: list = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else json.dumps(content, default=str)
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "tool":
            out.append(ToolMessage(content=content,
                                   tool_call_id=m.get("tool_call_id") or m.get("id") or ""))
        elif role == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except (json.JSONDecodeError, ValueError):
                    args = {}
                calls.append({"name": fn.get("name") or "", "args": args if isinstance(args, dict) else {},
                              "id": tc.get("id") or "", "type": "tool_call"})
            out.append(AIMessage(content=content, tool_calls=calls))
        else:
            out.append(HumanMessage(content=content))
    return out


def _bind_kwargs(tools: list[dict] | None, tool_choice: Any, response_format: Any,
                 extra: dict | None = None) -> dict:
    """The OpenAI-shaped kwargs forwarded through `.bind()`, unchanged; only
    omitted-vs-None is normalised (binding tools=None is not binding no tools)."""
    kwargs: dict[str, Any] = {}
    if tools:
        kwargs["tools"] = tools
    if tool_choice is not None and tools:            # tool_choice without tools is a 400
        kwargs["tool_choice"] = tool_choice
    response_format = response_format_for_round(tool_choice, response_format)
    if response_format is not None:
        kwargs["response_format"] = response_format
    for k, v in (extra or {}).items():
        if v is None or (k == "parallel_tool_calls" and not tools):
            continue
        kwargs[k] = v
    return kwargs


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else p.get("text", "")
                       for p in content if isinstance(p, (str, dict)))
    return "" if content is None else str(content)


def _sdk_tool_calls(message: Any) -> list[ChatCompletionMessageToolCall] | None:
    """LangChain tool calls (args a dict) -> OpenAI tool calls (arguments a string)."""
    calls = []
    for tc in getattr(message, "tool_calls", None) or []:
        if not isinstance(tc, dict) or not tc.get("name"):
            continue
        args = tc.get("args")
        calls.append(ChatCompletionMessageToolCall(
            id=tc.get("id") or f"call_{uuid.uuid4().hex[:24]}", type="function",
            function=Function(name=tc["name"],
                              arguments=args if isinstance(args, str) else json.dumps(args or {}, default=str))))
    return calls or None


def _usage_from_message(message: Any) -> CompletionUsage | None:
    meta = getattr(message, "usage_metadata", None)
    if not isinstance(meta, dict) or (meta.get("input_tokens") is None and meta.get("output_tokens") is None):
        return None
    prompt, completion = int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)
    return CompletionUsage(prompt_tokens=prompt, completion_tokens=completion,
                           total_tokens=int(meta.get("total_tokens") or (prompt + completion)))


def _completion_from_message(message: Any, model: str) -> ChatCompletion:
    """LangChain AIMessage -> OpenAI ChatCompletion; the SDK parses and validates."""
    tool_calls = _sdk_tool_calls(message)
    content = _content_text(getattr(message, "content", ""))
    meta = getattr(message, "response_metadata", None) or {}
    finish_reason = meta.get("finish_reason") if isinstance(meta, dict) else None
    if tool_calls:
        finish_reason = "tool_calls"
    elif finish_reason not in ("stop", "length", "content_filter", "tool_calls"):
        finish_reason = "stop"
    return ChatCompletion(
        id=getattr(message, "id", None) or f"chatcmpl_{uuid.uuid4().hex[:24]}",
        choices=[Choice(index=0, finish_reason=finish_reason, message=ChatCompletionMessage(
            role="assistant", content=content or None, tool_calls=tool_calls))],
        created=int(time.time()), model=model, object="chat.completion",
        usage=_usage_from_message(message))


class _SafeChainStream:
    """Async-iterable over the model's real token stream, as ChatCompletionChunks:
    a role delta first, a finish_reason terminator last. The gap between chunks is
    bounded - a long answer is slow, a stalled transport is not."""

    # A chunk repeating at least this much of the answer so far is a re-send.
    _RESEND_MIN = 20

    def __init__(self, *, agen, model: str, log: Any) -> None:
        self._agen = agen
        self._model = model
        self._log = log
        self._id = f"chatcmpl_{uuid.uuid4().hex[:24]}"
        self._created = int(time.time())
        self._sent_role = False
        self._saw_tool_call = False
        self._done = False
        self._text = ""                          # the content forwarded so far

    def __aiter__(self) -> "_SafeChainStream":
        return self

    def _chunk(self, delta: ChoiceDelta, finish_reason: str | None = None) -> ChatCompletionChunk:
        return ChatCompletionChunk(id=self._id, created=self._created, model=self._model,
                                   object="chat.completion.chunk",
                                   choices=[ChunkChoice(index=0, delta=delta,
                                                        finish_reason=finish_reason)])

    def _new_text(self, text: str) -> str:
        """What is new in a chunk's text. Some builds end a structured answer with
        one chunk carrying the WHOLE answer again; forwarded as is, the SDK reads
        `{...}{...}` - not JSON. A chunk that repeats everything sent so far
        contributes only what follows it."""
        if len(self._text) >= self._RESEND_MIN and text.startswith(self._text):
            self._log.log("safechain_stream_resend", {"resent_chars": len(self._text)})
            text = text[len(self._text):]
        self._text += text
        return text

    async def __anext__(self) -> ChatCompletionChunk:
        if not self._sent_role:
            self._sent_role = True
            return self._chunk(ChoiceDelta(role="assistant"))
        if self._done:
            raise StopAsyncIteration
        while True:
            try:
                raw = await asyncio.wait_for(self._agen.__anext__(),
                                             timeout=_SAFECHAIN_CALL_TIMEOUT_S)
            except StopAsyncIteration:
                self._done = True
                return self._chunk(ChoiceDelta(),
                                   finish_reason="tool_calls" if self._saw_tool_call else "stop")
            except asyncio.TimeoutError as e:
                self._done = True
                raise TimeoutError(f"safechain stream stalled for more than "
                                   f"{_SAFECHAIN_CALL_TIMEOUT_S:.0f}s between chunks") from e
            tool_calls = []
            for tc in getattr(raw, "tool_call_chunks", None) or []:
                if isinstance(tc, dict):           # partial arguments: the SDK reassembles
                    args = tc.get("args")
                    tool_calls.append(ChoiceDeltaToolCall(
                        index=tc.get("index") or 0, id=tc.get("id") or None, type="function",
                        function=ChoiceDeltaToolCallFunction(
                            name=tc.get("name") or None,
                            arguments=args if isinstance(args, str) else None)))
            text = self._new_text(_content_text(getattr(raw, "content", "")))
            if not text and not tool_calls:
                continue
            self._saw_tool_call = self._saw_tool_call or bool(tool_calls)
            return self._chunk(ChoiceDelta(content=text or None, tool_calls=tool_calls or None))

    async def close(self) -> None:
        self._done = True
        aclose = getattr(self._agen, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
