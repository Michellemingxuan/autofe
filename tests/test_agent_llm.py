"""The safechain client (copied from AgenticSys_v2) and the structured-answer guard,
without safechain: its modules are stood in for by small fakes."""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from agent.llm import firewall as fw
from agent.llm import safechain as sc


class _Msg:
    def __init__(self, content="", tool_calls=None, tool_call_chunks=None, **_):
        self.content = content
        self.tool_calls = tool_calls or []
        self.tool_call_chunks = tool_call_chunks or []
        self.response_metadata = {"finish_reason": "stop"}
        self.usage_metadata = None
        self.id = None


class _Chain:
    """`template | model`: ainvoke and astream go to the fake model."""

    def __init__(self, model):
        self.model = model

    async def ainvoke(self, _inputs):
        return await self.model.answer()

    def astream(self, _inputs):
        return self.model.stream()


class _Model:
    def __init__(self, delays=(), reply=None):
        self.delays = list(delays)
        self.reply = reply
        self.calls = 0

    def bind(self, **_):
        return self

    async def answer(self):
        self.calls += 1
        await asyncio.sleep(self.delays.pop(0) if self.delays else 0)
        return self.reply or _Msg(content="ok")

    async def stream(self):
        raise AssertionError("a streamed call is made as a plain call")
        yield


@pytest.fixture
def fake_safechain(monkeypatch, tmp_path):
    class _Template:
        @staticmethod
        def from_messages(_messages):
            return _Template()

        def __or__(self, model):
            return _Chain(model)

    mods = {
        "safechain": types.ModuleType("safechain"),
        "safechain.prompts": types.SimpleNamespace(ValidChatPromptTemplate=_Template),
        "langchain_core": types.ModuleType("langchain_core"),
        "langchain_core.prompts": types.SimpleNamespace(MessagesPlaceholder=lambda name: name),
        "langchain_core.messages": types.SimpleNamespace(
            AIMessage=_Msg, HumanMessage=_Msg, SystemMessage=_Msg, ToolMessage=_Msg),
    }
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)
    return fw.LlmLog(tmp_path)


def _client(model, log, stall_s=40.0):
    client = sc.SafeChainAsyncOpenAI(model_name="m", firewall=fw.Firewall(log), stall_s=stall_s)
    client._llm = model                        # skip amodel(): already built
    return client


def _events(log):
    return [json.loads(line)["event"] for line in log.path.read_text().splitlines()] \
        if log.path.exists() else []


def test_a_stalled_call_is_reissued_and_answers(fake_safechain, monkeypatch):
    model = _Model(delays=[5.0, 0.0])          # the first attempt wedges, the second does not
    client = _client(model, fake_safechain, stall_s=0.05)
    reply = asyncio.run(client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}]))
    assert reply.choices[0].message.content == "ok" and model.calls == 2
    assert "safechain_call_stalled" in _events(fake_safechain)


async def _read(client):
    stream = await client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    return [c async for c in stream]


def test_a_stalled_stream_is_reissued_and_answers(fake_safechain, monkeypatch):
    model = _Model(delays=[5.0, 0.0])
    chunks = asyncio.run(_read(_client(model, fake_safechain, stall_s=0.05)))
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "ok"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason == "stop" and model.calls == 2
    assert "safechain_call_stalled" in _events(fake_safechain)


def test_a_streamed_tool_call_arrives_whole(fake_safechain):
    call = {"name": "screen_feature", "args": {"name": "pay_to_spend_90d"}, "id": "call_1"}
    chunks = asyncio.run(_read(_client(_Model(reply=_Msg(tool_calls=[call])), fake_safechain)))
    [delta] = [c.choices[0].delta.tool_calls for c in chunks if c.choices[0].delta.tool_calls]
    assert delta[0].function.name == "screen_feature" and delta[0].id == "call_1"
    assert json.loads(delta[0].function.arguments) == {"name": "pay_to_spend_90d"}
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_a_firewall_rejection_is_retried_with_guidance(fake_safechain):
    class _Blocked(_Model):
        async def answer(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Error code: 403 - blocked")
            return _Msg(content="ok")

    model = _Blocked()
    reply = asyncio.run(_client(model, fake_safechain).chat.completions.create(
        model="m", messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]))
    assert reply.choices[0].message.content == "ok" and model.calls == 2
    assert "firewall_rejection" in _events(fake_safechain)


def test_long_digit_runs_are_masked_before_they_leave():
    assert fw.sanitize_message("card 378282246310005 case 40290638201") == \
        "card ***MASKED*** case 40290638201"


def test_a_structured_answer_sent_twice_is_read_once():
    from agents.exceptions import ModelBehaviorError

    from agent.agent import FirstAnswer
    from agent.tools import Ideas

    schema = FirstAnswer(Ideas)
    one = json.dumps({"ideas": [{"name": "a", "level": "L1", "lens": "trend",
                                 "description": "d", "data": "x", "beyond_scope": False}]})
    assert len(schema.validate_json(one + one).ideas) == 1
    assert len(schema.validate_json(one + "\n" + one).ideas) == 1
    for bad in (one + " and some words", '{"ideas": [}'):
        with pytest.raises(ModelBehaviorError):
            schema.validate_json(bad)
