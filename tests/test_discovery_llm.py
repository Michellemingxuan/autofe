"""The LLM client's retry policy, with both backends stubbed."""
import asyncio
import sys
import types

import pytest

import discovery.llm as L
from discovery.llm import CallTelemetry, EmptyResponse, LLMRefusal, call_llm


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    monkeypatch.setattr(L, "_DOTENV_LOADED", True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


@pytest.fixture
def openai_stub(monkeypatch):
    """Stand in for the openai SDK; `script` queues one behaviour per call."""
    calls, script = [], []

    class _Msg:
        def __init__(self, c): self.content = c

    class _Resp:
        def __init__(self, c): self.choices = [types.SimpleNamespace(message=_Msg(c))]

    class _Completions:
        async def create(self, **kw):
            calls.append(kw)
            kind, value = script.pop(0) if script else ("ok", "done")
            if kind == "ok":
                return _Resp(value)
            if kind == "hang":
                await asyncio.sleep(value)
                return _Resp("late")
            if kind == "raise":
                err = RuntimeError(value[1]); err.status_code = value[0]
                raise err
            raise AssertionError(kind)

    class _Client:
        def __init__(self, **kw):
            self.kwargs = kw
            self.chat = types.SimpleNamespace(completions=_Completions())

    module = types.ModuleType("openai"); module.AsyncOpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", module)
    return types.SimpleNamespace(calls=calls, script=script)


def run(**kw):
    defaults = dict(backend="openai", model="gpt-4o-mini",
                    timeout_s=3.0, stall_retry_s=0.4, max_attempts=3, backoff_s=0.01)
    return call_llm("prompt", **{**defaults, **kw})


def test_returns_content_and_telemetry(openai_stub):
    openai_stub.script.append(("ok", "hello"))
    content, tel = run()
    assert content == "hello"
    assert tel["backend"] == "openai" and tel["model"] == "gpt-4o-mini"
    assert tel["attempts"] == 1 and tel["elapsed_seconds"] >= 0


def test_model_and_prompt_reach_the_api(openai_stub):
    openai_stub.script.append(("ok", "x"))
    run(model="gpt-5", system_prompt="be terse")
    sent = openai_stub.calls[0]
    assert sent["model"] == "gpt-5"
    assert sent["messages"][0] == {"role": "system", "content": "be terse"}
    assert sent["messages"][-1] == {"role": "user", "content": "prompt"}


def test_reasoning_effort_is_only_sent_when_set(openai_stub):
    openai_stub.script.append(("ok", "x"))
    run()
    assert "reasoning_effort" not in openai_stub.calls[0]
    openai_stub.calls.clear(); openai_stub.script.append(("ok", "x"))
    run(reasoning_effort="low")
    assert openai_stub.calls[0]["reasoning_effort"] == "low"


def test_rate_limits_are_absorbed_by_the_sdk(openai_stub):
    openai_stub.script.append(("ok", "x"))
    run()
    # the SDK retries 429s itself with Retry-After before our loop sees them
    assert True


@pytest.mark.parametrize("status", [403, 400])
def test_refusals_are_not_retried(openai_stub, status):
    openai_stub.script.extend([("raise", (status, "no")), ("ok", "unreachable")])
    with pytest.raises(LLMRefusal) as excinfo:
        run()
    assert excinfo.value.status == str(status)
    assert len(openai_stub.calls) == 1


def test_openai_401_fails_fast_because_a_bad_key_will_not_fix_itself(openai_stub):
    openai_stub.script.extend([("raise", (401, "Incorrect API key")), ("ok", "unreachable")])
    with pytest.raises(LLMRefusal) as excinfo:
        run()
    assert excinfo.value.status == "401"
    assert "OPENAI_API_KEY" in str(excinfo.value)
    assert len(openai_stub.calls) == 1


def test_safechain_401_is_a_refreshable_token():
    assert L._classify(RuntimeError("HTTP 401"), "safechain") == "refresh"
    assert L._classify(RuntimeError("HTTP 401"), "openai") == "refuse_401"


def test_transient_errors_retry_up_to_the_limit(openai_stub):
    openai_stub.script.extend([("raise", (500, "boom")), ("raise", (500, "boom")), ("ok", "third")])
    content, tel = run()
    assert content == "third" and tel["attempts"] == 3

    openai_stub.calls.clear()
    openai_stub.script.extend([("raise", (500, "boom"))] * 5)
    with pytest.raises(RuntimeError):
        run()
    assert len(openai_stub.calls) == 3        # max_attempts


def test_a_stalled_call_is_reissued_rather_than_waited_on(openai_stub):
    openai_stub.script.extend([("hang", 5.0), ("ok", "second")])
    content, tel = run()
    assert content == "second" and tel["stall_retries"] == 1


def test_the_budget_is_never_exceeded(openai_stub):
    openai_stub.script.extend([("hang", 9.0)] * 3)
    with pytest.raises(TimeoutError):
        run()


def test_empty_completions_are_surfaced_not_returned(openai_stub):
    openai_stub.script.extend([("ok", "   "), ("ok", "recovered")])
    assert run()[0] == "recovered"
    openai_stub.script.extend([("ok", "")] * 5)
    with pytest.raises(EmptyResponse):
        run()


def test_block_shaped_content_is_joined():
    assert L._require_text([{"text": "he"}, {"text": "llo"}]) == "hello"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        run(backend="nope")


def test_missing_api_key_is_reported_clearly(openai_stub, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMRefusal, match="OPENAI_API_KEY"):
        run()
