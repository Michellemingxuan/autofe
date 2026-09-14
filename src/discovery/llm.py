"""One bounded, retrying LLM call.

Two backends behind one function:

* ``openai``    the openai SDK directly - development, and any environment with
                an API key. No langchain dependency.
* ``safechain`` ``amodel()`` plus an LCEL chain - the private deployment.

The reliability behaviour is the same on both and is not decoration. It was
established against safechain in production, where calls do not degrade
gracefully - they *stall*. Normal latency is 2-13s, while a wedged call either
rides out to ~126-131s or dies at a phase fence. So a call still running at
``stall_retry_s`` is treated as wedged and re-issued rather than waited on, and
the overall budget stays large enough that the retry can outlast a stall that
does eventually resolve.

Three details are load-bearing and easy to lose:

* ``nest_asyncio`` must be applied at import. safechain's ``TokenUtil.get_token``
  bridges sync to async with ``asyncio.run(...)``, which raises "asyncio.run()
  cannot be called from a running event loop" the moment it runs inside one -
  and ``amodel()`` is awaited inside exactly that.
* The chain runs through ``ainvoke`` so ``asyncio.wait_for`` genuinely aborts an
  in-flight request. A synchronous ``invoke`` in a thread pool cannot be
  interrupted, which is what turns a stall into a permanently wedged run.
* 401 means different things per backend: an expired token on safechain, worth
  rebuilding and retrying; a wrong API key on openai, where retrying only burns
  the budget. 403 and 400 are refusals on both - never retried.

Which model to ask is configuration, not environment: it comes from
``LLMConfig`` in the run's YAML. The environment supplies only the credential
and, optionally, network patience.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

__all__ = [
    "call_llm",
    "acall_llm",
    "CallTelemetry",
    "LLMRefusal",
    "EmptyResponse",
]

# safechain's token bridge calls asyncio.run() from inside a running loop;
# patching here is what makes `await amodel(...)` work at all. A no-op when
# applied twice, and simply absent in environments that do not need it.
_NEST_ASYNCIO_APPLIED = False
try:
    import nest_asyncio as _nest_asyncio

    _nest_asyncio.apply()
    _NEST_ASYNCIO_APPLIED = True
except ImportError:  # pragma: no cover - only the safechain env needs it
    _nest_asyncio = None

_DOTENV_LOADED = False


class LLMRefusal(RuntimeError):
    """The backend refused the request (400/403, or a bad key). Do not retry."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


class EmptyResponse(RuntimeError):
    """The model returned no content. Transient in practice; worth a retry."""


@dataclass
class CallTelemetry:
    """What a call actually did, recorded alongside every generated feature."""

    backend: str = ""
    model: str = ""
    reasoning_effort: str | None = None
    attempts: int = 0
    elapsed_seconds: float = 0.0
    build_seconds: float = 0.0
    stall_retries: int = 0
    token_refreshes: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "attempts": self.attempts,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "build_seconds": round(self.build_seconds, 3),
            "stall_retries": self.stall_retries,
            "token_refreshes": self.token_refreshes,
            "nest_asyncio": _NEST_ASYNCIO_APPLIED,
            "errors": self.errors,
        }


def _load_credentials() -> None:
    """
    Put .env into the environment before a backend is touched.

    safechain's amodel() acquires a token from the environment and the openai SDK
    reads OPENAI_API_KEY from it, so without this both fail with auth errors that
    look like library bugs.

    override=False deliberately: an explicitly exported variable must beat .env,
    or a command like ``OPENAI_API_KEY=sk-dummy python run.py`` silently uses the
    real key and bills a run that was meant to be a dry one.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(), override=False)
    except ImportError:  # pragma: no cover - credentials may come from the shell
        pass
    _DOTENV_LOADED = True


def _classify(error: Exception, backend: str) -> str:
    """Retry policy for one failure: refuse_4xx, refresh, or retry."""
    # The openai SDK carries a real status code; safechain surfaces HTTP codes
    # only in the message text, so fall back to matching that.
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if not isinstance(status, int):
        text = str(error)
        status = next((code for code in (403, 400, 401) if str(code) in text), None)

    if status == 403:
        return "refuse_403"
    if status == 400:
        return "refuse_400"
    if status == 401:
        return "refresh" if backend == "safechain" else "refuse_401"
    return "retry"


def _require_text(content: Any) -> str:
    """Refuse to pass an empty completion on as if it were an answer."""
    if isinstance(content, list):
        # Some builds return a list of content blocks rather than a string.
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    if not isinstance(content, str) or not content.strip():
        raise EmptyResponse(f"Model returned no usable content: {content!r}")
    return content


async def _build_openai(
    model: str, reasoning_effort: str | None, system_prompt: str, budget_s: float
) -> Callable[[str], Awaitable[str]]:
    """An invoker over the openai SDK. No langchain involved."""
    _load_credentials()
    try:
        from openai import AsyncOpenAI
    except ImportError as error:
        raise NotImplementedError(
            "The openai package is not installed. pip install 'mllite[discovery]', "
            "or set the backend to 'safechain'."
        ) from error

    if not os.environ.get("OPENAI_API_KEY"):
        raise LLMRefusal("401", "OPENAI_API_KEY is not set; put it in .env or export it.")

    # max_retries lets the SDK absorb 429s with its own Retry-After backoff
    # before our loop ever sees them.
    client = AsyncOpenAI(max_retries=8)
    preamble = [{"role": "system", "content": system_prompt}] if system_prompt else []

    async def invoke(prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [*preamble, {"role": "user", "content": prompt}],
        }
        # Only reasoning models accept this; gpt-4o rejects it with a 400.
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        response = await client.chat.completions.create(**kwargs)
        return _require_text(response.choices[0].message.content)

    return invoke


async def _build_safechain(
    model: str, reasoning_effort: str | None, system_prompt: str, budget_s: float
) -> Callable[[str], Awaitable[str]]:
    """An invoker over safechain. amodel() is async because it acquires a token."""
    _load_credentials()
    try:
        from safechain.core.model import amodel
    except ImportError as error:  # pragma: no cover - private env only
        raise NotImplementedError(
            "safechain is not installed here. Set the backend to 'openai'."
        ) from error

    try:
        llm = await asyncio.wait_for(amodel(model), timeout=budget_s)
    except asyncio.TimeoutError as error:
        # An unbounded build hangs with no error at all, which reads as a run
        # that is simply stuck.
        raise TimeoutError(
            f"safechain amodel() build did not return within {budget_s:.0f}s"
        ) from error

    # amodel() takes only a model id; per-call options ride on .bind(), which
    # forwards them to the endpoint verbatim.
    if reasoning_effort:
        llm = llm.bind(reasoning_effort=reasoning_effort)

    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:  # pragma: no cover - older environments
        from langchain.prompts import ChatPromptTemplate

    chain = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{prompt_text}"),
    ]) | llm

    async def invoke(prompt: str) -> str:
        reply = await chain.ainvoke({"prompt_text": prompt})
        return _require_text(getattr(reply, "content", reply))

    return invoke


_BUILDERS = {"openai": _build_openai, "safechain": _build_safechain}


async def acall_llm(
    prompt: str,
    *,
    backend: str,
    model: str,
    reasoning_effort: str | None = None,
    system_prompt: str = "",
    timeout_s: float = 180.0,
    stall_retry_s: float = 40.0,
    max_attempts: int = 3,
    backoff_s: float = 5.0,
    telemetry: CallTelemetry | None = None,
) -> str:
    """Async form of :func:`call_llm`."""
    if backend not in _BUILDERS:
        raise ValueError(f"Unknown LLM backend {backend!r}. Use 'openai' or 'safechain'.")

    telemetry = telemetry if telemetry is not None else CallTelemetry()
    telemetry.backend, telemetry.model = backend, model
    telemetry.reasoning_effort = reasoning_effort

    deadline = time.perf_counter() + timeout_s

    def remaining() -> float:
        return deadline - time.perf_counter()

    build_started = time.perf_counter()
    invoke = await _BUILDERS[backend](
        model, reasoning_effort, system_prompt, max(remaining(), 0.0)
    )
    telemetry.build_seconds = time.perf_counter() - build_started

    attempt = 0
    while True:
        attempt += 1
        telemetry.attempts = attempt

        # The first attempt gets the short stall fence; later ones get whatever
        # is left, so total wall clock stays inside the budget.
        budget = stall_retry_s if (attempt == 1 and 0 < stall_retry_s < timeout_s) else remaining()
        if budget <= 0:
            raise TimeoutError(
                f"LLM call exhausted its {timeout_s:.0f}s budget after {attempt - 1} attempt(s)"
            )

        try:
            return await asyncio.wait_for(invoke(prompt), timeout=budget)

        except asyncio.TimeoutError:
            telemetry.errors.append(f"attempt {attempt}: timed out after {budget:.0f}s")
            if remaining() <= 0:
                raise TimeoutError(f"LLM call did not return within {timeout_s:.0f}s") from None
            telemetry.stall_retries += 1      # wedged request, not slow work
            continue

        except Exception as error:  # noqa: BLE001 - reclassified immediately
            kind = _classify(error, backend)
            telemetry.errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")

            if kind.startswith("refuse"):
                status = kind.split("_")[1]
                hint = "check OPENAI_API_KEY" if status == "401" else f"{backend} refused the request"
                raise LLMRefusal(status, f"{hint}: {error}") from error

            if kind == "refresh":
                telemetry.token_refreshes += 1
                invoke = await _BUILDERS[backend](
                    model, reasoning_effort, system_prompt, max(remaining(), 0.0)
                )

            if attempt >= max(1, max_attempts) or remaining() <= 0:
                raise

            if backoff_s > 0 and kind != "refresh":
                await asyncio.sleep(min(backoff_s * attempt, max(remaining() - 1, 0)))


def call_llm(prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    """
    Send one prompt; return ``(content, telemetry)``.

    Raises LLMRefusal on a refusal, TimeoutError once the budget is spent, or the
    last transport error after the attempt limit.
    """
    telemetry = CallTelemetry()
    started = time.perf_counter()
    try:
        content = asyncio.run(acall_llm(prompt, telemetry=telemetry, **kwargs))
    finally:
        telemetry.elapsed_seconds = time.perf_counter() - started
    return content, telemetry.as_dict()
