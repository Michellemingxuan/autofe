"""The model the agent runs on: openai directly, or safechain.

Both give the openai-agents SDK an ``AsyncOpenAI``-shaped client, so nothing
above this package knows which one it has. The safechain client
(:mod:`agent.llm.safechain`, with :mod:`agent.llm.firewall`) is copied from
AgenticSys_v2 and revised here - no checkout of it is needed.
"""

from __future__ import annotations

from pathlib import Path

from agents import OpenAIChatCompletionsModel

from validation.config import LLMConfig

__all__ = ["build_model"]

# One safechain client per (model, log folder) for the process: its model is
# built once - a token acquisition - and reused by every agent and run after.
_SAFECHAIN: dict[tuple[str, str, float, float], object] = {}


def build_model(llm: LLMConfig, log_dir: str | Path = "outputs/agent/llm_logs"
                ) -> OpenAIChatCompletionsModel:
    if llm.backend == "openai":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # pragma: no cover
            pass
        from openai import AsyncOpenAI

        client = AsyncOpenAI(max_retries=8, timeout=llm.timeout_s)
        return OpenAIChatCompletionsModel(model=llm.model, openai_client=client)

    if llm.backend == "safechain":
        from agents import set_tracing_disabled

        from agent.llm.firewall import Firewall, LlmLog
        from agent.llm.safechain import SafeChainAsyncOpenAI

        set_tracing_disabled(True)           # no OpenAI key here: trace export only adds noise
        key = (llm.model, str(log_dir), llm.timeout_s, llm.stall_retry_s)
        if key not in _SAFECHAIN:
            _SAFECHAIN[key] = SafeChainAsyncOpenAI(
                model_name=llm.model, firewall=Firewall(LlmLog(log_dir)),
                call_s=llm.timeout_s, stall_s=llm.stall_retry_s)
        return OpenAIChatCompletionsModel(model=llm.model, openai_client=_SAFECHAIN[key])

    raise ValueError(f"unknown llm backend {llm.backend!r}; use openai or safechain")
