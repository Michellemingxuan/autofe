"""The model the agent runs on: openai directly, or safechain via AgenticSys_v2.

Both give the openai-agents SDK an ``AsyncOpenAI``-shaped client, so nothing
above this module knows which one it has. safechain's client lives in
AgenticSys_v2 (it is ~1000 lines of firewall and transport code) and is reused
from there rather than copied: set ``AGENTICSYS_V2_PATH`` to that checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from agents import OpenAIChatCompletionsModel

from validation.config import LLMConfig

__all__ = ["build_model"]


def build_model(llm: LLMConfig) -> OpenAIChatCompletionsModel:
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
        root = os.environ.get("AGENTICSYS_V2_PATH")
        if not root or not Path(root).exists():
            raise RuntimeError("the safechain backend reuses AgenticSys_v2's client: set "
                               "AGENTICSYS_V2_PATH to that checkout")
        if root not in sys.path:
            sys.path.insert(0, root)
        from llm.factory import build_session_clients  # type: ignore[import-not-found]
        from llm.firewall_stack import FirewallStack  # type: ignore[import-not-found]
        from logger.event_logger import EventLogger  # type: ignore[import-not-found]

        firewall = FirewallStack(logger=EventLogger(session_id="autofe-agent",
                                                    log_dir="outputs/agent/llm_logs"))
        return build_session_clients(firewall, model_name=llm.model, backend="safechain").model

    raise ValueError(f"unknown llm backend {llm.backend!r}; use openai or safechain")
