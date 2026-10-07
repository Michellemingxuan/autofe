"""The firewall around every LLM call: masking, payload shaping, rejection, a gate.

Copied from AgenticSys_v2 (``llm/firewall_stack.py``, at 0966d9f) and cut to
what one agent needs: there is a single run at a time here, so one semaphore
replaces the orchestrator / specialist pools, and the log is a JSONL file in the
run folder instead of AgenticSys_v2's EventLogger and node trace.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

__all__ = ["FIREWALL_GUIDANCE", "FirewallRejection", "Firewall", "LlmLog",
           "sanitize_message", "response_format_for_round"]

_CASE_ID_RE = re.compile(r"CASE-\d+")
# 13+ digits: masks a 15-digit card number, lets an 11-12 digit id through.
_DIGIT_RUN_RE = re.compile(r"\d{13,}")

FIREWALL_GUIDANCE = (
    "[IMPORTANT: Your previous response was blocked by the content firewall. "
    "Avoid: raw account numbers, PII, role-injection patterns like [SYSTEM] or "
    "[USER], code execution keywords (exec, eval, import). Use masked identifiers "
    "and descriptive language instead of raw numeric values.]"
)


def sanitize_message(message: str) -> str:
    """Mask identifiers: long digit runs (13+ digits) and CASE-\\d+ tokens."""
    return _DIGIT_RUN_RE.sub("***MASKED***", _CASE_ID_RE.sub("[CASE-ID]", message))


def forces_tool_call(tool_choice: Any) -> bool:
    """True when this `tool_choice` obliges the model to call a tool."""
    if isinstance(tool_choice, str):
        return tool_choice == "required"
    if isinstance(tool_choice, dict):
        return tool_choice.get("type") == "function"
    return False


def response_format_for_round(tool_choice: Any, response_format: Any) -> Any:
    """`response_format`, or None on a round that must call a tool: such a round
    cannot give the final structured answer, and on safechain the schema's mere
    presence routes the call through OpenAI's auto-parse path."""
    if response_format is None:
        return None
    return None if forces_tool_call(tool_choice) else response_format


class FirewallRejection(Exception):
    """Raised when a firewall rule blocks an LLM response."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"FirewallRejection({code}): {message}")


class LlmLog:
    """One JSON line per noteworthy LLM event - a stall, a rejection, a slow gate."""

    def __init__(self, folder: str | Path):
        self.path = Path(folder) / f"llm_{time.strftime('%Y%m%d')}.jsonl"

    def log(self, event: str, payload: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps({"ts": round(time.time(), 3), "event": event, **payload},
                                    default=str) + "\n")
        except OSError:
            pass                                     # logging must not break a call


class Firewall:
    """What both transports share: the log, the rejection retry count, and a gate
    that caps concurrent calls (``LLM_CONCURRENCY``, default 4)."""

    def __init__(self, logger: LlmLog, max_retries: int = 2):
        self.logger = logger
        self.max_retries = max_retries
        self.cap = max(1, int(os.environ.get("LLM_CONCURRENCY", "4")))
        self._semaphore: asyncio.Semaphore | None = None
        self._loop: Any = None

    @asynccontextmanager
    async def gate(self) -> AsyncIterator[None]:
        # Each run has its own event loop (asyncio.run per run): a semaphore is
        # bound to the loop it was made on, so make one per loop.
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore, self._loop = asyncio.Semaphore(self.cap), loop
        t0 = time.perf_counter()
        async with self._semaphore:
            waited_ms = int((time.perf_counter() - t0) * 1000)
            if waited_ms >= 100:
                self.logger.log("llm_gate_wait", {"waited_ms": waited_ms, "cap": self.cap})
            yield
