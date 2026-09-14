"""What every proposal strategy shares.

A strategy decides only *how it asks* for features. The framework owns the
sample, the screen, the guards, the history and the stopping rule, which is what
lets several published methods sit side by side without reimplementing any of it.

Two shapes cover four of the five methods:

    single phase   one call returns code                      (caafe)
    two phase      one call returns ideas in prose, then one
                   call per idea returns code                 (elfgym, ferg, featllm)

The fifth (promptfe) searches an operator grammar rather than asking in prose,
so it implements :class:`Proposer` directly.

Two-phase costs more calls but separates *what to build* from *how to write it*,
which is the published claim: a model reasoning about domain relationships in
prose is not simultaneously fighting pandas syntax.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from discovery.llm import call_llm
from discovery.prompt import extract_blocks

__all__ = ["LLMSettings", "PromptContext", "StrategyBase", "TwoPhaseProposer", "parse_ideas"]

# ELF-Gym keeps only proposal lines beginning with "* ", and handles each
# proposed feature independently. Numbered and dashed lists are accepted too,
# since a model asked for bullets does not always produce the requested marker.
_IDEA_LINE = re.compile(r"^\s*(?:[*\-•]|\d+[.)])\s+(.{3,})$")


@dataclass
class LLMSettings:
    """Transport settings, mirrored from validation.config.LLMConfig."""

    backend: str = "openai"
    model: str = "gpt-4o-mini"
    reasoning_effort: str | None = None
    system_prompt: str = ""
    timeout_s: float = 180.0
    stall_retry_s: float = 40.0
    max_attempts: int = 3
    backoff_s: float = 5.0

    @classmethod
    def from_config(cls, cfg: Any) -> "LLMSettings":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: getattr(cfg, k) for k in names if hasattr(cfg, k)})

    def as_kwargs(self, system_prompt: str | None = None) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "system_prompt": self.system_prompt,
            "timeout_s": self.timeout_s,
            "stall_retry_s": self.stall_retry_s,
            "max_attempts": self.max_attempts,
            "backoff_s": self.backoff_s,
        }
        # A strategy may carry its own system instruction - FeRG's is part of the
        # published method - without the config having to know about it.
        if system_prompt and not self.system_prompt:
            payload["system_prompt"] = system_prompt
        return payload


@dataclass
class PromptContext:
    """The prompt's fixed parts, plus the example rows that rotate per round.

    Everything here is settled before the loop starts except the column block:
    that is rendered once per shot batch, and a round picks the batch matching
    its index. A run with one batch behaves exactly as before.
    """

    task_description: str
    #: One rendered column block per shot batch. Never empty.
    column_contexts: list[str]
    metric_name: str
    metric_explanation: str = ""
    n_rows: int | None = None
    redundancy_max_abs: float | None = None
    target: str = "the target"
    task_type: str = "binary classification"
    domain: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.column_contexts, str):
            # Accepting a bare string keeps the single-batch case readable and
            # stops a caller silently iterating a string into characters.
            self.column_contexts = [self.column_contexts]
        if not self.column_contexts:
            raise ValueError("PromptContext needs at least one column block")

    def context_for(self, round_index: int) -> str:
        """The column block for this round, cycling if rounds outrun batches."""
        return self.column_contexts[(max(0, round_index) - 1) % len(self.column_contexts)]

    @property
    def column_context(self) -> str:
        """The first batch, for callers that do not rotate."""
        return self.column_contexts[0]


def parse_ideas(reply: str, limit: int | None = None) -> list[str]:
    """
    Pull one idea per list item out of a prose reply.

    Code fences are stripped first: a model asked for ideas often volunteers code
    as well, and feeding that back as an "idea" produces a second-phase prompt
    asking it to write code for a block of code.
    """
    without_code = re.sub(r"```.*?```", "", reply, flags=re.DOTALL)
    ideas: list[str] = []
    for line in without_code.splitlines():
        match = _IDEA_LINE.match(line)
        if match:
            idea = match.group(1).strip().rstrip(".")
            if idea and idea not in ideas:
                ideas.append(idea)
    return ideas[:limit] if limit else ideas


class StrategyBase:
    """Shared plumbing: one LLM call, and saving what was said."""

    name = "base"
    system_prompt = ""
    #: Which round is being proposed, so a prompt can pick its shot batch.
    #: Set by propose(); the default keeps a directly-built prompt valid.
    round_index = 1

    def __init__(
        self,
        context: PromptContext,
        llm: LLMSettings,
        *,
        output_dir: str | Path | None = None,
    ):
        self.context = context
        self.llm = llm
        self.output_dir = Path(output_dir) if output_dir else None

    def _ask(self, prompt: str) -> tuple[str, dict[str, Any]]:
        return call_llm(prompt, **self.llm.as_kwargs(self.system_prompt))

    def _save(self, round_index: int, name: str, text: str) -> None:
        """Persist a prompt or reply. The prompt is the experiment."""
        if not self.output_dir:
            return
        directory = self.output_dir / f"round_{round_index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text, encoding="utf-8")


class TwoPhaseProposer(StrategyBase):
    """
    Ideas first, then code for each idea.

    Subclasses supply the two prompts. The second phase runs once per idea, so a
    round costs 1 + len(ideas) calls; ideas that yield no usable block are
    dropped here rather than travelling on as empty candidates.
    """

    def ideas_prompt(self, history: Sequence[dict[str, Any]], n_features: int) -> str:
        raise NotImplementedError

    def code_prompt(self, idea: str, index: int) -> str:
        raise NotImplementedError

    def propose(
        self,
        *,
        history: Sequence[dict[str, Any]],
        proposed_names: Sequence[str],
        n_features: int,
        round_index: int,
    ) -> tuple[list[str], dict[str, Any]]:
        self.proposed_names = list(proposed_names)   # subclasses may mention these
        self.round_index = round_index               # so code_prompt can rotate shots

        prompt = self.ideas_prompt(history, n_features)
        reply, ideas_call = self._ask(prompt)
        self._save(round_index, "phase1_prompt.txt", prompt)
        self._save(round_index, "phase1_reply.txt", reply)

        ideas = parse_ideas(reply, limit=n_features)
        blocks: list[str] = []
        code_calls: list[dict[str, Any]] = []
        transcript: list[str] = []

        for index, idea in enumerate(ideas, start=1):
            code_prompt = self.code_prompt(idea, index)
            code_reply, call = self._ask(code_prompt)
            code_calls.append(call)
            transcript.append(
                f"### idea {index}\n{idea}\n\n--- prompt ---\n{code_prompt}"
                f"\n\n--- reply ---\n{code_reply}"
            )
            found = extract_blocks(code_reply)
            if found:
                blocks.append(found[0])

        self._save(round_index, "phase2.txt", "\n\n".join(transcript))

        return blocks, {
            "strategy": self.name,
            "round": round_index,
            "requested": n_features,
            "ideas": ideas,
            "returned": len(blocks),
            "llm": ideas_call,
            "llm_code_calls": code_calls,
        }
