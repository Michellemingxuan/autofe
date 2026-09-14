"""CAAFE: ask for features, show what happened, ask again.

The original method proposes one feature per call and feeds the evaluation back
into the next prompt. Here a round asks for a batch instead - the feedback
mechanism is unchanged, it just arrives in groups - because a batch costs one
call rather than N and lets the proposer diversify within a round.

The strategy owns only *how it asks*. Screening, guards, bookkeeping and
stopping belong to the framework, which is what lets the other four methods drop
in beside this one without reimplementing any of it.

Every prompt and reply is written to disk when an output directory is given. The
prompt is the experiment: a run whose prompts were not kept cannot be explained
afterwards, let alone reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from discovery.prompt import build_prompt, extract_blocks, format_history
from discovery.strategies.base import LLMSettings, PromptContext, StrategyBase

__all__ = ["CaafeProposer"]


class CaafeProposer(StrategyBase):
    """Asks an LLM for a batch of features, showing it the history so far."""

    name = "caafe"

    def propose(
        self,
        *,
        history: Sequence[dict[str, Any]],
        proposed_names: Sequence[str],
        n_features: int,
        round_index: int,
    ) -> tuple[list[str], dict[str, Any]]:
        prompt = build_prompt(
            self.context.task_description,
            self.context.context_for(round_index),
            format_history(history, self.context.metric_name),
            self.context.metric_name,
            metric_explanation=self.context.metric_explanation,
            already_proposed=proposed_names,
            n_rows=self.context.n_rows,
            n_features=n_features,
            redundancy_max_abs=self.context.redundancy_max_abs,
        )

        reply, telemetry = self._ask(prompt)
        blocks = extract_blocks(reply)
        self._save(round_index, "prompt.txt", prompt)
        self._save(round_index, "reply.txt", reply)

        return blocks, {
            "strategy": self.name,
            "round": round_index,
            "requested": n_features,
            "returned": len(blocks),
            "prompt_chars": len(prompt),
            "llm": telemetry,
        }
