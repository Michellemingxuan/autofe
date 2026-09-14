"""Proposal strategies.

Each decides only *how it asks* for features. The framework owns the sample, the
screen, the guards, the history and the stopping rule, which is what lets five
published methods coexist without reimplementing any of it.

    caafe      one batch per round, scored feedback folded into the next prompt
    elfgym     bulleted ideas, then code per idea, handled independently
    ferg       domain-framed rationale, then "write down python code"
    featllm    conditions per answer class, then one binary indicator each
    promptfe   a fixed operator grammar, guided by a score-ranked leaderboard

The first four ask in prose; promptfe searches an algebra. Two of them are
reconstructions rather than transcriptions - see the fidelity notes in
elfgym.py and promptfe.py before treating a run as a published-method result.
"""

from discovery.strategies.base import (
    LLMSettings,
    PromptContext,
    StrategyBase,
    TwoPhaseProposer,
    parse_ideas,
)
from discovery.strategies.caafe import CaafeProposer
from discovery.strategies.elfgym import ElfGymProposer
from discovery.strategies.featllm import FeatLlmProposer
from discovery.strategies.ferg import FergProposer
from discovery.strategies.promptfe import PromptFeProposer

#: Every strategy the discovery stage can run, by config name.
REGISTRY = {
    CaafeProposer.name: CaafeProposer,
    ElfGymProposer.name: ElfGymProposer,
    FergProposer.name: FergProposer,
    FeatLlmProposer.name: FeatLlmProposer,
    PromptFeProposer.name: PromptFeProposer,
}

__all__ = [
    "REGISTRY",
    "LLMSettings",
    "PromptContext",
    "StrategyBase",
    "TwoPhaseProposer",
    "parse_ideas",
    "CaafeProposer",
    "ElfGymProposer",
    "FergProposer",
    "FeatLlmProposer",
    "PromptFeProposer",
]
