"""All five strategies: shape of the asking, not quality of the answers."""
import numpy as np
import pandas as pd
import pytest

from discovery.strategies import (
    REGISTRY,
    CaafeProposer,
    ElfGymProposer,
    FeatLlmProposer,
    FergProposer,
    LLMSettings,
    PromptContext,
    PromptFeProposer,
    parse_ideas,
)

CODE_REPLY = """```python
# ('A over B', 'ratio')
# Usefulness: relative magnitude
df["ratio"] = df["a"] / df["b"].clip(lower=0.1)
```end"""


@pytest.fixture
def context():
    return PromptContext(
        task_description="predict bankruptcy from financial ratios",
        # A bare string is accepted and means a single batch of example rows.
        column_contexts='df["a"] (float64; continuous) - Total debt\ndf["b"] (float64) - Equity',
        metric_name="adjusted Gini",
        target="bankruptcy",
        domain="corporate finance",
        redundancy_max_abs=0.95,
        n_rows=3409,
    )


@pytest.fixture
def spy(monkeypatch):
    """Record every prompt; return a scripted reply per call."""
    calls = []
    script = []

    def _call(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        reply = script.pop(0) if script else CODE_REPLY
        return reply, {"backend": "stub", "model": "stub", "attempts": 1}

    monkeypatch.setattr("discovery.strategies.base.call_llm", _call)
    return type("Spy", (), {"calls": calls, "script": script})()


def make(cls, context, **kw):
    return cls(context, LLMSettings(backend="openai", model="stub"), **kw)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_all_five_methods_are_registered():
    assert set(REGISTRY) == {"caafe", "elfgym", "ferg", "featllm", "promptfe"}
    for name, cls in REGISTRY.items():
        assert cls.name == name


@pytest.mark.parametrize("cls", list(REGISTRY.values()))
def test_every_strategy_returns_blocks_and_metadata(cls, context, spy):
    spy.script.extend([CODE_REPLY] * 12)
    blocks, meta = make(cls, context).propose(
        history=[], proposed_names=[], n_features=2, round_index=1)
    assert isinstance(blocks, list)
    assert meta["strategy"] == cls.name
    assert meta["round"] == 1 and meta["requested"] == 2


@pytest.mark.parametrize("cls", list(REGISTRY.values()))
def test_every_strategy_states_the_task_and_the_columns(cls, context, spy):
    spy.script.extend([CODE_REPLY] * 12)
    make(cls, context).propose(history=[], proposed_names=[], n_features=1, round_index=1)
    first = spy.calls[0]["prompt"]
    assert 'df["a"]' in first          # identifiers, never descriptions alone
    assert "bankruptcy" in first


@pytest.mark.parametrize("cls", list(REGISTRY.values()))
def test_every_strategy_writes_its_prompts_down(cls, context, spy, tmp_path):
    spy.script.extend([CODE_REPLY] * 12)
    make(cls, context, output_dir=tmp_path).propose(
        history=[], proposed_names=[], n_features=1, round_index=3)
    saved = list((tmp_path / "round_003").glob("*.txt"))
    assert saved, "a run whose prompts were not kept cannot be explained"


# --------------------------------------------------------------------------- #
# caafe: one call, batch of blocks
# --------------------------------------------------------------------------- #
def test_caafe_asks_once_for_the_whole_batch(context, spy):
    spy.script.append(CODE_REPLY + "\n" + CODE_REPLY.replace("ratio", "ratio2"))
    blocks, meta = make(CaafeProposer, context).propose(
        history=[], proposed_names=[], n_features=2, round_index=1)
    assert len(spy.calls) == 1
    assert len(blocks) == 2
    assert "2 additive columns" in spy.calls[0]["prompt"]


# --------------------------------------------------------------------------- #
# two-phase: ideas, then code per idea
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [ElfGymProposer, FergProposer])
def test_two_phase_asks_for_ideas_then_code_for_each(cls, context, spy):
    spy.script.append("* idea one about leverage\n* idea two about liquidity")
    spy.script.extend([CODE_REPLY, CODE_REPLY])
    blocks, meta = make(cls, context).propose(
        history=[], proposed_names=[], n_features=2, round_index=1)

    assert len(spy.calls) == 3                    # 1 ideas + 2 code
    assert meta["ideas"] == ["idea one about leverage", "idea two about liquidity"]
    assert len(blocks) == 2
    assert "idea one about leverage" in spy.calls[1]["prompt"]
    assert "idea two about liquidity" in spy.calls[2]["prompt"]


def test_ideas_beyond_the_batch_size_are_not_expanded(context, spy):
    spy.script.append("* leverage over equity\n* cash over debt\n* margin trend\n* asset turnover\n* liquidity gap")
    spy.script.extend([CODE_REPLY] * 5)
    blocks, meta = make(ElfGymProposer, context).propose(
        history=[], proposed_names=[], n_features=2, round_index=1)
    assert len(meta["ideas"]) == 2 and len(spy.calls) == 3


def test_an_idea_that_yields_no_code_is_dropped(context, spy):
    spy.script.append("* good idea\n* bad idea")
    spy.script.extend([CODE_REPLY, "I cannot write that."])
    blocks, meta = make(FergProposer, context).propose(
        history=[], proposed_names=[], n_features=2, round_index=1)
    assert len(meta["ideas"]) == 2 and len(blocks) == 1


def test_ferg_carries_its_published_system_instruction(context, spy):
    spy.script.extend(["* an idea", CODE_REPLY])
    make(FergProposer, context).propose(
        history=[], proposed_names=[], n_features=1, round_index=1)
    assert "automated ML engineer" in spy.calls[0]["system_prompt"]
    assert "write down python code" in spy.calls[1]["prompt"]


def test_parse_ideas_ignores_code_the_model_volunteered():
    reply = '* real idea\n```python\ndf["x"] = 1\n```\n* another idea'
    assert parse_ideas(reply) == ["real idea", "another idea"]


@pytest.mark.parametrize("marker", ["* ", "- ", "1. ", "2) "])
def test_parse_ideas_accepts_the_list_markers_models_actually_use(marker):
    assert parse_ideas(f"{marker}leverage over equity") == ["leverage over equity"]


# --------------------------------------------------------------------------- #
# featllm: conditions per class -> one indicator each
# --------------------------------------------------------------------------- #
def test_featllm_turns_each_condition_into_one_indicator(context, spy):
    spy.script.append(
        "Step 1. leverage relates to distress.\n"
        "Answer: bankrupt\n* df[\"a\"] > 2.0\n* df[\"b\"] < 0.5\n"
    )
    spy.script.extend([CODE_REPLY, CODE_REPLY])
    blocks, meta = make(FeatLlmProposer, context).propose(
        history=[], proposed_names=[], n_features=2, round_index=1)

    assert len(spy.calls) == 3
    assert meta["conditions"] == ['bankrupt: df["a"] > 2.0', 'bankrupt: df["b"] < 0.5']
    assert len(blocks) == 2
    # the published two-step framing is preserved
    assert "extracting conditions for each answer class" in spy.calls[0]["prompt"]
    assert "Step 1." in spy.calls[0]["prompt"] and "Step 2." in spy.calls[0]["prompt"]
    # and the second phase asks for a binary indicator, not a ratio
    assert "binary indicator" in spy.calls[1]["prompt"]
    assert "astype(int)" in spy.calls[1]["prompt"]


def test_featllm_pairs_conditions_with_their_class(context, spy):
    spy.script.append("Answer: solvent\n* df[\"a\"] < 1\nAnswer: bankrupt\n* df[\"a\"] > 3")
    spy.script.extend([CODE_REPLY] * 2)
    _, meta = make(FeatLlmProposer, context).propose(
        history=[], proposed_names=[], n_features=2, round_index=1)
    assert meta["conditions"] == ['solvent: df["a"] < 1', 'bankrupt: df["a"] > 3']


# --------------------------------------------------------------------------- #
# promptfe: operator grammar and a score-ranked leaderboard
# --------------------------------------------------------------------------- #
def test_promptfe_lists_the_operator_grammar(context, spy):
    spy.script.append(CODE_REPLY)
    make(PromptFeProposer, context).propose(
        history=[], proposed_names=[], n_features=1, round_index=1)
    prompt = spy.calls[0]["prompt"]
    for operator in ("log", "sqrt_abs", "min_max", "reciprocal"):
        assert operator in prompt
    assert "No features have been scored yet" in prompt


def test_promptfe_ranks_the_leaderboard_by_score_not_by_time(context, spy):
    history = [
        {"round": 1, "feature_name": "weak", "expression": "df['a'] + df['b']", "delta": 0.001},
        {"round": 1, "feature_name": "best", "expression": "df['a'] / df['b']", "delta": 0.050},
        {"round": 1, "feature_name": "broken", "expression": "df['a'] / 0", "error": "non-finite"},
    ]
    spy.script.append(CODE_REPLY)
    make(PromptFeProposer, context).propose(
        history=history, proposed_names=[], n_features=1, round_index=2)
    prompt = spy.calls[0]["prompt"]
    assert prompt.index("df['a'] / df['b']") < prompt.index("df['a'] + df['b']")
    assert "rejected before they could be scored" in prompt
    assert "Error: non-finite" in prompt


def test_promptfe_returns_no_more_than_asked(context, spy):
    spy.script.append("\n".join([CODE_REPLY] * 4))
    blocks, meta = make(PromptFeProposer, context).propose(
        history=[], proposed_names=[], n_features=2, round_index=1)
    assert len(blocks) == 2 and meta["returned"] == 2


# --------------------------------------------------------------------------- #
# shared guarantees
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", list(REGISTRY.values()))
def test_every_strategy_warns_against_the_epsilon_guard(cls, context, spy):
    """The failure that silently destroyed a model; every method must say it."""
    spy.script.extend(["* an idea\n* another", CODE_REPLY, CODE_REPLY])
    make(cls, context).propose(history=[], proposed_names=[], n_features=1, round_index=1)
    assert any("1e-6" in call["prompt"] or "1e6" in call["prompt"] for call in spy.calls)


@pytest.mark.parametrize("cls", list(REGISTRY.values()))
def test_every_strategy_states_the_redundancy_limit(cls, context, spy):
    spy.script.extend(["* an idea", CODE_REPLY])
    make(cls, context).propose(history=[], proposed_names=[], n_features=1, round_index=1)
    assert any("0.95" in call["prompt"] for call in spy.calls)


# --------------------------------------------------------------------------- #
# rotating the example rows between rounds
# --------------------------------------------------------------------------- #
def _ctx(*blocks):
    return PromptContext(task_description="t", column_contexts=list(blocks),
                         metric_name="adjusted Gini")


def test_a_single_batch_is_used_for_every_round():
    context = _ctx("ONLY")
    assert [context.context_for(r) for r in (1, 2, 7)] == ["ONLY"] * 3


def test_each_round_takes_its_own_batch():
    context = _ctx("FIRST", "SECOND")
    assert context.context_for(1) == "FIRST"
    assert context.context_for(2) == "SECOND"


def test_rounds_beyond_the_last_batch_cycle():
    """More rounds than batches reuses them rather than failing."""
    context = _ctx("A", "B")
    assert context.context_for(3) == "A"
    assert context.context_for(4) == "B"


def test_column_context_still_reads_the_first_batch():
    assert _ctx("A", "B").column_context == "A"


def test_a_bare_string_is_not_iterated_into_characters():
    """The trap this guards: a str is a sequence, so it would silently split."""
    context = PromptContext(task_description="t", column_contexts="whole block",
                            metric_name="m")
    assert context.column_contexts == ["whole block"]


def test_no_column_block_at_all_is_refused():
    with pytest.raises(ValueError, match="at least one column block"):
        PromptContext(task_description="t", column_contexts=[], metric_name="m")


@pytest.mark.parametrize("cls", list(REGISTRY.values()))
def test_every_strategy_shows_the_rounds_own_example_rows(cls, spy):
    """
    The regression this exists for.

    Rotation is only useful if the strategies actually read the rotating block.
    Each one renders its prompt differently - some in `propose`, some in helper
    methods two calls down - so a change to the plumbing can leave one of them
    pinned to batch one while the rest rotate.
    """
    spy.script.extend([CODE_REPLY] * 12)
    context = _ctx("BLOCK_ONE_MARKER", "BLOCK_TWO_MARKER")
    make(cls, context).propose(history=[], proposed_names=[], n_features=1,
                               round_index=2)

    prompts = "\n".join(call["prompt"] for call in spy.calls)
    assert "BLOCK_TWO_MARKER" in prompts
    assert "BLOCK_ONE_MARKER" not in prompts
