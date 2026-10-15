---
name: ideas
description: Generate a round of ideas with the idea engines - varied, grounded in data in reach, and new against the board.
---
# Ideas skill

An idea is not a feature. It is a claim about why some customers default
that the model does not yet see, and where that would show in the data.
Features are attempts to measure it; the board keeps score per idea.

## An idea is a hypothesis, one level above its features

| | Idea | Feature |
| --- | --- | --- |
| Name | a concept: `liquidity_buffer_depletion` | a measure: `open_to_buy_trend_3m` |
| Says | what behaviour or mechanism, and why it means risk | what is computed, over what window |
| How many | one | several can test it |

Write each idea with:
* `name` - the concept, in snake_case. No window, statistic or version in it
  (`_6m`, `_90d`, `_std`, `_v2` belong to features). A feature-like name is
  sent back.
* `description` - the hypothesis: who, doing what, and why it signals default
  risk the model misses; and where it came from, as the engine's guide asks.
  Not a formula.
* `measures` - 2 to 4 different ways to measure it, one line each: a
  candidate feature (or, at L3, data to request) - what, over what window,
  against what. Make them differ in kind, not only in window.
* `data` - the columns and sources it would draw on.

For example, an idea from `mechanism` (Minsky's fragility stages):

* name: `minsky_stage_shift`
* description: customers slide from paying in full, to paying the minimum, to
  borrowing to pay - each step down shows months before arrears, which the
  model's late-stage columns cannot see.
* measures: share of the last 6 statements paid in full vs at the minimum;
  months since the customer last paid in full; minimum-only payments in a
  month a new tradeline opened.

## Use the engines, not free brainstorming

Each idea comes from one engine, and says so (`engine`). Read an engine's
method with `engine` before you use it for the first time in a run.

| engine | starts from | moves |
| --- | --- | --- |
| `mechanism` | a theory of why people default (its theory library) | forward: cause → decisions → traces |
| `contrastive` | cases the incumbent gets wrong, look-alike pairs | backward: difference → explanation |
| `analogy` | a measure proven in another field | sideways: map the relations |
| `morphological` | the board and the model columns on seven axes | fill an empty cell, or mutate one axis |
| `others` | any route the four miss - practitioners' know-how, earlier model generations, the product's rules, the market | open: name the route |

* The round's message names its **focus engines**, drawn by how well each
  engine's ideas have done. At least the stated number of ideas come from them.
* Use several engines in a round, and several **lenses** (the kind of
  signal: trend, recency, concentration ...). The checks count both.
* In `description`, say where the idea came from, as the engine's guide asks:
  the theory and the step of its chain, the contrast, the source field and
  the mapping, the grid cell - or, for `others`, the route.
* `others` is there to be creative beyond the four. Practitioner knowledge -
  what collections, underwriting and fraud teams notice first - is one route;
  any route you can name is welcome.

## Practical: every idea must be buildable

* Only data in reach: the model columns, the run's linked sources, and - for
  an L3 idea - a scope or data beyond it. Name the columns in `data`.
* On the direction's question, at the model's grain (one row per id) and
  before each row's as-of date.
* A reviewer can explain it in one sentence. An idea that needs a page to
  explain is two ideas, or none.

## New against the board

The brief's **Earlier directions** lists the ideas on the board with their
engine, standing and P(useful):

* **✓ active** - it carried signal. Deepen it by changing one axis - another
  window, anchor, comparison or population - not by renaming it.
* **· low** - measured, no signal yet. Bring it back only with a change that
  answers its "why not".

Nothing on the board is forbidden. An idea that reads like an earlier one is
linked to it and starts from its record, so one whose features failed is
drawn late - worked after the fresher and stronger ideas, not never.

Two ideas of one round that say the same thing are sent back: merge them, or
make one measure something else.

## Then, per feature

Build from an idea's measures, and say which idea each proposal tests:
`screen_feature(..., idea="minsky_stage_shift")` - the same for
`screen_request` and `propose_new_data`. Name the feature for what it
computes (`full_pay_share_6m`), not after the idea; one idea can have several
features. Each screen reply shows the run's ideas - every round's - with their
standing and `work_next`: the order to work them in, drawn by Thompson sampling
from each idea's record. An earlier round's idea that carried signal stays in
the order: its next measure is often worth more than a new idea's first. An idea whose features carried no signal draws low and moves
back; follow the order unless you have a reason not to, and say the reason.
