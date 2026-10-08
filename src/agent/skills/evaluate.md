---
name: evaluate
description: Read a screen result and decide the next attempt - refine, move on, go to a deeper level, or ask for data.
---
# Evaluate skill

## What a screen result says

* `gini_gain` - adjusted Gini of base + feature minus base alone, from a model
  fit on screen_train and scored on screen_valid.
* `capture_gain` - the same, for capture rate at the top of the ranking.
* `gates` - the analyst's minimum gains for this run. Verified means every
  gate set is cleared.
* `coverage` - share of screen rows with a value. Low coverage caps what a
  feature can add.
* `reason` - why it failed: a script error, an infinite value, redundancy with
  a base column (|rho| over the limit), or a gain below a gate.

The screen sample is small. Gini gains within about +/-0.005 are noise; treat
them as "no evidence either way", not as wins or losses. Capture rate is
noisier still.

## Deciding the next attempt

* **Script error** - fix it; probe first if the error is about data shape.
* **Redundant** - the base set already has it. Do not rephrase the same
  quantity; change what is measured.
* **Small gain** - one refinement at most (another window, a ratio instead of
  a level). If that does not move it, the idea is spent.
* **Verified** - build on it in a *different* direction rather than variants of
  it: a near-copy will be redundant with it in the final evaluation.
* **Signal seems to need data nobody has** - L3: `screen_request` within a
  scope, `propose_new_data` beyond scope.

Spread the attempts over different hypotheses. Several distinct verified
features are worth more than one feature tuned five ways.

## Finishing

Call `report_findings(summary)` when the target of K results is reached or the
attempts are used, or earlier if the direction is exhausted. The summary says, per verified feature, what it measures and why
it works; what was tried and failed, briefly; and any data request made.
