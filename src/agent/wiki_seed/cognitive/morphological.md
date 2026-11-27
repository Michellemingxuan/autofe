---
name: morphological
title: Morphological search
question: Given what we've tried, which combinations are still untested?
knowledge: The system's own trial history (the logging board), and the base features decomposed onto the same axes
direction: Iterative - decompose, find empty cells or mutate one axis, recombine
basis: Morphological analysis (Zwicky, 1969); combinational and exploratory creativity (Boden, 2004); conceptual blending (Fauconnier & Turner, 2002); representational change (Ohlsson, 1992)
---
# Morphological search

Treat a feature as a choice on seven axes. Place what exists - the model's
columns and the board's features - on the grid, then generate from the cells
nobody has filled, or by changing one axis of something that worked.

## The axes

| Axis | Question | Values (examples) |
| --- | --- | --- |
| measure | What is counted or summed? | spend, payments, balance, transactions, inquiries, fees, declines |
| source | Where does it come from? | model columns, each linked source, a scope, bureau |
| anchor | From which moment is time counted? | as-of date, last payment, last statement, last limit change, last bureau inquiry, account open |
| window | Over how long? | 7, 30, 90, 180, 365 days; last n cycles; since the anchor |
| comparison | Against what? | nothing (level), own past, population, peer segment, limit or capacity, another source |
| statistic | Which summary? | sum, count, mean, share, trend (slope), volatility, max, recency, entropy, autocorrelation |
| population | Over which subset? | all transactions, a category, a channel, merchants new to the customer, weekends, after pay day |

## How to use it

1. **Decompose** the features that exist onto the axes: a few model columns,
   and every feature on the board, verified or not.
2. **Find an empty cell**: a combination nobody has tried that still makes
   sense - "payments × since last inquiry × trend".
3. **Or mutate one axis** of a feature that worked: keep six, change one.
   "Spend vs population" verified → change the comparison to "own past", or
   the anchor to "since last bureau inquiry".
4. **Avoid the dead cells**: a cell whose features the board marks as
   explained by the model, or with no signal, stays empty unless you change
   what made them fail.

## Example idea

Verified **spend vs population** → mutate the comparison axis to **own
past** (the same spend against the customer's own 12-month median), or the
anchor to **since last bureau inquiry**.

## In the idea

`engine: morphological`. In the description, give the cell - measure,
source, anchor, window, comparison, statistic, population - and, for a
mutation, the feature it comes from and the axis changed.
