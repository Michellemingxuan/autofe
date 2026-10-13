---
name: contrastive
title: Contrastive abduction
question: Why did P default rather than Q? What does the incumbent miss?
knowledge: The data - the incumbent's errors (clean-looking defaulters, high-risk survivors), matched look-alike pairs, surprising screen results
direction: Backward - an observed difference, to its best explanation
basis: Abduction (Peirce); contrastive explanation (Lipton, 1990; Miller, 2019); method of difference (Mill, 1843); anomalous data (Chinn & Brewer, 1993)
---
# Contrastive abduction

Start from cases, not theory. Find two customers who look alike on the
model's columns but ended differently, and ask what separates them.
Whatever does is, by construction, invisible to the incumbent.

## How to use it

1. **Find the contrast.** Read the labelled shots (`shots`) and sample rows:
   * a defaulter who looks clean on the model's columns;
   * a survivor who looks risky;
   * a pair alike on the base features, one bad and one good.
   A screen result that surprised you is a contrast too: a feature that
   should have worked and did not, or the reverse.
2. **List the differences** you can see between them - in the extra sources,
   in timing, in sequence, in what is missing.
3. **Explain** the difference that best accounts for the outcome, and say
   why the model's columns cannot see it.
4. **Turn the explanation into a measure** that separates the pair, then
   ask whether it would separate similar pairs too, not only this one.

## Example idea

**Look-alike pairs differ in when they pay, not how much** - pay day relative
to the due date, drifting later over recent cycles, at the same paid amount.

## In the idea

`engine: contrastive`. In the description, say the contrast you started
from - "clean-looking defaulters in shots X ...".
