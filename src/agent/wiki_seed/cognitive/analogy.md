---
name: analogy
title: Cross-domain analogy
question: Which proven analytical shape from another field maps onto this problem?
knowledge: Other fields - corporate credit, ecology and complex systems, fraud, churn, survival analysis
direction: Lateral - the relational structure of a source field, mapped onto this target
basis: Structure mapping (Gentner, 1983); analogical reasoning (Holyoak & Thagard, 1995); analogy in scientific labs (Dunbar, 1997)
---
# Cross-domain analogy

Borrow a measure that works elsewhere. Map its *relations*, not its surface:
what plays the role of the firm's assets, the ecosystem's state, the churner's
engagement, here?

## How to use it

1. **Pick a source field** and a measure proven there.
2. **Write the mapping**: each part of the source measure and what plays its
   role in this data. A part with no counterpart breaks the analogy - pick
   another.
3. **Keep the relation**: the ratio, the threshold, the dynamics. That is what
   carries the signal over.

## Shapes worth borrowing

| Source field | Shape | Here |
| --- | --- | --- |
| Corporate credit | Merton distance-to-default: buffer ÷ volatility | liquidity buffer ÷ volatility of monthly net outflow |
| Complex systems | Critical slowing down (Scheffer et al., 2009): variance and autocorrelation rise before a tipping point | rising variance and lag-1 autocorrelation of the monthly payment ratio |
| Churn | Engagement decay: activity falling against its own baseline | share of usual merchants or categories no longer visited |
| Fraud | Velocity and novelty: many new things fast | new merchants, channels or devices in a short window |
| Survival analysis | Hazard and time-since-event | months since the last full payment, since the last limit change |
| Portfolio risk | Concentration (Herfindahl) | spend concentrated in few categories, or debt in one lender |
| Physiology | Heart-rate variability: a healthy system varies, a strained one goes rigid or erratic | regularity of payment timing and amounts |

## Example idea

**Critical slowing down** - the variance and lag-1 autocorrelation of the
monthly payment ratio over the last 12 months, rising before arrears.

## In the idea

`engine: analogy`. In the description, name the source field and the mapping -
"Merton: buffer = ..., volatility = ...".
