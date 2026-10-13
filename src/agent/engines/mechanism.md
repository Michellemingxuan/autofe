---
name: mechanism
title: Mechanism simulation
question: Given how typical defaulters get there, what should we observe, and how early?
knowledge: Financial, banking and behavioural theory on why people default - the theory library below
direction: Forward - an assumed cause chain, to the decisions it forces, to the traces they leave
basis: Mental simulation (Kahneman & Tversky, 1982); causal models (Sloman, 2005; Pearl, 2009); theory of mind (Premack & Woodruff, 1978)
---
# Mechanism simulation

Pick a mechanism from the library, then run it forward in your head: who is
this customer, what happened to them, what do they do next, and which of
those actions leave a trace in the data you have - payments, spend, bureau.

## How to use it

1. **Pick a theory** from the library that fits the direction.
2. **Simulate the chain** one step at a time: cause → strain → coping →
   choosing which creditors to pay → missed payment. Say what the customer
   decides at each step.
3. **Find the earliest trace.** A mature model is dense at the late stages
   (arrears, utilisation at the limit). The value is in the early ones: what
   the customer does months before the first missed payment.
4. **Ground it.** Name the columns or the source that would show the trace.
   No data for it → an L3 idea, or another theory.

## Theory library

| Theory domain | Assumed cause chain | Key references | Expected trace (payment / spend / bureau) |
| --- | --- | --- | --- |
| Liquidity shock and buffer depletion | Income or expense shock → buffer drawn down → no liquidity left → missed payment | Ganong & Noel (2020); Gross & Souleles (2002) | Falling total open-to-buy across bureau tradelines; rise in cash-like spend |
| Financial fragility stages | Hedge (pays in full) → speculative (pays interest or the minimum) → Ponzi (borrows to pay) → default | Minsky (1986, 1992) | Paid/statement ratio stepping down by stage; new tradelines opening alongside minimum payments |
| Life-event triggers | Job loss, divorce or similar → income drop → strain | Sullivan, Warren & Westbrook (2000) | Abrupt loss of routine, income-linked categories such as commuting or workday spend |
| Over-indebtedness / capacity | Debt service outgrows income capacity → arrears | 5 Cs of credit (capacity) | Rising total bureau balances against stable internal spend; balance growth across lenders |
| Debt prioritisation | Under strain, the customer ranks creditors and pays some first | Gathergood et al. (2019) | Other lenders' balances falling while ours holds or grows |
| Repayment heuristics | Anchoring on the minimum → slow paydown → balance builds | Stewart (2009); Keys & Wang (2019) | Payments clustering at exactly the minimum due; switch from full to minimum payer |
| Time preference | Present bias → overborrowing → repayment shortfall | Laibson (1997); Meier & Sprenger (2010) | Spend surges right after a payment or a limit increase |
| Attention and avoidance | Bad news → avoiding the account → late or missed payments | Karlsson, Loewenstein & Seppi (2009); Sicherman et al. (2016) | Payments drifting toward the due date; early or partial payments stopping |
| Cognitive scarcity | Strain narrows attention → worse financial decisions | Mullainathan & Shafir (2013) | Spend fragmenting into many small purchases; rising fees and charges |
| Credit-seeking / adverse selection | Riskier borrowers seek more credit and accept worse terms | Stiglitz & Weiss (1981) | Inquiry bursts; several new tradelines in a short window |
| Willingness vs ability | Capacity intact, but the customer chooses not to pay us | Guiso, Sapienza & Zingales (2013) | Bureau shows other accounts current while payments to us weaken |
| First-party fraud (bust-out) | Build a clean history → raise limits → max out → abandon the account | Industry practice (fraud operations) | Long clean history, then a sudden velocity spike and the limit maxed out |

## Example idea

**Minsky stage shift** - the share of recent months the customer paid in full,
paid only the minimum, or paid while new tradelines opened elsewhere. A step
down the stages (hedge → speculative → Ponzi) shows before arrears do.

## In the idea

`engine: mechanism`. In the description, name the theory and the step of the
chain the trace belongs to - "Minsky, speculative → Ponzi: ...".
