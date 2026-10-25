# From a direction to features: an example

**The goal:** find new features that make a mature credit risk model better at spotting who will default. Years of work have already covered balance, utilisation, payments and bureau data, so the value is in signals the model doesn't have yet.

**The analyst's direction:** *"Catch money trouble early, in how people live, not just how they pay."*

| Idea | Idea engine: how the agent got there | Feature | Level | What the feature tells the model |
|---|---|---|---|---|
| **1 · Cashing in rewards.** Points saved for a holiday get turned into a statement credit, because they need the money now. | **Mechanism simulation:** thinking about why people behave this way | `points_to_credit_share` | L2 | More of their points go to paying the bill, not to travel or shopping |
| | | `points_burn_vs_earn_3m` | L2 | They spend points as fast as they earn them, after years of saving them |
| | | `first_cash_redemption` | L2 | They cash in points for the first time ever |
| **2 · Breaking weekly routines.** The weekday coffee, the commute and the lunch near the office stop. A lost job often shows here first. | **Contrastive abduction:** comparing people who defaulted with similar people who didn't | `weekday_daytime_drop` | L2 | They spend much less on weekday mornings and lunchtimes than before |
| | | `commute_merchants_stop` | L2 | Train, toll or parking charges they had every week suddenly stop |
| | | `category_mix_shrink` | L1 | They spend at fewer kinds of places than they used to |
| **3 · Avoiding the account.** People in trouble avoid facing the bill: they log in less, leave statements unopened and switch off autopay, much like customers about to leave. | **Cross-domain analogy:** borrowed from models that predict customers leaving | `autopay_switched_off` | L2 | They turn off automatic payments after years of using them |
| | | `app_logins_drop` | L3 | They open the app far less than before. Needs digital activity data we don't have yet |
| | | `statement_unopened_streak` | L3 | They stop opening their statement emails. Needs email data we don't have yet |
| **4 · Shifting to riskier shops.** Some shops' customers default much more often than others'; a shift toward them is a warning. | **Cross-domain analogy:** borrowed from fraud detection, which scores merchants by risk | `risky_merchant_share` | L2 | More of their spending goes to shops whose customers often default |
| | | `new_risky_merchants_3m` | L2 | They start shopping at such places for the first time |
| | | `cash_advance_store_visits` | L2 | They start using pawn shops, cheque cashers or payday lenders |

**A feature is kept only if it makes the model better** at spotting future defaulters, and it must do so on data the agent never saw. If it adds nothing the model doesn't already know, it's dropped.

**Levels:**
- **L1:** made from data the model already uses.
- **L2:** also uses other data we already have, such as card transactions or rewards records.
- **L3:** needs new data. The agent writes the query to pull it, and the analyst approves it.

**Idea engines:** each idea records the engine that produced it. Each round, the agent leans more on the engines whose ideas have worked before.
- **Mechanism simulation:** starts from why people default and works forward to the traces it would leave.
- **Contrastive abduction:** starts from the cases the current model gets wrong and works backward to what explains them.
- **Cross-domain analogy:** borrows a measure proven in another field, such as fraud, churn or corporate credit.
- **Morphological search:** looks for combinations of data, window and transformation that haven't been tried yet.
- **Others:** any other route, such as practitioners' know-how or how the product works. The idea says which route it took.

All feature names are illustrative.
