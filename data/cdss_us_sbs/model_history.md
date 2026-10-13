# Model history

Current model: Gen9

Every variable earlier generations of the model used, tried or dropped - and
why. The agent reads this: a new feature that reads like a variable here must
say what differs (check C0), and the `others` engine uses it for ideas - a
dropped variable comes back only with a change that answers why it was dropped.

One row per variable. `status` is one of: in model, dropped, tried. `why` says
why it was dropped or not taken (instability, leakage, cost, no lift ...).

| generation | variable | description | status | why |
| --- | --- | --- | --- | --- |
