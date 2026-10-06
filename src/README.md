# src

Four packages. Two produce candidate features, one judges them, one prepares data.

```
preprocessing/   prepare a dataset once: splits, clustering shots       (used by data/*/prepare.ipynb)
validation/      judge features: data quality → selection → models →   (the pipeline; `autofe` CLI)
                 analysis → verdict, on train / valid / test
discovery/       the earlier way to find features - a fixed loop: one   (a stage of the pipeline,
                 prompt per round, the LLM writes code, the screen       when discovery.enabled)
                 scores it. Five strategies (CAAFE, FeatLLM, ...)
agent/           the Model Agent - an LLM that picks tools step by step (its own server, CLI and UI)
```

How they depend on each other - nothing points back up the list:

```
agent ──uses──▶ discovery.screen   (the screen: score a column against base, with its guards)
  │                                  the same yardstick as discovery, so gains compare
  ├──uses──▶ validation             (config, data reading, metrics; the pipeline for Evaluate)
  └──uses──▶ preprocessing.shots    (to generate clustering shots)
discovery ──uses──▶ validation      (data, metrics, the booster)
validation ──runs──▶ discovery      (only as an optional stage, when discovery.enabled)
```

## agent/

```
agent.py         builds the agent and runs its conversation (stages, streaming)
composer/        puts the prompt together: brief and message templates, filled-in sections  ← start here
skills/          the skills, appended to every brief in full
tools/           what the agent can do; each tool checks, runs and records
session.py       one run's state, parameters, events and approvals
workspace.py     what the agent can see: screen rows, sources, the CAS scope
memory.py        earlier runs' features and requests; refuses repeats
execution.py     runs agent-written code in a child process (_child.py), guarded
evaluate.py      an evaluation: verified features through the validation pipeline
server.py, cli.py, setup.py, events.py, llm.py, synthetic.py   - the app around it
```

The prompt is put together by `agent/composer/`; print one with
`PYTHONPATH=src python -m agent.composer -c configs/synthetic_agent.yaml`.
The earlier loop's prompt is `discovery/prompt.py`.
