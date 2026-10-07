# The prompt composer

It puts together what the LLM reads. A run's prompt is its **brief** (the system
instructions, one per kind of run) and a few **messages** (the user turns). The
templates are in `templates/`; the skills live beside the tools, in `agent/skills/`.

| File | What it is | Used by |
|---|---|---|
| `templates/feature_engineer.md` | the brief of a direction - L1 / L2 features, and an L3 share in a mixed run | Discover |
| `templates/data_scout.md` | the brief of a data-request run - L3 only | Discover, L3 ticked alone |
| `templates/linkage_writer.md` | the brief of a linkage job | Setup → Propose linkage |
| `templates/messages.md` | the user turns: the task, `ideas`, `propose`, `sent_back`, `nudge` | every run |
| `../skills/*.md` | the skills - `data_sourcing`, `feature`, `evaluate` - appended to the brief in full | every brief |
| `sections.py` | the parts filled in from the workspace and the run - see below | `__init__.compose` |

A brief is a template: `{fields}` are filled in by `compose()` in `__init__.py`.
Plain fields are values (`{K}`, `{id_col}`, `{task_description}` ...); the longer
ones are whole sections, built in `sections.py`:

| Field | Section | Comes from |
|---|---|---|
| `{skills}` | the run's skills, in full | `agent/skills/*.md` |
| `{memory}` | what earlier directions proposed - not to be repeated | `agent/memory.py`: the run folder |
| `{ideas}` | the ideas stage: the lenses, this run's focus, the L3 rule and the CAS list | `agent/tools/ideas.py` |
| `{current_data}` | the model columns and linked sources a challenge checks against | the workspace |
| `{columns}` | the columns a feature script can read - base and the run's sources, with descriptions and examples | the workspace: column descriptions, sources' sample JSON |
| `{scope_notes}` | the analyst's notes on the CAS scope, verbatim | Setup → Scope notes |
| `{gates}` | the analyst's minimum gains, in words | the run's parameters |
| `{quota}` | a mixed run's target, split across levels | the run's parameters |
| `{shot_list}` | the shot categories | Setup → Shots |

Two more things reach the model, kept beside the code they describe:

* **tool descriptions** - the docstrings of the wrappers in `agent/tools/__init__.py`
  (`for_agent`): what each tool does and what each argument means;
* **tool replies** - each tool's result, or why it refused and what to do next
  (`next`, `error`), in `agent/tools/*.py`.

## See it

```bash
# the brief and first message a run would get - nothing is run or written
PYTHONPATH=src python -m agent.composer -c configs/synthetic_agent.yaml --levels L1 L2
PYTHONPATH=src python -m agent.composer -c configs/synthetic_agent.yaml --levels L3 -k 5
PYTHONPATH=src python -m agent.composer -c configs/synthetic_agent.yaml --linkage spends
```

Every run also keeps the brief it was given: `outputs/agent/<run_id>/brief.md`.
