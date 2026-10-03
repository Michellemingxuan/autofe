# Stage Memory Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the notebook's monolithic validation call with individually runnable stages and process-memory checkpoints.

**Architecture:** The notebook calls the existing stage functions directly and stores their normal result objects. A small macOS-compatible helper records current and peak RSS after garbage collection, and the final cell assembles the existing `PipelineResult` interface so all later analysis cells remain unchanged.

**Tech Stack:** Jupyter notebook JSON, Python standard library, pandas, existing `validation` package APIs.

---

### Task 1: Replace The Monolithic Validation Cell

**Files:**
- Modify: `notebooks/usage_cdss_us_sbs.ipynb`

- [ ] **Step 1: Add diagnostic setup**

Replace the single `Pipeline.run(...)` cell with a setup cell that imports `gc`,
`resource`, `subprocess`, and `time`; imports `PipelineResult`, the logging helper,
and all five stage functions; creates a normal `Pipeline` output directory; and
defines `memory_checkpoint(label)`.

The helper must run `gc.collect()`, obtain current RSS from macOS `ps`, obtain
peak RSS from `resource.getrusage`, append a row to `memory_checkpoints`, and
display the latest row. Each row stores elapsed seconds, current RSS in MiB,
peak RSS in MiB, and current-RSS change from the preceding checkpoint. Record a
`baseline` checkpoint before running any stage.

- [ ] **Step 2: Add data-quality and feature-selection cells**

Use one executable cell for data quality and a separate executable cell for
feature selection. Run `run_data_quality(widened, cfg_validate)`, checkpoint
memory, apply the same data-quality and feature-selection gate rules used by
`Pipeline.run`, run `run_feature_selection`, checkpoint again, and write the
corresponding CSV artifacts through the `Pipeline` artifact helpers.

- [ ] **Step 3: Add modeling and analysis cells**

Use one executable cell for modeling and a separate executable cell for
analysis. Run `run_modeling` with the diagnostic output's `models` directory,
write tuning trials, checkpoint memory, then run `run_analysis`, write analysis
artifacts, and checkpoint memory again. Preserve `models` because analysis
requires their predictions and model paths.

- [ ] **Step 4: Add verdict and result-assembly cell**

Run `run_verdict`, write verdict artifacts, checkpoint memory, construct a
`PipelineResult` with `config`, `output_dir`, `dataset`, `discovery`,
`data_quality`, `feature_selection`, `models`, `analysis`, `verdicts`, `batch`,
and `elapsed_seconds`, write the memory checkpoint table, and display the full
table and output path.

- [ ] **Step 5: Validate notebook structure and syntax**

Run a JSON/notebook validation script that asserts every cell has a supported
type and `metadata.language`, then compiles each Python code cell after removing
IPython magic lines.

Expected: valid JSON and every diagnostic cell compiles.

- [ ] **Step 6: Run focused stage tests**

Run:

```bash
pytest -q tests/test_data.py tests/test_distribution_check.py \
  tests/test_feature_selection.py tests/test_pipeline.py
```

Expected: all selected tests pass. Do not execute the large notebook because it
uses external discovery and is itself the memory workload under investigation.