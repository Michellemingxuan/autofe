# Stage Memory Diagnostic Design

## Goal

Replace the notebook's single validation `Pipeline.run(...)` call with an
explicit, stage-by-stage diagnostic flow. The flow must identify which stage
causes resident memory to jump while preserving the notebook's downstream
`result`-based analysis.

## Scope

This is a notebook-only diagnostic. Production pipeline behavior and public
stage APIs remain unchanged.

## Flow

The validation section will contain separate executable cells for:

1. Diagnostic setup and baseline memory measurement.
2. Data quality.
3. Feature selection and application of its feature gate.
4. Model training.
5. Outcome analysis, including SHAP when configured.
6. Verdict creation and assembly of a `PipelineResult` for later cells.

Each stage records elapsed wall time, current resident set size (RSS), peak RSS,
and the change in current RSS from the preceding checkpoint. Garbage collection
runs before each measurement so short-lived Python objects do not obscure the
stage boundary. RSS is measured at the process level because `tracemalloc`
cannot see the native allocations used by NumPy, XGBoost, and SHAP.

## Data And Artifacts

The diagnostic starts from the existing in-memory `widened` dataset and uses the
same loaded config. It creates a normal timestamped pipeline output directory,
saves model files for SHAP analysis, and writes the main per-stage CSV artifacts.
The final cell constructs `result` with the same fields used by the remaining
notebook cells: dataset, data quality, feature selection, models, analysis,
verdicts, batch verdict, output directory, and elapsed time.

## Memory Interpretation

The checkpoint table identifies the first stage after which RSS rises sharply.
A high retained RSS does not by itself prove a live Python reference: native
allocators may keep freed arenas for reuse. If this pass identifies a stage but
cannot distinguish retained objects from allocator behavior, the next diagnostic
will isolate only that stage in a fresh process.

## Failure Behavior

Every stage is a separate notebook cell. If a stage fails or exhausts memory,
earlier results and checkpoints remain available, and the user can restart from
the failing stage after adjusting its configuration.

## Validation

Validate notebook JSON structure, compile every Python code cell, and run the
existing focused stage tests. The full large-data notebook is not executed as
part of automated validation because it invokes external discovery services and
is the workload under diagnosis.