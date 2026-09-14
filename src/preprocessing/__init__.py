"""Raw archives in, modeling tables out.

Run once per dataset, before any discovery or validation. A dataset folder holds
its raw input, the script that shapes it, and the result:

    data/<name>/prepare.py         the recipe - what a config cannot express
    configs/<name>.yaml            the target, the id column, the candidates
    data/<name>/modeling.parquet   the table the pipeline reads

Only the logic that is identical for every dataset lives here, where it can be
tested once instead of being copied into each script:

    from preprocessing import fetch_archive, sanitize_columns, write_modeling_table

Column names are kept as the archive published them, sanitized to snake_case.
That is a deliberate departure from the g-autofe benchmark, which renamed every
feature to ``X1..Xn`` to keep names out of the model's prior knowledge: here the
names are exactly what a proposer needs, since real-world knowledge about a
column is the whole point of asking a language model for features at all.
"""

from preprocessing.dataset import (
    BuildReport,
    declared_candidates,
    fetch_archive,
    load_config_from,
    print_report,
    resolve_path,
    sanitize,
    sanitize_columns,
    write_modeling_table,
)

__all__ = [
    "sanitize",
    "sanitize_columns",
    "fetch_archive",
    "declared_candidates",
    "resolve_path",
    "write_modeling_table",
    "print_report",
    "load_config_from",
    "BuildReport",
]
