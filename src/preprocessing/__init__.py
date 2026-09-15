"""Raw archives in; fixed splits and few-shot rows out.

Run once per dataset, before any discovery or validation. A dataset folder holds
its raw input, the notebook that shapes it, and the result:

    data/<name>/prepare.ipynb    input -> profile -> split -> few-shot -> write
    data/<name>/raw/             the downloaded input, cached
    data/<name>/train.csv        the fixed split the pipeline reads
    data/<name>/valid.csv
    data/<name>/test.csv
    data/<name>/few_shot.csv     the example rows a discovery run shows
    configs/<name>.yaml          the target, the id column, the candidates

Each notebook is specific to its dataset. Only the few pieces that are identical
for every one live here:

    from preprocessing import stratified_split, build_shot_batches, write_splits

Column names are kept as the archive published them, sanitized to snake_case.
That is a deliberate departure from the g-autofe benchmark, which renamed every
feature to ``X1..Xn`` to keep names out of the model's prior knowledge: here the
names are exactly what a proposer needs, since real-world knowledge about a
column is the whole point of asking a language model for features at all.
"""

from preprocessing.dataset import (
    SPLITS,
    build_sample,
    declared_candidates,
    fetch_archive,
    resolve_path,
    sanitize,
    sanitize_columns,
    stratified_split,
    write_splits,
)
from preprocessing.shots import build_shot_batches

__all__ = [
    "SPLITS",
    "sanitize",
    "sanitize_columns",
    "fetch_archive",
    "stratified_split",
    "build_sample",
    "build_shot_batches",
    "declared_candidates",
    "resolve_path",
    "write_splits",
]
