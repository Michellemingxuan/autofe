"""One place to decide how work is fanned out.

Stages never import joblib directly; they call ``parallel_map`` so that the
backend, worker count and thread budgeting stay consistent across the pipeline.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, List, Sequence, TypeVar

from joblib import Parallel, delayed

from validation.logging_utils import get_logger

T = TypeVar("T")
R = TypeVar("R")

logger = get_logger(__name__)


def resolve_n_jobs(n_jobs: int) -> int:
    """Turn -1/0/None into a concrete positive worker count."""
    cpus = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, cpus + 1 + n_jobs)
    return max(1, min(n_jobs, cpus))


def threads_per_worker(n_jobs: int, n_tasks: int) -> int:
    """Split available cores between process workers and each worker's threads.

    XGBoost spawns its own thread pool; without this the outer fan-out and the
    inner OpenMP pool oversubscribe the box and everything slows down.
    """
    cpus = os.cpu_count() or 1
    workers = max(1, min(resolve_n_jobs(n_jobs), max(1, n_tasks)))
    return max(1, cpus // workers)


def parallel_map(
    func: Callable[..., R],
    items: Sequence[T],
    n_jobs: int = -1,
    backend: str = "loky",
    desc: str = "task",
    log_progress: bool = False,
    **kwargs,
) -> List[R]:
    """Apply ``func`` to every item, in parallel unless there is nothing to gain.

    ``kwargs`` are passed through to every call.
    """
    items = list(items)
    if not items:
        return []
    workers = resolve_n_jobs(n_jobs)
    if workers == 1 or len(items) == 1 or backend == "sequential":
        logger.debug("running %d %s(s) sequentially", len(items), desc)
        results = []
        for completed, item in enumerate(items, start=1):
            results.append(func(item, **kwargs))
            if log_progress:
                logger.info("%s: %d/%d completed", desc, completed, len(items))
        return results

    logger.info("running %d %s(s) across %d worker(s) [%s]", len(items), desc, workers, backend)
    if not log_progress:
        return Parallel(n_jobs=workers, backend=backend)(
            delayed(func)(item, **kwargs) for item in items
        )

    completed_results = Parallel(n_jobs=workers, backend=backend, return_as="generator_unordered")(
        delayed(_indexed_call)(index, func, item, kwargs) for index, item in enumerate(items)
    )
    ordered_results: List[R] = [None] * len(items)  # type: ignore[list-item]
    for completed, (index, result) in enumerate(completed_results, start=1):
        ordered_results[index] = result
        logger.info("%s: %d/%d completed", desc, completed, len(items))
    return ordered_results


def _indexed_call(index: int, func: Callable[..., R], item: T, kwargs: dict) -> tuple[int, R]:
    return index, func(item, **kwargs)


def chunked(items: Sequence[T], size: int) -> Iterable[List[T]]:
    """Split a sequence into consecutive chunks of at most ``size``."""
    size = max(1, int(size))
    items = list(items)
    for start in range(0, len(items), size):
        yield items[start:start + size]
