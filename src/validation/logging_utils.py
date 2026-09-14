"""Logging helpers shared by every stage."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure the root logger once, optionally teeing to a file."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FMT))
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(_FMT))
        root.addHandler(file_handler)

    # shap/numba are chatty at INFO
    logging.getLogger("shap").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def timed(logger: logging.Logger, label: str):
    """Log wall time around a block of work."""
    start = time.perf_counter()
    logger.info("%s ...", label)
    try:
        yield
    finally:
        logger.info("%s done in %.1fs", label, time.perf_counter() - start)
