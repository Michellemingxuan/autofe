"""Validation: decide whether newly proposed features earn their place.

The second half of the loop. `discovery` proposes candidate features; this
package puts them through data quality, screening, per-feature model builds,
outcome analysis and the verdict gates, and says which survive.

Distributed as the `mllite` package; imported as `validation`.
"""

from validation.config import Config, load_config
from validation.data import (
    Dataset,
    build_dataset,
    prepare_dataset_from_frames,
)
from validation.pipeline import Pipeline, PipelineResult, run_pipeline

__all__ = [
    "Config",
    "load_config",
    "Dataset",
    "build_dataset",
    "prepare_dataset_from_frames",
    "Pipeline",
    "PipelineResult",
    "run_pipeline",
]
__version__ = "0.1.0"
