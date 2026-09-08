"""mllite: a modular, parallel pipeline for evaluating newly proposed features."""

from mllite.config import Config, load_config
from mllite.data import (
    Dataset,
    build_dataset,
    prepare_dataset,
    prepare_dataset_from_frames,
)
from mllite.pipeline import Pipeline, PipelineResult, run_pipeline

__all__ = [
    "Config",
    "load_config",
    "Dataset",
    "build_dataset",
    "prepare_dataset",
    "prepare_dataset_from_frames",
    "Pipeline",
    "PipelineResult",
    "run_pipeline",
]
__version__ = "0.1.0"
