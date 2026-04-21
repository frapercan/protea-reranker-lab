"""protea-reranker-lab: frozen-feature reranker experimentation."""

from .schemas import DatasetSpec, ManifestV1, SCHEMA_VERSION
from .builder import build_dataset
from .experiment import (
    DatasetRef,
    ExperimentSpec,
    ModelSpec,
    SweepRef,
    TrainingSpec,
)
from .runner import resolve_dataset, run_experiment

__version__ = "0.1.0"

__all__ = [
    "DatasetSpec",
    "ManifestV1",
    "SCHEMA_VERSION",
    "build_dataset",
    "DatasetRef",
    "ExperimentSpec",
    "ModelSpec",
    "SweepRef",
    "TrainingSpec",
    "resolve_dataset",
    "run_experiment",
]
