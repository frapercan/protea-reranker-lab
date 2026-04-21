"""protea-reranker-lab: frozen-feature reranker experimentation."""

from .schemas import (
    DatasetSpec,
    ManifestV1,
    SCHEMA_VERSION,
    compute_feature_schema_sha,
    compute_schema_sha,
)
from .builder import build_dataset
from .experiment import (
    DatasetRef,
    ExperimentSpec,
    ModelSpec,
    SweepRef,
    TrainingSpec,
)
from .runner import resolve_dataset, run_experiment

__version__ = "0.2.0"

__all__ = [
    "DatasetSpec",
    "ManifestV1",
    "SCHEMA_VERSION",
    "build_dataset",
    "compute_feature_schema_sha",
    "compute_schema_sha",
    "DatasetRef",
    "ExperimentSpec",
    "ModelSpec",
    "SweepRef",
    "TrainingSpec",
    "resolve_dataset",
    "run_experiment",
]
