"""protea-reranker-lab: frozen-feature reranker experimentation."""

from .contracts import FeatureBuildContext, KnnContext
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
from .sequences import ParquetFeatureSequence
from .staging import StageResult, stage_for_training

__version__ = "0.3.0"

__all__ = [
    "DatasetSpec",
    "FeatureBuildContext",
    "KnnContext",
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
    "ParquetFeatureSequence",
    "StageResult",
    "resolve_dataset",
    "run_experiment",
    "stage_for_training",
]
