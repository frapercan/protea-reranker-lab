"""Experiment spec — one YAML per research run.

An :class:`ExperimentSpec` composes a dataset reference (pre-built manifest
*or* inline :class:`DatasetSpec` to materialize on demand), a model config,
training knobs, and an optional sweep backend. It is the reproducibility
unit: two runs with the same spec hash should produce comparable results
(modulo W&B bayes sampling).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .schemas import DatasetSpec


EXPERIMENT_SCHEMA_VERSION = "v1"


class DatasetRef(BaseModel):
    """Either a pre-built dataset (by manifest path) or a build recipe."""

    manifest: Path | None = None
    spec: DatasetSpec | None = None

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _exactly_one(self) -> "DatasetRef":
        if (self.manifest is None) == (self.spec is None):
            raise ValueError("DatasetRef: set exactly one of 'manifest' or 'spec'")
        return self


class ModelSpec(BaseModel):
    kind: Literal["lgbm_reranker"] = "lgbm_reranker"
    defaults: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class TrainingSpec(BaseModel):
    cell: str
    val_strategy: Literal["protein_group", "temporal", "none"] = "temporal"
    val_fraction: float = 0.2
    val_holdout_snapshot: str | None = None
    neg_pos_ratio: float | None = None
    seed: int = 42

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _temporal_needs_holdout(self) -> "TrainingSpec":
        if self.val_strategy == "temporal" and not self.val_holdout_snapshot:
            raise ValueError("val_strategy='temporal' requires val_holdout_snapshot")
        return self


class SweepRef(BaseModel):
    backend: Literal["wandb", "local_grid", "none"] = "none"
    project: str | None = None
    config: Path | None = None

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _needs_config(self) -> "SweepRef":
        if self.backend == "local_grid" and self.config is None:
            raise ValueError("sweep backend='local_grid' requires 'config' path")
        if self.backend == "wandb" and self.project is None:
            raise ValueError("sweep backend='wandb' requires 'project'")
        return self


class ExperimentSpec(BaseModel):
    schema_version: str = EXPERIMENT_SCHEMA_VERSION
    name: str
    description: str | None = None
    dataset: DatasetRef
    model: ModelSpec = Field(default_factory=ModelSpec)
    training: TrainingSpec
    sweep: SweepRef = Field(default_factory=SweepRef)
    output_dir: Path | None = None
    tags: list[str] = Field(default_factory=list)
    keep_staging: bool = False

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentSpec":
        data = yaml.safe_load(Path(path).read_text()) or {}
        data.setdefault("schema_version", EXPERIMENT_SCHEMA_VERSION)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        payload = self.model_dump(mode="json", exclude_none=True)
        Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))

    def hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"output_dir", "description", "tags"})
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]
