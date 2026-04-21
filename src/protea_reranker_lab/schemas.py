"""Data contracts for the lab: dataset spec, build manifest, schema hashing.

The row-level schema (52 features + reserved cols) lives in ``reranker.py``
and is enforced via the parquet column set plus ``schema_sha`` in the manifest.
Per-row pydantic validation is deliberately skipped — millions of rows make
it prohibitive; we validate the *shape*, not each sample.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .reranker import ALL_FEATURES, FEATURE_FAMILIES


SCHEMA_VERSION = "v1"

RESERVED_COLUMNS: tuple[str, ...] = (
    "protein_accession", "go_term_id", "label",
    "category", "aspect", "snapshot_pair",
)


def required_columns(families: list[str] | None = None,
                     drop: list[str] | None = None) -> list[str]:
    drop_set = set(drop or [])
    if families is None:
        feats = list(ALL_FEATURES)
    else:
        feats = []
        for fam in families:
            feats.extend(FEATURE_FAMILIES[fam])
    seen: set[str] = set()
    out: list[str] = []
    for col in (*RESERVED_COLUMNS, *feats):
        if col in drop_set or col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def compute_schema_sha(columns: list[str]) -> str:
    blob = "|".join(sorted(columns)).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


class DatasetSpec(BaseModel):
    """What the user requests from the builder. Hashable → reproducible."""

    name: str
    source_manifest: Path
    enabled_feature_families: list[str] | None = None
    drop_features: list[str] = Field(default_factory=list)
    train_snapshot_pairs: list[str] | None = None
    eval_snapshot_pair: str | None = None
    format: Literal["parquet"] = "parquet"
    seed: int = 42

    model_config = {"frozen": True}

    @field_validator("enabled_feature_families")
    @classmethod
    def _check_families(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        unknown = set(v) - set(FEATURE_FAMILIES)
        if unknown:
            raise ValueError(f"unknown feature families: {sorted(unknown)}")
        return v

    def hash(self) -> str:
        payload = self.model_dump(exclude={"source_manifest", "name"}, mode="json")
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


class ManifestV1(BaseModel):
    """Written to ``<dataset_dir>/manifest.json`` after each build."""

    schema_version: str = SCHEMA_VERSION
    name: str
    k: int
    embedding_config_id: str
    ontology_snapshot_id: str
    annotation_source: str | None = None
    train_snapshot_pairs: list[str]
    eval_snapshot_pair: str
    schema_sha: str
    n_train_rows: int | None = None
    n_eval_rows: int | None = None
    format: Literal["parquet"] = "parquet"
    spec_hash: str | None = None
    parent_schema_sha: str | None = None
    feature_families: list[str] | None = None

    model_config = {"extra": "ignore"}

    @classmethod
    def load(cls, path: str | Path) -> "ManifestV1":
        data: dict[str, Any] = json.loads(Path(path).read_text())
        data.setdefault("schema_version", SCHEMA_VERSION)
        return cls.model_validate(data)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))
