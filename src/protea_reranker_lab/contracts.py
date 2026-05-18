"""Public contract surface consumed by PROTEA (producer side).

PROTEA imports only from this module. It is intentionally pydantic-only
(no LightGBM, no sklearn, no pandas runtime) so installing the lab as a
dev dependency of PROTEA stays cheap.

Context types (KnnContext, FeatureBuildContext) are sourced from
protea-contracts>=0.2.0 and re-exported here so PROTEA can import a
single consistent surface regardless of whether it depends directly on
protea-contracts.
"""

from __future__ import annotations

from protea_contracts.contexts import FeatureBuildContext, KnnContext

from .reranker import ALL_FEATURES, FEATURE_FAMILIES
from .schemas import (
    RESERVED_COLUMNS,
    SCHEMA_VERSION,
    DatasetSpec,
    ManifestV1,
    compute_feature_schema_sha,
    compute_schema_sha,
    required_columns,
)

__all__ = [
    "ALL_FEATURES",
    "DatasetSpec",
    "FeatureBuildContext",
    "FEATURE_FAMILIES",
    "KnnContext",
    "ManifestV1",
    "RESERVED_COLUMNS",
    "SCHEMA_VERSION",
    "compute_feature_schema_sha",
    "compute_schema_sha",
    "required_columns",
]
