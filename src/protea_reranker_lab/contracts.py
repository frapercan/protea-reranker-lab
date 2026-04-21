"""Public contract surface consumed by PROTEA (producer side).

PROTEA imports only from this module. It is intentionally pydantic-only
(no LightGBM, no sklearn, no pandas runtime) so installing the lab as a
dev dependency of PROTEA stays cheap.
"""

from __future__ import annotations

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
    "FEATURE_FAMILIES",
    "ManifestV1",
    "RESERVED_COLUMNS",
    "SCHEMA_VERSION",
    "compute_feature_schema_sha",
    "compute_schema_sha",
    "required_columns",
]
