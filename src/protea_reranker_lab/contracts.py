"""Public contract surface consumed by PROTEA (producer side).

PROTEA imports only from this module. It is intentionally pydantic-only
(no LightGBM, no sklearn, no pandas runtime) so installing the lab as a
dev dependency of PROTEA stays cheap.

All feature-schema symbols are re-exported from ``protea-contracts`` (the
canonical source of truth).
"""

from __future__ import annotations

from protea_contracts import (
    ALL_FEATURES,
    FEATURE_FAMILIES,
    RESERVED_COLUMNS,
    SCHEMA_VERSION,
    compute_feature_schema_sha,
    compute_schema_sha,
    required_columns,
)

from .schemas import DatasetSpec, ManifestV1

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
