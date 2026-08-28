"""Public contract surface consumed by PROTEA (producer side).

PROTEA imports only from this module. It is intentionally pydantic-only
(no LightGBM, no sklearn, no pandas runtime) so installing the lab as a
dev dependency of PROTEA stays cheap.

All feature-schema symbols and context types (KnnContext, FeatureBuildContext)
are re-exported from ``protea-contracts>=0.2.0`` so PROTEA can import a
single consistent surface.
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
from protea_contracts.contexts import FeatureBuildContext, KnnContext

from .schemas import DatasetSpec, ManifestV1

#: Families present in the contracts catalogue that the lab has NOT adopted
#: into its default training set.
#:
#: ``ALL_FEATURES`` is a catalogue of every column the schema defines, not a
#: recommendation of what to train on. Those two meanings were the same thing
#: while the lab tracked contracts v3, so the lab used ``ALL_FEATURES``
#: directly as its default feature set. They stopped being the same thing at
#: v6, which added 24 columns across six families. Training on the catalogue
#: would therefore have silently changed the default booster the moment the
#: pin moved, with no test failing, which is the exact failure mode the lab
#: guards against elsewhere.
#:
#: Each family is held out for a stated reason:
#:
#: ``lineage``
#:     NO-GO (DEFAULT EXCLUDE) under the ruling in
#:     ``docs/FEATURE_LEAKAGE_AUDIT_UNIVERSAL.md``. The columns are temporally
#:     honest, but the ruling holds them out pending a controlled ablation
#:     showing a positive Fmax delta. Contracts v6 put them back in the
#:     catalogue; that is a catalogue decision and does not overturn the
#:     lab's ruling.
#: ``classifier``, ``self_prior``, ``association``, ``protst_text``
#:     Declared but not produced (``FeatureStatus.DECLARED_ABSENT`` in
#:     ``FEATURE_DOCS``). Nothing writes them, so training on them would feed
#:     the booster nine all-NaN columns.
#: ``interpro``
#:     Produced, but never evaluated in this lab. Adopting a new signal is a
#:     research decision that belongs in its own slice with its own ablation,
#:     not a side effect of a dependency bump.
#:
#: To train WITH any of these, set ``enabled_feature_families`` explicitly on
#: the config. That path is unchanged and reaches every family in the
#: catalogue.
UNADOPTED_FEATURE_FAMILIES = (
    "lineage",
    "interpro",
    "classifier",
    "self_prior",
    "association",
    "protst_text",
)


def _default_training_features() -> list[str]:
    """The catalogue minus the families the lab has not adopted.

    Derived rather than listed, so a column added to an already-adopted
    family is picked up automatically, while a whole new family has to be
    adopted deliberately. Under contracts v6 this returns exactly the 54
    columns the v3 catalogue held, in the same order, so the schema sha is
    unchanged and every stored booster keeps its meaning.
    """
    held_out = {
        column
        for family in UNADOPTED_FEATURE_FAMILIES
        for column in FEATURE_FAMILIES.get(family, ())
    }
    return [column for column in ALL_FEATURES if column not in held_out]


#: The lab's default training feature set. Use this, not ``ALL_FEATURES``,
#: wherever the question is "what should the booster train on by default".
DEFAULT_TRAINING_FEATURES = _default_training_features()

__all__ = [
    "ALL_FEATURES",
    "DEFAULT_TRAINING_FEATURES",
    "UNADOPTED_FEATURE_FAMILIES",
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
