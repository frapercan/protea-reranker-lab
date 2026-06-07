"""Regression guard: feature count and pool-context features for protea-contracts v3.

History:
- protea-contracts 4e898af / v0.3.0: 56 features (52 base + 4 lineage_*).
  These were used by the v27-binary champion (cafaeval Fmax 0.7291).
- protea-contracts v1.0.0 (F-RERANK-UNIVERSAL.2): 54 features.
  The 4 lineage_* columns are DEFAULT EXCLUDED from ALL_FEATURES per the
  leakage ruling in docs/FEATURE_LEAKAGE_AUDIT_UNIVERSAL.md.
  Two new pool-context features are added:
    - k_context (numeric, family k_neighborhood): KNN neighbourhood size.
    - plm_id (categorical, family plm_context): PLM embedding identifier.

This is an accepted one-way door: phase3a model schema_sha is invalidated
by the v3 ALL_FEATURES change.  Do not revert to the 56-feature count.
"""

from __future__ import annotations

from protea_contracts import ALL_FEATURES, FEATURE_FAMILIES


# v3 canonical count: 52 (v2 base without lineage_*) + k_context + plm_id = 54
EXPECTED_TOTAL_FEATURES = 54
POOL_CONTEXT_FEATURES = ["k_context", "plm_id"]
LINEAGE_FEATURES = [
    "lineage_is_ancestor_of_known",
    "lineage_is_descendant_of_known",
    "lineage_ancestor_of_count",
    "lineage_descendant_of_count",
]


def test_all_features_count() -> None:
    """ALL_FEATURES must have exactly 54 entries (protea-contracts v3 / v1.0.0).

    54 = 52 (v2 base without lineage_*) + k_context + plm_id.

    If this fails:
    - A contracts downgrade restored lineage_* (56 features) or removed the
      pool-context columns (52 features).
    - Restore with: protea-contracts @ path/to/rerank2-contracts (v1.0.0).
    """
    assert len(ALL_FEATURES) == EXPECTED_TOTAL_FEATURES, (
        f"Feature count regressed: expected {EXPECTED_TOTAL_FEATURES}, "
        f"got {len(ALL_FEATURES)}.  "
        f"Expected protea-contracts v1.0.0 (v3 schema).  "
        f"Current ALL_FEATURES: {ALL_FEATURES!r}"
    )


def test_pool_context_features_in_all_features() -> None:
    """k_context and plm_id must appear in ALL_FEATURES (v3+)."""
    missing = [f for f in POOL_CONTEXT_FEATURES if f not in ALL_FEATURES]
    assert not missing, (
        f"Pool-context features missing from ALL_FEATURES: {missing}.  "
        f"Requires protea-contracts v1.0.0."
    )


def test_plm_context_family_registered() -> None:
    """'plm_context' must be a registered feature family (v3+)."""
    assert "plm_context" in FEATURE_FAMILIES, (
        "'plm_context' family not found in FEATURE_FAMILIES.  "
        "Requires protea-contracts v1.0.0."
    )
    assert FEATURE_FAMILIES["plm_context"] == ["plm_id"]


def test_k_neighborhood_family_registered() -> None:
    """'k_neighborhood' must be a registered feature family (v3+)."""
    assert "k_neighborhood" in FEATURE_FAMILIES, (
        "'k_neighborhood' family not found in FEATURE_FAMILIES.  "
        "Requires protea-contracts v1.0.0."
    )
    assert FEATURE_FAMILIES["k_neighborhood"] == ["k_context"]


def test_lineage_features_excluded_from_all_features() -> None:
    """lineage_* must NOT appear in ALL_FEATURES (DEFAULT EXCLUDE ruling).

    The leakage ruling in docs/FEATURE_LEAKAGE_AUDIT_UNIVERSAL.md mandates
    DEFAULT EXCLUDE for the 4 lineage columns pending a dedicated ablation.
    If this test fails, a contracts change re-added them.
    """
    present = [f for f in LINEAGE_FEATURES if f in ALL_FEATURES]
    assert not present, (
        f"Lineage features re-appeared in ALL_FEATURES: {present}.  "
        f"The leakage ruling requires DEFAULT EXCLUDE until a controlled "
        f"ablation clears them.  See docs/FEATURE_LEAKAGE_AUDIT_UNIVERSAL.md."
    )
