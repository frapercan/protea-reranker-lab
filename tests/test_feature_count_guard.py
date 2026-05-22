"""Regression guard: feature count must stay at 56 (52 base + 4 lineage).

The v27-binary champion (cafaeval Fmax 0.7291) was trained with 56 features
including the four lineage features from protea_contracts.  Any accidental
rollback of protea-contracts or removal of the lineage family would silently
drop the LK/PK cells to incomparable Fmax values, invalidating the thesis
chapter-6 comparison table.

This test pins the exact count and the four lineage column names so any
regression is caught at CI time before a sweep is launched.
"""

from __future__ import annotations

from protea_contracts import ALL_FEATURES, FEATURE_FAMILIES


EXPECTED_TOTAL_FEATURES = 56
EXPECTED_LINEAGE_FEATURES = [
    "lineage_is_ancestor_of_known",
    "lineage_is_descendant_of_known",
    "lineage_ancestor_of_count",
    "lineage_descendant_of_count",
]


def test_all_features_count() -> None:
    """ALL_FEATURES must have exactly 56 entries (52 base + 4 lineage).

    If this fails it means protea-contracts was rolled back below v0.3.0 or
    the lineage family was removed.  Restore via:
      pyproject.toml: protea-contracts @ git+...@v0.3.0
      then: poetry lock && poetry install
    """
    assert len(ALL_FEATURES) == EXPECTED_TOTAL_FEATURES, (
        f"Feature count regressed: expected {EXPECTED_TOTAL_FEATURES}, "
        f"got {len(ALL_FEATURES)}.  The 4 lineage features (lineage_*) were "
        f"likely dropped by a protea-contracts downgrade.  Current ALL_FEATURES "
        f"has {len(ALL_FEATURES)} entries."
    )


def test_lineage_features_present_in_all_features() -> None:
    """Each lineage column must appear in ALL_FEATURES."""
    missing = [f for f in EXPECTED_LINEAGE_FEATURES if f not in ALL_FEATURES]
    assert not missing, (
        f"Lineage features missing from ALL_FEATURES: {missing}.  "
        f"Bump protea-contracts to >=v0.3.0."
    )


def test_lineage_family_registered() -> None:
    """'lineage' must be a registered feature family."""
    assert "lineage" in FEATURE_FAMILIES, (
        "'lineage' family not found in FEATURE_FAMILIES.  "
        "Requires protea-contracts>=v0.3.0."
    )


def test_lineage_family_members() -> None:
    """The lineage family must declare exactly the 4 canonical columns."""
    actual = FEATURE_FAMILIES.get("lineage", [])
    assert sorted(actual) == sorted(EXPECTED_LINEAGE_FEATURES), (
        f"lineage family mismatch: expected {sorted(EXPECTED_LINEAGE_FEATURES)}, "
        f"got {sorted(actual)}"
    )


def test_lineage_features_are_last_numeric() -> None:
    """Lineage columns must appear as a contiguous block at the tail of ALL_FEATURES.

    They must not be interleaved or moved, as LightGBM boosters are sensitive
    to column ordering and any position change invalidates saved models.
    """
    indices = [ALL_FEATURES.index(f) for f in EXPECTED_LINEAGE_FEATURES]
    # All four must be present (index() raises ValueError otherwise)
    assert len(indices) == 4
    # They must be contiguous
    assert indices == list(range(min(indices), min(indices) + 4)), (
        f"Lineage features are not contiguous in ALL_FEATURES: positions {indices}"
    )
