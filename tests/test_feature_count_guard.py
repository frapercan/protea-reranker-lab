"""Regression guard: the lab's default training feature set.

History:
- protea-contracts 4e898af / v0.3.0: 56 features (52 base + 4 lineage_*).
  These were used by the v27-binary champion (cafaeval Fmax 0.7291).
- protea-contracts v1.0.0 (F-RERANK-UNIVERSAL.2): 54 features.
  The 4 lineage_* columns are DEFAULT EXCLUDED per the leakage ruling in
  docs/FEATURE_LEAKAGE_AUDIT_UNIVERSAL.md. Two pool-context features are
  added: k_context (family k_neighborhood) and plm_id (family plm_context).
- protea-contracts v1.7.0 (v6 schema): the catalogue grew to 78 columns
  across six new families, and put lineage_* back into ALL_FEATURES.

That last step is why these tests no longer assert on ALL_FEATURES.
ALL_FEATURES is a catalogue of every column the schema defines, including
nine that nothing produces yet. It is not a recommendation of what to train
on. The lab's default training set is
``protea_reranker_lab.contracts.DEFAULT_TRAINING_FEATURES``, which holds out
the families listed in UNADOPTED_FEATURE_FAMILIES, and that is what these
tests guard.

The 54-column count is an accepted one-way door: phase3a model schema_sha is
invalidated by the v3 change. Do not revert to the 56-feature count.
"""

from __future__ import annotations

from protea_contracts import ALL_FEATURES, FEATURE_FAMILIES, compute_schema_sha

from protea_reranker_lab.contracts import (
    DEFAULT_TRAINING_FEATURES,
    UNADOPTED_FEATURE_FAMILIES,
)

# v3 canonical count: 52 (v2 base without lineage_*) + k_context + plm_id = 54
EXPECTED_TOTAL_FEATURES = 54

# The sha of the 54-column set as it stood under contracts v3. Pinned so a
# family the lab adopts, or a column added to one it already trains on, has
# to be an explicit decision rather than a side effect of a contracts bump.
EXPECTED_SCHEMA_SHA = "a0986dedd912"

POOL_CONTEXT_FEATURES = ["k_context", "plm_id"]
LINEAGE_FEATURES = [
    "lineage_is_ancestor_of_known",
    "lineage_is_descendant_of_known",
    "lineage_ancestor_of_count",
    "lineage_descendant_of_count",
]


def test_default_training_feature_count() -> None:
    """The lab default must hold exactly 54 columns.

    If this fails, a contracts bump added a family the lab has not adopted
    (add it to UNADOPTED_FEATURE_FAMILIES, or adopt it deliberately), or a
    column was added to a family the lab already trains on.
    """
    assert len(DEFAULT_TRAINING_FEATURES) == EXPECTED_TOTAL_FEATURES, (
        f"Default training feature count moved: expected "
        f"{EXPECTED_TOTAL_FEATURES}, got {len(DEFAULT_TRAINING_FEATURES)}.  "
        f"Current set: {DEFAULT_TRAINING_FEATURES!r}"
    )


def test_default_training_schema_sha_unchanged() -> None:
    """The default set must still hash to the v3 sha.

    This is the assertion that makes a contracts bump safe: as long as it
    holds, every stored booster and every cached artefact keeps its meaning.
    """
    assert compute_schema_sha(list(DEFAULT_TRAINING_FEATURES)) == EXPECTED_SCHEMA_SHA


def test_default_training_features_are_catalogue_subset() -> None:
    """Every default column must still exist in the contracts catalogue."""
    missing = [f for f in DEFAULT_TRAINING_FEATURES if f not in ALL_FEATURES]
    assert not missing, f"Columns dropped from the contracts catalogue: {missing}"


def test_unadopted_families_are_real_families() -> None:
    """A held-out family must exist, or the hold-out silently does nothing."""
    unknown = [f for f in UNADOPTED_FEATURE_FAMILIES if f not in FEATURE_FAMILIES]
    assert not unknown, (
        f"UNADOPTED_FEATURE_FAMILIES names families that no longer exist: "
        f"{unknown}.  A renamed family would let its columns back into the "
        f"default training set unnoticed."
    )


def test_pool_context_features_in_default_training_set() -> None:
    """k_context and plm_id must be trained on (v3+)."""
    missing = [f for f in POOL_CONTEXT_FEATURES if f not in DEFAULT_TRAINING_FEATURES]
    assert not missing, f"Pool-context features missing: {missing}."


def test_plm_context_family_registered() -> None:
    """'plm_context' must be a registered feature family (v3+)."""
    assert "plm_context" in FEATURE_FAMILIES
    assert FEATURE_FAMILIES["plm_context"] == ["plm_id"]


def test_k_neighborhood_family_registered() -> None:
    """'k_neighborhood' must be a registered feature family (v3+)."""
    assert "k_neighborhood" in FEATURE_FAMILIES
    assert FEATURE_FAMILIES["k_neighborhood"] == ["k_context"]


def test_lineage_features_excluded_from_default_training_set() -> None:
    """lineage_* must NOT be trained on by default (DEFAULT EXCLUDE ruling).

    The leakage ruling in docs/FEATURE_LEAKAGE_AUDIT_UNIVERSAL.md mandates
    DEFAULT EXCLUDE for the 4 lineage columns pending a dedicated ablation.
    Contracts v6 returned them to ALL_FEATURES, which is a catalogue
    decision; the ruling is a lab decision and is enforced here.
    """
    present = [f for f in LINEAGE_FEATURES if f in DEFAULT_TRAINING_FEATURES]
    assert not present, (
        f"Lineage features re-entered the default training set: {present}.  "
        f"The leakage ruling requires DEFAULT EXCLUDE until a controlled "
        f"ablation clears them.  See docs/FEATURE_LEAKAGE_AUDIT_UNIVERSAL.md."
    )


def test_lineage_is_still_reachable_when_asked_for() -> None:
    """Holding lineage out of the default must not make it unreachable.

    The ruling is "excluded by default pending an ablation", so the ablation
    has to be able to switch it on.
    """
    assert FEATURE_FAMILIES["lineage"] == LINEAGE_FEATURES
