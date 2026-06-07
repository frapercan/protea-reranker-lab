"""Tests for seeded K-augmentation (F-RERANK-UNIVERSAL.4).

Covers:
- BYTE-IDENTICAL candidate selection across two runs with the same seed
  (explicit acceptance gate from the slice spec).
- Different seeds produce different selections.
- K-augmentation bounds validation.
- filter_rows_by_k_map preserves rows correctly.
- inference_k_sources policy behaviour.
- ExperimentSpec.hash() changes when K-aug params change.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.k_augmentation import (
    CANONICAL_K_VALUES,
    KAugPolicy,
    build_protein_k_map,
    filter_rows_by_k_map,
    inference_k_sources,
    sample_k_per_protein,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_proteins(n: int = 50, seed: int = 0) -> np.ndarray:
    return np.array([f"P{i:05d}" for i in range(n)], dtype=object)


# ---------------------------------------------------------------------------
# Byte-identical seed gate (acceptance criterion)
# ---------------------------------------------------------------------------


def test_byte_identical_k_draw_same_seed() -> None:
    """Two runs with the SAME seed produce BYTE-IDENTICAL K draws.

    This is the explicit acceptance criterion from F-RERANK-UNIVERSAL.4:
    'two runs with the same seed produce a BYTE-IDENTICAL candidate selection'.
    """
    proteins = _synthetic_proteins(60)
    policy = KAugPolicy(seed=42, k_min=3, k_max=10)

    draw_a = sample_k_per_protein(proteins, policy)
    draw_b = sample_k_per_protein(proteins, policy)

    np.testing.assert_array_equal(draw_a, draw_b, err_msg=(
        "Byte-identical seed gate FAILED: two runs with seed=42 produced "
        "different K draws. The RNG path is not deterministic."
    ))


def test_k_draw_changes_with_different_seed() -> None:
    """Different seeds must produce different K draws (sanity gate)."""
    proteins = _synthetic_proteins(60)
    policy_a = KAugPolicy(seed=1, k_min=3, k_max=10)
    policy_b = KAugPolicy(seed=2, k_min=3, k_max=10)

    draw_a = sample_k_per_protein(proteins, policy_a)
    draw_b = sample_k_per_protein(proteins, policy_b)

    assert not np.array_equal(draw_a, draw_b), (
        "Seeds 1 and 2 produced identical K draws; RNG seed is being ignored."
    )


def test_byte_identical_protein_k_map_same_seed() -> None:
    """build_protein_k_map is byte-identical across two calls with same seed."""
    proteins = _synthetic_proteins(30)
    # Extend to a row-aligned array (multiple rows per protein).
    row_proteins = np.repeat(proteins, 3)
    policy = KAugPolicy(seed=42, k_min=3, k_max=5)

    map_a = build_protein_k_map(row_proteins, policy)
    map_b = build_protein_k_map(row_proteins, policy)

    assert map_a == map_b, (
        "build_protein_k_map is not deterministic for the same seed."
    )


def test_byte_identical_filter_mask_same_seed() -> None:
    """filter_rows_by_k_map with the same k_map produces the same mask."""
    proteins = np.repeat(_synthetic_proteins(20), 3)
    k_contexts = np.tile(np.array([3, 5, 10], dtype=np.int32), 20)
    policy = KAugPolicy(seed=42, k_min=3, k_max=5)
    k_map = build_protein_k_map(proteins, policy)

    mask_a = filter_rows_by_k_map(proteins, k_contexts, k_map)
    mask_b = filter_rows_by_k_map(proteins, k_contexts, k_map)

    np.testing.assert_array_equal(mask_a, mask_b)


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_k_aug_policy_k_min_gt_k_max_raises() -> None:
    with pytest.raises(ValueError, match="k_min"):
        KAugPolicy(seed=42, k_min=10, k_max=3)


def test_k_aug_policy_invalid_k_min_raises() -> None:
    with pytest.raises(ValueError, match="members of"):
        KAugPolicy(seed=42, k_min=4, k_max=5)


def test_k_aug_policy_invalid_k_max_raises() -> None:
    with pytest.raises(ValueError, match="members of"):
        KAugPolicy(seed=42, k_min=3, k_max=7)


def test_k_aug_policy_invalid_inference_policy_raises() -> None:
    with pytest.raises(ValueError, match="inference_policy"):
        KAugPolicy(seed=42, k_min=3, k_max=10, inference_policy="random")


@pytest.mark.parametrize("k_min,k_max,expected_eligible", [
    (3, 3, [3]),
    (3, 5, [3, 5]),
    (3, 10, [3, 5, 10]),
    (5, 10, [5, 10]),
    (10, 10, [10]),
])
def test_eligible_k_values(k_min: int, k_max: int, expected_eligible: list[int]) -> None:
    policy = KAugPolicy(seed=42, k_min=k_min, k_max=k_max)
    assert policy.eligible_k_values == expected_eligible


# ---------------------------------------------------------------------------
# sample_k_per_protein
# ---------------------------------------------------------------------------


def test_k_draws_are_within_bounds() -> None:
    """All drawn K values must be in the eligible set."""
    proteins = _synthetic_proteins(100)
    policy = KAugPolicy(seed=42, k_min=3, k_max=10)
    draws = sample_k_per_protein(proteins, policy)
    eligible = set(policy.eligible_k_values)
    assert set(draws.tolist()).issubset(eligible), (
        f"Draws outside eligible set: {set(draws.tolist()) - eligible}"
    )


def test_k_draws_shape_matches_unique_proteins() -> None:
    """Output length equals number of unique proteins."""
    row_proteins = np.repeat(_synthetic_proteins(15), 4)
    policy = KAugPolicy(seed=1, k_min=3, k_max=5)
    draws = sample_k_per_protein(row_proteins, policy)
    assert len(draws) == 15


# ---------------------------------------------------------------------------
# filter_rows_by_k_map
# ---------------------------------------------------------------------------


def test_filter_keeps_all_when_k_map_is_max() -> None:
    """When k_map assigns K=10 to all proteins, no rows are dropped."""
    proteins = np.array(["P001", "P001", "P001", "P002", "P002", "P002"], dtype=object)
    k_contexts = np.array([3, 5, 10, 3, 5, 10], dtype=np.int32)
    k_map = {"P001": 10, "P002": 10}
    mask = filter_rows_by_k_map(proteins, k_contexts, k_map)
    assert mask.all()


def test_filter_drops_k10_when_k_map_ceiling_is_5() -> None:
    """When k_map ceiling is 5, k_context=10 rows are dropped."""
    proteins = np.array(["P001", "P001", "P001"], dtype=object)
    k_contexts = np.array([3, 5, 10], dtype=np.int32)
    k_map = {"P001": 5}
    mask = filter_rows_by_k_map(proteins, k_contexts, k_map)
    expected = np.array([True, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_filter_keeps_absent_protein_rows() -> None:
    """Rows for proteins absent from k_map are always kept (conservative fallback)."""
    proteins = np.array(["P999", "P999"], dtype=object)
    k_contexts = np.array([5, 10], dtype=np.int32)
    k_map: dict[str, int] = {}
    mask = filter_rows_by_k_map(proteins, k_contexts, k_map)
    assert mask.all()


# ---------------------------------------------------------------------------
# inference_k_sources
# ---------------------------------------------------------------------------


def test_inference_fixed_returns_all_k() -> None:
    policy = KAugPolicy(seed=42, k_min=3, k_max=10)
    result = inference_k_sources([3, 5, 10], policy)
    assert result == [3, 5, 10]


def test_inference_adaptive_returns_max_k() -> None:
    policy = KAugPolicy(seed=42, k_min=3, k_max=10, inference_policy="adaptive")
    result = inference_k_sources([3, 5, 10], policy)
    assert result == [10]


def test_inference_adaptive_single_k() -> None:
    policy = KAugPolicy(seed=42, k_min=5, k_max=5, inference_policy="adaptive")
    result = inference_k_sources([5], policy)
    assert result == [5]


def test_inference_empty_k_values() -> None:
    policy = KAugPolicy(seed=42, k_min=3, k_max=5)
    assert inference_k_sources([], policy) == []


# ---------------------------------------------------------------------------
# ExperimentSpec hash changes with K-aug params (reproducibility gate)
# ---------------------------------------------------------------------------


def _make_training_spec(**kwargs):
    """Build a TrainingSpec with val_strategy=none to avoid temporal holdout req."""
    from protea_reranker_lab.experiment import TrainingSpec
    defaults = {"cell": "nk-mfo", "val_strategy": "none", "val_fraction": 0.0}
    defaults.update(kwargs)
    return TrainingSpec(**defaults)


def test_spec_hash_changes_with_k_aug_seed() -> None:
    """ExperimentSpec.hash() changes when k_aug_seed changes."""
    from protea_reranker_lab.experiment import (
        DatasetRef,
        ExperimentSpec,
    )
    from pathlib import Path

    spec_base = ExperimentSpec(
        name="test",
        dataset=DatasetRef(manifest=Path("dummy/manifest.json")),
        training=_make_training_spec(k_aug_seed=42, k_aug_bounds=None),
    )
    spec_other = ExperimentSpec(
        name="test",
        dataset=DatasetRef(manifest=Path("dummy/manifest.json")),
        training=_make_training_spec(k_aug_seed=99, k_aug_bounds=None),
    )
    assert spec_base.hash() != spec_other.hash(), (
        "Spec hash did not change when k_aug_seed changed; reproducibility gate broken."
    )


def test_spec_hash_changes_with_k_aug_bounds() -> None:
    """ExperimentSpec.hash() changes when k_aug_bounds changes."""
    from protea_reranker_lab.experiment import (
        DatasetRef,
        ExperimentSpec,
    )
    from pathlib import Path

    spec_no_aug = ExperimentSpec(
        name="test",
        dataset=DatasetRef(manifest=Path("dummy/manifest.json")),
        training=_make_training_spec(k_aug_bounds=None),
    )
    spec_with_aug = ExperimentSpec(
        name="test",
        dataset=DatasetRef(manifest=Path("dummy/manifest.json")),
        training=_make_training_spec(k_aug_bounds=(3, 5)),
    )
    assert spec_no_aug.hash() != spec_with_aug.hash()


def test_spec_hash_stable_same_k_aug_params() -> None:
    """ExperimentSpec.hash() is stable (identical) for identical K-aug params."""
    from protea_reranker_lab.experiment import (
        DatasetRef,
        ExperimentSpec,
    )
    from pathlib import Path

    spec_a = ExperimentSpec(
        name="test",
        dataset=DatasetRef(manifest=Path("dummy/manifest.json")),
        training=_make_training_spec(k_aug_seed=7, k_aug_bounds=(3, 10), k_inference_policy="fixed"),
    )
    spec_b = ExperimentSpec(
        name="test",
        dataset=DatasetRef(manifest=Path("dummy/manifest.json")),
        training=_make_training_spec(k_aug_seed=7, k_aug_bounds=(3, 10), k_inference_policy="fixed"),
    )
    assert spec_a.hash() == spec_b.hash()
