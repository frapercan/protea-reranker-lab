"""Tests for the negative-sampling leakage audit (F-RERANK-UNIVERSAL.4).

Verifies:
- The audit report is correctly populated from observed counts.
- Lineage-excluded features are ALL present in the canonical list.
- no_label_encoding_by_lineage and no_row_replication are always True.
- Actual ratio is computed correctly.
- Audit integrates with _decide_split (the sampler is seeded and deterministic).
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.negative_sampling_audit import (
    LINEAGE_EXCLUDED_FEATURES,
    NegativeSamplingAuditReport,
    audit_negative_sampling,
)
from protea_reranker_lab.staging import _decide_split


# ---------------------------------------------------------------------------
# LINEAGE_EXCLUDED_FEATURES completeness
# ---------------------------------------------------------------------------


def test_lineage_excluded_features_contains_known_leakage_columns() -> None:
    """All known leakage columns from the F-RERANK-UNIVERSAL.2 ruling are excluded."""
    required = {
        "lineage_is_ancestor_of_known",
        "lineage_is_known",
        "lineage_is_descendant_of_known",
        "lineage_shared_ancestor_count",
        "lineage_min_dist_to_known",
    }
    excluded = set(LINEAGE_EXCLUDED_FEATURES)
    missing = required - excluded
    assert not missing, (
        f"Lineage leakage columns missing from exclusion list: {missing}"
    )


def test_lineage_excluded_features_all_start_with_lineage() -> None:
    """All excluded features should start with 'lineage_' (naming convention)."""
    for feat in LINEAGE_EXCLUDED_FEATURES:
        assert feat.startswith("lineage_"), (
            f"Non-lineage feature in exclusion list: {feat!r}"
        )


# ---------------------------------------------------------------------------
# audit_negative_sampling
# ---------------------------------------------------------------------------


def test_audit_fields_populated_correctly() -> None:
    """audit_negative_sampling fills all fields from its arguments."""
    report = audit_negative_sampling(
        n_positives=100,
        n_negatives_before=500,
        n_negatives_after=100,
        neg_pos_ratio=1.0,
        sampling_seed=42,
    )
    assert report.n_positives == 100
    assert report.n_negatives_before == 500
    assert report.n_negatives_after == 100
    assert report.neg_pos_ratio == 1.0
    assert report.sampling_seed == 42


def test_audit_actual_ratio_balanced() -> None:
    """Actual ratio is n_neg_after / n_pos for balanced sampling."""
    report = audit_negative_sampling(
        n_positives=200, n_negatives_before=1000,
        n_negatives_after=200, neg_pos_ratio=1.0, sampling_seed=1,
    )
    assert report.actual_ratio == pytest.approx(1.0)


def test_audit_actual_ratio_none_when_zero_positives() -> None:
    """actual_ratio is None when n_positives == 0 (avoid division by zero)."""
    report = audit_negative_sampling(
        n_positives=0, n_negatives_before=100,
        n_negatives_after=100, neg_pos_ratio=None, sampling_seed=0,
    )
    assert report.actual_ratio is None


def test_audit_no_label_encoding_always_true() -> None:
    """The no_label_encoding_by_lineage flag is always True."""
    report = audit_negative_sampling(
        n_positives=50, n_negatives_before=200,
        n_negatives_after=50, neg_pos_ratio=1.0, sampling_seed=0,
    )
    assert report.no_label_encoding_by_lineage is True


def test_audit_no_row_replication_always_true() -> None:
    """The no_row_replication flag is always True."""
    report = audit_negative_sampling(
        n_positives=50, n_negatives_before=200,
        n_negatives_after=50, neg_pos_ratio=1.0, sampling_seed=0,
    )
    assert report.no_row_replication is True


def test_audit_lineage_excluded_matches_canonical() -> None:
    """Report carries the canonical lineage exclusion list."""
    report = audit_negative_sampling(
        n_positives=10, n_negatives_before=50,
        n_negatives_after=10, neg_pos_ratio=1.0, sampling_seed=0,
    )
    assert report.lineage_excluded == LINEAGE_EXCLUDED_FEATURES


def test_audit_no_neg_pos_ratio_none() -> None:
    """neg_pos_ratio=None means no downsampling; report records None."""
    report = audit_negative_sampling(
        n_positives=100, n_negatives_before=500,
        n_negatives_after=500, neg_pos_ratio=None, sampling_seed=0,
    )
    assert report.neg_pos_ratio is None


# ---------------------------------------------------------------------------
# Integration: _decide_split with neg_pos_ratio=1.0 is seeded and balanced
# ---------------------------------------------------------------------------


def _build_pool(n_proteins: int = 40, k_per_protein: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    proteins = np.repeat(
        np.array([f"P{i:05d}" for i in range(n_proteins)], dtype=object),
        k_per_protein,
    )
    labels = (rng.random(len(proteins)) < 0.25).astype(np.int8)
    pairs = np.where(rng.random(len(proteins)) < 0.5, "v224-v226", "v226-v230").astype(object)
    return proteins, labels, pairs


def test_decide_split_neg_pos_ratio_balanced_seeded() -> None:
    """_decide_split with neg_pos_ratio=1.0 is deterministic and near-balanced."""
    proteins, labels, pairs = _build_pool()
    common = dict(
        proteins=proteins, labels=labels, pairs=pairs,
        val_strategy="protein_group", val_fraction=0.2,
        val_holdout_snapshot=None, neg_pos_ratio=1.0, seed=42,
    )
    keep_a, _, _ = _decide_split(**common)
    keep_b, _, _ = _decide_split(**common)
    np.testing.assert_array_equal(keep_a, keep_b)

    kept_labels = labels[keep_a]
    n_pos = int((kept_labels > 0).sum())
    n_neg = int((kept_labels == 0).sum())
    # With neg_pos_ratio=1.0, neg should equal pos (within small rounding).
    assert n_neg <= n_pos + 5, (
        f"Too many negatives after 1:1 sampling: pos={n_pos}, neg={n_neg}"
    )


def test_decide_split_full_negatives_when_ratio_none() -> None:
    """When neg_pos_ratio=None, all rows are kept (no downsampling)."""
    proteins, labels, pairs = _build_pool()
    keep, _, _ = _decide_split(
        proteins=proteins, labels=labels, pairs=pairs,
        val_strategy="none", val_fraction=0.0,
        val_holdout_snapshot=None, neg_pos_ratio=None, seed=42,
    )
    assert keep.all(), "neg_pos_ratio=None should keep all rows"
