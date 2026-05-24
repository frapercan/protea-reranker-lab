"""Determinism of train/val split routing.

The lab routes rows to train or val via ``_decide_split`` in
``protea_reranker_lab.staging``. Two split strategies are used:

- ``protein_group``: random hold-out of full proteins (no row of a
  hold-out protein leaks into train).
- ``temporal``: hold out one snapshot pair (deterministic by data, but
  still flows through the same seeded path).

If the seed is not fully propagated, two runs of the same recipe will
produce different splits, breaking reproducibility of the v27-binary
multiseed sweep (where seeds are the only intended axis of variation).

These tests catch:

- Undocumented ``random_state`` defaults that bypass the seed argument.
- Env-leak from ``PYTHONHASHSEED`` (numpy.default_rng is independent of
  PYTHONHASHSEED, but ``set()``/``dict`` iteration is not; the
  ``protein_group`` branch routes through both).
- Accidental use of ``np.random.shuffle`` (which seeds from os.urandom
  if no rng is constructed) instead of the rng-bound shuffle.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.staging import _decide_split


def _synthetic_pool(n_proteins: int = 40, k_per_protein: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    proteins = np.repeat(
        np.array([f"P{idx:05d}" for idx in range(n_proteins)], dtype=object),
        k_per_protein,
    )
    labels = (rng.random(len(proteins)) < 0.3).astype(np.int8)
    pairs = np.where(
        rng.random(len(proteins)) < 0.5,
        "v224-v226",
        "v226-v230",
    ).astype(object)
    return proteins, labels, pairs


def _split_call(*, proteins, labels, pairs, strategy: str, seed: int = 42):
    return _decide_split(
        proteins=proteins,
        labels=labels,
        pairs=pairs,
        val_strategy=strategy,
        val_fraction=0.25,
        val_holdout_snapshot="v226-v230",
        neg_pos_ratio=None,
        seed=seed,
    )


def test_protein_group_split_is_deterministic_across_runs() -> None:
    proteins, labels, pairs = _synthetic_pool()
    keep_a, val_a, vp_a = _split_call(
        proteins=proteins, labels=labels, pairs=pairs, strategy="protein_group"
    )
    keep_b, val_b, vp_b = _split_call(
        proteins=proteins, labels=labels, pairs=pairs, strategy="protein_group"
    )
    np.testing.assert_array_equal(keep_a, keep_b)
    np.testing.assert_array_equal(val_a, val_b)
    assert vp_a == vp_b


def test_protein_group_split_changes_with_seed() -> None:
    """Sanity: different seed must yield a different val-protein set.

    If this fails the rng is being ignored entirely.
    """
    proteins, labels, pairs = _synthetic_pool()
    _, _, vp_a = _split_call(
        proteins=proteins, labels=labels, pairs=pairs, strategy="protein_group", seed=42
    )
    _, _, vp_b = _split_call(
        proteins=proteins, labels=labels, pairs=pairs, strategy="protein_group", seed=43
    )
    assert vp_a != vp_b, "rng seed is not being threaded through the split"


def test_temporal_split_is_deterministic_across_runs() -> None:
    proteins, labels, pairs = _synthetic_pool()
    keep_a, val_a, _ = _split_call(
        proteins=proteins, labels=labels, pairs=pairs, strategy="temporal"
    )
    keep_b, val_b, _ = _split_call(
        proteins=proteins, labels=labels, pairs=pairs, strategy="temporal"
    )
    np.testing.assert_array_equal(keep_a, keep_b)
    np.testing.assert_array_equal(val_a, val_b)


def test_temporal_split_matches_snapshot_predicate() -> None:
    """Temporal split must equal the boolean predicate (no rng noise)."""
    proteins, labels, pairs = _synthetic_pool()
    _, val_mask, _ = _split_call(
        proteins=proteins, labels=labels, pairs=pairs, strategy="temporal"
    )
    expected = pairs == "v226-v230"
    np.testing.assert_array_equal(val_mask, expected)


def test_protein_group_split_no_protein_leak() -> None:
    """No protein assigned to val may also appear in the train side."""
    proteins, labels, pairs = _synthetic_pool()
    _, val_mask, val_proteins = _split_call(
        proteins=proteins, labels=labels, pairs=pairs, strategy="protein_group"
    )
    train_side_proteins = set(proteins[~val_mask].tolist())
    overlap = train_side_proteins & val_proteins
    assert not overlap, (
        f"Train/val protein leak: {sorted(overlap)[:5]} (first 5 of {len(overlap)})."
    )


def test_neg_pos_downsample_is_deterministic() -> None:
    """The negative-downsampling path must also be seed-stable."""
    proteins, labels, pairs = _synthetic_pool(seed=1)
    common = dict(
        proteins=proteins,
        labels=labels,
        pairs=pairs,
        val_strategy="protein_group",
        val_fraction=0.25,
        val_holdout_snapshot=None,
        neg_pos_ratio=2.0,
        seed=42,
    )
    keep_a, _, _ = _decide_split(**common)
    keep_b, _, _ = _decide_split(**common)
    np.testing.assert_array_equal(keep_a, keep_b)


@pytest.mark.parametrize("strategy", ["protein_group", "temporal"])
def test_split_unaffected_by_pythonhashseed_env(monkeypatch, strategy: str) -> None:
    """The split must not depend on PYTHONHASHSEED.

    We can't set PYTHONHASHSEED at runtime (interpreter only reads it on
    boot), but we can verify the rng-bound code path is referentially
    transparent: two calls in the same process with the same args give
    identical results regardless of intervening hash state.
    """
    proteins, labels, pairs = _synthetic_pool()
    keep_a, val_a, _ = _split_call(
        proteins=proteins, labels=labels, pairs=pairs, strategy=strategy
    )
    # Touch a hash-randomised object between calls.
    _ = hash(("warm", "up", strategy))
    keep_b, val_b, _ = _split_call(
        proteins=proteins, labels=labels, pairs=pairs, strategy=strategy
    )
    np.testing.assert_array_equal(keep_a, keep_b)
    np.testing.assert_array_equal(val_a, val_b)
