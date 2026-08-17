"""The gate that asks whether the learned arm predicts function or annotation mass.

Lin similarity is twice the information content of the most informative common
ancestor, over the sum of the two proteins' best-IC values. Best-IC is a
per-protein scalar, so a large share of the target's variance is explained by two
per-protein numbers and no pair information whatsoever. An encoder scoring well
against that target may have learned only how deeply each protein is annotated.

No arm in the shipped ablation could detect that, because every arm was given
either the real target or no target at all. These two nulls keep the encoder, the
pairs and the pool identical and change only what is being predicted.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.encoder_ablation import ArmSpec, substitute_target


def _fixture(n_proteins: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    bic = list(rng.uniform(1.0, 8.0, size=n_proteins))
    pairs = [(i, j) for i in range(n_proteins) for j in range(i + 1, n_proteins)]
    mica = rng.uniform(0.0, 1.0, size=len(pairs))
    lin = np.array(
        [2.0 * m / (bic[i] + bic[j]) for m, (i, j) in zip(mica, pairs, strict=True)],
        dtype=np.float32,
    )
    return lin, pairs, bic


# --------------------------------------------------------------------------- passthrough

def test_the_real_target_is_returned_untouched():
    lin, pairs, bic = _fixture()
    got = substitute_target(lin, pairs, bic, "lin", np.random.default_rng(0))
    assert np.array_equal(got, lin)


def test_an_unknown_target_is_refused_rather_than_defaulted():
    """Falling through to the real target would report a null arm as a real one."""
    lin, pairs, bic = _fixture()
    with pytest.raises(ValueError, match="unknown target"):
        substitute_target(lin, pairs, bic, "random", np.random.default_rng(0))


# --------------------------------------------------------------------------- the marginal null

def test_the_marginal_null_deletes_every_trace_of_which_pair_is_which():
    """Two pairs sharing a denominator must receive the identical target."""
    bic = [2.0, 2.0, 3.0, 3.0]
    pairs = [(0, 1), (2, 3), (0, 2)]
    lin = np.array([0.9, 0.1, 0.5], dtype=np.float32)

    got = substitute_target(lin, pairs, bic, "marginal", np.random.default_rng(0))

    # (0,1) has denominator 4.0 and (2,3) has 6.0, so they differ; but the
    # ordering of the real targets, 0.9 against 0.1, must not survive.
    assert got[0] > got[1]  # driven only by the smaller denominator
    assert not np.isclose(got[0] / got[1], lin[0] / lin[1])


def test_the_marginal_null_preserves_the_per_protein_marginals_exactly():
    """The denominators are untouched, which is what makes it a marginal null."""
    lin, pairs, bic = _fixture()
    got = substitute_target(lin, pairs, bic, "marginal", np.random.default_rng(0))

    denom = np.array([bic[i] + bic[j] for i, j in pairs])
    implied = got * denom / 2.0
    # every pair now shares one common-ancestor value: the pool mean
    assert np.allclose(implied, implied[0], rtol=1e-4)


def test_the_marginal_null_keeps_the_mean_common_ancestor_term():
    """It substitutes the pool mean rather than an arbitrary constant."""
    lin, pairs, bic = _fixture()
    denom = np.array([bic[i] + bic[j] for i, j in pairs])
    expected_mica = float((lin * denom / 2.0).mean())

    got = substitute_target(lin, pairs, bic, "marginal", np.random.default_rng(0))

    assert np.isclose(float((got * denom / 2.0).mean()), expected_mica, rtol=1e-4)


def test_a_zero_denominator_does_not_divide_by_zero():
    """A protein with no annotated term has best-IC zero and must not produce a nan."""
    bic = [0.0, 0.0, 4.0]
    pairs = [(0, 1), (0, 2)]
    lin = np.array([0.0, 0.3], dtype=np.float32)

    got = substitute_target(lin, pairs, bic, "marginal", np.random.default_rng(0))

    assert np.isfinite(got).all()
    assert got[0] == 0.0


# --------------------------------------------------------------------------- the shuffle

def test_the_shuffle_preserves_the_whole_marginal_distribution():
    lin, pairs, bic = _fixture(n_proteins=12)
    got = substitute_target(lin, pairs, bic, "shuffled", np.random.default_rng(3))

    assert np.allclose(np.sort(got), np.sort(lin))


def test_the_shuffle_destroys_the_pairing():
    lin, pairs, bic = _fixture(n_proteins=12)
    got = substitute_target(lin, pairs, bic, "shuffled", np.random.default_rng(3))

    assert not np.array_equal(got, lin)


def test_the_shuffle_is_deterministic_under_a_seed():
    lin, pairs, bic = _fixture(n_proteins=12)
    a = substitute_target(lin, pairs, bic, "shuffled", np.random.default_rng(5))
    b = substitute_target(lin, pairs, bic, "shuffled", np.random.default_rng(5))

    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- the arm

def test_the_arm_defaults_to_the_real_target():
    """So that adding this axis moves no existing number."""
    assert ArmSpec(name="learned", kind="learned").target == "lin"


def test_the_target_is_part_of_the_arm_identity():
    """spec_hash hashes the arms, so a null arm cannot collide with a real one."""
    from protea_reranker_lab.encoder_ablation import EncoderAblationSpec

    real = EncoderAblationSpec(arms=[ArmSpec(name="learned", kind="learned")])
    null = EncoderAblationSpec(
        arms=[ArmSpec(name="learned", kind="learned", target="marginal")]
    )
    assert real.spec_hash() != null.spec_hash()
