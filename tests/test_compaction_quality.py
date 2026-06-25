"""Unit tests for the compaction-quality primitives (gold kernels + compactions).

These pin the contract the study relies on: the gold max-sim is symmetric and bounded,
each compaction produces a code of the advertised byte budget, and the similarity fns
return sane values. Pure-numpy, no GPU, no DB.
"""

from __future__ import annotations

import numpy as np

from protea_reranker_lab.compaction_quality import (
    BUCKET_NAMES,
    cosine_pairwise_fn,
    late_interaction_maxsim,
    length_bucket,
    max_pool,
    mean_pool,
    multivector_codes,
    multivector_pairwise_fn,
    pca_fit_transform,
    sdr_union_codes,
    sinkhorn_ot_sim,
    tanimoto_pairwise_fn,
)


def _toy_residues(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    # three proteins of different lengths in 32-d residue space
    return [
        rng.standard_normal((40, 32)).astype(np.float32),
        rng.standard_normal((600, 32)).astype(np.float32),
        rng.standard_normal((2100, 32)).astype(np.float32),
    ]


def test_length_bucket_edges() -> None:
    assert length_bucket(1) == "short"
    assert length_bucket(318) == "short"
    assert length_bucket(319) == "medium"
    assert length_bucket(969) == "medium"
    assert length_bucket(970) == "long"
    assert length_bucket(1959) == "long"
    assert length_bucket(1960) == "very_long"
    assert set(BUCKET_NAMES) == {"short", "medium", "long", "very_long"}


def test_late_interaction_symmetric_and_self_is_one() -> None:
    rng = np.random.default_rng(1)
    p = rng.standard_normal((30, 16)).astype(np.float32)
    q = rng.standard_normal((50, 16)).astype(np.float32)
    s_pq = late_interaction_maxsim(p, q)
    s_qp = late_interaction_maxsim(q, p)
    assert abs(s_pq - s_qp) < 1e-5  # symmetric
    assert abs(late_interaction_maxsim(p, p) - 1.0) < 1e-4  # self-sim = 1
    assert -1.0 <= s_pq <= 1.0


def test_sinkhorn_ot_self_is_max() -> None:
    rng = np.random.default_rng(2)
    p = rng.standard_normal((25, 16)).astype(np.float32)
    q = rng.standard_normal((25, 16)).astype(np.float32)
    # self-OT cost is near zero (identical sets), so sim ~ 0 and >= cross sim
    assert sinkhorn_ot_sim(p, p) >= sinkhorn_ot_sim(p, q)


def test_pool_shapes_and_budgets() -> None:
    res = _toy_residues()
    d = res[0].shape[1]
    assert mean_pool(res).shape == (3, d)
    assert max_pool(res).shape == (3, d)


def test_pca_reduces_dim() -> None:
    # need at least r+1 samples for r real components (the study uses ~6000)
    rng = np.random.default_rng(3)
    mm = rng.standard_normal((50, 32)).astype(np.float32)
    p = pca_fit_transform(mm, 8)
    assert p.shape == (50, 8)


def test_sdr_union_budget_and_popcount() -> None:
    res = _toy_residues()
    sdr, nbytes = sdr_union_codes(res, per_residue_k=4, final_k=10)
    assert nbytes == 2 * 10  # k uint16 indices
    # every protein code has exactly final_k active bits (or all dims if k>=d)
    assert (sdr.sum(axis=1) == 10).all()
    sim = tanimoto_pairwise_fn(sdr)
    assert abs(sim(0, 0) - 1.0) < 1e-6  # self-Tanimoto = 1
    assert 0.0 <= sim(0, 1) <= 1.0


def test_multivector_budget_and_sim() -> None:
    res = _toy_residues()
    codes, nbytes = multivector_codes(res, m=4, seed=0)
    d = res[0].shape[1]
    assert nbytes == 2 * 4 * d  # M codes of d float16
    # short protein (40 res) keeps <= M codes; long ones get exactly M centroids
    assert codes[1].shape[0] == 4
    sim = multivector_pairwise_fn(codes)
    assert -1.0 <= sim(0, 1) <= 1.0
    assert abs(sim(1, 1) - 1.0) < 1e-3


def test_cosine_pairwise_self_one() -> None:
    res = _toy_residues()
    mm = mean_pool(res)
    sim = cosine_pairwise_fn(mm)
    assert abs(sim(0, 0) - 1.0) < 1e-5
