"""Unit tests for the SDR learned-attention-pool numeric helpers (torch-free surface).

The torch-dependent training (:func:`fit_attention_pool`) is exercised by the factorial
script under GPU; here we lock the pure-numpy pieces that the rest of the pipeline relies on.
"""

from __future__ import annotations

import numpy as np

from protea_reranker_lab.sdr_pool import PoolSpec, l2n, sample_pairs, topk_real


def test_l2n_unit_norm_rows() -> None:
    X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    Y = l2n(X)
    # row 0 -> norm 5 -> (0.6, 0.8); row 1 (zero) stays zero; row 2 already unit
    assert np.allclose(np.linalg.norm(Y[[0, 2]], axis=1), 1.0)
    assert np.allclose(Y[1], 0.0)


def test_topk_real_keeps_top_magnitude_values() -> None:
    X = np.array([[0.1, -0.9, 0.3, -0.05, 0.8]], dtype=np.float32)
    out = topk_real(X, 2)
    # top-2 by |.| are -0.9 (idx1) and 0.8 (idx4); their VALUES are kept, rest zero
    assert out[0, 1] == np.float32(-0.9)
    assert out[0, 4] == np.float32(0.8)
    assert (out == 0).sum() == 3
    assert (out != 0).sum() == 2


def test_topk_real_k_geq_dim_is_identity() -> None:
    X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert np.array_equal(topk_real(X, 5), X)


def test_sample_pairs_distinct_and_bounded() -> None:
    rng = np.random.default_rng(0)
    pairs = sample_pairs(10, 1000, rng)
    # only 45 distinct unordered pairs exist for n=10
    assert len(pairs) == 45
    assert len(set(pairs)) == 45
    for i, j in pairs:
        assert i < j


def test_pool_spec_defaults() -> None:
    s = PoolSpec()
    assert s.dict_dim == 2048
    assert s.top_k == 128
    assert s.attn_dim == 256
