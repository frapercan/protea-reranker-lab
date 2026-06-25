"""Unit tests for the chunk-attention encoder math (no DB, no heavy training loop)."""
from __future__ import annotations

import numpy as np

from protea_reranker_lab.chunk_attn_encoder import (
    _l2n,
    _pad_chunks,
    _sample_pairs,
    build_attn_encoder,
    topk_real,
)


def test_topk_real_keeps_largest_magnitude_with_sign():
    X = np.array([[0.1, -0.9, 0.2, 0.05]], dtype=np.float32)
    out = topk_real(X, 2)
    assert out[0, 1] == -0.9
    assert out[0, 2] == 0.2
    assert out[0, 0] == 0.0 and out[0, 3] == 0.0


def test_l2n_zero_row_is_safe():
    X = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = _l2n(X)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)


def test_pad_chunks_truncates_and_masks():
    d = 3
    per_chunk = [
        [np.ones(d, np.float32), 2 * np.ones(d, np.float32)],  # 2 chunks
        [np.ones(d, np.float32)],  # 1 chunk
    ]
    cap = 4
    Xpad, mask = _pad_chunks(per_chunk, cap, d)
    assert Xpad.shape == (2, cap, d)
    assert mask.shape == (2, cap)
    assert mask[0].tolist() == [True, True, False, False]
    assert mask[1].tolist() == [True, False, False, False]
    # padding region is zero
    assert np.allclose(Xpad[0, 2:], 0.0)


def test_pad_chunks_caps_long_proteins():
    d = 2
    per_chunk = [[np.full(d, i, np.float32) for i in range(10)]]  # 10 chunks
    cap = 4
    Xpad, mask = _pad_chunks(per_chunk, cap, d)
    assert Xpad.shape == (1, cap, d)
    assert mask[0].tolist() == [True, True, True, True]  # truncated to cap


def test_sample_pairs_unique_and_count():
    rng = np.random.default_rng(0)
    pairs = _sample_pairs(40, 80, rng)
    assert len(pairs) == 80
    assert len(set(pairs)) == 80
    assert all(i != j for i, j in pairs)


def test_attn_encoder_shapes_and_mask_invariance():
    import pytest

    torch = pytest.importorskip("torch")

    in_dim, dict_dim, att_dim, heads, cap = 6, 24, 8, 1, 5
    enc = build_attn_encoder(in_dim, dict_dim, att_dim, heads).eval()
    B = 3
    M = torch.randn(B, cap, in_dim)
    mask = torch.ones(B, cap, dtype=torch.bool)
    mask[0, 3:] = False  # protein 0 has only 3 real chunks
    with torch.no_grad():
        out = enc(M, mask)
    assert out.shape == (B, dict_dim)
    assert torch.isfinite(out).all()

    # masked (padding) chunk values must NOT affect the output: scramble the padded region
    M2 = M.clone()
    M2[0, 3:] = 999.0
    with torch.no_grad():
        out2 = enc(M2, mask)
    assert torch.allclose(out[0], out2[0], atol=1e-4)


def test_attn_encoder_multihead_shape():
    import pytest

    torch = pytest.importorskip("torch")

    in_dim, dict_dim, heads, cap = 6, 24, 4, 5
    enc = build_attn_encoder(in_dim, dict_dim, 8, heads).eval()
    M = torch.randn(2, cap, in_dim)
    mask = torch.ones(2, cap, dtype=torch.bool)
    with torch.no_grad():
        out = enc(M, mask)
    assert out.shape == (2, dict_dim)
