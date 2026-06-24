"""Unit tests for the encoder-ablation study math + data layer (no DB, no cafaeval)."""
from __future__ import annotations

import numpy as np

from protea_reranker_lab.encoder_ablation import (
    EncoderAblationSpec,
    knn_transfer,
    l2n,
    load_gt,
    sample_pairs,
    topk_real,
)


def test_l2n_unit_norm():
    X = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2n(X)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)  # zero row stays zero (no div-by-zero)


def test_topk_real_keeps_largest_magnitude_with_sign():
    X = np.array([[0.1, -0.9, 0.2, 0.05]], dtype=np.float32)
    out = topk_real(X, 2)
    # top-2 by |.| are -0.9 and 0.2, with sign preserved; others zeroed
    assert out[0, 1] == -0.9
    assert out[0, 2] == 0.2
    assert out[0, 0] == 0.0 and out[0, 3] == 0.0


def test_topk_real_full_k_is_identity():
    X = np.random.RandomState(0).randn(3, 5).astype(np.float32)
    assert np.allclose(topk_real(X, 5), X)


def test_sample_pairs_unique_and_count():
    rng = np.random.default_rng(0)
    pairs = sample_pairs(50, 100, rng)
    assert len(pairs) == 100
    assert len(set(pairs)) == 100
    assert all(i != j for i, j in pairs)


def test_knn_transfer_identical_query_recovers_neighbor_terms():
    # 2 reference proteins with distinct embeddings + GO closures; a query == ref 0
    R = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    Q = np.array([[1.0, 0.0]], dtype=np.float32)  # identical to ref 0
    ref_closures = [{"GO:1"}, {"GO:2"}]
    tix = {"GO:1": 0, "GO:2": 1}
    scores = knn_transfer(Q, R, ref_closures, tix, knn=1).toarray()
    # query's single nearest neighbor is ref 0 -> GO:1 scored, GO:2 not
    assert scores[0, 0] > 0
    assert scores[0, 1] == 0


def test_load_gt_aspect_mapping(tmp_path):
    for cat, rows in {
        "NK": [("P1", "GO:1", "F")],
        "LK": [("P2", "GO:2", "P")],
        "PK": [("P3", "GO:3", "C")],
    }.items():
        lines = ["EntryID\tterm\taspect"] + [f"{a}\t{g}\t{s}" for a, g, s in rows]
        (tmp_path / f"groundtruth_{cat}.tsv").write_text("\n".join(lines) + "\n")
    cells = load_gt(tmp_path)
    assert ("nk", "mfo") in cells  # F -> mfo
    assert ("lk", "bpo") in cells  # P -> bpo
    assert ("pk", "cco") in cells  # C -> cco
    assert cells[("nk", "mfo")]["pairs"] == {("P1", "GO:1")}


def test_spec_hash_deterministic_and_arm_sensitive():
    s1 = EncoderAblationSpec()
    s2 = EncoderAblationSpec()
    assert s1.spec_hash() == s2.spec_hash()
    s3 = EncoderAblationSpec(ref_n=12345)
    assert s3.spec_hash() != s1.spec_hash()
