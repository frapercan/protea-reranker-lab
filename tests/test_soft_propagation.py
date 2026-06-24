"""Tests for soft two-way Pmin/Pmax score propagation (R2.1).

Small-DAG fixture:

    GO:ROOT
      -> GO:MID
           -> GO:LEAF
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.hierarchical_correction import (
    build_children_map,
    check_hierarchical_consistency,
)
from protea_reranker_lab.soft_propagation import soft_propagate_scores

PARENT_MAP = {
    "GO:MID": frozenset({"GO:ROOT"}),
    "GO:LEAF": frozenset({"GO:MID"}),
}
CHILDREN_MAP = build_children_map(PARENT_MAP)


def _consistent(proteins, go_terms, scores) -> bool:
    res = check_hierarchical_consistency(
        proteins=proteins, go_terms=go_terms, scores=scores,
        children_map=CHILDREN_MAP,
    )
    return res["n_violations"] == 0


class TestSoftPropagateScores:
    def test_yields_parent_ge_child(self) -> None:
        # Inconsistent: leaf 0.9 > mid 0.3 > root 0.2.
        proteins = np.array(["P1", "P1", "P1"])
        go_terms = np.array(["GO:LEAF", "GO:MID", "GO:ROOT"])
        scores = np.array([0.9, 0.3, 0.2], dtype=np.float32)
        out = soft_propagate_scores(proteins, go_terms, scores, PARENT_MAP)
        assert _consistent(proteins, go_terms, out)

    def test_idempotent_on_consistent_input(self) -> None:
        # Already consistent: root 0.9 >= mid 0.5 >= leaf 0.2.
        proteins = np.array(["P1", "P1", "P1"])
        go_terms = np.array(["GO:ROOT", "GO:MID", "GO:LEAF"])
        scores = np.array([0.9, 0.5, 0.2], dtype=np.float32)
        once = soft_propagate_scores(proteins, go_terms, scores, PARENT_MAP)
        # Consistent input unchanged...
        np.testing.assert_allclose(once, scores, atol=1e-6)
        # ...and applying again is a no-op (idempotent).
        twice = soft_propagate_scores(proteins, go_terms, once, PARENT_MAP)
        np.testing.assert_allclose(twice, once, atol=1e-6)

    def test_blend_pmax_pmin_hand_worked(self) -> None:
        # root 0.6, mid 0.2, leaf 0.8. Two-protein-independent check of blend.
        # Pmax (bottom-up): leaf 0.8 -> mid >= 0.8 -> root >= 0.8.
        #   pmax: root=0.8, mid=0.8, leaf=0.8
        # Pmin (top-down): root 0.6 caps mid -> mid=min(0.2,0.6)=0.2;
        #   then leaf capped by mid 0.2 -> leaf=0.2.
        #   pmin: root=0.6, mid=0.2, leaf=0.2
        # Blend 0.7/0.3:
        #   root = 0.7*0.8 + 0.3*0.6 = 0.74
        #   mid  = 0.7*0.8 + 0.3*0.2 = 0.62
        #   leaf = 0.7*0.8 + 0.3*0.2 = 0.62
        # Already parent>=child after blend, so projection is identity.
        proteins = np.array(["P1", "P1", "P1"])
        go_terms = np.array(["GO:ROOT", "GO:MID", "GO:LEAF"])
        scores = np.array([0.6, 0.2, 0.8], dtype=np.float32)
        out = soft_propagate_scores(
            proteins, go_terms, scores, PARENT_MAP, pmax_weight=0.7,
        )
        assert out[0] == pytest.approx(0.74, abs=1e-5)
        assert out[1] == pytest.approx(0.62, abs=1e-5)
        assert out[2] == pytest.approx(0.62, abs=1e-5)
        assert _consistent(proteins, go_terms, out)

    def test_pmax_weight_extremes(self) -> None:
        proteins = np.array(["P1", "P1", "P1"])
        go_terms = np.array(["GO:ROOT", "GO:MID", "GO:LEAF"])
        scores = np.array([0.6, 0.2, 0.8], dtype=np.float32)
        pure_pmax = soft_propagate_scores(
            proteins, go_terms, scores, PARENT_MAP, pmax_weight=1.0,
        )
        # Pure Pmax raises everything to 0.8.
        np.testing.assert_allclose(pure_pmax, [0.8, 0.8, 0.8], atol=1e-5)
        pure_pmin = soft_propagate_scores(
            proteins, go_terms, scores, PARENT_MAP, pmax_weight=0.0,
        )
        # Pure Pmin: root 0.6, mid 0.2, leaf 0.2 (already consistent).
        np.testing.assert_allclose(pure_pmin, [0.6, 0.2, 0.2], atol=1e-5)

    def test_independent_proteins(self) -> None:
        proteins = np.array(["P1", "P1", "P2", "P2"])
        go_terms = np.array(["GO:ROOT", "GO:MID", "GO:ROOT", "GO:MID"])
        scores = np.array([0.2, 0.9, 0.9, 0.2], dtype=np.float32)
        out = soft_propagate_scores(proteins, go_terms, scores, PARENT_MAP)
        assert _consistent(proteins, go_terms, out)

    def test_invalid_weight_raises(self) -> None:
        proteins = np.array(["P1"])
        go_terms = np.array(["GO:ROOT"])
        scores = np.array([0.5], dtype=np.float32)
        with pytest.raises(ValueError):
            soft_propagate_scores(
                proteins, go_terms, scores, PARENT_MAP, pmax_weight=1.5,
            )

    def test_empty(self) -> None:
        out = soft_propagate_scores(
            np.empty(0, dtype=object),
            np.empty(0, dtype=object),
            np.empty(0, dtype=np.float32),
            PARENT_MAP,
        )
        assert len(out) == 0

    def test_output_float32(self) -> None:
        proteins = np.array(["P1", "P1"])
        go_terms = np.array(["GO:ROOT", "GO:MID"])
        scores = np.array([0.5, 0.3], dtype=np.float64)
        out = soft_propagate_scores(proteins, go_terms, scores, PARENT_MAP)
        assert out.dtype == np.float32
