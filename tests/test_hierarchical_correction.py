"""Tests for post-hoc hierarchical-consistency score correction (F-RERANK-UNIVERSAL.5)."""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.hierarchical_correction import (
    build_children_map,
    check_hierarchical_consistency,
    correct_scores_hierarchical,
)


# Minimal GO DAG fixture:
# GO:0000003 (ancestor)
#   -> GO:0000002 (intermediate)
#       -> GO:0000001 (leaf)
# parent_map: {child: {direct_parents}}
PARENT_MAP = {
    "GO:0000001": frozenset({"GO:0000002"}),
    "GO:0000002": frozenset({"GO:0000003"}),
}


@pytest.fixture
def children_map() -> dict[str, set[str]]:
    return build_children_map(PARENT_MAP)


class TestBuildChildrenMap:
    def test_basic_inversion(self) -> None:
        cm = build_children_map(PARENT_MAP)
        assert "GO:0000002" in cm
        assert "GO:0000001" in cm["GO:0000002"]
        assert "GO:0000003" in cm
        assert "GO:0000002" in cm["GO:0000003"]

    def test_empty_parent_map(self) -> None:
        cm = build_children_map({})
        assert cm == {}

    def test_multiple_parents(self) -> None:
        pm = {
            "GO:A": frozenset({"GO:B", "GO:C"}),
        }
        cm = build_children_map(pm)
        assert "GO:A" in cm.get("GO:B", set())
        assert "GO:A" in cm.get("GO:C", set())


class TestCorrectScoresHierarchical:
    def test_parent_raised_when_child_higher(self, children_map) -> None:
        # Leaf has score 0.9, intermediate 0.3, ancestor 0.2
        # After correction: intermediate should be 0.9, ancestor should be 0.9
        proteins = np.array(["P1", "P1", "P1"])
        go_terms = np.array(["GO:0000001", "GO:0000002", "GO:0000003"])
        scores = np.array([0.9, 0.3, 0.2], dtype=np.float32)
        corrected, n = correct_scores_hierarchical(
            proteins=proteins,
            go_terms=go_terms,
            scores=scores,
            children_map=children_map,
        )
        # Intermediate must be >= max(child=0.9)
        assert corrected[1] >= corrected[0] - 1e-6, (
            f"Intermediate {corrected[1]:.4f} < leaf {corrected[0]:.4f}"
        )
        # Ancestor must be >= intermediate
        assert corrected[2] >= corrected[1] - 1e-6, (
            f"Ancestor {corrected[2]:.4f} < intermediate {corrected[1]:.4f}"
        )
        assert n > 0

    def test_already_consistent_unchanged(self, children_map) -> None:
        # Ancestor > Intermediate > Leaf: already consistent, no corrections needed
        proteins = np.array(["P1", "P1", "P1"])
        go_terms = np.array(["GO:0000001", "GO:0000002", "GO:0000003"])
        scores = np.array([0.2, 0.5, 0.9], dtype=np.float32)
        corrected, n = correct_scores_hierarchical(
            proteins=proteins,
            go_terms=go_terms,
            scores=scores,
            children_map=children_map,
        )
        # Scores should not decrease
        assert corrected[0] == pytest.approx(0.2, abs=1e-5)
        assert corrected[1] == pytest.approx(0.5, abs=1e-5)
        assert corrected[2] == pytest.approx(0.9, abs=1e-5)
        assert n == 0

    def test_empty_input(self, children_map) -> None:
        corrected, n = correct_scores_hierarchical(
            proteins=np.empty(0, dtype=object),
            go_terms=np.empty(0, dtype=object),
            scores=np.empty(0, dtype=np.float32),
            children_map=children_map,
        )
        assert len(corrected) == 0
        assert n == 0

    def test_independent_proteins(self, children_map) -> None:
        # Two proteins: P1 has high leaf, P2 has low leaf
        proteins = np.array(["P1", "P1", "P2", "P2"])
        go_terms = np.array(["GO:0000001", "GO:0000002", "GO:0000001", "GO:0000002"])
        scores = np.array([0.9, 0.1, 0.2, 0.8], dtype=np.float32)
        corrected, n = correct_scores_hierarchical(
            proteins=proteins,
            go_terms=go_terms,
            scores=scores,
            children_map=children_map,
        )
        # P1: GO:0000002 must be raised to 0.9
        p1_intermediate = corrected[1]
        assert p1_intermediate >= 0.9 - 1e-5

        # P2: GO:0000002 already 0.8 >= 0.2, no correction needed
        p2_intermediate = corrected[3]
        assert p2_intermediate == pytest.approx(0.8, abs=1e-5)

    def test_unknown_terms_ignored(self, children_map) -> None:
        # Terms not in the children_map should just pass through
        proteins = np.array(["P1", "P1"])
        go_terms = np.array(["GO:UNKNOWN1", "GO:UNKNOWN2"])
        scores = np.array([0.9, 0.1], dtype=np.float32)
        corrected, n = correct_scores_hierarchical(
            proteins=proteins,
            go_terms=go_terms,
            scores=scores,
            children_map=children_map,
        )
        # Should remain unchanged (no DAG edges)
        assert corrected[0] == pytest.approx(0.9, abs=1e-5)
        assert corrected[1] == pytest.approx(0.1, abs=1e-5)

    def test_output_dtype_float32(self, children_map) -> None:
        proteins = np.array(["P1", "P1"])
        go_terms = np.array(["GO:0000001", "GO:0000002"])
        scores = np.array([0.5, 0.3], dtype=np.float64)
        corrected, _ = correct_scores_hierarchical(
            proteins=proteins,
            go_terms=go_terms,
            scores=scores,
            children_map=children_map,
        )
        assert corrected.dtype == np.float32


class TestCheckHierarchicalConsistency:
    def test_detects_violations(self, children_map) -> None:
        proteins = np.array(["P1", "P1", "P1"])
        go_terms = np.array(["GO:0000001", "GO:0000002", "GO:0000003"])
        # Child (GO:0000001) > parent (GO:0000002): violation
        scores = np.array([0.9, 0.3, 0.2], dtype=np.float32)
        result = check_hierarchical_consistency(
            proteins=proteins, go_terms=go_terms, scores=scores,
            children_map=children_map,
        )
        assert result["n_violations"] > 0

    def test_no_violations_after_correction(self, children_map) -> None:
        proteins = np.array(["P1", "P1", "P1"])
        go_terms = np.array(["GO:0000001", "GO:0000002", "GO:0000003"])
        scores = np.array([0.9, 0.3, 0.2], dtype=np.float32)
        corrected, _ = correct_scores_hierarchical(
            proteins=proteins, go_terms=go_terms, scores=scores,
            children_map=children_map,
        )
        result = check_hierarchical_consistency(
            proteins=proteins, go_terms=go_terms, scores=corrected,
            children_map=children_map,
        )
        assert result["n_violations"] == 0

    def test_empty_input(self, children_map) -> None:
        result = check_hierarchical_consistency(
            proteins=np.empty(0, dtype=object),
            go_terms=np.empty(0, dtype=object),
            scores=np.empty(0, dtype=np.float32),
            children_map=children_map,
        )
        assert result["n_violations"] == 0
        assert result["n_checked_pairs"] == 0
