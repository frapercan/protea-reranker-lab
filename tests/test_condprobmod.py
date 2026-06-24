"""Tests for CondProbMod conditional-probability hierarchical modelling (R2.1).

The reconstruction tests pin the parent-product recursion against a
hand-worked small-DAG example:

    GO:ROOT          (root)
      -> GO:MID      (child of ROOT)
           -> GO:LEAF  (child of MID)
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.condprobmod import (
    build_conditional_mask,
    go_roots,
    reconstruct_marginal_probs,
)

# child -> direct parents
PARENT_MAP = {
    "GO:MID": frozenset({"GO:ROOT"}),
    "GO:LEAF": frozenset({"GO:MID"}),
}


class TestGoRoots:
    def test_identifies_root(self) -> None:
        assert go_roots(PARENT_MAP) == frozenset({"GO:ROOT"})

    def test_empty(self) -> None:
        assert go_roots({}) == frozenset()

    def test_multi_parent_root_detection(self) -> None:
        pm = {"GO:C": frozenset({"GO:A", "GO:B"})}
        # A and B have no parents -> both roots; C has parents -> not a root.
        assert go_roots(pm) == frozenset({"GO:A", "GO:B"})


class TestReconstructMarginal:
    def test_hand_worked_chain(self) -> None:
        # Conditional edge probs: P(ROOT)=0.8, P(MID|ROOT)=0.5, P(LEAF|MID)=0.5
        # Marginals: ROOT=0.8, MID=0.5*0.8=0.40, LEAF=0.5*0.40=0.20
        proteins = np.array(["P1", "P1", "P1"])
        go_terms = np.array(["GO:ROOT", "GO:MID", "GO:LEAF"])
        cond = np.array([0.8, 0.5, 0.5], dtype=np.float32)
        out = reconstruct_marginal_probs(proteins, go_terms, cond, PARENT_MAP)
        assert out[0] == pytest.approx(0.8, abs=1e-6)
        assert out[1] == pytest.approx(0.40, abs=1e-6)
        assert out[2] == pytest.approx(0.20, abs=1e-6)

    def test_marginal_monotone_decreasing_down_chain(self) -> None:
        # With conditionals in [0,1], each marginal <= its parent's marginal.
        proteins = np.array(["P1", "P1", "P1"])
        go_terms = np.array(["GO:ROOT", "GO:MID", "GO:LEAF"])
        cond = np.array([0.9, 0.7, 0.6], dtype=np.float32)
        out = reconstruct_marginal_probs(proteins, go_terms, cond, PARENT_MAP)
        assert out[0] >= out[1] >= out[2]

    def test_multi_parent_takes_max_lineage(self) -> None:
        # GO:C has two parents A (marginal 0.9) and B (marginal 0.2).
        # P(C|parent)=0.5 -> max lineage: 0.5 * 0.9 = 0.45.
        pm = {"GO:C": frozenset({"GO:A", "GO:B"})}
        proteins = np.array(["P1", "P1", "P1"])
        go_terms = np.array(["GO:A", "GO:B", "GO:C"])
        cond = np.array([0.9, 0.2, 0.5], dtype=np.float32)
        out = reconstruct_marginal_probs(proteins, go_terms, cond, pm)
        assert out[2] == pytest.approx(0.45, abs=1e-6)

    def test_absent_parent_keeps_conditional_as_marginal(self) -> None:
        # MID present but its parent ROOT absent for this protein -> MID is
        # treated as a local root, marginal == conditional.
        proteins = np.array(["P1", "P1"])
        go_terms = np.array(["GO:MID", "GO:LEAF"])
        cond = np.array([0.6, 0.5], dtype=np.float32)
        out = reconstruct_marginal_probs(proteins, go_terms, cond, PARENT_MAP)
        assert out[0] == pytest.approx(0.6, abs=1e-6)  # MID local root
        assert out[1] == pytest.approx(0.30, abs=1e-6)  # 0.5 * 0.6

    def test_independent_proteins(self) -> None:
        proteins = np.array(["P1", "P1", "P2", "P2"])
        go_terms = np.array(["GO:ROOT", "GO:MID", "GO:ROOT", "GO:MID"])
        cond = np.array([0.8, 0.5, 0.4, 0.5], dtype=np.float32)
        out = reconstruct_marginal_probs(proteins, go_terms, cond, PARENT_MAP)
        assert out[1] == pytest.approx(0.40, abs=1e-6)  # P1 MID = 0.5*0.8
        assert out[3] == pytest.approx(0.20, abs=1e-6)  # P2 MID = 0.5*0.4

    def test_empty(self) -> None:
        out = reconstruct_marginal_probs(
            np.empty(0, dtype=object),
            np.empty(0, dtype=object),
            np.empty(0, dtype=np.float32),
            PARENT_MAP,
        )
        assert len(out) == 0

    def test_output_clipped_and_float32(self) -> None:
        proteins = np.array(["P1"])
        go_terms = np.array(["GO:ROOT"])
        cond = np.array([1.5], dtype=np.float32)  # out of range on purpose
        out = reconstruct_marginal_probs(proteins, go_terms, cond, PARENT_MAP)
        assert out.dtype == np.float32
        assert out[0] == pytest.approx(1.0, abs=1e-6)

    def test_cycle_guard_does_not_hang(self) -> None:
        # Pathological cyclic map: A->B, B->A. Should terminate, not recurse
        # forever. The exact value is unimportant; termination is the contract.
        pm = {"GO:A": frozenset({"GO:B"}), "GO:B": frozenset({"GO:A"})}
        proteins = np.array(["P1", "P1"])
        go_terms = np.array(["GO:A", "GO:B"])
        cond = np.array([0.5, 0.5], dtype=np.float32)
        out = reconstruct_marginal_probs(proteins, go_terms, cond, pm)
        assert len(out) == 2
        assert np.all(np.isfinite(out))


class TestBuildConditionalMask:
    def test_root_always_kept(self) -> None:
        proteins = np.array(["P1"])
        go_terms = np.array(["GO:ROOT"])
        labels = np.array([0.0], dtype=np.float32)
        mask, _ = build_conditional_mask(proteins, go_terms, labels, PARENT_MAP)
        assert mask[0]  # roots always learnable

    def test_child_kept_only_when_parent_active(self) -> None:
        # P1: ROOT positive -> MID is learnable. P2: ROOT negative -> MID masked.
        proteins = np.array(["P1", "P1", "P2", "P2"])
        go_terms = np.array(["GO:ROOT", "GO:MID", "GO:ROOT", "GO:MID"])
        labels = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        mask, _ = build_conditional_mask(proteins, go_terms, labels, PARENT_MAP)
        assert mask[0]   # P1 ROOT (root)
        assert mask[1]   # P1 MID (parent ROOT active)
        assert mask[2]   # P2 ROOT (root)
        assert not mask[3]  # P2 MID (parent ROOT inactive)

    def test_uses_parent_signal_when_given(self) -> None:
        # Labels all zero but a prediction column marks ROOT active for P1.
        proteins = np.array(["P1", "P1"])
        go_terms = np.array(["GO:ROOT", "GO:MID"])
        labels = np.array([0.0, 0.0], dtype=np.float32)
        signal = np.array([0.7, 0.0], dtype=np.float32)
        mask, _ = build_conditional_mask(
            proteins, go_terms, labels, PARENT_MAP, parent_signal=signal,
        )
        assert mask[0]  # root
        assert mask[1]  # parent ROOT signalled 0.7 > 0

    def test_threshold_respected(self) -> None:
        proteins = np.array(["P1", "P1"])
        go_terms = np.array(["GO:ROOT", "GO:MID"])
        labels = np.array([0.0, 0.0], dtype=np.float32)
        signal = np.array([0.2, 0.0], dtype=np.float32)
        mask, _ = build_conditional_mask(
            proteins, go_terms, labels, PARENT_MAP,
            parent_signal=signal, threshold=0.5,
        )
        assert mask[0]
        assert not mask[1]  # parent signal 0.2 below threshold 0.5

    def test_empty(self) -> None:
        mask, target = build_conditional_mask(
            np.empty(0, dtype=object),
            np.empty(0, dtype=object),
            np.empty(0, dtype=np.float32),
            PARENT_MAP,
        )
        assert len(mask) == 0
        assert len(target) == 0
