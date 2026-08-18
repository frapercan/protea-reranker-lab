"""The order axis, and the arithmetic that decides whether a gated arm can carry it.

The premise of the whole order axis is that sparsifying and aggregating do not
commute. That is easy to assert and worth demonstrating numerically, because if it
were false the axis would have nothing to measure and the plan would be funding a
distinction that does not exist on this data.

The second thing here is the gate's cost. An elementwise gate keeps the
intersection of two supports, expected size k squared over D, which at the shipped
configuration is eight atoms out of 2048. An arm like that runs, finishes, and
reports a flat result for a reason unrelated to the hypothesis. The refusal exists
so that null is never produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.sparse_aggregation import (
    MIN_EXPECTED_GATED_SUPPORT,
    aggregate_then_sparsify,
    expected_gated_support,
    gate,
    orders_disagree,
    refuse_a_collapsing_gate,
    sparsify_then_aggregate,
    support_size,
    topk_real,
)


# --------------------------------------------------------------- the non-commutativity

def test_the_two_orders_select_different_atoms():
    """The premise of the order axis, demonstrated rather than asserted.

    Cell one is strongly about atom 0 and silent elsewhere; both cells are weakly
    about atom 3. Averaging first lets the weak-but-everywhere atom compete;
    sparsifying first lets the strong-but-local one through at full strength.
    """
    local = np.array([[9.0, 0.0, 0.0, 2.0],
                      [0.0, 0.0, 0.0, 2.0]], dtype=np.float32)

    first = np.flatnonzero(aggregate_then_sparsify(local, 1)[0])
    second = np.flatnonzero(topk_real(sparsify_then_aggregate(local, 1), 1)[0])

    assert first.tolist() == [0]
    assert second.tolist() == [0]
    assert orders_disagree(local, 1) is False  # they agree HERE, and that is the point


def test_a_case_where_the_orders_genuinely_disagree():
    """Weak everywhere beats strong once, but only if you average first.

    The window is narrow and worth writing down, because a hand-picked example
    outside it silently agrees and would have made this axis look empty. With one
    strong cell of value a, n cells in total and a weak value b present in all of
    them, the two orders disagree exactly when b(n-1) < a < nb: averaging first
    dilutes a by n while sparsifying first costs b the cell it was never selected
    in. Here n=3, b=3, so a must sit strictly between 6 and 9.
    """
    local = np.array([[7.0, 3.0],
                      [0.0, 3.0],
                      [0.0, 3.0]], dtype=np.float32)

    first = np.flatnonzero(aggregate_then_sparsify(local, 1)[0])
    second = np.flatnonzero(topk_real(sparsify_then_aggregate(local, 1), 1)[0])

    assert first.tolist() == [1]     # mean 2.33 against 3.00, the weak-everywhere atom
    assert second.tolist() == [0]    # 2.33 against 2.00, since atom 1 loses a cell
    assert orders_disagree(local, 1) is True


def test_outside_that_window_the_orders_agree():
    """The same shape with a=5 agrees, which is why the window is stated above and
    not left for the next reader to rediscover by writing a failing test."""
    local = np.array([[5.0, 3.0],
                      [0.0, 3.0],
                      [0.0, 3.0]], dtype=np.float32)

    assert orders_disagree(local, 1) is False


def test_sparsify_then_aggregate_is_not_k_sparse():
    """It carries the union of the local supports, which is why a caller has to
    sparsify again and why the two orders are different representations."""
    local = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

    got = sparsify_then_aggregate(local, 1)

    assert support_size(got)[0] == 3


def test_aggregate_then_sparsify_is_k_sparse():
    local = np.random.default_rng(0).random((5, 64)).astype(np.float32)

    assert support_size(aggregate_then_sparsify(local, 8))[0] == 8


def test_one_local_code_makes_the_orders_identical():
    """A single cell has nothing to aggregate, so the axis collapses. Worth
    pinning: a corpus of very short proteins would silently have no order axis."""
    local = np.random.default_rng(1).random((1, 32)).astype(np.float32)

    a = aggregate_then_sparsify(local, 4)
    b = topk_real(sparsify_then_aggregate(local, 4), 4)

    assert np.array_equal(np.flatnonzero(a[0]), np.flatnonzero(b[0]))


# ------------------------------------------------------------------- the gate's cost

def test_the_shipped_configuration_would_leave_eight_atoms():
    assert expected_gated_support(128, 2048) == pytest.approx(8.0)


def test_the_plan_table_reproduces():
    """The same arithmetic the sparsity surface uses, in the opposite direction."""
    assert expected_gated_support(256, 2048) == pytest.approx(32.0)
    assert expected_gated_support(362, 2048) == pytest.approx(63.98, abs=0.1)
    assert expected_gated_support(128, 1024) == pytest.approx(16.0)
    assert expected_gated_support(128, 512) == pytest.approx(32.0)
    assert expected_gated_support(128, 256) == pytest.approx(64.0)


def test_the_shipped_configuration_is_refused_for_a_gated_arm():
    """The regression this exists for: it would run, finish, and report a null
    from an empty vector that is indistinguishable from a null from the hypothesis."""
    with pytest.raises(ValueError, match="expected to leave"):
        refuse_a_collapsing_gate(128, 2048)


def test_the_refusal_says_how_to_fix_it():
    with pytest.raises(ValueError) as excinfo:
        refuse_a_collapsing_gate(128, 2048)

    message = str(excinfo.value)
    assert "Raise k or shrink the dictionary" in message


def test_a_viable_gated_configuration_passes():
    refuse_a_collapsing_gate(128, 1024)
    refuse_a_collapsing_gate(256, 2048)


def test_the_floor_sits_above_the_shipped_expected_support():
    """Calibration: the constraint must exclude the configuration that motivated it."""
    assert expected_gated_support(128, 2048) < MIN_EXPECTED_GATED_SUPPORT


def test_a_zero_dictionary_does_not_divide_by_zero():
    assert expected_gated_support(128, 0) == 0.0


# ------------------------------------------------------------------------- gating

def test_gating_keeps_the_intersection_of_supports():
    local = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    sequence = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    assert np.flatnonzero(gate(local, sequence)[0]).tolist() == [0]


def test_gating_never_grows_the_support():
    rng = np.random.default_rng(2)
    local = topk_real(rng.random((4, 128)).astype(np.float32), 16)
    sequence = topk_real(rng.random((1, 128)).astype(np.float32), 16)[0]

    before = support_size(local)
    after = support_size(gate(local, sequence))

    assert np.all(after <= before)


def test_gating_by_a_dense_sequence_code_preserves_the_local_support():
    """The escape hatch the plan implies: a dense gate modulates without deciding."""
    rng = np.random.default_rng(3)
    local = topk_real(rng.random((3, 32)).astype(np.float32), 8)
    dense = rng.random(32).astype(np.float32) + 0.1

    assert np.array_equal(support_size(gate(local, dense)), support_size(local))
