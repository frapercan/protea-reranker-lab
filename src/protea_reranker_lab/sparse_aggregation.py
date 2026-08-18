"""Aggregating sparse codes, where the aggregator chooses SUPPORT rather than weights.

The learned-representation plan asks for two things together, composition and
interaction, as a two by two rather than as a sum. This module holds the
operations both cells need, and the arithmetic that says which configurations can
carry them.

WHY THIS IS NOT THE EXPERIMENT THAT ALREADY LOST

Learned aggregation over DENSE residues has been measured three ways and never
decisively won: per-residue attention softpool came in at 0.6293 and 0.6205
against learned-mean at 0.6327, and a set encoder was null to marginal. Proposing
a learned aggregator again needs a reason the answer should differ, or it is a
repeat with a new name.

The reason is that in dense space an aggregator chooses WEIGHTS, and mean is
already close to the best weighting because the thing being averaged is a cloud of
comparable vectors. Over sparse codes an aggregator chooses SUPPORT. The mean of
several top-k codes is not a top-k code at all: it is a dense vector of many small
entries, so the operation everyone calls "mean pooling" silently leaves the
representation class. That is the whole content of

    top-k(mean(x_i))  !=  mean(top-k(x_i))

and it is why the order is an axis rather than an implementation detail. The dense
result does not transfer, because in dense space the two orders do not exist.

WHAT THE GATE COSTS, BEFORE IT IS RUN

An elementwise gate has the intersection of two supports, whose expected size is
k squared over D. At the shipped 2048 by 128 that is EIGHT atoms, which is close
to empty and would report a flat result for a reason that has nothing to do with
the hypothesis. So a gated arm is not another point on an existing sparsity
surface; D and k have to be chosen together, before the run rather than discovered
when everything comes out flat. ``refuse_a_collapsing_gate`` makes that a
condition instead of a footnote.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

#: Fewest atoms a gated code may be expected to carry. Below this the arm is not
#: measuring the hypothesis, it is measuring an empty vector. Eight, the shipped
#: configuration's expected support, sits under it deliberately.
MIN_EXPECTED_GATED_SUPPORT = 16


def topk_real(X: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest entries per row, zero the rest, values preserved."""
    if k >= X.shape[1]:
        return X
    out = np.zeros_like(X)
    idx = np.argpartition(-X, k - 1, axis=1)[:, :k]
    np.put_along_axis(out, idx, np.take_along_axis(X, idx, axis=1), axis=1)
    return out


def aggregate_then_sparsify(local: np.ndarray, k: int) -> np.ndarray:
    """Sequence granularity: average the local codes, then take the top k.

    The average runs over dense projections, so every atom any part of the protein
    activated can still compete for a place. A feature that is strong in one cell
    and absent elsewhere is diluted by the averaging and may lose to one that is
    weakly present throughout.
    """
    return topk_real(local.mean(axis=0, keepdims=True), k)


def sparsify_then_aggregate(local: np.ndarray, k: int) -> np.ndarray:
    """Cell or residue granularity: take the top k of each local code, then average.

    Each part of the protein commits to its own k atoms first, so a feature strong
    in one cell survives at full strength into the average and one that is weakly
    present everywhere may never be selected anywhere. The result is generally NOT
    k-sparse: it carries the union of the local supports, which is why the caller
    usually sparsifies again and why the two orders are different representations
    rather than two routes to one.
    """
    return topk_real(local, k).mean(axis=0, keepdims=True)


def expected_gated_support(k: int, dictionary: int) -> float:
    """Expected atoms surviving an elementwise gate of two independent k-sparse codes.

    The collision number, k squared over D. It is the same arithmetic the sparsity
    surface uses for accidental overlap, applied in the opposite direction, and the
    two agreeing is a reason to trust both.
    """
    if dictionary <= 0:
        return 0.0
    return (k * k) / dictionary


def refuse_a_collapsing_gate(k: int, dictionary: int,
                             floor: int = MIN_EXPECTED_GATED_SUPPORT) -> None:
    """Stop a gated arm that would carry almost no atoms.

    A gated arm at 2048 by 128 is expected to keep eight, and it would run, finish,
    and report a flat result that says nothing about gating. Refusing is better
    than measuring, because a null from an empty vector is indistinguishable in the
    output from a null from the hypothesis being wrong.
    """
    expected = expected_gated_support(k, dictionary)
    if expected >= floor:
        return
    raise ValueError(
        f"a gate at dictionary {dictionary} with k={k} is expected to leave "
        f"{expected:.1f} atoms, under the {floor} needed for the arm to be "
        "measuring anything. Raise k or shrink the dictionary: the gated arm needs "
        "them chosen together, since the support of a gate is the intersection of "
        "two supports and its expected size is k squared over D"
    )


def gate(local: np.ndarray, sequence: np.ndarray) -> np.ndarray:
    """Modulate each local code by the sequence code, elementwise.

    Not an outer product: at 2048 atoms with 128 active that is 16,384 non-zeros
    per residue and about 10 million per protein, which is dead before it starts.
    Elementwise stays in the same dimension and says something defensible, that the
    global type of the protein decides which local features count.

    The support shrinks to the intersection, which is the point and the cost.
    """
    return local * sequence.reshape(1, -1)


def support_size(X: np.ndarray) -> np.ndarray:
    """Non-zeros per row, which is what the sparsity claims are actually about."""
    return np.count_nonzero(X, axis=1)


def orders_disagree(local: np.ndarray, k: int) -> bool:
    """Whether the two orders select different supports on this input.

    Exposed as a function because the non-commutativity is the premise of the whole
    order axis, and a premise that holds in the abstract can be empty on real data:
    if a corpus's local codes are near-identical to each other, both orders return
    the same atoms and the axis has nothing to measure. Worth checking on the
    actual pool before funding the arm.
    """
    a = np.flatnonzero(aggregate_then_sparsify(local, k)[0])
    b = np.flatnonzero(topk_real(sparsify_then_aggregate(local, k), k)[0])
    return set(a.tolist()) != set(b.tolist())
