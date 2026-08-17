"""The rest of the battery: what the target says, as against whether it says anything.

The two nulls ask whether the encoder learned anything at all. These ask whether
what it learned is the thing the campaign is about.

``bpo`` exists because the most informative common ancestor is chosen with no
aspect filter, while molecular-function terms carry higher information content
than biological-process ones. So the supervision may be dominated by the aspect
the campaign cares least about, and nothing in the shipped run would say so.

``jaccard`` exists because cafaeval weights by information accretion and Lin
weights by information content, and both come from the same frequency table. The
learned arm is the only one that was ever told the metric's weights, so some of
its gain may be alignment with the scorer rather than representation.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.encoder_ablation import (
    _jaccard_pairwise,
    _restrict_to_aspect,
    build_target,
    mica_aspect_histogram,
)


class _Dag:
    """Only the surface these functions touch: term to aspect code."""

    def __init__(self, aspect: dict[str, str]) -> None:
        self.aspect = aspect


# --------------------------------------------------------------------------- the pre-flight

def test_the_histogram_names_the_aspect_of_the_winning_ancestor():
    dag = _Dag({"GO:P1": "P", "GO:F1": "F", "GO:C1": "C"})
    closures = [
        frozenset({"GO:P1", "GO:F1"}),
        frozenset({"GO:P1", "GO:F1"}),
    ]
    # the molecular-function term is more informative, so it wins the argmax
    ic = {"GO:P1": 1.0, "GO:F1": 9.0, "GO:C1": 1.0}

    got = mica_aspect_histogram(closures, ic, [(0, 1)], dag)

    assert got["F"] == 1
    assert got["P"] == 0


def test_a_pair_with_no_common_ancestor_is_counted_and_not_dropped():
    """Dropping them would make the histogram a share of an unstated denominator."""
    dag = _Dag({"GO:P1": "P", "GO:F1": "F"})
    closures = [frozenset({"GO:P1"}), frozenset({"GO:F1"})]

    got = mica_aspect_histogram(closures, {"GO:P1": 1.0, "GO:F1": 1.0}, [(0, 1)], dag)

    assert got["none"] == 1
    assert sum(got.values()) == 1


def test_the_histogram_totals_the_pair_count():
    dag = _Dag({f"GO:P{i}": "P" for i in range(4)})
    closures = [frozenset({"GO:P0", "GO:P1"}), frozenset({"GO:P1"}), frozenset({"GO:P2"})]
    ic = {f"GO:P{i}": float(i) for i in range(4)}
    pairs = [(0, 1), (0, 2), (1, 2)]

    got = mica_aspect_histogram(closures, ic, pairs, dag)

    assert sum(got.values()) == len(pairs)


# --------------------------------------------------------------------------- jaccard

def test_jaccard_is_intersection_over_union():
    closures = [frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})]
    got = _jaccard_pairwise(closures, [(0, 1)])
    assert np.isclose(got[0], 2 / 4)


def test_jaccard_of_identical_closures_is_one():
    closures = [frozenset({"a", "b"}), frozenset({"a", "b"})]
    assert np.isclose(_jaccard_pairwise(closures, [(0, 1)])[0], 1.0)


def test_jaccard_of_two_empty_closures_is_zero_and_not_a_nan():
    closures = [frozenset(), frozenset()]
    got = _jaccard_pairwise(closures, [(0, 1)])
    assert got[0] == 0.0
    assert np.isfinite(got).all()


def test_jaccard_ignores_information_content_entirely():
    """That is the point: it prices what the weighting was worth."""
    closures = [frozenset({"rare", "common"}), frozenset({"rare"})]
    a = _jaccard_pairwise(closures, [(0, 1)])
    # the same sets under any weighting give the same answer, because none is used
    assert np.isclose(a[0], 1 / 2)


# --------------------------------------------------------------------------- aspect restriction

def test_restriction_keeps_only_the_named_aspect():
    dag = _Dag({"GO:P1": "P", "GO:F1": "F", "GO:C1": "C"})
    closures = [frozenset({"GO:P1", "GO:F1", "GO:C1"})]

    got = _restrict_to_aspect(closures, dag, "P")

    assert got == [frozenset({"GO:P1"})]


def test_restriction_can_empty_a_closure_without_failing():
    """A protein annotated only in molecular function has no biological process."""
    dag = _Dag({"GO:F1": "F"})
    got = _restrict_to_aspect([frozenset({"GO:F1"})], dag, "P")
    assert got == [frozenset()]


def test_an_unknown_term_is_dropped_rather_than_kept():
    """A term absent from the ontology snapshot has no aspect and cannot be assigned one."""
    dag = _Dag({"GO:P1": "P"})
    got = _restrict_to_aspect([frozenset({"GO:P1", "GO:UNKNOWN"})], dag, "P")
    assert got == [frozenset({"GO:P1"})]


# --------------------------------------------------------------------------- dispatch

def test_an_unknown_target_is_refused_by_the_builder_too():
    dag = _Dag({})
    with pytest.raises(ValueError, match="unknown target"):
        build_target([frozenset()], {}, [(0, 0)], [0.0], dag, "nonsense",
                     np.random.default_rng(0))


def test_jaccard_routes_without_touching_information_content():
    """It must not require an ic table, since its whole point is not to use one."""
    dag = _Dag({})
    closures = [frozenset({"a", "b"}), frozenset({"b"})]

    got = build_target(closures, {}, [(0, 1)], [0.0, 0.0], dag, "jaccard",
                       np.random.default_rng(0))

    assert np.isclose(got[0], 0.5)
