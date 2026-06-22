"""Unit tests for the SDR (sparse distributed representation) primitives."""

from __future__ import annotations

import numpy as np

from protea_reranker_lab.sdr import (
    GoDag,
    cosine_dense,
    information_content,
    kwta_active_set,
    kwta_binarise,
    lin_pairwise,
    propagate,
    resnik_pairwise,
    tanimoto_dense,
)


# ---------------------------------------------------------------------------
# k-WTA + Tanimoto
# ---------------------------------------------------------------------------
def test_kwta_active_set_picks_top_magnitude() -> None:
    vec = np.array([0.1, -0.9, 0.3, -0.05, 0.8])
    # top-2 by magnitude are |-0.9| and |0.8| -> indices 1 and 4
    assert kwta_active_set(vec, 2).tolist() == [1, 4]


def test_kwta_active_set_k_geq_dim_returns_all() -> None:
    vec = np.array([1.0, 2.0, 3.0])
    assert kwta_active_set(vec, 5).tolist() == [0, 1, 2]


def test_kwta_binarise_has_exactly_k_active_bits() -> None:
    emb = np.random.default_rng(0).normal(size=(7, 16))
    sdr = kwta_binarise(emb, k=4)
    assert sdr.shape == (7, 16)
    assert sdr.dtype == np.uint8
    assert (sdr.sum(axis=1) == 4).all()


def test_tanimoto_identical_sets_is_one() -> None:
    sdr = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.uint8)
    t = tanimoto_dense(sdr, k=2)
    assert np.isclose(t[0, 1], 1.0)


def test_tanimoto_disjoint_sets_is_zero() -> None:
    sdr = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    t = tanimoto_dense(sdr, k=2)
    assert np.isclose(t[0, 1], 0.0)


def test_tanimoto_partial_overlap_formula() -> None:
    # k=2, intersection=1 -> 1 / (2*2 - 1) = 1/3
    sdr = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    t = tanimoto_dense(sdr, k=2)
    assert np.isclose(t[0, 1], 1.0 / 3.0)


def test_cosine_identical_vectors_is_one() -> None:
    emb = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])  # parallel
    c = cosine_dense(emb)
    assert np.isclose(c[0, 1], 1.0)


def test_cosine_orthogonal_vectors_is_zero() -> None:
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    c = cosine_dense(emb)
    assert np.isclose(c[0, 1], 0.0)


# ---------------------------------------------------------------------------
# GO DAG + information content + semantic similarity
# ---------------------------------------------------------------------------
_TINY_OBO = """format-version: 1.2

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0000100
name: childA
namespace: biological_process
is_a: GO:0008150

[Term]
id: GO:0000200
name: childB
namespace: biological_process
is_a: GO:0008150

[Term]
id: GO:0000110
name: grandchildA
namespace: biological_process
is_a: GO:0000100

[Term]
id: GO:0000999
name: obsolete-term
namespace: biological_process
is_obsolete: true

[Typedef]
id: part_of
name: part of
"""


def _build_dag(tmp_path) -> GoDag:
    p = tmp_path / "tiny.obo"
    p.write_text(_TINY_OBO)
    return GoDag.from_obo(p)


def test_dag_parses_parents_and_aspect(tmp_path) -> None:
    dag = _build_dag(tmp_path)
    assert dag.aspect["GO:0000100"] == "P"
    assert dag.parents["GO:0000100"] == frozenset({"GO:0008150"})
    # obsolete terms are dropped
    assert "GO:0000999" not in dag.parents


def test_dag_ancestors_include_self_and_transitive(tmp_path) -> None:
    dag = _build_dag(tmp_path)
    anc = dag.ancestors("GO:0000110")
    assert anc == frozenset({"GO:0000110", "GO:0000100", "GO:0008150"})


def test_propagate_closure(tmp_path) -> None:
    dag = _build_dag(tmp_path)
    closure = propagate(["GO:0000110"], dag)
    assert "GO:0008150" in closure and "GO:0000100" in closure


def test_information_content_root_is_zero_children_positive(tmp_path) -> None:
    dag = _build_dag(tmp_path)
    # corpus where childA appears in 2 proteins but grandchildA in only 1, so
    # the rarer (deeper) term must carry strictly more information.
    sets = [
        propagate(["GO:0000110"], dag),  # closure includes childA + grandchildA
        propagate(["GO:0000100"], dag),  # closure includes childA only
        propagate(["GO:0000200"], dag),  # childB
    ]
    ic = information_content(sets, dag)
    assert np.isclose(ic.get("GO:0008150", 0.0), 0.0)  # root: appears in all -> IC 0
    assert ic["GO:0000110"] > ic["GO:0000100"] > 0.0  # rarer/deeper = more informative


def test_resnik_and_lin_share_common_ancestor(tmp_path) -> None:
    dag = _build_dag(tmp_path)
    closures = [propagate(["GO:0000110"], dag), propagate(["GO:0000200"], dag)]
    ic = information_content(closures, dag)
    # MICA of grandchildA and childB is the root (IC 0) -> Resnik 0
    res = resnik_pairwise(closures, ic, [(0, 1)])
    assert np.isclose(res[0], 0.0)
    # same protein vs itself: MICA is its own most specific term -> Resnik > 0
    self_res = resnik_pairwise(closures, ic, [(0, 0)])
    assert self_res[0] > 0.0
    best_ic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in closures]
    lin = lin_pairwise(closures, ic, [(0, 0)], best_ic)
    assert np.isclose(lin[0], 1.0)  # self-similarity under Lin is 1
