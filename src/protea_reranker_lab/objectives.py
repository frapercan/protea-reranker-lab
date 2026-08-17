"""What the code cosine is regressed onto, and the nulls that price it.

Split out of ``encoder_ablation`` because it is a separate concern and because
that module had reached the file-size ceiling. Everything here answers one
question: given a pool of proteins and a set of pairs, what scalar should the
encoder be asked to predict, and what would it score if the scalar carried no
information.

The five inputs travel together everywhere, so they are one object. That is not
tidiness: the information content and the best-IC vector must be computed from
the SAME closures the pairs index into, and passing them separately is how they
come apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from protea_reranker_lab.sdr import GoDag, information_content, lin_pairwise

__all__ = [
    "TargetInputs",
    "build_target",
    "substitute_target",
    "mica_aspect_histogram",
]


@dataclass(frozen=True)
class TargetInputs:
    """The pool and the pairs a target is built over.

    ``ic`` and ``bic`` must be derived from ``closures``; a target built from one
    pool's closures and another's information content is not wrong in a way
    anything would report.
    """

    closures: list[frozenset[str]]
    ic: dict[str, float]
    pairs: list[tuple[int, int]]
    bic: list[float]
    dag: GoDag


def mica_aspect_histogram(inputs: TargetInputs) -> dict[str, int]:
    """Which aspect the most informative common ancestor belongs to, per pair.

    A pre-flight, not an arm. ``_max_common_ancestor_ic`` applies no aspect
    filter, while molecular-function terms carry higher information content than
    biological-process ones because the branch is shallower and better annotated.
    So the supervision may be dominated by the aspect the campaign cares least
    about, and the encoder would be optimised for it without anything saying so.

    Costs one pass over the pairs and answers whether the ``bpo`` arm is worth
    fitting at all.
    """
    closures, ic, dag = inputs.closures, inputs.ic, inputs.dag
    counts: dict[str, int] = {"P": 0, "F": 0, "C": 0, "none": 0}
    for i, j in inputs.pairs:
        common = closures[i] & closures[j]
        if not common:
            counts["none"] += 1
            continue
        best = max(common, key=lambda term: ic.get(term, 0.0))
        counts[dag.aspect.get(best, "none")] = counts.get(dag.aspect.get(best, "none"), 0) + 1
    return counts

def _jaccard_pairwise(
    closures: list[frozenset[str]], pairs: list[tuple[int, int]]
) -> np.ndarray:
    """Overlap of two propagated closures, with every term weighted equally.

    Prices metric alignment. cafaeval weights by information accretion and Lin
    weights by information content, and both come from the same frequency table,
    so the learned arm is the only one that was ever told the metric's weights.
    An arm trained on an unweighted target and scored by the weighted metric says
    how much of the gain was that alignment rather than the representation.
    """
    out = np.empty(len(pairs), dtype=np.float32)
    for n, (i, j) in enumerate(pairs):
        a, b = closures[i], closures[j]
        union = len(a | b)
        out[n] = (len(a & b) / union) if union else 0.0
    return out

def _restrict_to_aspect(closures: list[frozenset[str]], dag: GoDag, aspect: str) -> list[frozenset[str]]:
    """Drop every term outside one aspect, so the target speaks only about it."""
    return [frozenset(t for t in c if dag.aspect.get(t) == aspect) for c in closures]

def build_target(
    inputs: TargetInputs, target: str, rng: np.random.Generator
) -> np.ndarray:
    """The scalar the code cosine is regressed onto, for one arm.

    Everything except the target is held identical across arms: the same pool,
    the same pairs, the same seed. So a difference between two arms is a
    difference in what was asked for and nothing else.
    """
    closures, ic, pairs, bic, dag = (
        inputs.closures, inputs.ic, inputs.pairs, inputs.bic, inputs.dag
    )
    if target == "jaccard":
        return _jaccard_pairwise(closures, pairs)
    if target == "bpo":
        # The information content has to be recomputed on the restricted
        # closures. Reusing the full-ontology IC would weight biological-process
        # terms by frequencies that counted molecular-function annotations too.
        restricted = _restrict_to_aspect(closures, dag, "P")
        ic_p = information_content(restricted, dag)
        bic_p = [max((ic_p.get(t, 0.0) for t in c), default=0.0) for c in restricted]
        return np.asarray(lin_pairwise(restricted, ic_p, pairs, bic_p), dtype=np.float32)
    lin = np.asarray(lin_pairwise(closures, ic, pairs, bic), dtype=np.float32)
    return substitute_target(lin, pairs, bic, target, rng)

def substitute_target(
    lin: np.ndarray, pairs: list[tuple[int, int]], bic: list[float],
    target: str, rng: np.random.Generator,
) -> np.ndarray:
    """Replace the regression target with a null, keeping everything else fixed.

    Lin similarity is twice the information content of the most informative
    common ancestor over the sum of the two proteins' best-IC values. Best-IC is
    a PER-PROTEIN scalar, so a large share of the target's variance is explained
    by two per-protein numbers and no pair information at all. An encoder that
    scores well may have learned that and nothing else.

    ``marginal`` tests exactly that. It recovers the common-ancestor term from
    the Lin values, replaces it with its pool mean, and rebuilds the target over
    the untouched denominators. Both per-protein marginals survive; every trace
    of which pair is which is gone. If this recovers a large share of the real
    target's gain, the headline is an annotation-mass prior rather than a
    functional representation.

    ``shuffled`` permutes the real targets across pairs. It preserves the
    target's whole marginal distribution and destroys its relationship to the
    inputs, so any gain over the raw embedding under this condition is an offset
    that every other number carries too.
    """
    if target == "lin":
        return lin
    if target == "shuffled":
        return rng.permutation(lin)
    if target == "marginal":
        denom = np.array([bic[i] + bic[j] for i, j in pairs], dtype=np.float64)
        safe = np.where(denom > 0.0, denom, 1.0)
        mica = lin * safe / 2.0
        rebuilt = 2.0 * float(mica.mean()) / safe
        return np.where(denom > 0.0, rebuilt, 0.0).astype(np.float32)
    raise ValueError(f"unknown target {target!r}; choose lin, marginal or shuffled")
