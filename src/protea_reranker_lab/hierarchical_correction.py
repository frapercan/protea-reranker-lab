"""Post-hoc true-path / hierarchical-consistency score correction.

Applies the GO DAG constraint: a parent term's score must be >= the maximum
score of any of its children for the same protein. This ensures that
predictions are "hierarchically consistent" (True-Path-Rule in scoring form).

Algorithm (bottom-up over the DAG)::

    For each protein independently, propagate scores UP the DAG:
        score[parent] = max(score[parent], max(score[child] for child in children))
    Repeat until no score changes (convergence), or for at most ``max_passes``
    sweeps.

This is the complement of the True-Path-Rule label propagation in
:mod:`protea_reranker_lab.propagation`. Where propagation makes labels
consistent BEFORE training, score correction makes predictions consistent
AFTER scoring.

Usage::

    import numpy as np
    from protea_reranker_lab.hierarchical_correction import (
        build_children_map, correct_scores_hierarchical,
    )

    parent_map = {"GO:0000001": {"GO:0000002"}, ...}
    children_map = build_children_map(parent_map)

    corrected = correct_scores_hierarchical(
        proteins=proteins,
        go_terms=go_terms,
        scores=scores,
        children_map=children_map,
    )

The corrected scores satisfy parent_score >= max(child_scores) for
every protein after the operation.
"""

from __future__ import annotations

import numpy as np


def build_children_map(
    parent_map: dict[str, frozenset[str]],
) -> dict[str, set[str]]:
    """Invert a parent_map (child -> parents) to a children map (parent -> children).

    Parameters
    ----------
    parent_map:
        Mapping from child GO ID to its direct parents (as returned by
        :func:`~protea_reranker_lab.propagation.load_parent_map`).

    Returns
    -------
    Mapping from parent GO ID to the set of its direct children.
    """
    children: dict[str, set[str]] = {}
    for child, parents in parent_map.items():
        for parent in parents:
            children.setdefault(parent, set()).add(child)
    return children


def _propagate_score_map(
    score_map: dict[str, float],
    children_map: dict[str, set[str]],
    max_passes: int,
) -> None:
    """Mutate ``score_map`` in place: raise each parent to max(children).

    Repeats until convergence or ``max_passes`` iterations.
    """
    for _ in range(max_passes):
        changed = False
        for go_id, s in list(score_map.items()):
            children = children_map.get(go_id)
            if children is None:
                continue
            max_child = max(
                (score_map[c] for c in children if c in score_map),
                default=None,
            )
            if max_child is not None and max_child > s:
                score_map[go_id] = max_child
                changed = True
        if not changed:
            break


def _correct_one_protein(
    idxs: np.ndarray,
    go_terms: np.ndarray,
    out: np.ndarray,
    children_map: dict[str, set[str]],
    max_passes: int,
) -> int:
    """Apply bottom-up correction for one protein. Returns n_corrections."""
    gos_p = go_terms[idxs]
    score_map: dict[str, float] = {str(g): float(out[i]) for g, i in zip(gos_p, idxs)}
    _propagate_score_map(score_map, children_map, max_passes)
    n_corr = 0
    for i, g in zip(idxs, gos_p):
        new_score = score_map.get(str(g))
        if new_score is not None and new_score > out[i]:
            out[i] = np.float32(new_score)
            n_corr += 1
    return n_corr


def correct_scores_hierarchical(
    proteins: np.ndarray,
    go_terms: np.ndarray,
    scores: np.ndarray,
    children_map: dict[str, set[str]],
    max_passes: int = 20,
) -> tuple[np.ndarray, int]:
    """Apply bottom-up hierarchical score correction.

    For each protein, propagates the maximum score upward so that every
    parent term's score >= max score of any of its direct children (for
    the same protein). This implements the CAFA True-Path-Rule in scoring
    space.

    Returns ``(corrected_scores, n_corrections)`` where ``n_corrections``
    is the number of score values raised during the operation.
    """
    if len(proteins) == 0:
        return scores.copy(), 0
    out = scores.astype(np.float32).copy()
    total = 0
    for protein in np.unique(proteins):
        idxs = np.flatnonzero(proteins == protein)
        total += _correct_one_protein(idxs, go_terms, out, children_map, max_passes)
    return out, total


def check_hierarchical_consistency(
    proteins: np.ndarray,
    go_terms: np.ndarray,
    scores: np.ndarray,
    children_map: dict[str, set[str]],
) -> dict[str, int]:
    """Count parent-child score violations for diagnostics.

    A violation is when ``score[parent] < score[child]`` for the same protein
    and both terms are present in the prediction set.

    Returns a dict with keys ``"n_violations"`` and ``"n_checked_pairs"``.
    """
    n_violations = 0
    n_checked = 0
    unique_proteins = np.unique(proteins)

    for protein in unique_proteins:
        mask = proteins == protein
        idxs = np.flatnonzero(mask)
        score_map = {str(go_terms[i]): float(scores[i]) for i in idxs}
        for go_id, s in score_map.items():
            children = children_map.get(go_id)
            if children is None:
                continue
            for child in children:
                if child in score_map:
                    n_checked += 1
                    if score_map[child] > s:
                        n_violations += 1

    return {"n_violations": n_violations, "n_checked_pairs": n_checked}
