"""Soft, hierarchy-aware two-way Pmin / Pmax score propagation.

Hard True-Path-Rule *label* propagation was already shown to be negligible
for PROTEA (see ``propagation.py`` and the CAFA-6 Kaggle levers note). This
module implements the **soft** variant that operates in *score* space and
keeps both directions of the GO DAG honest:

Pmax (leaf-to-root, bottom-up)
    A parent's score is raised to at least the maximum score of any of its
    children. This is the True-Path-Rule in scoring form and is what
    :func:`protea_reranker_lab.hierarchical_correction.correct_scores_hierarchical`
    already does. We re-derive it here so the two directions can be blended
    in a single pass.

Pmin (root-to-leaf, top-down)
    A child's score is capped at its parent's score (a term cannot be more
    likely than its most-likely lineage). This is the dual constraint: it
    pulls *down* over-confident leaves whose ancestry is weak.

The two passes pull in opposite directions, so we **blend** them rather than
apply one then the other. With weight ``pmax_weight`` (default 0.7) the
final score is::

    final = pmax_weight * pmax_score + (1 - pmax_weight) * pmin_score

The blend is then projected onto the consistency constraint (one cheap
bottom-up max sweep) so the returned scores satisfy ``parent >= child`` for
every present edge. On inputs that are already consistent the pass is the
identity (idempotent), because both ``pmax`` and ``pmin`` leave a consistent
score map untouched and the blend of a value with itself is the value.

Pure numpy + stdlib; safe under the lab's no-heavy-deps CI rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hierarchical_correction import build_children_map

DEFAULT_PMAX_WEIGHT = 0.7


@dataclass(frozen=True)
class _SoftPropConfig:
    """Bundled DAG maps + blend params for the per-protein soft pass."""

    parent_map: dict[str, frozenset[str]]
    children_map: dict[str, set[str]]
    pmax_weight: float
    max_passes: int


def _pmax_map(
    score_map: dict[str, float],
    children_map: dict[str, set[str]],
    max_passes: int,
) -> dict[str, float]:
    """Bottom-up: raise each parent to max(children). Returns a new map."""
    out = dict(score_map)
    for _ in range(max_passes):
        changed = False
        for go_id in list(out):
            children = children_map.get(go_id)
            if not children:
                continue
            present = [out[c] for c in children if c in out]
            if present:
                m = max(present)
                if m > out[go_id]:
                    out[go_id] = m
                    changed = True
        if not changed:
            break
    return out


def _pmin_map(
    score_map: dict[str, float],
    parent_map: dict[str, frozenset[str]],
    max_passes: int,
) -> dict[str, float]:
    """Top-down: cap each child at min over present parents. Returns a new map."""
    out = dict(score_map)
    for _ in range(max_passes):
        changed = False
        for go_id in list(out):
            parents = parent_map.get(go_id)
            if not parents:
                continue
            present = [out[p] for p in parents if p in out]
            if present:
                cap = min(present)
                if cap < out[go_id]:
                    out[go_id] = cap
                    changed = True
        if not changed:
            break
    return out


def _soft_propagate_one_protein(
    block: np.ndarray,
    go_terms: np.ndarray,
    scores: np.ndarray,
    cfg: _SoftPropConfig,
    out: np.ndarray,
) -> None:
    """Blend pmax + pmin for one protein and project onto consistency."""
    score_map: dict[str, float] = {str(go_terms[i]): float(scores[i]) for i in block}
    pmax = _pmax_map(score_map, cfg.children_map, cfg.max_passes)
    pmin = _pmin_map(score_map, cfg.parent_map, cfg.max_passes)
    w = cfg.pmax_weight
    blended = {gid: w * pmax[gid] + (1.0 - w) * pmin[gid] for gid in score_map}
    # Project the blend onto parent>=child so the output is hierarchy-consistent.
    consistent = _pmax_map(blended, cfg.children_map, cfg.max_passes)
    for i in block:
        gid = str(go_terms[i])
        out[i] = np.float32(consistent[gid])


def soft_propagate_scores(
    proteins: np.ndarray,
    go_terms: np.ndarray,
    scores: np.ndarray,
    parent_map: dict[str, frozenset[str]],
    *,
    pmax_weight: float = DEFAULT_PMAX_WEIGHT,
    max_passes: int = 50,
) -> np.ndarray:
    """Apply the soft two-way Pmin/Pmax propagation, blended ``pmax_weight``.

    For each protein independently, compute the bottom-up Pmax map and the
    top-down Pmin map, blend them ``pmax_weight`` / ``1 - pmax_weight``, and
    project the blend back onto the ``parent >= child`` constraint. The
    result is hierarchy-consistent and idempotent on already-consistent
    inputs.

    Parameters
    ----------
    proteins, go_terms, scores:
        Parallel 1-D arrays describing the candidate rows.
    parent_map:
        ``{child: frozenset(direct parents)}`` (as from
        :func:`protea_reranker_lab.propagation.load_parent_map`).
    pmax_weight:
        Blend weight on the bottom-up Pmax direction (default 0.7); the
        top-down Pmin direction gets ``1 - pmax_weight``.
    max_passes:
        Convergence cap for each directional sweep.

    Returns
    -------
    ``np.ndarray`` of propagated scores aligned to the input rows
    (dtype float32).
    """
    if not 0.0 <= pmax_weight <= 1.0:
        raise ValueError(f"pmax_weight must be in [0, 1], got {pmax_weight}")
    n = len(scores)
    if n == 0:
        return scores.astype(np.float32).copy()

    cfg = _SoftPropConfig(
        parent_map=parent_map,
        children_map=build_children_map(parent_map),
        pmax_weight=pmax_weight,
        max_passes=max_passes,
    )
    out = scores.astype(np.float32).copy()
    order = np.argsort(proteins, kind="stable")
    sorted_prot = proteins[order]
    boundaries = np.flatnonzero(
        np.concatenate(([True], sorted_prot[1:] != sorted_prot[:-1], [True]))
    )
    for b in range(len(boundaries) - 1):
        block = order[boundaries[b]:boundaries[b + 1]]
        _soft_propagate_one_protein(block, go_terms, scores, cfg, out)
    return out
