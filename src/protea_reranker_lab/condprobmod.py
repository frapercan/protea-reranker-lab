"""Conditional-probability hierarchical modelling (ProtBoost "CondProbMod").

This is the single biggest external ablation lever reported by ProtBoost
(~+0.04 IA-Fmax). The idea has two halves that mirror each other:

Training half (mask)
    A child term is only *learnable* in the context of its parent. We
    therefore train the booster on the **conditional** target
    ``P(term | parent)`` and only on the rows whose parent term carries a
    non-zero signal (a non-zero prediction or a positive annotation). Rows
    whose parent is certainly absent contribute nothing to the conditional
    estimate (the child is structurally impossible there under the True
    Path Rule), so masking them out concentrates the booster's capacity on
    the decisions that matter.

Reconstruction half (parent-product)
    At inference the booster emits the per-edge conditional
    ``P(term | parent)``. The *marginal* per-term probability is then
    rebuilt by walking the GO DAG from the roots downward and multiplying
    along the path::

        P(term) = P(term | parent) * P(parent)

    with the GO roots (terms with no parent in the map) seeded at their own
    conditional value (which there equals the marginal). When a term has
    several parents we take the maximum reconstructed marginal over its
    parents, matching the True-Path-Rule semantics used elsewhere in the
    lab (a term is supported if *any* lineage supports it).

Both halves operate on flat ``(protein, go_term, value)`` triples keyed by
a ``parent_map`` (``{child: frozenset(direct parents)}``) as produced by
:func:`protea_reranker_lab.propagation.load_parent_map`. The module is pure
numpy + stdlib so it imports cleanly under the lab's no-heavy-deps CI rule.

Usage::

    from protea_reranker_lab.condprobmod import (
        build_conditional_mask, reconstruct_marginal_probs,
    )

    # Training: keep only rows whose parent has signal, and convert the
    # target to the conditional P(child | parent).
    mask, cond_target = build_conditional_mask(
        proteins, go_terms, labels, parent_map,
    )
    booster.fit(X[mask], cond_target[mask])

    # Inference: booster emits conditional edge probabilities; rebuild the
    # marginal per-term probability by the parent-product recursion.
    marginal = reconstruct_marginal_probs(
        proteins, go_terms, cond_pred, parent_map,
    )
"""

from __future__ import annotations

import numpy as np


def go_roots(parent_map: dict[str, frozenset[str]]) -> frozenset[str]:
    """Return the set of GO terms that act as roots within ``parent_map``.

    A root is any term that appears in the DAG (as a child key or as a
    parent of some child) but has no parent of its own in the map. These
    are seeded with ``P(term) = P(term | parent) = marginal`` during
    reconstruction because there is no parent to multiply by.
    """
    all_terms: set[str] = set()
    children_with_parents: set[str] = set()
    for child, parents in parent_map.items():
        all_terms.add(child)
        all_terms.update(parents)
        if parents:
            children_with_parents.add(child)
    return frozenset(all_terms - children_with_parents)


def _max_parent_value(
    go_id: str,
    value_map: dict[str, float],
    parent_map: dict[str, frozenset[str]],
) -> float:
    """Maximum *present* parent value for ``go_id`` (0.0 if no parent present).

    Only parents that exist in ``value_map`` count; a parent absent from a
    protein's candidate set contributes no signal.
    """
    parents = parent_map.get(go_id)
    if not parents:
        return 0.0
    present = [value_map[p] for p in parents if p in value_map]
    return max(present) if present else 0.0


def build_conditional_mask(
    proteins: np.ndarray,
    go_terms: np.ndarray,
    labels: np.ndarray,
    parent_map: dict[str, frozenset[str]],
    *,
    parent_signal: np.ndarray | None = None,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the CondProbMod training mask and conditional targets.

    A ``(protein, go_term)`` row is kept (mask ``True``) when the term is a
    GO root (always learnable, it has no parent to condition on) **or** when
    at least one of its parents present for the same protein carries signal
    strictly above ``threshold``. Signal is taken from ``parent_signal`` if
    given (e.g. a prior prediction column) and otherwise from ``labels``
    (train on the conditional given a positive parent annotation).

    The returned ``cond_target`` equals ``labels`` for the kept rows; the
    masked-out rows keep their original label but should be excluded via the
    mask. (The marginal->conditional rescaling that some formulations apply
    is identity here because the kept rows already condition on a present,
    signalled parent; the booster learns ``P(child | parent present)``
    directly.)

    Parameters
    ----------
    proteins, go_terms, labels:
        Parallel 1-D arrays of equal length describing the candidate rows.
    parent_map:
        ``{child: frozenset(direct parents)}``.
    parent_signal:
        Optional per-row signal used to decide whether a row's *parent* is
        active. When ``None`` the binary ``labels`` are used.
    threshold:
        A parent counts as active when its signal is ``> threshold``.

    Returns
    -------
    ``(mask, cond_target)`` where ``mask`` is a boolean array (rows to train
    on) and ``cond_target`` is the conditional target aligned to the input
    rows.
    """
    n = len(labels)
    if n == 0:
        return np.zeros(0, dtype=bool), labels.copy()

    signal = labels if parent_signal is None else parent_signal

    mask = np.zeros(n, dtype=bool)
    # Per-protein map of go_id -> signal, so a parent's activity is judged
    # within the same protein's candidate set.
    for block in _protein_blocks(proteins):
        _mask_one_protein(block, go_terms, signal, parent_map, threshold, mask)

    return mask, labels.copy()


def _protein_blocks(proteins: np.ndarray) -> list[np.ndarray]:
    """Group row indices by protein (stable order within each group)."""
    order = np.argsort(proteins, kind="stable")
    sorted_prot = proteins[order]
    boundaries = np.flatnonzero(
        np.concatenate(([True], sorted_prot[1:] != sorted_prot[:-1], [True]))
    )
    return [order[boundaries[b]:boundaries[b + 1]] for b in range(len(boundaries) - 1)]


def _mask_one_protein(
    block: np.ndarray,
    go_terms: np.ndarray,
    signal: np.ndarray,
    parent_map: dict[str, frozenset[str]],
    threshold: float,
    mask: np.ndarray,
) -> None:
    """Set ``mask`` True for this protein's learnable rows (root / active parent).

    A row is a GO root when it has no parents in ``parent_map`` (absent key
    or empty parent set); roots are always learnable.
    """
    value_map: dict[str, float] = {str(go_terms[i]): float(signal[i]) for i in block}
    for i in block:
        gid = str(go_terms[i])
        if not parent_map.get(gid):
            mask[i] = True
        elif _max_parent_value(gid, value_map, parent_map) > threshold:
            mask[i] = True


def reconstruct_marginal_probs(
    proteins: np.ndarray,
    go_terms: np.ndarray,
    cond_probs: np.ndarray,
    parent_map: dict[str, frozenset[str]],
) -> np.ndarray:
    """Rebuild marginal per-term probabilities from conditional edge probs.

    Implements the parent-product recursion ``P(term) = P(term | parent) *
    P(parent)`` walking the GO DAG from roots downward, per protein. When a
    term has several parents the maximum reconstructed marginal over its
    present parents is used (True-Path-Rule "any supporting lineage").
    Terms absent from a protein's candidate set are treated as absent
    parents and contribute no multiplier.

    Parameters
    ----------
    proteins, go_terms, cond_probs:
        Parallel 1-D arrays; ``cond_probs`` holds the booster's conditional
        ``P(term | parent)`` for each row.
    parent_map:
        ``{child: frozenset(direct parents)}``.

    Returns
    -------
    ``np.ndarray`` of marginal probabilities aligned to the input rows
    (dtype float32). Values are clipped to ``[0, 1]``.
    """
    n = len(cond_probs)
    if n == 0:
        return cond_probs.astype(np.float32).copy()

    out = np.zeros(n, dtype=np.float32)
    for block in _protein_blocks(proteins):
        _reconstruct_one_protein(block, go_terms, cond_probs, parent_map, out)
    np.clip(out, 0.0, 1.0, out=out)
    return out


def _reconstruct_one_protein(
    block: np.ndarray,
    go_terms: np.ndarray,
    cond_probs: np.ndarray,
    parent_map: dict[str, frozenset[str]],
    out: np.ndarray,
) -> None:
    """Fill ``out`` for one protein's rows via memoised parent-product."""
    cond_map: dict[str, float] = {str(go_terms[i]): float(cond_probs[i]) for i in block}
    idx_map: dict[str, int] = {str(go_terms[i]): int(i) for i in block}
    marginal_cache: dict[str, float] = {}
    visiting: set[str] = set()

    def marginal(gid: str) -> float:
        cached = marginal_cache.get(gid)
        if cached is not None:
            return cached
        cond = cond_map.get(gid)
        if cond is None:
            # Term not present for this protein: no signal to contribute.
            return 0.0
        parents = parent_map.get(gid)
        present_parents = [p for p in (parents or ()) if p in cond_map]
        if not present_parents or gid in visiting:
            # Root, orphan, or a cycle guard: the conditional equals the
            # marginal (nothing to multiply by).
            result = cond
        else:
            visiting.add(gid)
            best_parent = max(marginal(p) for p in present_parents)
            visiting.discard(gid)
            result = cond * best_parent
        marginal_cache[gid] = result
        return result

    for gid, i in idx_map.items():
        out[i] = np.float32(marginal(gid))
