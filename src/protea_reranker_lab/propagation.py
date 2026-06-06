"""True-Path-Rule label propagation helpers used by :mod:`staging`.

A ``parent_map.json`` (produced by ``scripts/export_parent_map.py``) maps each
GO term to its direct is_a / part_of parents. ``propagate_labels_to_ancestors``
expands every protein's leaf positives up that DAG so any row whose
``(protein, go_term)`` is in the closure is re-labelled positive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_parent_map(path: Path | str) -> dict[str, frozenset[str]]:
    """Read a parent_map.json (as produced by ``scripts/export_parent_map.py``)
    and return ``{child_go: frozenset(direct parents)}``."""
    with open(path) as f:
        payload = json.load(f)
    if "parents" in payload:
        raw = payload["parents"]
    else:
        raw = payload
    return {child: frozenset(parents) for child, parents in raw.items()}


def propagate_labels_to_ancestors(
    proteins: np.ndarray,
    go_terms: np.ndarray,
    labels: np.ndarray,
    parent_map: dict[str, frozenset[str]],
) -> tuple[np.ndarray, int]:
    """True-Path-Rule label propagation.

    For each protein, expand its set of leaf positives with every is_a /
    part_of ancestor; any row whose ``(protein, go_term)`` is in the closure
    is then re-labelled as positive.

    Returns ``(new_labels, n_promoted)`` where ``n_promoted`` counts rows
    flipped from 0 to 1.
    """
    if labels.size == 0:
        return labels.copy(), 0

    ancestor_cache: dict[str, frozenset[str]] = {}

    def _ancestors(go_id: str) -> frozenset[str]:
        cached = ancestor_cache.get(go_id)
        if cached is not None:
            return cached
        seen: set[str] = set()
        stack = [go_id]
        while stack:
            cur = stack.pop()
            for p in parent_map.get(cur, ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        result = frozenset(seen)
        ancestor_cache[go_id] = result
        return result

    closure: dict[str, set[str]] = {}
    pos_idx = np.flatnonzero(labels > 0)
    for i in pos_idx:
        prot = proteins[i]
        gid = go_terms[i]
        bucket = closure.setdefault(prot, set())
        bucket.add(gid)
        bucket.update(_ancestors(gid))

    new_labels = labels.copy()
    n_promoted = 0
    for i in range(len(labels)):
        if new_labels[i]:
            continue
        cl = closure.get(proteins[i])
        if cl is not None and go_terms[i] in cl:
            new_labels[i] = 1
            n_promoted += 1
    return new_labels, n_promoted
