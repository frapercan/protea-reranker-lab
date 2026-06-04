"""Information-Accretion sample weighting for reranker training (palanca 1).

Maps each training row's GO term to an Information Accretion value
``IA(go)`` (Clark and Radivojac 2013) and turns it into a LightGBM sample
weight. The honest hypothesis (see ``docs/source/metrics.rst`` and
``docs/source/adr/D40-ia-aligned-training.rst``) is that aligning the loss
with IA lifts the IA-weighted metric (wFmax / S_min) in the informative
region that propagation cannot fake.

Two modes, probed explicitly:

``positives``
    Weight positives by ``1 + scale * IA(go)``; every negative keeps weight
    ``1.0``. Tells the tree that missing a deep (high-IA) true term hurts
    more than missing a shallow one, without disturbing the negative mass
    set by ``neg_pos_ratio``.

``all``
    Weight every row by ``1 + scale * IA(go)``. A false positive on a
    specific (high-IA) term is also more expensive, so both error types are
    re-priced by term informativeness.

``none``
    No weighting (weight array is ``None``; LightGBM defaults to uniform).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IA_MODES = ("none", "positives", "all")


def load_ia_table(path: Path | str) -> dict[str, float]:
    """Parse a two-column ``GO:xxxxxxx\\tIA`` table into a dict.

    Blank lines and lines that do not parse to ``(go_id, float)`` are
    skipped. Negative or NaN IA values are clamped to ``0.0`` so a malformed
    row can never produce a negative sample weight.
    """
    table: dict[str, float] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            go_id = parts[0]
            try:
                ia = float(parts[1])
            except ValueError:
                continue
            if not np.isfinite(ia) or ia < 0.0:
                ia = 0.0
            table[go_id] = ia
    return table


def ia_weights(
    go_terms: np.ndarray,
    labels: np.ndarray,
    ia_table: dict[str, float],
    *,
    mode: str,
    scale: float = 1.0,
) -> np.ndarray | None:
    """Per-row sample weights from IA, aligned to ``go_terms`` / ``labels``.

    Returns ``None`` for ``mode == "none"`` (LightGBM then uses uniform
    weights). Terms absent from ``ia_table`` contribute ``IA = 0`` (weight
    ``1.0``), the same neutral floor as a negative.
    """
    if mode == "none":
        return None
    if mode not in IA_MODES:
        raise ValueError(f"unknown ia_weighting mode {mode!r}; pick one of {IA_MODES}")
    if go_terms.shape[0] != labels.shape[0]:
        raise ValueError(
            f"go_terms ({go_terms.shape[0]}) and labels ({labels.shape[0]}) "
            "length mismatch"
        )

    ia = np.fromiter(
        (ia_table.get(g, 0.0) for g in go_terms),
        count=go_terms.shape[0],
        dtype=np.float64,
    )
    weights = 1.0 + scale * ia
    if mode == "positives":
        weights = np.where(labels > 0, weights, 1.0)
    return weights.astype(np.float32, copy=False)
