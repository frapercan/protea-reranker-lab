"""Per-cell CAFA Fmax on numpy arrays.

Mirrors PROTEA's ``run_cafa_evaluation`` fmax logic but operates on flat
numpy arrays plus a contiguous ``group_sizes`` vector — no pandas, no
DataFrame materialisation. Used after :func:`predict_streaming`.
"""

from __future__ import annotations

import numpy as np


def fmax_per_protein_group(
    scores: np.ndarray,
    labels: np.ndarray,
    group_sizes: np.ndarray,
    *,
    n_thresholds: int = 101,
) -> float:
    """Protein-averaged Fmax.

    Inputs are flat row-aligned arrays (rows sorted by protein, contiguous
    per protein) and ``group_sizes`` gives the number of rows per protein.
    """
    if scores.size == 0 or group_sizes.size == 0:
        return 0.0

    edges = np.empty(group_sizes.size + 1, dtype=np.int64)
    edges[0] = 0
    np.cumsum(group_sizes, out=edges[1:])

    lo, hi = float(scores.min()), float(scores.max())
    thresholds = np.array([lo]) if hi <= lo else np.linspace(lo, hi, n_thresholds)

    starts = edges[:-1]

    pos_mask = labels > 0
    n_pos_per_group = np.add.reduceat(pos_mask.astype(np.int64), starts)
    has_pos = n_pos_per_group > 0

    best = 0.0
    for t in thresholds:
        pred = scores >= t
        n_pred_per_group = np.add.reduceat(pred.astype(np.int64), starts)
        tp_per_group = np.add.reduceat((pred & pos_mask).astype(np.int64), starts)

        with np.errstate(divide="ignore", invalid="ignore"):
            precisions = np.where(
                n_pred_per_group > 0,
                tp_per_group / np.maximum(n_pred_per_group, 1),
                0.0,
            )
            recalls = np.where(
                has_pos,
                tp_per_group / np.maximum(n_pos_per_group, 1),
                0.0,
            )

        m_pred = n_pred_per_group > 0
        if not m_pred.any():
            continue
        p = float(precisions[m_pred].mean()) if m_pred.any() else 0.0
        r = float(recalls[has_pos].mean()) if has_pos.any() else 0.0
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f > best:
                best = f
    return float(best)


def fmax_binary(
    y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 101,
) -> float:
    if y_true.size == 0:
        return 0.0
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    pos = y_true > 0
    best = 0.0
    for t in thresholds:
        pred = y_score >= t
        tp = int((pred & pos).sum())
        if tp == 0:
            continue
        fp = int((pred & ~pos).sum())
        fn = int((~pred & pos).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f > best:
                best = f
    return float(best)
