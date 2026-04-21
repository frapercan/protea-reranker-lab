"""Per-cell CAFA Fmax on held-out predictions.

Mirrors PROTEA's ``run_cafa_evaluation`` fmax logic but takes pandas input
directly so it works on parquet dumps. For a full apples-to-apples number
against PROTEA, use ``cafaeval`` externally — this is the fast in-process
metric used during sweeps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fmax_binary(y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 101) -> float:
    """Protein-averaged Fmax as used in CAFA: sweeps thresholds, averages per
    protein precision/recall, returns max F1 over the sweep.

    Expects ``y_true`` binary labels and ``y_score`` continuous scores, both
    flat over (protein × candidate GO term). The function itself does NOT
    group by protein — the caller should pass a single cell's predictions,
    or call ``fmax_per_protein_group`` for the protein-averaged variant.
    """
    if len(y_true) == 0:
        return 0.0
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    pos = y_true > 0
    best = 0.0
    for t in thresholds:
        pred = y_score >= t
        tp = np.sum(pred & pos)
        if tp == 0:
            continue
        fp = np.sum(pred & ~pos)
        fn = np.sum(~pred & pos)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f > best:
                best = f
    return float(best)


def fmax_per_protein_group(
    df: pd.DataFrame, score_col: str = "score", *, n_thresholds: int = 101
) -> float:
    """Protein-averaged Fmax: for each threshold, compute precision and recall
    per protein, then average, then take the max F1 over the sweep.

    ``df`` must have ``protein_accession``, ``label``, and ``score_col`` columns.
    """
    if df.empty:
        return 0.0
    scores = df[score_col].to_numpy()
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        thresholds = np.array([lo])
    else:
        thresholds = np.linspace(lo, hi, n_thresholds)

    grouped = df.groupby("protein_accession", sort=False)
    prot_pos = grouped["label"].sum().to_numpy()
    any_positive = prot_pos > 0

    best = 0.0
    for t in thresholds:
        pred = df[score_col] >= t
        tp = df.loc[pred, :].groupby("protein_accession", sort=False)["label"].sum()
        n_pred = pred.groupby(df["protein_accession"], sort=False).sum()
        precisions = (tp / n_pred).replace([np.inf, np.nan], 0.0)
        if n_pred.sum() == 0:
            continue
        recalls = (tp / grouped["label"].sum()).replace([np.inf, np.nan], 0.0)
        p = precisions[n_pred > 0].mean() if (n_pred > 0).any() else 0.0
        r = recalls[any_positive].mean() if any_positive.any() else 0.0
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f > best:
                best = float(f)
    return best


def eval_cells(
    df: pd.DataFrame, score_col: str = "score", *, n_thresholds: int = 101
) -> dict[tuple[str, str], float]:
    """Compute Fmax for every (category, aspect) cell present in ``df``."""
    out: dict[tuple[str, str], float] = {}
    for (cat, asp), sub in df.groupby(["category", "aspect"], sort=False):
        out[(cat, asp)] = fmax_per_protein_group(
            sub, score_col=score_col, n_thresholds=n_thresholds
        )
    return out


def avg_fmax(cells: dict[tuple[str, str], float]) -> float:
    vals = [v for v in cells.values() if v is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0
