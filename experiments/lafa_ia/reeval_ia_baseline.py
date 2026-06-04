#!/usr/bin/env python
"""F-LAFA-IA.0 -- IA-weighted re-evaluation of the existing seed42 runs.

Re-runs ``cafaeval.cafa_eval`` with ``ia=`` over the EXISTING cafaeval
prediction/ground-truth pairs already on disk for the v26-binary champion
study, for BOTH the reranker arm and the KNN baseline arm, and records the
four reference numbers per cell:

    internal Fmax  -- lab proxy (no propagation), from the study results.csv
    cafaeval Fmax  -- propagated CAFA Fmax (prop=fill, norm=cafa), unweighted
    wFmax          -- IA-weighted Fmax (the ``f`` value at the f_w-best tau)
    S_min          -- minimum semantic distance (unweighted and IA-weighted)

No training is run and no predictions are regenerated. The cafaeval call
mirrors the lab sweeps exactly except for adding ``ia=``:
    prop="fill", norm="cafa", no_orphans=True, max_terms=500,
    th_step=0.001, n_cpu=1.

Why study_v26_binary and not phase3d_K3_prostt5: the IA-ALIGNED brief
table cites prostt5/K3 reranker numbers, but only study_v26_binary
(bench-v1-K5-v226-binary) retains the KNN baseline predictions on disk
(``<cell>/baseline/pred_dir`` + ``baseline/gt.tsv``). phase3d stores the
KNN baseline only as a hardcoded Fmax constant, so its baseline arm is
not IA-re-evaluable without regenerating predictions, which this slice
explicitly must not do. The reranker arm is the same v26-binary recipe in
both; the baseline arm is the KNN-only vote score.

cafaeval is imported from the source tree (the PROTEA poetry venv that the
original sweeps used no longer exists). Run with the lab venv plus the
cafaeval-protea ``src`` on ``PYTHONPATH``::

    PYTHONPATH=$CAFA/src .venv/bin/python \
        experiments/lafa_ia/reeval_ia_baseline.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
# Inputs that live outside this worktree (gitignored large artefacts: the
# OBO and the on-disk runs). They are read-only here. The IA table is
# tracked and resolved relative to this worktree.
DEV_LAB = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab")
OBO = DEV_LAB / "datasets" / "bench-v1-K5" / "go.obo"
IA = REPO / "datasets" / "ia" / "IA-swissprot-exp-v227.txt"
STUDY = DEV_LAB / "runs" / "study_v26_binary" / "cafaeval"
RESULTS_CSV = STUDY / "results.csv"

CELLS = ["lk-bpo", "lk-mfo", "lk-cco"]
SEED = 42
OUT_DIR = REPO / "runs" / "lafa_ia"
OUT_JSON = OUT_DIR / "reeval_ia_baseline.json"


def _internal_fmax() -> dict[str, float]:
    """Lab proxy Fmax (no propagation) per cell, from the study results.csv.

    Only the reranker arm has a stored internal proxy; the KNN baseline
    candidate-score parquet was not retained, so its internal proxy is not
    recoverable here.
    """
    out: dict[str, float] = {}
    with RESULTS_CSV.open() as fh:
        for row in csv.DictReader(fh):
            if row["cell"] in CELLS and row.get("lab_fmax"):
                out[row["cell"]] = float(row["lab_fmax"])
    return out


def _eval_one(pred_dir: Path, gt: Path) -> dict[str, float | None]:
    """cafaeval (with ia=) over one prediction/gt pair; return the metrics."""
    from cafaeval.evaluation import cafa_eval

    df, best = cafa_eval(
        str(OBO), str(pred_dir), str(gt), ia=str(IA),
        prop="fill", norm="cafa", no_orphans=True,
        max_terms=500, th_step=0.001, n_cpu=1,
    )
    cafaeval_fmax = float(best["f"]["f"].iloc[0])
    wfmax = float(best["f_w"]["f"].iloc[0]) if "f_w" in best else None
    smin = float(best["s"]["s"].iloc[0])
    smin_w = float(df["s_w"].min()) if "s_w" in df.columns else None
    return {
        "cafaeval_fmax": cafaeval_fmax,
        "wfmax": wfmax,
        "smin": smin,
        "smin_weighted": smin_w,
    }


def main() -> None:
    for path in (OBO, IA, RESULTS_CSV):
        if not path.exists():
            raise SystemExit(f"required input missing: {path}")

    internal = _internal_fmax()
    rows: list[dict[str, Any]] = []

    for cell in CELLS:
        cell_dir = STUDY / cell
        rr_pred = cell_dir / "pred_dir"
        rr_gt = cell_dir / "gt.tsv"
        bl_pred = cell_dir / "baseline" / "pred_dir"
        bl_gt = cell_dir / "baseline" / "gt.tsv"

        reranker = _eval_one(rr_pred, rr_gt)
        reranker["internal_fmax"] = internal.get(cell)
        reranker["arm"] = "reranker"
        reranker["cell"] = cell

        baseline = _eval_one(bl_pred, bl_gt)
        baseline["internal_fmax"] = None  # candidate-score parquet not retained
        baseline["arm"] = "knn_baseline"
        baseline["cell"] = cell

        rows.append(reranker)
        rows.append(baseline)
        print(
            f"{cell:>7s} reranker  Fmax={reranker['cafaeval_fmax']:.4f} "
            f"wFmax={reranker['wfmax']:.4f} Smin={reranker['smin']:.4f} "
            f"Smin_w={reranker['smin_weighted']:.4f}",
            flush=True,
        )
        print(
            f"{cell:>7s} knn-base  Fmax={baseline['cafaeval_fmax']:.4f} "
            f"wFmax={baseline['wfmax']:.4f} Smin={baseline['smin']:.4f} "
            f"Smin_w={baseline['smin_weighted']:.4f}",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "study": "study_v26_binary",
        "dataset": "bench-v1-K5-v226-binary",
        "seed": SEED,
        "ia_table": str(IA.relative_to(REPO)),
        "obo": "datasets/bench-v1-K5/go.obo",
        "cafaeval_call": {
            "prop": "fill", "norm": "cafa", "no_orphans": True,
            "max_terms": 500, "th_step": 0.001, "n_cpu": 1, "ia": True,
        },
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
