#!/usr/bin/env python
"""Phase 5 driver — cafaeval re-validation of v9 winners.

For each cell winner from F1, re-aligns ``(protein, go_term_id, label)`` with
the trainer's ``predictions.parquet`` (same crc32-bucket sort that staging
applied), writes CAFA-format TSVs, and calls ``cafaeval.cafa_eval`` from the
PROTEA venv (the lab venv does not ship cafaeval).

Output:
    runs/study_v9/cafaeval/<cell>/{pred.tsv,gt.tsv,out/,metrics.json}
    runs/study_v9/cafaeval/results.csv  -- per-cell {lab_fmax, cafaeval_fmax, delta}

The cafa_eval call mirrors PROTEA's ``run_cafa_evaluation``:
    prop="fill", norm="cafa", no_orphans=True, max_terms=500, th_step=0.001

IA path is resolved via the band-registry bridge (band "v226" for the
v9/study bench).  Set LAB_IA_V226=/path/to/IA_cafa6.tsv to override.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

# Registry bridge: resolves (band, cutoff) -> (OBO path, IA path).
# Import is done inside the script so it works from the scripts/ pythonpath.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from protea_reranker_lab.band_registry_bridge import (  # noqa: E402
    BandMismatchError,
    resolve_band_artifacts,
)

REPO = Path(__file__).resolve().parents[1]
SOURCE_EVAL = REPO / "datasets" / "bench-v1-K5" / "eval.parquet"
RESULTS_CSV = REPO / "runs" / "study_v9" / "replication" / "results.csv"
OUT_DIR = REPO / "runs" / "study_v9" / "cafaeval"
PROTEA_VENV = Path("/home/frapercan/.cache/pypoetry/virtualenvs/protea-wR7652Au-py3.12")

# Resolve OBO and IA for the v226 bench band (the band this script evaluates).
# LAB_OBO_V226 / LAB_IA_V226 env vars override the search-list if set.
try:
    OBO_PATH, IA_PATH = resolve_band_artifacts("v226")
except (FileNotFoundError, BandMismatchError) as _e:
    # Fall back gracefully at import time; main() will report the error.
    OBO_PATH = REPO / "datasets" / "bench-v1-K5" / "go.obo"
    IA_PATH = Path(os.environ.get("LAB_IA_V226", ""))


def _winners(rows: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        cell = r["spec"].rsplit("_seed", 1)[0]
        try:
            f = float(r["fmax"])
        except (TypeError, ValueError):
            continue
        if cell not in best or f > best[cell]["fmax"]:
            best[cell] = {**r, "fmax": f, "cell": cell}
    return best


def _extract_alignment(cell: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-derive (protein, go, label) for a cell in the same bucket-sort order
    that ``staging.stage_for_training`` produced. crc32 % 32 → bucket; within
    each bucket, stable sort by ``protein_accession``.
    """
    from protea_reranker_lab.data import iter_batches

    cat, asp = cell.split("-", 1)
    prots, gos, labs = [], [], []
    for batch in iter_batches(
        SOURCE_EVAL,
        columns=["protein_accession", "go_term_id", "label"],
        category=cat, aspect=asp, batch_size=200_000,
    ):
        prots.append(batch.column("protein_accession").to_numpy(zero_copy_only=False))
        gos.append(batch.column("go_term_id").to_numpy(zero_copy_only=False))
        labs.append(batch.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False))
    proteins = np.concatenate(prots)
    go_terms = np.concatenate(gos)
    labels = np.concatenate(labs)

    buckets = np.fromiter(
        (zlib.crc32(p.encode("ascii")) % 32 for p in proteins),
        count=len(proteins), dtype=np.int32,
    )
    # primary=bucket, secondary=protein, both stable ascending
    order = np.lexsort((proteins, buckets))
    return proteins[order], go_terms[order], labels[order]


def _write_pred_tsv(path: Path, proteins: np.ndarray,
                    gos: np.ndarray, scores: np.ndarray) -> None:
    """Cafaeval prediction file: protein\\tgo\\tscore (no header)."""
    with path.open("w") as f:
        for p, g, s in zip(proteins, gos, scores):
            f.write(f"{p}\t{g}\t{s:.6f}\n")


def _write_gt_tsv(path: Path, proteins: np.ndarray,
                  gos: np.ndarray, labels: np.ndarray) -> None:
    """Cafaeval ground-truth: protein\\tgo (positives only, no header)."""
    pos = labels > 0
    with path.open("w") as f:
        for p, g in zip(proteins[pos], gos[pos]):
            f.write(f"{p}\t{g}\n")


def _run_cafa_eval(cell: str, cell_dir: Path) -> dict[str, Any]:
    """Invoke cafaeval via the PROTEA venv (lab venv lacks the package).

    Writes a Python driver to a temp file and executes it; returns the parsed
    metrics dict (one entry per namespace).
    """
    pred_dir = cell_dir / "pred_dir"
    pred_dir.mkdir(exist_ok=True)
    # cafaeval treats each TSV in pred_dir as one "prediction"; we have one.
    # Atomic move: link the existing pred.tsv into pred_dir/.
    src = cell_dir / "pred.tsv"
    dst = pred_dir / f"{cell}.tsv"
    if not dst.exists() or dst.read_bytes() != src.read_bytes():
        dst.write_bytes(src.read_bytes())

    out_dir = cell_dir / "out"
    out_dir.mkdir(exist_ok=True)

    driver = cell_dir / "_driver.py"
    driver.write_text(_DRIVER_SRC.format(
        obo=str(OBO_PATH),
        pred_dir=str(pred_dir),
        gt=str(cell_dir / "gt.tsv"),
        ia=str(IA_PATH),
        out_json=str(cell_dir / "_raw_metrics.json"),
    ))

    import subprocess
    proc = subprocess.run(
        [str(PROTEA_VENV / "bin" / "python"), str(driver)],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cafaeval failed for {cell}:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )

    raw = json.loads((cell_dir / "_raw_metrics.json").read_text())
    return raw


_DRIVER_SRC = '''
import json
import signal
from cafaeval.evaluation import cafa_eval

# cafaeval forks worker processes; reset signal handlers to defaults so SIGTERM
# from the pool actually terminates them.
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGINT, signal.SIG_DFL)

df, dfs_best = cafa_eval(
    "{obo}",
    "{pred_dir}",
    "{gt}",
    ia="{ia}",
    prop="fill",
    norm="cafa",
    no_orphans=True,
    max_terms=500,
    th_step=0.001,
    n_cpu=1,
    weighted_only=False,
)

# Extract all metrics per namespace from dfs_best.
metrics = {{}}
for metric_kind, df_best in dfs_best.items():
    rec = df_best.reset_index().to_dict(orient="records")
    metrics[metric_kind] = rec

with open("{out_json}", "w") as f:
    json.dump(metrics, f, indent=2, default=str)
'''


def _extract_fmax(raw: dict) -> dict[str, float]:
    """Pick the protein-centric Fmax per namespace from cafaeval output.

    cafaeval returns ``dfs_best`` with one entry per (norm, metric); the entry
    keyed ``f`` (Fmax) under norm ``cafa`` is the protein-centric Fmax.
    """
    out: dict[str, float] = {}
    # dfs_best keys are tuples-as-strings; the metric name is in the records.
    for metric_kind, records in raw.items():
        for rec in records:
            ns = rec.get("ns") or rec.get("namespace") or "unknown"
            f_value = rec.get("f") or rec.get("Fmax") or rec.get("fmax")
            if f_value is None:
                continue
            try:
                f_val = float(f_value)
            except (TypeError, ValueError):
                continue
            key = f"{metric_kind}__{ns}"
            out[key] = f_val
    return out


_ASPECT_TO_NS = {
    "bpo": "biological_process",
    "mfo": "molecular_function",
    "cco": "cellular_component",
}


def _pick_cell_fmax(fmax_dict: dict[str, float], cell: str) -> float | None:
    """Pick the cafaeval Fmax that matches the cell's aspect.

    Multi-namespace cells (where some predictions land in a sibling namespace
    via the OBO lookup) report several ``f__<ns>``; we want the one matching
    the cell aspect, not whatever happens to come first.
    """
    aspect = cell.split("-", 1)[1] if "-" in cell else cell
    target_ns = _ASPECT_TO_NS.get(aspect)
    if target_ns is not None:
        target_key = f"f__{target_ns}"
        if target_key in fmax_dict:
            return fmax_dict[target_key]
    # fall back to any ``f__*``
    for k, v in fmax_dict.items():
        if k.startswith("f__"):
            return v
    return None


def main() -> int:
    print(f"[info] OBO: {OBO_PATH}")
    print(f"[info] IA : {IA_PATH}")
    if not OBO_PATH.exists():
        print(f"[err] OBO not found at {OBO_PATH}; download it first")
        return 2
    if not IA_PATH or not Path(IA_PATH).exists():
        print(
            f"[err] IA file not found at {IA_PATH}. "
            "Set LAB_IA_V226=/path/to/IA_cafa6.tsv or ensure the file is in "
            "one of the standard search roots."
        )
        return 2
    if not RESULTS_CSV.exists():
        print(f"[err] {RESULTS_CSV} missing -- run F1 first")
        return 2

    rows = list(csv.DictReader(RESULTS_CSV.open()))
    winners = _winners(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flat_rows: list[dict[str, Any]] = []

    for cell in sorted(winners):
        winner = winners[cell]
        pred_pq = REPO / winner["output_dir"] / "predictions.parquet"
        if not pred_pq.exists():
            print(f"[skip] {cell}: missing {pred_pq}")
            continue

        cell_dir = OUT_DIR / cell
        cell_dir.mkdir(parents=True, exist_ok=True)

        # 1. Read predictions (bucket-sorted order)
        pred_t = pq.read_table(pred_pq, columns=["protein_accession", "label", "score"])
        proteins_p = pred_t.column("protein_accession").to_numpy(zero_copy_only=False)
        labels_p = pred_t.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
        scores = pred_t.column("score").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

        # 2. Re-align (protein, go, label)
        proteins_a, gos, labels_a = _extract_alignment(cell)
        if not np.array_equal(proteins_p, proteins_a):
            print(f"[err] {cell}: protein alignment mismatch")
            continue
        if not np.array_equal(labels_p, labels_a):
            print(f"[err] {cell}: label alignment mismatch — staging vs predictions diverged")
            continue

        # 3. Write TSVs
        pred_tsv = cell_dir / "pred.tsv"
        gt_tsv = cell_dir / "gt.tsv"
        _write_pred_tsv(pred_tsv, proteins_p, gos, scores)
        _write_gt_tsv(gt_tsv, proteins_p, gos, labels_p)
        n_pos = int((labels_p > 0).sum())

        # 4. Run cafaeval via PROTEA venv
        print(f"[run] {cell}  rows={len(proteins_p):,}  positives={n_pos:,}  "
              f"lab_fmax={winner['fmax']:.4f}", flush=True)
        try:
            raw = _run_cafa_eval(cell, cell_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        fmax_dict = _extract_fmax(raw)
        cafa_f = _pick_cell_fmax(fmax_dict, cell)
        (cell_dir / "metrics.json").write_text(json.dumps({
            "cell": cell,
            "lab_fmax": winner["fmax"],
            "n_rows": len(proteins_p),
            "n_positives": n_pos,
            "cafaeval_fmax": cafa_f,
            "cafaeval_fmax_per_namespace": fmax_dict,
        }, indent=2))
        flat_rows.append({
            "cell": cell,
            "lab_fmax": winner["fmax"],
            "cafaeval_fmax": cafa_f,
            "delta": (winner["fmax"] - cafa_f) if cafa_f is not None else None,
            "n_positives": n_pos,
        })
        print(f"  cafaeval fmax: {cafa_f}  (full per-ns: {fmax_dict})")

    if flat_rows:
        flat_csv = OUT_DIR / "results.csv"
        with flat_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            w.writeheader()
            w.writerows(flat_rows)
        print(f"[done] {len(flat_rows)} cells → {flat_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
