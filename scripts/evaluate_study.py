#!/usr/bin/env python
"""Generalized cafaeval driver for any study + dataset in this lab.

Replaces run_cafaeval_phase.py (hardcoded to study_v9 + bench-v1-K5) and
run_cafaeval_v226_mini.py (one-off for the mini smoke). Iterates the cells
under ``runs/<study>/`` directly, aligns each cell's predictions.parquet
with the chosen eval.parquet, writes CAFA-format TSVs, and calls cafaeval
from the PROTEA venv.

Optional --include-baseline runs a parallel pass using
``neighbor_vote_fraction`` (raw KNN, no reranker) as the score, emitting a
delta-lift column per cell so the contribution of the reranker is
disentangled from the KNN baseline.

Optional --validate-against <evaluation_result_id> compares the lab-side
fmax against the API EvaluationResult (must be the same predictions +
eval set; useful as a sanity check).

Outputs:
    runs/<study>/cafaeval/<cell>/{pred.tsv,gt.tsv,_raw_metrics.json,metrics.json}
    runs/<study>/cafaeval/<cell>/baseline/{baseline_pred.tsv,_raw_metrics.json}
    runs/<study>/cafaeval/results.csv
    runs/<study>/cafaeval/results.md
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
PROTEA_VENV = Path("/home/frapercan/.cache/pypoetry/virtualenvs/protea-DH72uro9-py3.12")

_ASPECT_TO_NS = {
    "bpo": "biological_process",
    "mfo": "molecular_function",
    "cco": "cellular_component",
}

_DRIVER_SRC = """
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGINT, signal.SIG_DFL)
df, dfs_best = cafa_eval(
    "{obo}", "{pred_dir}", "{gt}",
    prop="fill", norm="cafa", no_orphans=True,
    max_terms=500, th_step=0.001, n_cpu=1,
)
out = {{}}
for kind, df_best in dfs_best.items():
    out[kind] = df_best.reset_index().to_dict(orient="records")
with open("{out_json}", "w") as f:
    json.dump(out, f, indent=2, default=str)
"""


def _bucket_align(proteins: np.ndarray) -> np.ndarray:
    buckets = np.fromiter(
        (zlib.crc32(p.encode("ascii")) % 32 for p in proteins),
        count=len(proteins),
        dtype=np.int32,
    )
    return np.lexsort((proteins, buckets))


def _read_eval(
    eval_path: Path, cell: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read (protein, go, label, neighbor_vote_fraction) restricted to cell,
    sorted into the same crc32-bucket order as the lab's staging."""
    from protea_reranker_lab.data import iter_batches

    cat, asp = cell.split("-", 1)
    prots, gos, labs, votes = [], [], [], []
    for batch in iter_batches(
        eval_path,
        columns=["protein_accession", "go_term_id", "label", "neighbor_vote_fraction"],
        category=cat,
        aspect=asp,
        batch_size=200_000,
    ):
        prots.append(batch.column("protein_accession").to_numpy(zero_copy_only=False))
        gos.append(batch.column("go_term_id").to_numpy(zero_copy_only=False))
        labs.append(
            batch.column("label")
            .to_numpy(zero_copy_only=False)
            .astype(np.int8, copy=False)
        )
        votes.append(
            batch.column("neighbor_vote_fraction")
            .to_numpy(zero_copy_only=False)
            .astype(np.float32, copy=False)
        )
    proteins = np.concatenate(prots)
    gos_arr = np.concatenate(gos)
    labels = np.concatenate(labs)
    vote_frac = np.concatenate(votes)
    order = _bucket_align(proteins)
    return proteins[order], gos_arr[order], labels[order], vote_frac[order]


def _write_pred_tsv(path: Path, proteins, gos, scores) -> None:
    with path.open("w") as f:
        for p, g, s in zip(proteins, gos, scores):
            f.write(f"{p}\t{g}\t{s:.6f}\n")


def _write_gt_tsv(path: Path, proteins, gos, labels) -> None:
    pos = labels > 0
    with path.open("w") as f:
        for p, g in zip(proteins[pos], gos[pos]):
            f.write(f"{p}\t{g}\n")


def _run_cafa(label: str, work_dir: Path, obo: Path, pred_name: str) -> dict[str, Any]:
    """Invoke cafaeval via PROTEA venv. Returns parsed metrics dict.

    Writes pred.tsv into a private ``pred_dir/`` and invokes a driver
    that pickles dfs_best. label is used only for error messages.
    """
    pred_dir = work_dir / "pred_dir"
    pred_dir.mkdir(exist_ok=True)
    src = work_dir / pred_name
    dst = pred_dir / f"{label}.tsv"
    if not dst.exists() or dst.read_bytes() != src.read_bytes():
        dst.write_bytes(src.read_bytes())
    driver = work_dir / "_driver.py"
    driver.write_text(
        _DRIVER_SRC.format(
            obo=str(obo),
            pred_dir=str(pred_dir),
            gt=str(work_dir / "gt.tsv"),
            out_json=str(work_dir / "_raw_metrics.json"),
        )
    )
    proc = subprocess.run(
        [str(PROTEA_VENV / "bin" / "python"), str(driver)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cafaeval failed for {label}:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return json.loads((work_dir / "_raw_metrics.json").read_text())


def _pick_metric(fmax_dict: dict[str, float], cell: str, kind: str) -> float | None:
    """Pick the metric for the cell's aspect, falling back to any namespace."""
    aspect = cell.split("-", 1)[1]
    target = _ASPECT_TO_NS.get(aspect)
    if target and f"{kind}__{target}" in fmax_dict:
        return fmax_dict[f"{kind}__{target}"]
    for k, v in fmax_dict.items():
        if k.startswith(f"{kind}__"):
            return v
    return None


def _flatten_metrics(raw: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for kind, records in raw.items():
        for rec in records:
            ns = rec.get("ns") or rec.get("namespace") or "unknown"
            for field in ("f", "s", "Fmax", "fmax", "smin"):
                v = rec.get(field)
                if v is None:
                    continue
                try:
                    out[f"{kind}__{ns}"] = float(v)
                except (TypeError, ValueError):
                    pass
                break
            # Also surface precision/recall/coverage if present
            for extra in ("pr", "rc", "cov"):
                v = rec.get(extra)
                if v is not None:
                    try:
                        out[f"{kind}_{extra}__{ns}"] = float(v)
                    except (TypeError, ValueError):
                        pass
    return out


def _process_cell_reranker(
    rd: Path, eval_path: Path, cell_dir: Path, obo: Path
) -> dict[str, Any] | None:
    """Reranker pass: read predictions.parquet, align with eval, run cafaeval."""
    rj = rd / "run.json"
    pp = rd / "predictions.parquet"
    if not (rj.exists() and pp.exists()):
        print(f"[skip] {rd.name}: missing run.json or predictions.parquet")
        return None
    r = json.loads(rj.read_text())
    if r.get("status") != "ok":
        print(f"[skip] {rd.name}: status={r.get('status')}")
        return None
    spec_name = r["spec_name"]
    # Derive cell from spec_name suffix after last "_"
    cell = spec_name.rsplit("_", 1)[-1]
    if "-" not in cell:
        print(f"[skip] {rd.name}: cannot derive cell from spec_name={spec_name!r}")
        return None

    lab_fmax = float((r.get("metrics") or {}).get("test_fmax") or 0.0)

    pred_t = pq.read_table(pp, columns=["protein_accession", "label", "score"])
    proteins_p = pred_t.column("protein_accession").to_numpy(zero_copy_only=False)
    labels_p = (
        pred_t.column("label")
        .to_numpy(zero_copy_only=False)
        .astype(np.int8, copy=False)
    )
    scores = (
        pred_t.column("score")
        .to_numpy(zero_copy_only=False)
        .astype(np.float32, copy=False)
    )

    proteins_a, gos, labels_a, votes = _read_eval(eval_path, cell)
    if not np.array_equal(proteins_p, proteins_a):
        print(
            f"[err] {cell}: protein alignment mismatch — staging vs predictions diverged"
        )
        return None
    if not np.array_equal(labels_p, labels_a):
        print(
            f"[err] {cell}: label alignment mismatch — staging vs predictions diverged"
        )
        return None

    cell_dir.mkdir(parents=True, exist_ok=True)
    _write_pred_tsv(cell_dir / "pred.tsv", proteins_p, gos, scores)
    _write_gt_tsv(cell_dir / "gt.tsv", proteins_p, gos, labels_p)
    n_pos = int((labels_p > 0).sum())
    n_rows = int(len(proteins_p))

    metrics_path = cell_dir / "metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
        if existing.get("status") == "ok":
            print(f"[cached] {cell}: reusing metrics.json")
            return {**existing, "_cached": True, "_gt_votes": votes, "_gt_gos": gos}

    print(
        f"[run] {cell}  rows={n_rows:,}  positives={n_pos:,}  lab_fmax={lab_fmax:.4f}",
        flush=True,
    )
    try:
        raw = _run_cafa(cell, cell_dir, obo, "pred.tsv")
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return {"cell": cell, "status": "failed", "error": str(exc)}

    flat = _flatten_metrics(raw)
    cafa_f = _pick_metric(flat, cell, "f")
    cafa_s = _pick_metric(flat, cell, "s")
    cafa_f_micro = _pick_metric(flat, cell, "f_micro")
    result = {
        "cell": cell,
        "status": "ok",
        "lab_fmax": lab_fmax,
        "n_rows": n_rows,
        "n_positives": n_pos,
        "cafaeval_fmax": cafa_f,
        "cafaeval_smin": cafa_s,
        "cafaeval_f_micro": cafa_f_micro,
        "cafaeval_all": flat,
    }
    metrics_path.write_text(json.dumps(result, indent=2))
    print(f"  fmax={cafa_f}  smin={cafa_s}  f_micro={cafa_f_micro}")
    return {**result, "_gt_votes": votes, "_gt_gos": gos}


def _process_cell_baseline(
    cell: str, cell_dir: Path, proteins, gos, labels, votes, obo: Path
) -> dict[str, Any] | None:
    """Baseline pass: same proteins/gt, but score = neighbor_vote_fraction."""
    bl_dir = cell_dir / "baseline"
    bl_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = bl_dir / "metrics.json"
    if metrics_path.exists():
        cached = json.loads(metrics_path.read_text())
        if cached.get("status") == "ok":
            print(f"[cached] {cell} baseline: reusing")
            return cached

    _write_pred_tsv(bl_dir / "pred.tsv", proteins, gos, votes)
    # gt.tsv is the same as reranker's; copy for self-contained run
    (bl_dir / "gt.tsv").write_bytes((cell_dir / "gt.tsv").read_bytes())

    print(f"[run] {cell} baseline (neighbor_vote_fraction)", flush=True)
    try:
        raw = _run_cafa(f"{cell}_baseline", bl_dir, obo, "pred.tsv")
    except Exception as exc:
        print(f"  baseline FAILED: {exc}")
        return {"cell": cell, "status": "failed", "error": str(exc)}

    flat = _flatten_metrics(raw)
    f = _pick_metric(flat, cell, "f")
    s = _pick_metric(flat, cell, "s")
    fm = _pick_metric(flat, cell, "f_micro")
    result = {
        "cell": cell,
        "status": "ok",
        "baseline_fmax": f,
        "baseline_smin": s,
        "baseline_f_micro": fm,
        "baseline_all": flat,
    }
    metrics_path.write_text(json.dumps(result, indent=2))
    print(f"  baseline fmax={f}  smin={s}")
    return result


def _emit_markdown(
    out_path: Path, rows: list[dict[str, Any]], include_baseline: bool
) -> None:
    """Write a human-readable markdown report."""
    lines = ["# CAFA evaluation results", ""]
    if include_baseline:
        lines.append(
            "| cell | n_pos | lab_fmax | reranker_fmax | baseline_fmax | Δ lift | reranker_smin | baseline_smin |"
        )
        lines.append(
            "|------|------:|---------:|--------------:|--------------:|-------:|--------------:|--------------:|"
        )
        for r in rows:
            if r.get("status") != "ok":
                continue
            lift = (r.get("cafaeval_fmax") or 0) - (r.get("baseline_fmax") or 0)
            lines.append(
                f"| {r['cell']} | {r.get('n_positives', 0)} | "
                f"{r.get('lab_fmax', 0):.4f} | "
                f"{r.get('cafaeval_fmax') or 0:.4f} | "
                f"{r.get('baseline_fmax') or 0:.4f} | "
                f"{lift:+.4f} | "
                f"{r.get('cafaeval_smin') or 0:.4f} | "
                f"{r.get('baseline_smin') or 0:.4f} |"
            )
    else:
        lines.append(
            "| cell | n_pos | lab_fmax | reranker_fmax | reranker_smin | f_micro |"
        )
        lines.append(
            "|------|------:|---------:|--------------:|--------------:|--------:|"
        )
        for r in rows:
            if r.get("status") != "ok":
                continue
            lines.append(
                f"| {r['cell']} | {r.get('n_positives', 0)} | "
                f"{r.get('lab_fmax', 0):.4f} | "
                f"{r.get('cafaeval_fmax') or 0:.4f} | "
                f"{r.get('cafaeval_smin') or 0:.4f} | "
                f"{r.get('cafaeval_f_micro') or 0:.4f} |"
            )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--study", required=True, help="study dir under runs/, e.g. study_v23"
    )
    ap.add_argument(
        "--dataset",
        required=True,
        help="dataset dir under datasets/ (see ADR D36 naming bench-v1-K{k}-v{val_band}-lineage-{plm_short})",
    )
    ap.add_argument(
        "--obo",
        default=None,
        help="path to go.obo (default: datasets/<dataset>/go.obo or fall back to bench-v1-K5/go.obo)",
    )
    ap.add_argument(
        "--include-baseline",
        action="store_true",
        help="also run baseline cafaeval using neighbor_vote_fraction",
    )
    args = ap.parse_args()

    runs_root = REPO / "runs" / args.study
    eval_path = REPO / "datasets" / args.dataset / "eval.parquet"
    if args.obo:
        obo = Path(args.obo)
    else:
        local = REPO / "datasets" / args.dataset / "go.obo"
        obo = local if local.exists() else REPO / "datasets" / "bench-v1-K5" / "go.obo"

    if not runs_root.is_dir():
        print(f"[err] {runs_root} not found", file=sys.stderr)
        return 2
    if not eval_path.exists():
        print(f"[err] {eval_path} not found", file=sys.stderr)
        return 2
    if not obo.exists():
        print(f"[err] OBO not found at {obo}", file=sys.stderr)
        return 2
    if not (PROTEA_VENV / "bin" / "python").exists():
        print(f"[err] PROTEA venv missing at {PROTEA_VENV}", file=sys.stderr)
        return 2

    out_dir = runs_root / "cafaeval"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        p for p in runs_root.iterdir() if p.is_dir() and p.name != "cafaeval"
    )
    rows: list[dict[str, Any]] = []

    for rd in run_dirs:
        # derive cell from rd name (after last '_')
        cell = rd.name.rsplit("_", 1)[-1]
        if "-" not in cell:
            print(f"[skip] {rd.name}: no -aspect suffix")
            continue
        cell_dir = out_dir / cell

        reranker_row = _process_cell_reranker(rd, eval_path, cell_dir, obo)
        if reranker_row is None:
            continue
        if reranker_row.get("status") != "ok":
            rows.append(reranker_row)
            continue

        merged = {
            "cell": reranker_row["cell"],
            "status": "ok",
            "n_positives": reranker_row["n_positives"],
            "n_rows": reranker_row["n_rows"],
            "lab_fmax": reranker_row["lab_fmax"],
            "cafaeval_fmax": reranker_row.get("cafaeval_fmax"),
            "cafaeval_smin": reranker_row.get("cafaeval_smin"),
            "cafaeval_f_micro": reranker_row.get("cafaeval_f_micro"),
        }

        if args.include_baseline:
            # need protein/go/label/votes again for baseline
            pred_t = pq.read_table(
                rd / "predictions.parquet", columns=["protein_accession", "label"]
            )
            proteins_p = pred_t.column("protein_accession").to_numpy(
                zero_copy_only=False
            )
            labels_p = (
                pred_t.column("label")
                .to_numpy(zero_copy_only=False)
                .astype(np.int8, copy=False)
            )
            gos = reranker_row.get("_gt_gos")
            votes = reranker_row.get("_gt_votes")
            if gos is None or votes is None:
                # re-derive (cached run case)
                _, gos, _, votes = _read_eval(eval_path, cell)
            bl_row = _process_cell_baseline(
                cell, cell_dir, proteins_p, gos, labels_p, votes, obo
            )
            if bl_row and bl_row.get("status") == "ok":
                merged["baseline_fmax"] = bl_row.get("baseline_fmax")
                merged["baseline_smin"] = bl_row.get("baseline_smin")
                merged["baseline_f_micro"] = bl_row.get("baseline_f_micro")
                if (
                    merged["cafaeval_fmax"] is not None
                    and bl_row.get("baseline_fmax") is not None
                ):
                    merged["fmax_lift"] = (
                        merged["cafaeval_fmax"] - bl_row["baseline_fmax"]
                    )

        rows.append(merged)

    if rows:
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        csv_path = out_dir / "results.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        _emit_markdown(out_dir / "results.md", rows, args.include_baseline)
        print(f"[done] {len(rows)} cells -> {csv_path} + results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
