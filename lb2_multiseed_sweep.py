"""LB.2 multi-seed sweep: 6 cells x 3 seeds = 18 runs.

Replicates v23 config (no anc2vec, no pca) across seeds 42, 7, 137
for all NK+LK cells on bench-v1-K5-v226-lineage.

Writes:
  - runs/lb2_multiseed/<cell>_seed<n>/run.json (training artefact)
  - runs/lb2_multiseed/<cell>_seed<n>/cafaeval_metrics.json
  - runs/lb2_multiseed/cis.json (bootstrap 95% CIs)
  - runs/lb2_multiseed/summary.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# ─── Paths ──────────────────────────────────────────────────────────────────
WORKTREE = Path(__file__).resolve().parent
REPO_DATASETS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets")
DATASET_DIR = REPO_DATASETS / "bench-v1-K5-v226-lineage"
OBO_PATH = REPO_DATASETS / "bench-v1-K5" / "go.obo"
RUNS_ROOT = WORKTREE / "runs" / "lb2_multiseed"

# PROTEA venv for cafaeval (lab venv doesn't have cafaeval)
PROTEA_VENV = Path("/home/frapercan/.cache/pypoetry/virtualenvs/protea-DH72uro9-py3.12")
LAB_PYTHON = WORKTREE / ".venv" / "bin" / "python"

# ─── Config ─────────────────────────────────────────────────────────────────
CELLS = ["nk-mfo", "nk-bpo", "nk-cco", "lk-mfo", "lk-bpo", "lk-cco"]
SEEDS = [42, 7, 137]

# v23 drop_features (no anc2vec, no pca)
DROP_FEATURES = [
    "anc2vec_has_emb",
    "anc2vec_neighbor_cos",
    "anc2vec_neighbor_maxcos",
    "anc2vec_query_known_cos",
    "anc2vec_query_known_count",
    "anc2vec_query_known_maxcos",
    "emb_pca_query_0",
    "emb_pca_query_1",
    "emb_pca_query_10",
    "emb_pca_query_11",
    "emb_pca_query_12",
    "emb_pca_query_13",
    "emb_pca_query_14",
    "emb_pca_query_15",
    "emb_pca_query_2",
    "emb_pca_query_3",
    "emb_pca_query_4",
    "emb_pca_query_5",
    "emb_pca_query_6",
    "emb_pca_query_7",
    "emb_pca_query_8",
    "emb_pca_query_9",
]

ASPECT_TO_NS = {
    "bpo": "biological_process",
    "mfo": "molecular_function",
    "cco": "cellular_component",
}

# Baseline cafaeval fmax from study_v23 (seed=42, same config)
BASELINE_CAFAEVAL_FMAX = {
    "nk-mfo": 0.7112098758994384,
    "nk-bpo": 0.5598660370394628,
    "nk-cco": 0.7733465563847183,
    "lk-mfo": 0.6998213995564616,
    "lk-bpo": 0.6596985578537196,
    "lk-cco": 0.7433980049032997,
    "pk-mfo": 0.1608114912198337,
    "pk-bpo": 0.14959499221253486,
    "pk-cco": 0.3090467316947692,
}
BASELINE_RERANKER_FMAX = {  # from study_v23 results.csv (baseline = KNN-only)
    "nk-mfo": 0.6447077458266381,
    "nk-bpo": 0.5333487093513323,
    "nk-cco": 0.6999748236935452,
    "lk-mfo": 0.5815861789905945,
    "lk-bpo": 0.5844391871418759,
    "lk-cco": 0.7052765825164343,
    "pk-mfo": 0.48306719848909857,
    "pk-bpo": 0.4030745350373079,
    "pk-cco": 0.6009257401293558,
}


def _run_key(cell: str, seed: int) -> str:
    return f"{cell}_seed{seed}"


def _run_dir(cell: str, seed: int) -> Path:
    return RUNS_ROOT / _run_key(cell, seed)


def _make_spec_yaml(cell: str, seed: int, out_dir: Path) -> Path:
    spec = {
        "schema_version": "v1",
        "name": f"lb2_multiseed_{cell}_seed{seed}",
        "dataset": {
            "manifest": str(DATASET_DIR / "manifest.json"),
        },
        "model": {
            "kind": "lgbm_reranker",
            "defaults": {
                "objective": "lambdarank",
                "num_boost_round": 10000,
                "early_stopping_rounds": 100,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_data_in_leaf": 100,
                "drop_features": DROP_FEATURES,
            },
        },
        "training": {
            "cell": cell,
            "val_strategy": "protein_group",
            "val_fraction": 0.2,
            "seed": seed,
            "propagate_labels": False,
        },
        "sweep": {"backend": "none"},
        "output_dir": str(out_dir),
        "tags": ["lb2_multiseed", cell, f"seed{seed}", "no_anc2vec", "no_pca"],
        "keep_staging": False,
    }
    spec_path = out_dir / "spec_input.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return spec_path


def run_training(cell: str, seed: int) -> dict:
    out_dir = _run_dir(cell, seed)
    run_json = out_dir / "run.json"

    # Check if already complete
    if run_json.exists():
        try:
            report = json.loads(run_json.read_text())
            if report.get("status") == "ok":
                print(f"  [skip] {cell} seed={seed} already complete (fmax={report['metrics']['test_fmax']:.4f})")
                return report
        except Exception:
            pass

    spec_path = _make_spec_yaml(cell, seed, out_dir)
    print(f"  [train] {cell} seed={seed} → {out_dir}", flush=True)
    t0 = time.monotonic()

    run_script = WORKTREE / "scripts" / "run.py"
    proc = subprocess.run(
        [str(LAB_PYTHON), str(run_script), str(spec_path),
         "--datasets-root", str(REPO_DATASETS)],
        cwd=str(WORKTREE),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    dur = time.monotonic() - t0
    if proc.returncode != 0:
        print(f"  [FAIL] {cell} seed={seed} exit={proc.returncode} in {dur:.0f}s")
        print(f"  stderr: {proc.stderr[-500:]}")
        return {}
    try:
        report = json.loads(run_json.read_text())
        fmax = report.get("metrics", {}).get("test_fmax", 0)
        print(f"  [done] {cell} seed={seed} test_fmax={fmax:.4f} dur={dur:.0f}s")
        return report
    except Exception as e:
        print(f"  [FAIL] {cell} seed={seed} could not read run.json: {e}")
        return {}


# ─── cafaeval ────────────────────────────────────────────────────────────────

_DRIVER_SRC = '''
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
'''


def _extract_alignment(cell: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-derive (protein, go, label) in bucket-sort order for a cell."""
    sys.path.insert(0, str(WORKTREE / "src"))
    from protea_reranker_lab.data import iter_batches

    cat, asp = cell.split("-", 1)
    prots, gos, labs = [], [], []
    for batch in iter_batches(
        DATASET_DIR / "eval.parquet",
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
    order = np.lexsort((proteins, buckets))
    return proteins[order], go_terms[order], labels[order]


def run_cafaeval(cell: str, seed: int, run_report: dict) -> dict:
    out_dir = _run_dir(cell, seed)
    pred_pq = out_dir / "predictions.parquet"
    cafaeval_json = out_dir / "cafaeval_metrics.json"

    if cafaeval_json.exists():
        try:
            m = json.loads(cafaeval_json.read_text())
            if m.get("cafaeval_fmax") is not None:
                print(f"  [skip cafaeval] {cell} seed={seed} fmax={m['cafaeval_fmax']:.4f}")
                return m
        except Exception:
            pass

    if not pred_pq.exists():
        print(f"  [skip cafaeval] {cell} seed={seed}: no predictions.parquet")
        return {}

    print(f"  [cafaeval] {cell} seed={seed}", flush=True)

    # Read predictions
    pred_t = pq.read_table(str(pred_pq), columns=["protein_accession", "label", "score"])
    proteins_p = pred_t.column("protein_accession").to_numpy(zero_copy_only=False)
    labels_p = pred_t.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    scores = pred_t.column("score").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    # Re-align (protein, go, label)
    proteins_a, gos, labels_a = _extract_alignment(cell)
    if not np.array_equal(proteins_p, proteins_a):
        print(f"  [err] {cell} seed={seed}: protein alignment mismatch")
        return {}
    if not np.array_equal(labels_p, labels_a):
        print(f"  [err] {cell} seed={seed}: label mismatch")
        return {}

    # Write TSVs
    cell_dir = out_dir / "cafaeval"
    cell_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = cell_dir / "pred_dir"
    pred_dir.mkdir(exist_ok=True)

    pred_tsv = cell_dir / "pred.tsv"
    gt_tsv = cell_dir / "gt.tsv"
    with pred_tsv.open("w") as f:
        for p, g, s in zip(proteins_p, gos, scores):
            f.write(f"{p}\t{g}\t{s:.6f}\n")
    pos = labels_p > 0
    with gt_tsv.open("w") as f:
        for p, g in zip(proteins_p[pos], gos[pos]):
            f.write(f"{p}\t{g}\n")

    # Copy pred.tsv into pred_dir
    pred_dst = pred_dir / f"{cell}.tsv"
    pred_dst.write_bytes(pred_tsv.read_bytes())

    # Write cafaeval driver
    raw_json = cell_dir / "_raw_metrics.json"
    driver = cell_dir / "_driver.py"
    driver.write_text(_DRIVER_SRC.format(
        obo=str(OBO_PATH),
        pred_dir=str(pred_dir),
        gt=str(gt_tsv),
        out_json=str(raw_json),
    ))

    proc = subprocess.run(
        [str(PROTEA_VENV / "bin" / "python"), str(driver)],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        print(f"  [FAIL cafaeval] {cell} seed={seed}: {proc.stderr[-300:]}")
        return {}

    raw = json.loads(raw_json.read_text())

    # Parse fmax for the right namespace
    aspect = cell.split("-", 1)[1]
    target_ns = ASPECT_TO_NS.get(aspect)
    target_key = f"f__{target_ns}"
    cafaeval_fmax = None
    for metric_kind, records in raw.items():
        for rec in records:
            ns = rec.get("ns") or rec.get("namespace") or ""
            if ns == target_ns:
                for k in ("f", "Fmax", "fmax"):
                    if k in rec and rec[k] is not None:
                        try:
                            cafaeval_fmax = float(rec[k])
                        except (TypeError, ValueError):
                            pass
                        break
            if cafaeval_fmax is not None:
                break
        if cafaeval_fmax is not None:
            break

    # Fallback: find any f__ key
    if cafaeval_fmax is None:
        for metric_kind, records in raw.items():
            for rec in records:
                for k in ("f", "Fmax", "fmax"):
                    if k in rec and rec[k] is not None:
                        try:
                            cafaeval_fmax = float(rec[k])
                        except (TypeError, ValueError):
                            pass
                        if cafaeval_fmax is not None:
                            break
                if cafaeval_fmax is not None:
                    break
            if cafaeval_fmax is not None:
                break

    n_pos = int(pos.sum())
    metrics = {
        "cell": cell,
        "seed": seed,
        "cafaeval_fmax": cafaeval_fmax,
        "n_rows": len(proteins_p),
        "n_positives": n_pos,
    }
    cafaeval_json.write_text(json.dumps(metrics, indent=2))
    print(f"  [cafaeval done] {cell} seed={seed} cafaeval_fmax={cafaeval_fmax}")
    return metrics


# ─── Bootstrap CIs ──────────────────────────────────────────────────────────

def compute_bootstrap_cis(
    cell_seed_fmax: dict[str, dict[int, float]],
    n_iter: int = 10000,
    seed: int = 0,
) -> dict:
    """Simple bootstrap on per-cell per-seed Fmax values.

    Since we only have 3 seeds, we bootstrap the mean across seeds using
    standard bootstrap resampling of the 3 observations.
    """
    rng = np.random.default_rng(seed)
    cis = {}
    for cell, seed_fmax in sorted(cell_seed_fmax.items()):
        vals = np.array([v for v in seed_fmax.values() if v is not None], dtype=float)
        if len(vals) == 0:
            cis[cell] = {"mean": None, "ci_lo": None, "ci_hi": None, "std": None}
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        # bootstrap mean
        boot_means = np.array([
            rng.choice(vals, size=len(vals), replace=True).mean()
            for _ in range(n_iter)
        ])
        ci_lo = float(np.quantile(boot_means, 0.025))
        ci_hi = float(np.quantile(boot_means, 0.975))
        cis[cell] = {
            "n_seeds": len(vals),
            "values": vals.tolist(),
            "mean": mean,
            "std": std,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_width": ci_hi - ci_lo,
        }
    return cis


def main():
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    start_ts = time.time()
    print(f"\n=== LB.2 multi-seed sweep: {len(CELLS)} cells x {len(SEEDS)} seeds ===\n")

    # ── Phase 1: Training ────────────────────────────────────────────────────
    all_reports: dict[str, dict[int, dict]] = {}
    for cell in CELLS:
        all_reports[cell] = {}
        for seed in SEEDS:
            elapsed = time.time() - start_ts
            if elapsed > 3 * 3600:
                print(f"[HALT] 3h wall-clock exceeded at {elapsed/3600:.1f}h — stopping")
                sys.exit(1)
            report = run_training(cell, seed)
            all_reports[cell][seed] = report

    print("\n=== Phase 1 complete: training done ===\n")

    # ── Phase 2: cafaeval ────────────────────────────────────────────────────
    cafaeval_results: dict[str, dict[int, dict]] = {}
    for cell in CELLS:
        cafaeval_results[cell] = {}
        for seed in SEEDS:
            elapsed = time.time() - start_ts
            if elapsed > 3 * 3600:
                print(f"[HALT] 3h wall-clock exceeded at {elapsed/3600:.1f}h — stopping")
                sys.exit(1)
            run_report = all_reports[cell].get(seed, {})
            m = run_cafaeval(cell, seed, run_report)
            cafaeval_results[cell][seed] = m

    print("\n=== Phase 2 complete: cafaeval done ===\n")

    # ── Phase 3: Bootstrap CIs ───────────────────────────────────────────────
    cell_seed_fmax: dict[str, dict[int, float]] = {}
    for cell in CELLS:
        cell_seed_fmax[cell] = {}
        for seed in SEEDS:
            m = cafaeval_results[cell].get(seed, {})
            cell_seed_fmax[cell][seed] = m.get("cafaeval_fmax")

    cis = compute_bootstrap_cis(cell_seed_fmax, n_iter=10000, seed=99)
    cis_path = RUNS_ROOT / "cis.json"
    cis_path.write_text(json.dumps(cis, indent=2))
    print(f"Bootstrap CIs written → {cis_path}")

    # ── Phase 4: Summary ────────────────────────────────────────────────────
    # NK+LK selective avg (6 cells) + PK baselines (3 cells, use study_v23 values)
    nklk_fmax_means = [
        cis[cell]["mean"] for cell in CELLS if cis[cell]["mean"] is not None
    ]
    # PK cells: use study_v23 cafaeval values (unchanged, reranker hurts PK)
    pk_cells = ["pk-mfo", "pk-bpo", "pk-cco"]
    pk_fmax_values = [BASELINE_CAFAEVAL_FMAX[c] for c in pk_cells]

    all_fmax = nklk_fmax_means + pk_fmax_values
    selective_avg = float(np.mean(all_fmax)) if all_fmax else None

    # Compute CI on selective avg by propagating NK+LK uncertainty
    # For simplicity: bootstrap the mean of the 6 NK+LK per-cell means combined with 3 fixed PK
    nklk_vals_per_cell = [
        np.array(list(cell_seed_fmax[cell].values()), dtype=float)
        for cell in CELLS
        if any(v is not None for v in cell_seed_fmax[cell].values())
    ]
    rng = np.random.default_rng(42)
    n_iter = 10000
    boot_avgs = []
    for _ in range(n_iter):
        sample_cell_means = [
            rng.choice(vals, size=len(vals), replace=True).mean()
            for vals in nklk_vals_per_cell
        ]
        avg = np.mean(sample_cell_means + pk_fmax_values)
        boot_avgs.append(avg)
    boot_avgs = np.array(boot_avgs)
    sel_ci_lo = float(np.quantile(boot_avgs, 0.025))
    sel_ci_hi = float(np.quantile(boot_avgs, 0.975))

    summary = {
        "task": "lb2_multiseed",
        "cells": CELLS,
        "seeds": SEEDS,
        "n_runs": len(CELLS) * len(SEEDS),
        "per_cell_cafaeval_fmax_by_seed": {
            cell: {str(s): cell_seed_fmax[cell].get(s) for s in SEEDS}
            for cell in CELLS
        },
        "per_cell_ci": cis,
        "selective_avg_nklk_plus_pk": selective_avg,
        "selective_avg_ci_lo": sel_ci_lo,
        "selective_avg_ci_hi": sel_ci_hi,
        "champion_comparison": {
            "champion_single_seed_0_6408": 0.6408,
            "new_multiseed_avg": selective_avg,
            "delta": (selective_avg - 0.6408) if selective_avg is not None else None,
        },
        "pk_cells_fixed_from_study_v23": {c: BASELINE_CAFAEVAL_FMAX[c] for c in pk_cells},
        "elapsed_s": round(time.time() - start_ts, 1),
    }
    summary_path = RUNS_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary written → {summary_path}")

    # ── Print report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LB.2 multi-seed sweep RESULTS")
    print("=" * 60)
    print(f"Run set: lb2_multiseed (6 cells x 3 seeds = 18 runs)")
    print(f"Per-cell cafaeval Fmax (mean ± 95% bootstrap CI):")
    for cell in CELLS:
        ci = cis[cell]
        m = ci.get("mean")
        lo = ci.get("ci_lo")
        hi = ci.get("ci_hi")
        vals = ci.get("values", [])
        half = (hi - lo) / 2 if (hi is not None and lo is not None) else None
        if m is not None and half is not None:
            print(f"  {cell}: {m:.4f} ± {half:.4f}  [seeds: {vals}]")
        else:
            print(f"  {cell}: N/A")
    print(f"\nSelective avg (6 NK+LK + 3 PK baseline) ± CI: "
          f"{selective_avg:.4f} ± {(sel_ci_hi - sel_ci_lo)/2:.4f}")
    print(f"Champion comparison vs 0.6408 (single-seed):")
    delta = (selective_avg - 0.6408) if selective_avg is not None else None
    if delta is not None:
        direction = "+" if delta >= 0 else ""
        print(f"  Delta: {direction}{delta:.4f}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    summary = main()
    sys.exit(0)
