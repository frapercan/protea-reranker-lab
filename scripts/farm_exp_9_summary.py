#!/usr/bin/env python
"""FARM-EXP.9 (partial) summary: collect results and compute paired CIs vs pre-leakage runs.

Reads all run.json files from runs/transversal/farm_exp_9_rep_*/
and compares against the pre-leakage bench-v1-K5 fmax values from
experiments/farm_exp_9/cells_to_rerun.csv.

For each cell x seed triple that has both a pre-leakage fmax and a
new bench-v1-K5-filtered fmax, computes:
  - Per-seed delta (new - old)
  - Across-seed mean delta and seed-level bootstrap CI

IMPORTANT: The pre-leakage (bench-v1-K5) fmax values and the new
(bench-v1-K5-filtered) fmax values are NOT directly comparable because:
  1. The eval set changed (bench-v1-K5: eval v220-v229;
     bench-v1-K5-filtered: eval v220-v230 with anc2vec leakage filtered).
  2. The feature set changed: bench-v1-K5 used 52 features including
     anc2vec leakage features; bench-v1-K5-filtered uses 34 features
     (anc2vec and emb_pca families dropped).
The deltas are reported for completeness and to surface any anomalous cells,
but MUST NOT be interpreted as measuring the leakage correction effect alone.

Usage:
    python scripts/farm_exp_9_summary.py [--write]

Outputs:
    experiments/farm_exp_9/summary.json  (if --write)
    experiments/farm_exp_9/partial_ci.csv  (if --write)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TRANSVERSAL_DIR = REPO / "runs" / "transversal"
CELLS_CSV = REPO / "experiments" / "farm_exp_9" / "cells_to_rerun.csv"
OUT_DIR = REPO / "experiments" / "farm_exp_9"

CELLS = ["nk-bpo", "nk-mfo", "nk-cco", "lk-bpo", "lk-mfo", "lk-cco", "pk-bpo", "pk-mfo", "pk-cco"]
SEEDS = [42, 7, 137]

N_ITER = 10000
RNG_SEED = 42


def load_pre_leakage_fmax() -> dict[str, dict[int, float]]:
    """Load pre-leakage fmax from cells_to_rerun.csv.

    Returns dict: cell -> {seed: fmax}
    """
    data: dict[str, dict[int, float]] = {}
    if not CELLS_CSV.exists():
        return data
    with CELLS_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["source"] != "study_v9_replication":
                continue
            cell = row["cell"]
            try:
                seed = int(row["seed"])
                fmax = float(row["pre_leakage_fmax"]) if row["pre_leakage_fmax"] else None
            except (ValueError, KeyError):
                continue
            if fmax is None:
                continue
            data.setdefault(cell, {})[seed] = fmax
    return data


def load_new_fmax() -> dict[str, dict[int, float]]:
    """Load new bench-v1-K5-filtered fmax from run.json files.

    Returns dict: cell -> {seed: fmax}
    """
    data: dict[str, dict[int, float]] = {}
    if not TRANSVERSAL_DIR.exists():
        return data
    for run_dir in TRANSVERSAL_DIR.iterdir():
        if not run_dir.name.startswith("farm_exp_9_rep_"):
            continue
        run_json = run_dir / "run.json"
        if not run_json.exists():
            continue
        with run_json.open() as fh:
            d = json.load(fh)
        # Extract cell and seed from tags
        tags = d.get("spec_tags", [])
        cell = None
        seed = None
        for tag in tags:
            if tag in CELLS:
                cell = tag
            if tag.startswith("seed") and tag[4:].isdigit():
                seed = int(tag[4:])
        if cell is None or seed is None:
            continue
        fmax = d.get("metrics", {}).get("test_fmax")
        if fmax is not None:
            data.setdefault(cell, {})[seed] = float(fmax)
    return data


def paired_bootstrap_ci(
    new_vals: list[float],
    old_vals: list[float],
    n_iter: int = N_ITER,
    seed: int = RNG_SEED,
) -> dict[str, float]:
    """Seed-level paired bootstrap CI on the mean difference (new - old)."""
    rng = np.random.default_rng(seed)
    new_arr = np.array(new_vals)
    old_arr = np.array(old_vals)
    n = len(new_arr)
    diffs = new_arr - old_arr
    boot_means = np.array([
        np.mean(diffs[rng.integers(0, n, size=n)])
        for _ in range(n_iter)
    ])
    return {
        "new_fmax_mean": float(np.mean(new_arr)),
        "new_fmax_ci_lo": float(np.percentile(
            [np.mean(new_arr[rng.integers(0, n, size=n)]) for _ in range(n_iter)],
            2.5,
        )),
        "new_fmax_ci_hi": float(np.percentile(
            [np.mean(new_arr[rng.integers(0, n, size=n)]) for _ in range(n_iter)],
            97.5,
        )),
        "old_fmax_mean": float(np.mean(old_arr)),
        "paired_diff_mean": float(np.mean(diffs)),
        "paired_diff_ci_lo": float(np.percentile(boot_means, 2.5)),
        "paired_diff_ci_hi": float(np.percentile(boot_means, 97.5)),
    }


def build_per_cell_summary(
    pre_leakage: dict[str, dict[int, float]],
    new_fmax: dict[str, dict[int, float]],
) -> list[dict]:
    rows = []
    for cell in CELLS:
        cat, asp = cell.split("-", 1)
        old_seeds = pre_leakage.get(cell, {})
        new_seeds = new_fmax.get(cell, {})
        # Only include seeds present in both
        common_seeds = sorted(set(old_seeds) & set(new_seeds))
        if not common_seeds:
            rows.append({
                "cell": cell,
                "category": cat,
                "aspect": asp,
                "seeds_done": 0,
                "seeds_missing": sorted(set(SEEDS) - set(new_seeds)),
                "status": "pending",
                "new_fmax_mean": None,
                "new_fmax_ci_lo": None,
                "new_fmax_ci_hi": None,
                "old_fmax_mean": None,
                "paired_diff_mean": None,
                "paired_diff_ci_lo": None,
                "paired_diff_ci_hi": None,
                "sig_95": None,
                "note": "NOT_COMPARABLE: eval set + feature set changed",
            })
            continue
        new_vals = [new_seeds[s] for s in common_seeds]
        old_vals = [old_seeds[s] for s in common_seeds]
        n_iter_use = min(N_ITER, 1000) if len(common_seeds) < 2 else N_ITER
        ci = paired_bootstrap_ci(new_vals, old_vals, n_iter=n_iter_use)
        missing = sorted(set(SEEDS) - set(new_seeds))
        rows.append({
            "cell": cell,
            "category": cat,
            "aspect": asp,
            "seeds_done": len(common_seeds),
            "seeds_missing": missing,
            "status": "complete" if not missing else "partial",
            **ci,
            "sig_95": 1 if ci["paired_diff_ci_lo"] > 0 else 0,
            "note": "NOT_COMPARABLE: eval set + feature set changed between old and new run",
        })
    return rows


def _fmt(x: object, digits: int = 4) -> str:
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    if x is None:
        return "n/a"
    return str(x)


def print_table(rows: list[dict]) -> None:
    hdr = (
        f"{'cell':8s}  {'new_mean':>8s} [{' new_lo':>7s}, {' new_hi':>7s}]"
        f"  {'old_mean':>8s}  {'delta_mean':>10s} [{' d_lo':>7s}, {' d_hi':>7s}]"
        f"  {'seeds':>5s}  sig"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['cell']:8s}  {_fmt(r['new_fmax_mean']):>8s}"
            f" [{_fmt(r.get('new_fmax_ci_lo')):>7s},"
            f" {_fmt(r.get('new_fmax_ci_hi')):>7s}]"
            f"  {_fmt(r['old_fmax_mean']):>8s}"
            f"  {_fmt(r['paired_diff_mean']):>10s}"
            f" [{_fmt(r.get('paired_diff_ci_lo')):>7s},"
            f" {_fmt(r.get('paired_diff_ci_hi')):>7s}]"
            f"  {r['seeds_done']:>2d}/{len(SEEDS)}  "
            f"{'*' if r.get('sig_95') else ' '}"
        )


CSV_COLUMNS = [
    "cell", "category", "aspect", "seeds_done", "status",
    "new_fmax_mean", "new_fmax_ci_lo", "new_fmax_ci_hi",
    "old_fmax_mean", "paired_diff_mean", "paired_diff_ci_lo", "paired_diff_ci_hi",
    "sig_95", "note",
]


def write_outputs(rows: list[dict], summary: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Write CSV
    csv_path = OUT_DIR / "partial_ci.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({
                col: _fmt(r[col]) if isinstance(r[col], float) else r[col]
                for col in CSV_COLUMNS
                if col in r
            })
    print(f"[wrote] {csv_path.relative_to(REPO)}")
    # Write summary JSON
    json_path = OUT_DIR / "summary.json"
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[wrote] {json_path.relative_to(REPO)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true", help="write outputs to disk")
    args = p.parse_args(argv)

    pre_leakage = load_pre_leakage_fmax()
    new_fmax = load_new_fmax()

    rows = build_per_cell_summary(pre_leakage, new_fmax)

    print("\nFARM-EXP.9 (partial) — per-cell paired CI vs pre-leakage fmax")
    print("WARNING: Old and new rows are NOT comparable (eval set + features changed).")
    print()
    print_table(rows)

    complete = [r for r in rows if r["status"] == "complete"]
    partial = [r for r in rows if r["status"] == "partial"]
    pending = [r for r in rows if r["status"] == "pending"]
    print(f"\nStatus: complete={len(complete)}, partial={len(partial)}, pending={len(pending)}")

    # Champion flip detection (check if any cell with positive delta in old is now negative)
    champion_cells = {"nk-bpo", "nk-mfo", "nk-cco", "lk-bpo", "lk-mfo", "lk-cco"}
    for r in rows:
        if r["cell"] not in champion_cells:
            continue
        if r.get("paired_diff_ci_lo") is None:
            continue
        if r["paired_diff_ci_lo"] is not None and float(r["paired_diff_ci_lo"]) < 0 and float(r.get("paired_diff_ci_hi", 1)) < 0:
            print(f"\n[STOP-CONDITION] CI crosses zero for champion cell {r['cell']}: "
                  f"delta CI [{r['paired_diff_ci_lo']:.4f}, {r['paired_diff_ci_hi']:.4f}]. "
                  "This cell was in the champion table. SURFACE FOR HUMAN REVIEW.")

    summary = {
        "farm_exp_9_partial": True,
        "cells_total_scope": 94,
        "replication_cells": 27,
        "cells_done_this_pass": len(complete) + len(partial),
        "cells_remaining": len(pending),
        "rows": rows,
        "note": (
            "FARM-EXP.9 partial pass. Old fmax (bench-v1-K5, pre-leakage, 52 features) "
            "and new fmax (bench-v1-K5-filtered, 34 features, leakage families dropped) "
            "are NOT directly comparable due to eval set and feature set changes. "
            "Deltas are reported for anomaly detection only."
        ),
    }

    if args.write:
        write_outputs(rows, summary)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
