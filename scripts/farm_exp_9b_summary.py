#!/usr/bin/env python
"""FARM-EXP.9b summary: collect pass-2 results and update cells_to_rerun.csv + summary.json.

Extends farm_exp_9_summary.py to handle ablation (farm_exp_9_abl_*),
hparam (farm_exp_9_hp_*), and standalone (farm_exp_9_standalone_*) runs
in addition to replication runs (farm_exp_9_rep_*).

For each completed run, flips status to 'done' in cells_to_rerun.csv.
Writes partial_ci_pass2.csv with pass-2 replication CI results.
Updates summary.json with new totals.

Usage:
    python scripts/farm_exp_9b_summary.py [--write] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TRANSVERSAL_DIR = REPO / "runs" / "transversal"
CELLS_CSV = REPO / "experiments" / "farm_exp_9" / "cells_to_rerun.csv"
PARTIAL_CI_CSV = REPO / "experiments" / "farm_exp_9" / "partial_ci.csv"
PARTIAL_CI_PASS2_CSV = REPO / "experiments" / "farm_exp_9" / "partial_ci_pass2.csv"
SUMMARY_JSON = REPO / "experiments" / "farm_exp_9" / "summary.json"
OUT_DIR = REPO / "experiments" / "farm_exp_9"

CELLS = ["nk-bpo", "nk-mfo", "nk-cco", "lk-bpo", "lk-mfo", "lk-cco", "pk-bpo", "pk-mfo", "pk-cco"]
SEEDS = [42, 7, 137]
N_ITER = 10000
RNG_SEED = 42


# Mapping from run.json name pattern to cells_to_rerun.csv cell_id
def _cell_id_from_run_name(run_name: str) -> str | None:
    """Map run spec name to the CSV cell_id."""
    # farm_exp_9_rep_pk-cco_seed137 -> study_v9_rep_pk-cco_seed137
    m = re.match(r"farm_exp_9_rep_(.+)_seed(\d+)$", run_name)
    if m:
        cell, seed = m.group(1), m.group(2)
        return f"study_v9_rep_{cell}_seed{seed}"

    # farm_exp_9_abl_nk-bpo_alignment_nw -> study_v9_abl_nk-bpo_drop_alignment_nw
    m = re.match(r"farm_exp_9_abl_(.+?)_(.+)$", run_name)
    if m:
        cell, drop = m.group(1), m.group(2)
        return f"study_v9_abl_{cell}_drop_{drop}"

    # farm_exp_9_hp_L127_lr003_npr10 -> study_v9_hp_L127_lr003_npr10
    m = re.match(r"farm_exp_9_hp_(.+)$", run_name)
    if m:
        return f"study_v9_hp_{m.group(1)}"

    # farm_exp_9_standalone_nk-bpo_seed42 -> bench-v1-K5_nk-bpo_standalone
    m = re.match(r"farm_exp_9_standalone_nk-bpo", run_name)
    if m:
        return "bench-v1-K5_nk-bpo_standalone"

    return None


def _collect_completed_runs() -> dict[str, dict]:
    """Return dict: cell_id -> run.json data for completed (status=ok) runs."""
    completed: dict[str, dict] = {}
    if not TRANSVERSAL_DIR.exists():
        return completed
    for run_dir in TRANSVERSAL_DIR.iterdir():
        if not run_dir.name.startswith("farm_exp_9_"):
            continue
        run_json = run_dir / "run.json"
        if not run_json.exists():
            continue
        try:
            with run_json.open() as fh:
                d = json.load(fh)
        except json.JSONDecodeError:
            continue
        if d.get("status") != "ok":
            continue
        cell_id = _cell_id_from_run_name(run_dir.name)
        if cell_id is None:
            print(f"[warn] cannot map run dir {run_dir.name} to a cell_id")
            continue
        completed[cell_id] = d
    return completed


def _load_cells_csv() -> list[dict]:
    with CELLS_CSV.open() as fh:
        return list(csv.DictReader(fh))


def _write_cells_csv(rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with CELLS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_existing_summary() -> dict:
    if SUMMARY_JSON.exists():
        with SUMMARY_JSON.open() as fh:
            return json.load(fh)
    return {}


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
    new_boot = [np.mean(new_arr[rng.integers(0, n, size=n)]) for _ in range(n_iter)]
    return {
        "new_fmax_mean": float(np.mean(new_arr)),
        "new_fmax_ci_lo": float(np.percentile(new_boot, 2.5)),
        "new_fmax_ci_hi": float(np.percentile(new_boot, 97.5)),
        "old_fmax_mean": float(np.mean(old_arr)),
        "paired_diff_mean": float(np.mean(diffs)),
        "paired_diff_ci_lo": float(np.percentile(boot_means, 2.5)),
        "paired_diff_ci_hi": float(np.percentile(boot_means, 97.5)),
    }


def _fmt(x: object, digits: int = 4) -> str:
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    if x is None:
        return ""
    return str(x)


CSV_COLUMNS = [
    "cell", "category", "aspect", "seeds_done", "status",
    "new_fmax_mean", "new_fmax_ci_lo", "new_fmax_ci_hi",
    "old_fmax_mean", "paired_diff_mean", "paired_diff_ci_lo", "paired_diff_ci_hi",
    "sig_95", "note",
]


def _build_replication_ci_rows(
    completed: dict[str, dict],
    cells_rows: list[dict],
) -> list[dict]:
    """Build CI rows for replication cells only (pass 1+2 combined)."""
    # Build lookup: cell -> seed -> fmax from CSV pre-leakage
    pre_leakage: dict[str, dict[int, float]] = {}
    for r in cells_rows:
        if r["source"] != "study_v9_replication":
            continue
        cell = r["cell"]
        try:
            seed = int(r["seed"])
            fmax_str = r.get("pre_leakage_fmax", "")
            if fmax_str:
                pre_leakage.setdefault(cell, {})[seed] = float(fmax_str)
        except (ValueError, KeyError):
            continue

    # Build new fmax from completed runs
    new_fmax: dict[str, dict[int, float]] = {}
    for cell_id, d in completed.items():
        if not cell_id.startswith("study_v9_rep_"):
            continue
        tags = d.get("spec_tags", [])
        cell = None
        seed = None
        for tag in tags:
            if tag in CELLS:
                cell = tag
            if tag.startswith("seed") and tag[4:].isdigit():
                seed = int(tag[4:])
        if cell and seed is not None:
            fmax = d.get("metrics", {}).get("test_fmax")
            if fmax is not None:
                new_fmax.setdefault(cell, {})[seed] = float(fmax)

    rows = []
    for cell in CELLS:
        cat, asp = cell.split("-", 1)
        old_seeds = pre_leakage.get(cell, {})
        ns = new_fmax.get(cell, {})
        common = sorted(set(old_seeds) & set(ns))
        if not common:
            rows.append({
                "cell": cell, "category": cat, "aspect": asp,
                "seeds_done": 0,
                "status": "pending",
                "new_fmax_mean": None, "new_fmax_ci_lo": None, "new_fmax_ci_hi": None,
                "old_fmax_mean": None, "paired_diff_mean": None,
                "paired_diff_ci_lo": None, "paired_diff_ci_hi": None,
                "sig_95": None,
                "note": "NOT_COMPARABLE: eval set + feature set changed",
            })
            continue
        new_vals = [ns[s] for s in common]
        old_vals = [old_seeds[s] for s in common]
        n_iter_use = min(N_ITER, 1000) if len(common) < 2 else N_ITER
        ci = paired_bootstrap_ci(new_vals, old_vals, n_iter=n_iter_use)
        missing = sorted(set(SEEDS) - set(ns))
        rows.append({
            "cell": cell, "category": cat, "aspect": asp,
            "seeds_done": len(common),
            "status": "complete" if not missing else "partial",
            **ci,
            "sig_95": 1 if ci["paired_diff_ci_lo"] > 0 else 0,
            "note": "NOT_COMPARABLE: eval set + feature set changed between old and new run",
        })
    return rows


def _build_ablation_rows(
    completed: dict[str, dict],
    cells_rows: list[dict],
) -> list[dict]:
    """Summarise ablation runs: one row per cell_id with fmax and delta vs full model."""
    # Get full model fmax by cell (from rep runs)
    full_fmax_by_cell: dict[str, float] = {}
    for cell_id, d in completed.items():
        if not cell_id.startswith("study_v9_rep_"):
            continue
        tags = d.get("spec_tags", [])
        cell = None
        for tag in tags:
            if tag in CELLS:
                cell = tag
        if cell:
            fmax = d.get("metrics", {}).get("test_fmax")
            if fmax is not None:
                full_fmax_by_cell.setdefault(cell, []).append(float(fmax))

    # Average full fmax per cell
    full_fmax_mean = {c: float(np.mean(v)) for c, v in full_fmax_by_cell.items()}

    rows = []
    for cell_id, d in completed.items():
        if not cell_id.startswith("study_v9_abl_"):
            continue
        # Parse: study_v9_abl_{cell}_drop_{family}
        m = re.match(r"study_v9_abl_(.+)_drop_(.+)$", cell_id)
        if not m:
            continue
        cell, family = m.group(1), m.group(2)
        fmax = d.get("metrics", {}).get("test_fmax")
        if fmax is None:
            continue
        full = full_fmax_mean.get(cell)
        delta = float(fmax) - full if full is not None else None
        rows.append({
            "cell_id": cell_id,
            "cell": cell,
            "family_dropped": family,
            "fmax": float(fmax),
            "full_fmax": full,
            "delta_vs_full": delta,
        })
    return rows


def _build_hparam_rows(
    completed: dict[str, dict],
    cells_rows: list[dict],
) -> list[dict]:
    """Summarise hparam sweep runs."""
    # Get baseline (default hparams, nk-bpo seed42, rep run)
    baseline_fmax = None
    for cell_id, d in completed.items():
        if cell_id == "study_v9_rep_nk-bpo_seed42":
            baseline_fmax = d.get("metrics", {}).get("test_fmax")

    rows = []
    for cell_id, d in completed.items():
        if not cell_id.startswith("study_v9_hp_"):
            continue
        m = re.match(r"study_v9_hp_L(\d+)_lr(\d+)_npr(\w+)$", cell_id)
        if not m:
            continue
        leaves, lr_raw, npr_raw = m.groups()
        lr_map = {"003": 0.003, "005": 0.05, "01": 0.1}
        fmax = d.get("metrics", {}).get("test_fmax")
        if fmax is None:
            continue
        delta = float(fmax) - float(baseline_fmax) if baseline_fmax is not None else None
        rows.append({
            "cell_id": cell_id,
            "num_leaves": int(leaves),
            "learning_rate": lr_map.get(lr_raw, float(lr_raw)),
            "neg_pos_ratio": None if npr_raw == "none" else int(npr_raw),
            "fmax": float(fmax),
            "delta_vs_baseline": delta,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true", help="Write outputs to disk")
    p.add_argument("--dry-run", action="store_true", help="Show what would be written")
    args = p.parse_args(argv)

    # Collect completed runs
    completed = _collect_completed_runs()
    print(f"[info] Found {len(completed)} completed run.json files")

    cells_rows = _load_cells_csv()
    total = len(cells_rows)
    prev_done = sum(1 for r in cells_rows if r["status"] == "done")

    # Determine which cells are newly done
    newly_done: list[str] = []
    for row in cells_rows:
        cid = row["cell_id"]
        if row["status"] == "done":
            continue
        if cid in completed:
            newly_done.append(cid)
            row["status"] = "done"

    remaining = sum(1 for r in cells_rows if r["status"] != "done")
    done_total = sum(1 for r in cells_rows if r["status"] == "done")
    print(f"[info] Status: {prev_done} was done, {len(newly_done)} newly done, "
          f"{done_total} total done, {remaining} still pending")
    if newly_done:
        print(f"[info] Newly done this pass: {', '.join(newly_done[:10])}"
              + (f"...+{len(newly_done)-10}" if len(newly_done) > 10 else ""))

    # Build replication CI rows
    rep_ci_rows = _build_replication_ci_rows(completed, cells_rows)
    abl_rows = _build_ablation_rows(completed, cells_rows)
    hp_rows = _build_hparam_rows(completed, cells_rows)

    print("\n--- REPLICATION CI TABLE (all seeds completed so far) ---")
    for r in rep_ci_rows:
        n = r.get("new_fmax_mean")
        d_lo = r.get("paired_diff_ci_lo")
        d_hi = r.get("paired_diff_ci_hi")
        sig = "*" if r.get("sig_95") else " "
        seeds = r.get("seeds_done", 0)
        print(f"  {r['cell']:8s} seeds={seeds}/3 new_mean={_fmt(n)} "
              f"delta=[{_fmt(d_lo)}, {_fmt(d_hi)}] {sig}")

    print(f"\n--- ABLATION SUMMARY ({len(abl_rows)} cells done) ---")
    by_cell: dict[str, list] = {}
    for r in abl_rows:
        by_cell.setdefault(r["cell"], []).append(r)
    for cell in sorted(by_cell):
        cell_rows_sorted = sorted(by_cell[cell], key=lambda x: x["delta_vs_full"] or 0)
        print(f"  {cell}: {len(cell_rows_sorted)} ablations done")
        if cell_rows_sorted:
            worst = cell_rows_sorted[0]
            best = cell_rows_sorted[-1]
            print(f"    worst drop: {worst['family_dropped']} delta={_fmt(worst['delta_vs_full'])}")
            print(f"    best drop: {best['family_dropped']} delta={_fmt(best['delta_vs_full'])}")

    print(f"\n--- HPARAM SWEEP ({len(hp_rows)} cells done) ---")
    if hp_rows:
        best_hp = max(hp_rows, key=lambda x: x["fmax"])
        print(f"  Best: L={best_hp['num_leaves']} lr={best_hp['learning_rate']} "
              f"npr={best_hp['neg_pos_ratio']} fmax={_fmt(best_hp['fmax'])} "
              f"delta={_fmt(best_hp['delta_vs_baseline'])}")

    if args.write or args.dry_run:
        prefix = "[dry-run] " if args.dry_run else ""

        # Update cells_to_rerun.csv
        if not args.dry_run:
            _write_cells_csv(cells_rows)
        print(f"{prefix}[write] cells_to_rerun.csv (flipped {len(newly_done)} rows to done)")

        # Write pass-2 CI CSV (replication only)
        if not args.dry_run:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            with PARTIAL_CI_PASS2_CSV.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                for r in rep_ci_rows:
                    writer.writerow({
                        col: _fmt(r[col]) if isinstance(r[col], float) else r.get(col, "")
                        for col in CSV_COLUMNS
                    })
        print(f"{prefix}[write] partial_ci_pass2.csv ({len(rep_ci_rows)} replication rows)")

        # Write ablation CSV
        abl_csv = OUT_DIR / "ablation_summary_pass2.csv"
        if abl_rows and not args.dry_run:
            abl_fields = ["cell_id", "cell", "family_dropped", "fmax", "full_fmax", "delta_vs_full"]
            with abl_csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=abl_fields)
                writer.writeheader()
                for r in abl_rows:
                    writer.writerow({k: _fmt(r[k]) if isinstance(r[k], float) else r.get(k, "") for k in abl_fields})
        print(f"{prefix}[write] ablation_summary_pass2.csv ({len(abl_rows)} ablation rows)")

        # Write hparam CSV
        hp_csv = OUT_DIR / "hparam_sweep_summary.csv"
        if hp_rows and not args.dry_run:
            hp_fields = ["cell_id", "num_leaves", "learning_rate", "neg_pos_ratio", "fmax", "delta_vs_baseline"]
            with hp_csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=hp_fields)
                writer.writeheader()
                for r in hp_rows:
                    writer.writerow({k: _fmt(r[k]) if isinstance(r[k], float) else r.get(k, "") for k in hp_fields})
        print(f"{prefix}[write] hparam_sweep_summary.csv ({len(hp_rows)} hparam rows)")

        # Update summary.json
        existing = _load_existing_summary()
        updated_summary = dict(existing)
        updated_summary["farm_exp_9_partial"] = remaining > 0
        updated_summary["cells_total_scope"] = total
        updated_summary["cells_done_total"] = done_total
        updated_summary["cells_remaining"] = remaining
        updated_summary["pass2_newly_done"] = len(newly_done)
        # Keep pass-1 rows, extend with pass-2 data
        updated_summary["rows"] = rep_ci_rows
        updated_summary["ablation_summary"] = {
            "cells_done": len(abl_rows),
            "by_cell": {
                c: [
                    {"family_dropped": r["family_dropped"],
                     "fmax": r["fmax"],
                     "delta_vs_full": r["delta_vs_full"]}
                    for r in sorted(by_cell.get(c, []), key=lambda x: x["delta_vs_full"] or 0)
                ]
                for c in CELLS
                if c in {r["cell"] for r in abl_rows}
            }
        }
        if hp_rows:
            best_hp = max(hp_rows, key=lambda x: x["fmax"])
            updated_summary["hparam_sweep"] = {
                "cells_done": len(hp_rows),
                "best": best_hp,
                "all": hp_rows,
            }
        updated_summary["pass2_note"] = (
            "FARM-EXP.9b: second pass. Old fmax (bench-v1-K5, pre-leakage, 52 features) "
            "and new fmax (bench-v1-K5-filtered, 34 features) are NOT directly comparable. "
            "Deltas are anomaly-detection only."
        )

        if not args.dry_run:
            with SUMMARY_JSON.open("w") as fh:
                json.dump(updated_summary, fh, indent=2, default=str)
        print(f"{prefix}[write] summary.json (done={done_total}, remaining={remaining})")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
