#!/usr/bin/env python
"""LB.3 closure: per-cell x aspect paired bootstrap CI (champion vs baseline).

Computes 95% percentile-bootstrap confidence intervals on the paired
Fmax difference (champion minus baseline) for each cell x aspect
combination using the LB.2 multi-seed sweep data (3 seeds: 42, 7, 137).

Data source (default):
  The script embeds the canonical LB.2 per-seed cafaeval Fmax values
  from EXPERIMENTS.md (committed 2026-05-17). These are the leakage-fixed
  champion (run--plm=esmc_300m--k=5--rr=lgbm.per_cell_9--feat=v6+lineage-leakfree--eval=bench-v1-K5-v226-lineage--prop=fill--ens=none, NK+LK reranker + PK baseline fallback) and the
  KNN-only baseline, both on bench-v1-K5-v226-lineage (eval v226-v230).

  The raw prediction parquets are gitignored; the embedded summary table
  is the committed artefact that drives this script.  Pass --data-json
  to override with a custom file matching the documented schema.

Data schema (for --data-json override):
  JSON object with keys "cells" (list of cell names) and "data" (dict
  mapping cell_name -> {"champion": [s42, s7, s137],
  "baseline": [s42, s7, s137]}).

Bootstrap method:
  For each cell, treat the 3 seed values as individual observations of a
  paired (champion, baseline) experiment.  Draw N_ITER bootstrap resamples
  with replacement from the S=3 seed set (paired: same seed-index for
  both arms), recompute the mean difference on each resample, and take the
  2.5th / 97.5th quantile of the resulting N_ITER-length distribution.

  This is a seed-level paired bootstrap: the pairing structure is seed,
  not protein.  For a protein-level paired bootstrap on the raw prediction
  arrays, see src/protea_reranker_lab/bootstrap.py (requires the gitignored
  prediction parquets).

  N_ITER=10000, seed=42 unless overridden via CLI.

Output:
  experiments/lb3/per_cell_paired_ci.csv

  Columns:
    cell, category, aspect,
    champion_fmax_mean, champion_fmax_ci_lo, champion_fmax_ci_hi,
    baseline_fmax_mean, baseline_fmax_ci_lo, baseline_fmax_ci_hi,
    paired_diff_mean, paired_diff_ci_lo, paired_diff_ci_hi,
    n_seeds, n_iter, seed,
    sig_95  (1 if paired_diff_ci_lo > 0, else 0)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO / "experiments" / "lb3"
OUTPUT_CSV = OUTPUT_DIR / "per_cell_paired_ci.csv"

# ---------------------------------------------------------------------------
# Canonical LB.2 multi-seed cafaeval Fmax (seeds 42, 7, 137)
# Source: EXPERIMENTS.md, LB.2 multi-seed sweep section (2026-05-17)
# bench-v1-K5-v226-lineage, eval v226-v230
# Champion arm: leakage-fixed booster (run--feat=v6+lineage-leakfree, NK+LK cells reranked)
# Baseline arm: KNN-only (vote_count, no reranker)
# PK cells: champion == baseline (selective-deploy policy; reranker NOT used)
# ---------------------------------------------------------------------------

_LB2_DATA: dict[str, dict[str, list[float]]] = {
    "nk-mfo": {
        "champion": [0.7112, 0.7041, 0.7041],
        "baseline": [0.6447, 0.6447, 0.6447],
    },
    "nk-bpo": {
        "champion": [0.5599, 0.5571, 0.5618],
        "baseline": [0.5333, 0.5333, 0.5333],
    },
    "nk-cco": {
        "champion": [0.7733, 0.7830, 0.7758],
        "baseline": [0.7000, 0.7000, 0.7000],
    },
    "lk-mfo": {
        "champion": [0.6877, 0.6786, 0.6757],
        "baseline": [0.5816, 0.5816, 0.5816],
    },
    "lk-bpo": {
        "champion": [0.6472, 0.6421, 0.6485],
        "baseline": [0.5844, 0.5844, 0.5844],
    },
    "lk-cco": {
        "champion": [0.7434, 0.7252, 0.7417],
        "baseline": [0.7053, 0.7053, 0.7053],
    },
    # PK cells: selective-deploy policy; reranker NOT applied
    # champion == baseline (KNN fallback); see FARM-EXP.10
    "pk-mfo": {
        "champion": [0.4830, 0.4830, 0.4830],
        "baseline": [0.4830, 0.4830, 0.4830],
    },
    "pk-bpo": {
        "champion": [0.4030, 0.4030, 0.4030],
        "baseline": [0.4030, 0.4030, 0.4030],
    },
    "pk-cco": {
        "champion": [0.6010, 0.6010, 0.6010],
        "baseline": [0.6010, 0.6010, 0.6010],
    },
}

CELLS: tuple[str, ...] = (
    "nk-bpo", "nk-mfo", "nk-cco",
    "lk-bpo", "lk-mfo", "lk-cco",
    "pk-bpo", "pk-mfo", "pk-cco",
)

CSV_COLUMNS: list[str] = [
    "cell", "category", "aspect",
    "champion_fmax_mean", "champion_fmax_ci_lo", "champion_fmax_ci_hi",
    "baseline_fmax_mean", "baseline_fmax_ci_lo", "baseline_fmax_ci_hi",
    "paired_diff_mean", "paired_diff_ci_lo", "paired_diff_ci_hi",
    "n_seeds", "n_iter", "seed",
    "sig_95",
]


def paired_bootstrap_ci(
    champion: list[float],
    baseline: list[float],
    *,
    n_iter: int,
    seed: int,
) -> dict[str, float | int]:
    """Seed-level paired bootstrap CI.

    Parameters
    ----------
    champion:
        Per-seed cafaeval Fmax for the champion arm (S values).
    baseline:
        Per-seed cafaeval Fmax for the baseline arm (S values,
        same seed ordering as champion).
    n_iter:
        Number of bootstrap resamples.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys matching the CI sub-columns of the CSV.
    """
    champ_arr = np.asarray(champion, dtype=np.float64)
    base_arr = np.asarray(baseline, dtype=np.float64)
    if champ_arr.shape != base_arr.shape:
        raise ValueError(
            f"champion and baseline must have the same length; "
            f"got {champ_arr.shape} vs {base_arr.shape}"
        )
    s = champ_arr.size
    rng = np.random.default_rng(seed)
    # Draw resample indices: shape (n_iter, s)
    idx = rng.integers(0, s, size=(n_iter, s))
    champ_boot = champ_arr[idx].mean(axis=1)   # shape (n_iter,)
    base_boot = base_arr[idx].mean(axis=1)     # shape (n_iter,)
    diff_boot = champ_boot - base_boot          # paired difference per iter

    def _ci(arr: np.ndarray) -> tuple[float, float]:
        return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

    c_lo, c_hi = _ci(champ_boot)
    b_lo, b_hi = _ci(base_boot)
    d_lo, d_hi = _ci(diff_boot)

    return {
        "champion_fmax_mean": float(champ_arr.mean()),
        "champion_fmax_ci_lo": c_lo,
        "champion_fmax_ci_hi": c_hi,
        "baseline_fmax_mean": float(base_arr.mean()),
        "baseline_fmax_ci_lo": b_lo,
        "baseline_fmax_ci_hi": b_hi,
        "paired_diff_mean": float((champ_arr - base_arr).mean()),
        "paired_diff_ci_lo": d_lo,
        "paired_diff_ci_hi": d_hi,
    }


def compute_rows(
    data: dict[str, dict[str, list[float]]],
    *,
    n_iter: int,
    seed: int,
) -> list[dict[str, object]]:
    """Compute one result row per cell in CELLS order."""
    rows: list[dict[str, object]] = []
    for cell in CELLS:
        if cell not in data:
            raise KeyError(
                f"cell '{cell}' missing from data; "
                "check --data-json or the embedded _LB2_DATA table"
            )
        cell_data = data[cell]
        champion = cell_data["champion"]
        baseline = cell_data["baseline"]
        ci = paired_bootstrap_ci(champion, baseline, n_iter=n_iter, seed=seed)
        cat, asp = cell.split("-", 1)
        sig_95 = 1 if ci["paired_diff_ci_lo"] > 0 else 0
        rows.append({
            "cell": cell,
            "category": cat,
            "aspect": asp,
            **ci,
            "n_seeds": len(champion),
            "n_iter": n_iter,
            "seed": seed,
            "sig_95": sig_95,
        })
    return rows


def _fmt(x: object, digits: int = 4) -> str:
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _print_table(rows: list[dict[str, object]]) -> None:
    hdr = (
        f"{'cell':8s}  {'champ':>7s} [{' ci_lo':>7s}, {' ci_hi':>7s}]"
        f"  {'base':>7s} [{' ci_lo':>7s}, {' ci_hi':>7s}]"
        f"  {'delta':>7s} [{' ci_lo':>7s}, {' ci_hi':>7s}]  sig"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        champ = _fmt(r["champion_fmax_mean"])
        c_lo = _fmt(r["champion_fmax_ci_lo"])
        c_hi = _fmt(r["champion_fmax_ci_hi"])
        base = _fmt(r["baseline_fmax_mean"])
        b_lo = _fmt(r["baseline_fmax_ci_lo"])
        b_hi = _fmt(r["baseline_fmax_ci_hi"])
        diff = _fmt(r["paired_diff_mean"])
        d_lo = _fmt(r["paired_diff_ci_lo"])
        d_hi = _fmt(r["paired_diff_ci_hi"])
        sig = "*" if r["sig_95"] else " "
        print(
            f"{r['cell']:8s}  {champ:>7s} [{c_lo:>7s}, {c_hi:>7s}]"
            f"  {base:>7s} [{b_lo:>7s}, {b_hi:>7s}]"
            f"  {diff:>7s} [{d_lo:>7s}, {d_hi:>7s}]  {sig}"
        )


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                col: _fmt(r[col]) if isinstance(r[col], float) else r[col]
                for col in CSV_COLUMNS
            })


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--write", action="store_true",
        help=f"write the CSV to {OUTPUT_CSV.relative_to(REPO)} (else stdout only)",
    )
    p.add_argument(
        "--data-json", type=Path, default=None,
        help=(
            "JSON file with per-cell per-seed data overriding the embedded "
            "LB.2 canonical table. Schema: {\"cells\": [...], \"data\": "
            "{cell: {\"champion\": [...], \"baseline\": [...]}}}"
        ),
    )
    p.add_argument("--n-iter", type=int, default=10000,
                   help="number of bootstrap resamples (default: 10000)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility (default: 42)")
    args = p.parse_args(argv)

    data = _LB2_DATA
    if args.data_json is not None:
        payload = json.loads(args.data_json.read_text())
        data = payload["data"]

    rows = compute_rows(data, n_iter=args.n_iter, seed=args.seed)
    _print_table(rows)

    if args.write:
        write_csv(rows, OUTPUT_CSV)
        print(f"\n[wrote] {OUTPUT_CSV.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
