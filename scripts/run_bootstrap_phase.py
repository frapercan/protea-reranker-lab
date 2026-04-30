#!/usr/bin/env python
"""Phase 3 driver — paired bootstrap CIs for v9 winners vs KNN baseline.

Reads ``runs/study_v9/replication/results.csv``, picks the winning seed per
cell (highest fmax), and runs :func:`bootstrap_paired_fmax` against
``vote_count`` (the natural KNN-only score).

Output:
    runs/study_v9/bootstrap/<cell>.json   — full BootstrapResult per cell
    runs/study_v9/bootstrap/results.csv   — flat aggregate (one row per cell)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO / "runs" / "study_v9" / "replication" / "results.csv"
OUT_DIR = REPO / "runs" / "study_v9" / "bootstrap"

# Eval snapshot pair from the dataset manifest. Keep aligned with bench-v1-K5.
EVAL_SNAPSHOT_PAIR = "v220-v230"
SOURCE_EVAL = REPO / "datasets" / "bench-v1-K5" / "eval.parquet"


def _winners_per_cell(rows: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        spec = r["spec"]
        cell = spec.rsplit("_seed", 1)[0]
        try:
            fmax = float(r["fmax"])
        except (TypeError, ValueError):
            continue
        if cell not in best or fmax > float(best[cell]["fmax"]):
            best[cell] = {**r, "fmax": fmax, "cell": cell}
    return best


def main(argv: list[str] | None = None) -> int:
    from protea_reranker_lab.bootstrap import bootstrap_paired_fmax

    p = argparse.ArgumentParser()
    p.add_argument("--baseline-column", default="vote_count")
    p.add_argument(
        "--baseline-direction",
        choices=["higher_is_better", "lower_is_better"],
        default="higher_is_better",
    )
    p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", nargs="+", help="cell names to restrict to")
    args = p.parse_args(argv)

    if not RESULTS_CSV.exists():
        print(f"[err] missing {RESULTS_CSV}; run F1 first")
        return 2
    rows = list(csv.DictReader(RESULTS_CSV.open()))
    winners = _winners_per_cell(rows)
    cells = sorted(winners) if not args.only else [c for c in args.only if c in winners]
    if not cells:
        print(f"[err] no winners found ({len(rows)} rows in {RESULTS_CSV})")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flat_csv = OUT_DIR / "results.csv"
    flat_writer: csv.DictWriter | None = None

    for cell in cells:
        winner = winners[cell]
        predictions = REPO / winner["output_dir"] / "predictions.parquet"
        if not predictions.exists():
            print(f"[skip] {cell}: missing {predictions}")
            continue
        workdir = OUT_DIR / "_workdir" / cell
        out_path = OUT_DIR / f"{cell}.json"
        print(f"[run] cell={cell} winner_fmax={winner['fmax']:.4f} seed={winner['spec']}")
        result = bootstrap_paired_fmax(
            cell=cell,
            predictions_parquet=predictions,
            source_eval_parquet=SOURCE_EVAL,
            eval_snapshot_pair=EVAL_SNAPSHOT_PAIR,
            baseline_column=args.baseline_column,
            higher_is_better=args.baseline_direction == "higher_is_better",
            n_iter=args.n_iter,
            seed=args.seed,
            workdir=workdir,
        )
        out_path.write_text(result.to_json())
        row = json.loads(result.to_json())
        if flat_writer is None:
            flat_writer = csv.DictWriter(flat_csv.open("w", newline=""), fieldnames=list(row.keys()))
            flat_writer.writeheader()
        flat_writer.writerow(row)
        print(f"  fmax_v9={row['fmax_v9']:.4f} [{row['fmax_v9_ci_lo']:.4f}, "
              f"{row['fmax_v9_ci_hi']:.4f}]  baseline={row['fmax_baseline']:.4f} "
              f"[{row['fmax_baseline_ci_lo']:.4f}, {row['fmax_baseline_ci_hi']:.4f}]  "
              f"diff_mean={row['paired_diff_mean']:.4f} "
              f"p={row['p_value_one_sided']:.4f}")

    print(f"[done] {len(cells)} cells → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
