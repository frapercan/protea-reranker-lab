#!/usr/bin/env python
"""LR.1 closure: per-cell x aspect Fmax delta of the v22 lineage booster.

Writes ``runs/lr1/lineage_delta.csv`` and prints the table to stdout. The
delta is reranker-with-lineage minus baseline-no-lineage on the
``bench-v1-K5-v226-lineage`` dataset (eval v226-v230), reported per cell
(NK/LK/PK x BPO/MFO/CCO).

Sources:

- Reranker arm: ``runs/study_v23/<cell>/run.json``. study_v23 is the
  v22-architectural booster (lambdarank, lineage features kept, anc2vec
  and emb_pca families dropped) trained on bench-v1-K5-v226-lineage at
  seed 42, num_boost_round=10000, early_stopping_rounds=100. It is the
  same configuration that the lab uploaded to PROTEA as the nine
  ``v226full_lineage_<cell>`` ``RerankerModel`` rows (dataset_id
  ``3517bc8b-4562-49e0-8c67-99afc5fdc67f``, external_source
  ``protea-reranker-lab@28d9ce0-study_v23``), registered via
  ``POST /reranker-models/import-by-reference`` on 2026-05-14.

- Baseline arm: ``runs/study_v24_no_lineage/<cell>/run.json``. study_v24
  is the same training configuration with the four ``lineage_*``
  features additionally dropped, so the delta isolates the contribution
  of the lineage feature family.

Metric: ``test_fmax`` from each ``run.json`` (the lab Fmax: protein-grouped
Fmax without label propagation). The canonical cafaeval delta is recorded
separately in ``EXPERIMENTS.md`` under the LB.2 multi-seed sweep section.

Both source studies are seed=42 single-seed runs; this script reports
deltas as point estimates. The multi-seed seed-variance characterisation
is the LB.2 / FARM-EXP.8 work; LR.1 closes the booster training + import
loop, not the variance loop.

The script tolerates missing run.json files (prints "(missing)") so the
table regenerates even on a fresh worktree where the artefacts are
gitignored. The committed CSV is the canonical record.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RERANKER_RUNS = REPO / "runs" / "study_v23"
DEFAULT_BASELINE_RUNS = REPO / "runs" / "study_v24_no_lineage"
OUTPUT_DIR = REPO / "experiments" / "lr1"
OUTPUT_CSV = OUTPUT_DIR / "lineage_delta.csv"

CELLS: tuple[str, ...] = (
    "nk-bpo", "nk-mfo", "nk-cco",
    "lk-bpo", "lk-mfo", "lk-cco",
    "pk-bpo", "pk-mfo", "pk-cco",
)


def _read_fmax(run_dir: Path) -> float | None:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return None
    try:
        payload = json.loads(run_json.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "ok":
        return None
    metrics = payload.get("metrics") or {}
    raw = metrics.get("test_fmax")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _rows(reranker_runs: Path, baseline_runs: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for cell in CELLS:
        reranker_dir = reranker_runs / f"bench-v1-K5-v226-lineage_{cell}"
        baseline_dir = baseline_runs / f"bench-v1-K5-v226-nolineage_{cell}"
        reranker = _read_fmax(reranker_dir)
        baseline = _read_fmax(baseline_dir)
        delta = (reranker - baseline) if (reranker is not None and baseline is not None) else None
        out.append({
            "cell": cell,
            "category": cell.split("-", 1)[0],
            "aspect": cell.split("-", 1)[1],
            "reranker_lab_fmax": reranker,
            "baseline_lab_fmax": baseline,
            "delta": delta,
        })
    return out


def _fmt(x: float | None, digits: int = 4) -> str:
    return f"{x:.{digits}f}" if x is not None else "(missing)"


def _print_table(rows: list[dict[str, object]]) -> None:
    print(f"{'cell':8s}  {'reranker':>10s}  {'baseline':>10s}  {'delta':>9s}")
    print(f"{'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 9}")
    for r in rows:
        print(
            f"{r['cell']:8s}  {_fmt(r['reranker_lab_fmax']):>10s}  "
            f"{_fmt(r['baseline_lab_fmax']):>10s}  {_fmt(r['delta']):>9s}"
        )


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell", "category", "aspect", "reranker_lab_fmax", "baseline_lab_fmax", "delta"])
        for r in rows:
            writer.writerow([
                r["cell"], r["category"], r["aspect"],
                _fmt(r["reranker_lab_fmax"]) if r["reranker_lab_fmax"] is not None else "",
                _fmt(r["baseline_lab_fmax"]) if r["baseline_lab_fmax"] is not None else "",
                _fmt(r["delta"]) if r["delta"] is not None else "",
            ])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--write", action="store_true",
        help=f"write the CSV to {OUTPUT_CSV.relative_to(REPO)} (else stdout only)",
    )
    p.add_argument(
        "--reranker-runs", type=Path, default=DEFAULT_RERANKER_RUNS,
        help="directory holding study_v23 (reranker) run.json artefacts",
    )
    p.add_argument(
        "--baseline-runs", type=Path, default=DEFAULT_BASELINE_RUNS,
        help="directory holding study_v24_no_lineage (baseline) run.json artefacts",
    )
    args = p.parse_args(argv)

    rows = _rows(args.reranker_runs, args.baseline_runs)
    _print_table(rows)
    if args.write:
        _write_csv(rows, OUTPUT_CSV)
        print(f"\n[wrote] {OUTPUT_CSV.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
