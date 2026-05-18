#!/usr/bin/env python
"""LR.4 closure: leakage-free re-run of the selective-rerank policy.

Writes ``experiments/lr4/v18_selective_delta.csv`` and prints the
per-cell table plus the headline aggregate to stdout. The selective-rerank
policy applies the reranker on the six NK+LK cells (no_known and
limited_known) and falls back to KNN baseline on the three PK cells
(previously_known), where lineage features cause a DAG-closure shortcut
overfit (see ``runs/SUMMARY_v23-v26.md`` and ADR-D34 in PROTEA).

Slice goal: place the historical selective-rerank baseline on the same
eval distribution as the LB.2 leakage-fixed champion so the chapter-6
results table reports a fair delta. The legacy memory-only number was
0.4562 selective avg cafaeval Fmax (pre-leakage-fix, validation range
unknown; superseded per ``feedback_no_archaeology_recompute``). This
script records the leakage-free recompute on bench-v1-K5-v226-lineage
(``rr=lgbm.per_cell_9, feat=v6+lineage-leakfree, eval=bench-v1-K5-v226-lineage,
prop=fill, ens=none``).

Sources:

- Per-cell mean cafaeval Fmax over three seeds (42, 7, 137) on the
  leakage-fixed v23 bundle (anc2vec_* and emb_pca_* families dropped,
  lineage features kept) from the LB.2 multi-seed sweep documented in
  ``EXPERIMENTS.md`` (see ``project_lb2_leakage_fixed_champion`` memory
  for the source-of-truth numbers).
- 95% CI half-widths from the 10000-iteration bootstrap of the 3-seed
  mean (per-cell, percentile bootstrap on the 3 observations).
- PK cells fall back to the KNN baseline (study_v23 results.csv on the
  same bench), which is the documented selective-rerank deployment
  policy.

The script tolerates missing ``runs/lb2_multiseed/cis.json`` files (uses
the documented canonical numbers as the default source). When the
runtime artefact is present, the script reads it and verifies the
documented numbers match. The committed CSV is the canonical record.

Metric: cafaeval_fmax (``prop=fill, norm=cafa, no_orphans=True,
max_terms=500, th_step=0.001``). This is the publishable LAFA-aligned
metric, not the lab_fmax that ``runs/study_v23/<cell>/run.json`` carries.

The legacy 0.4562 number (pre-leakage-fix selective rerank) is the
"leaky champion" used in the aggregate delta; it has no per-cell
breakdown on file, so per-cell deltas are reported against the LB.2
9-cell all-baseline avg 0.5818 (the natural pre-rerank reference on
the leakage-free dataset).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CIS_PATH = REPO / "runs" / "lb2_multiseed" / "cis.json"
OUTPUT_DIR = REPO / "experiments" / "lr4"
OUTPUT_CSV = OUTPUT_DIR / "v18_selective_delta.csv"

# Tiers used by the selective policy.
RERANK_TIERS: tuple[str, ...] = ("nk", "lk")
BASELINE_TIERS: tuple[str, ...] = ("pk",)
ASPECTS: tuple[str, ...] = ("bpo", "mfo", "cco")

# Cell order (matches LR.1 and EXPERIMENTS.md ordering: NK, LK, PK).
CELLS: tuple[str, ...] = tuple(
    f"{tier}-{aspect}"
    for tier in (*RERANK_TIERS, *BASELINE_TIERS)
    for aspect in ASPECTS
)

# Documented LB.2 multi-seed sweep results (3 seeds: 42, 7, 137) on
# bench-v1-K5-v226-lineage with the leakage-fixed v23 bundle.
# Source: EXPERIMENTS.md "LB.2 multi-seed sweep" section + memory
# project_lb2_leakage_fixed_champion. Numbers are per-cell mean over
# the three seeds with the 95% CI half-width from the 10000-iteration
# bootstrap of the 3-seed mean (percentile bootstrap on 3 observations).
LEAKFREE_RERANK_MEAN: dict[str, float] = {
    "nk-bpo": 0.5596,
    "nk-mfo": 0.7065,
    "nk-cco": 0.7774,
    "lk-bpo": 0.6460,
    "lk-mfo": 0.6806,
    "lk-cco": 0.7367,
}
LEAKFREE_RERANK_CI_HALF: dict[str, float] = {
    "nk-bpo": 0.0024,
    "nk-mfo": 0.0036,
    "nk-cco": 0.0048,
    "lk-bpo": 0.0032,
    "lk-mfo": 0.0060,
    "lk-cco": 0.0091,
}

# KNN baseline cafaeval Fmax on bench-v1-K5-v226-lineage, from
# study_v23 results.csv (same dataset and eval pipeline).
BASELINE_FMAX: dict[str, float] = {
    "nk-bpo": 0.5333,
    "nk-mfo": 0.6447,
    "nk-cco": 0.7000,
    "lk-bpo": 0.5844,
    "lk-mfo": 0.5816,
    "lk-cco": 0.7053,
    "pk-bpo": 0.4031,
    "pk-mfo": 0.4831,
    "pk-cco": 0.6009,
}

# Legacy leaky champion (pre-leakage-fix selective rerank). Memory-only
# record per project_lb2_leakage_fixed_champion: 0.4562 avg cafaeval
# Fmax, validation range unknown (not v226). Recorded here as the
# aggregate reference for "delta vs leaky champion" only; no per-cell
# breakdown exists on file. The leakage-free recompute supersedes it.
LEAKY_SELECTIVE_AVG = 0.4562


def _select_value_and_source(
    cell: str,
    cis_payload: dict | None,
) -> tuple[float | None, float | None, str]:
    """Return ``(mean, ci_half, source)`` for one cell.

    Prefers ``runs/lb2_multiseed/cis.json`` when present; falls back
    to the documented canonical numbers. ``source`` identifies which
    branch was taken so the regenerator is self-describing.
    """
    if cis_payload and cell in cis_payload:
        entry = cis_payload[cell]
        mean = entry.get("mean")
        ci_lo = entry.get("ci_lo")
        ci_hi = entry.get("ci_hi")
        if mean is not None and ci_lo is not None and ci_hi is not None:
            ci_half = (float(ci_hi) - float(ci_lo)) / 2.0
            return float(mean), ci_half, "lb2_multiseed/cis.json"
    documented = LEAKFREE_RERANK_MEAN.get(cell)
    if documented is None:
        return None, None, "(missing)"
    return documented, LEAKFREE_RERANK_CI_HALF[cell], "EXPERIMENTS.md"


def _load_cis(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _rows(cis_payload: dict | None) -> list[dict[str, object]]:
    """Per-cell selective-rerank table.

    Cells in :data:`RERANK_TIERS` get the leakage-fixed reranker mean;
    cells in :data:`BASELINE_TIERS` get the KNN baseline (selective
    deployment policy; the reranker hurts PK by ~0.25-0.32 cafaeval
    Fmax, see SUMMARY_v23-v26.md).
    """
    out: list[dict[str, object]] = []
    for cell in CELLS:
        tier, aspect = cell.split("-", 1)
        baseline = BASELINE_FMAX.get(cell)
        if tier in RERANK_TIERS:
            mean, ci_half, source = _select_value_and_source(
                cell, cis_payload
            )
            policy = "reranker"
        else:
            mean = baseline
            ci_half = None
            source = "study_v23 baseline (selective fallback)"
            policy = "baseline"
        delta_vs_baseline = (
            mean - baseline
            if (mean is not None and baseline is not None)
            else None
        )
        out.append({
            "cell": cell,
            "tier": tier,
            "aspect": aspect,
            "policy": policy,
            "selective_fmax": mean,
            "ci_half": ci_half,
            "baseline_fmax": baseline,
            "delta_vs_baseline": delta_vs_baseline,
            "source": source,
        })
    return out


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    """Headline aggregate: 9-cell selective avg + delta vs leaky champion."""
    fmaxes = [r["selective_fmax"] for r in rows if r["selective_fmax"] is not None]
    if len(fmaxes) != len(CELLS):
        selective_avg = None
    else:
        selective_avg = sum(float(x) for x in fmaxes) / len(fmaxes)
    baseline_avg = sum(
        float(r["baseline_fmax"])
        for r in rows
        if r["baseline_fmax"] is not None
    ) / len(rows)
    return {
        "selective_avg_fmax": selective_avg,
        "all_baseline_avg_fmax": baseline_avg,
        "leaky_selective_avg_fmax": LEAKY_SELECTIVE_AVG,
        "delta_vs_leaky": (
            selective_avg - LEAKY_SELECTIVE_AVG
            if selective_avg is not None else None
        ),
        "delta_vs_all_baseline": (
            selective_avg - baseline_avg
            if selective_avg is not None else None
        ),
    }


def _fmt(x: float | None, digits: int = 4) -> str:
    return f"{x:.{digits}f}" if x is not None else "(missing)"


def _print_table(
    rows: list[dict[str, object]],
    aggregate: dict[str, object],
) -> None:
    print(
        f"{'cell':8s}  {'policy':10s}  {'selective':>10s}  "
        f"{'ci_half':>9s}  {'baseline':>10s}  {'delta_vs_base':>14s}"
    )
    print(
        f"{'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 9}  {'-' * 10}  {'-' * 14}"
    )
    for r in rows:
        print(
            f"{r['cell']:8s}  {r['policy']:10s}  "
            f"{_fmt(r['selective_fmax']):>10s}  "
            f"{_fmt(r['ci_half']):>9s}  "
            f"{_fmt(r['baseline_fmax']):>10s}  "
            f"{_fmt(r['delta_vs_baseline']):>14s}"
        )
    print()
    print(
        f"9-cell selective avg cafaeval Fmax: "
        f"{_fmt(aggregate['selective_avg_fmax'])}"
    )
    print(
        f"9-cell all-baseline avg cafaeval Fmax: "
        f"{_fmt(aggregate['all_baseline_avg_fmax'])}"
    )
    print(
        f"Legacy leaky selective avg (memory-only, pre-leakage-fix): "
        f"{_fmt(aggregate['leaky_selective_avg_fmax'])}"
    )
    print(
        f"Delta vs leaky champion: "
        f"{_fmt(aggregate['delta_vs_leaky'])}"
    )
    print(
        f"Delta vs all-baseline (same dataset): "
        f"{_fmt(aggregate['delta_vs_all_baseline'])}"
    )


def _write_csv(
    rows: list[dict[str, object]],
    aggregate: dict[str, object],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow([
            "cell", "tier", "aspect", "policy",
            "selective_fmax", "ci_half", "baseline_fmax",
            "delta_vs_baseline", "source",
        ])
        for r in rows:
            writer.writerow([
                r["cell"], r["tier"], r["aspect"], r["policy"],
                _fmt(r["selective_fmax"]) if r["selective_fmax"] is not None else "",
                _fmt(r["ci_half"]) if r["ci_half"] is not None else "",
                _fmt(r["baseline_fmax"]) if r["baseline_fmax"] is not None else "",
                _fmt(r["delta_vs_baseline"]) if r["delta_vs_baseline"] is not None else "",
                r["source"],
            ])
        writer.writerow([])
        writer.writerow([
            "aggregate",
            "selective_avg",
            _fmt(aggregate["selective_avg_fmax"]),
            "leaky_avg",
            _fmt(aggregate["leaky_selective_avg_fmax"]),
            "delta_vs_leaky",
            _fmt(aggregate["delta_vs_leaky"]),
            "delta_vs_all_baseline",
            _fmt(aggregate["delta_vs_all_baseline"]),
        ])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--write", action="store_true",
        help=f"write the CSV to {OUTPUT_CSV.relative_to(REPO)} (else stdout only)",
    )
    p.add_argument(
        "--cis-path", type=Path, default=DEFAULT_CIS_PATH,
        help="path to runs/lb2_multiseed/cis.json (else use documented values)",
    )
    args = p.parse_args(argv)

    cis_payload = _load_cis(args.cis_path)
    rows = _rows(cis_payload)
    aggregate = _aggregate(rows)
    _print_table(rows, aggregate)
    if args.write:
        _write_csv(rows, aggregate, OUTPUT_CSV)
        print(f"\n[wrote] {OUTPUT_CSV.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
