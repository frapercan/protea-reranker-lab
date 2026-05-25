#!/usr/bin/env python
"""Per-family feature importance aggregation for any study.

Walks ``runs/<study>/*/run.json``, groups each feature's importance by the
14 canonical families in :data:`protea_contracts.FEATURE_FAMILIES`, and
emits a CSV plus a markdown table. The previous ad-hoc bash one-liner
only surfaced ``lineage_imp_frac``; this lifts the whole tier of evidence
about WHICH feature families the booster actually leans on per cell.

Output:
    runs/<study>/family_importance.csv
    runs/<study>/family_importance.md

Notes:
    * A feature can live in multiple families (e.g. ``distance`` is in
      both ``knn`` and ``knn_distance``). We attribute it to ALL its
      families; row-sums per cell therefore exceed 100%. The "primary"
      column flags whichever family the feature is FIRST defined in.
    * ``feature_count`` per family is the count of features in that
      family that were actually present in the booster's feature
      importance map (after lab drop_features).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from protea_contracts import FEATURE_FAMILIES

REPO = Path(__file__).resolve().parents[1]


def _build_family_index() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return (feature -> [families], feature -> primary family)."""
    feat_to_families: dict[str, list[str]] = defaultdict(list)
    feat_to_primary: dict[str, str] = {}
    for fam, cols in FEATURE_FAMILIES.items():
        for c in cols:
            feat_to_families[c].append(fam)
            feat_to_primary.setdefault(c, fam)
    return feat_to_families, feat_to_primary


def _read_importance(run_json: Path) -> tuple[str, dict[str, float]]:
    r = json.loads(run_json.read_text())
    cell = r.get("spec_name", run_json.parent.name).rsplit("_", 1)[-1]
    fi = r.get("feature_importance") or {}
    if isinstance(fi, list):
        fi = dict(fi)
    return cell, {k: float(v) for k, v in fi.items()}


def _aggregate(
    importance: dict[str, float], feat_to_families: dict[str, list[str]]
) -> dict[str, float]:
    """Return family -> total importance (each feature contributes to all its families)."""
    fam_total: dict[str, float] = defaultdict(float)
    for feat, imp in importance.items():
        for fam in feat_to_families.get(feat, []):
            fam_total[fam] += imp
        if feat not in feat_to_families:
            fam_total["UNKNOWN"] += imp
    return fam_total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True, help="study dir under runs/")
    ap.add_argument(
        "--mode",
        choices=["fractional", "absolute"],
        default="fractional",
        help="report each family as fraction of cell total (default) or absolute gain",
    )
    args = ap.parse_args()

    runs_root = REPO / "runs" / args.study
    if not runs_root.is_dir():
        print(f"[err] {runs_root} missing", file=sys.stderr)
        return 2

    feat_to_families, _ = _build_family_index()
    family_order = list(FEATURE_FAMILIES.keys())

    cells: list[tuple[str, dict[str, float], float]] = []
    for rd in sorted(
        p for p in runs_root.iterdir() if p.is_dir() and p.name != "cafaeval"
    ):
        rj = rd / "run.json"
        if not rj.exists():
            continue
        cell, fi = _read_importance(rj)
        if not fi:
            print(f"[skip] {rd.name}: no feature_importance")
            continue
        total = sum(fi.values()) or 1.0
        fam_total = _aggregate(fi, feat_to_families)
        cells.append((cell, dict(fam_total), total))

    if not cells:
        print(
            f"[err] no runs with feature_importance under {runs_root}", file=sys.stderr
        )
        return 2

    # CSV
    csv_path = runs_root / "family_importance.csv"
    headers = ["cell", "total_importance"] + family_order
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for cell, fam_total, total in cells:
            row = [cell, f"{total:.0f}"]
            for fam in family_order:
                v = fam_total.get(fam, 0.0)
                if args.mode == "fractional":
                    row.append(f"{100 * v / total:.1f}")
                else:
                    row.append(f"{v:.0f}")
            cells_sorted = cells  # noqa: F841
            w.writerow(row)

    # Markdown
    md_path = runs_root / "family_importance.md"
    label = "%" if args.mode == "fractional" else "abs"
    lines = [
        f"# Feature importance by family — {args.study}",
        "",
        f"Each cell shows family contribution ({label}). Each feature is",
        "attributed to ALL families it belongs to per `FEATURE_FAMILIES`,",
        "so row-sums exceed 100% for fractional mode.",
        "",
    ]
    header = "| cell | " + " | ".join(family_order) + " |"
    sep = "|------|" + "|".join(["------:"] * len(family_order)) + "|"
    lines.append(header)
    lines.append(sep)
    for cell, fam_total, total in cells:
        parts = [cell]
        for fam in family_order:
            v = fam_total.get(fam, 0.0)
            if args.mode == "fractional":
                parts.append(f"{100 * v / total:.1f}")
            else:
                parts.append(f"{v:.0f}")
        lines.append("| " + " | ".join(parts) + " |")
    md_path.write_text("\n".join(lines) + "\n")

    print(f"[done] {len(cells)} cells -> {csv_path} + {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
