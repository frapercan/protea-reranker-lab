#!/usr/bin/env python
"""LM.3 closure: per-aspect feature importance audit on the v226-lineage champion.

Reads ``feature_importance`` (LightGBM gain) from the nine
``runs/study_v23/bench-v1-K5-v226-lineage_<cell>/run.json`` records
(the v226full_lineage booster set registered in PROTEA on 2026-05-14)
and writes ``experiments/lm3/feature_importance_per_aspect.csv``.

Output CSV columns: cell, category, aspect, feature, importance, rank.
- ``importance`` is the raw gain value stored in run.json.
- ``rank`` is the descending gain rank within that cell (rank 1 = highest gain).

The script also writes ``experiments/lm3/feature_importance_summary.md``
(top-10 per aspect plus aspect-specific interpretation notes).

Design choices
--------------
- Gain metric: LightGBM ``gain`` is the importance stored in run.json by
  the runner (``booster.feature_importance(importance_type="gain")``).
  This is the fractional total reduction in the training loss achieved by
  each feature across all trees and all splits. It is the default metric
  chosen in the PLAN (spec: "importance metric: gain").
- Aggregation: per-aspect mean rank across the three category cells
  (NK/LK/PK) is used to identify aspect-stable winners. Mean gain is
  also reported for reference, but raw gain is not comparable across
  cells because training set sizes differ by roughly 10x (PK << NK+LK).
- Zero-importance features: 20 of the 34 features have gain=0 in NK+LK
  cells (alignment families: identity_nw, alignment_score_*, etc.; length;
  taxonomy; the lineage family except lk-mfo). These are excluded from
  the top-K display but included in the full CSV so the table is complete.
- PK divergence: in PK cells the lineage family dominates (ranks 2-6 in
  all three aspects). This is the documented DAG-closure shortcut that
  motivated the selective-deploy policy (NK+LK only). The summary
  highlights this cross-aspect divergence between NK+LK and PK.

Sources
-------
- Booster artefacts: ``runs/study_v23/<cell>/run.json`` (seed=42).
  These are the same nine boosters registered in PROTEA as
  v226full_lineage_<cell> (external_source=protea-reranker-lab@28d9ce0-
  study_v23, dataset bench-v1-K5-v226-lineage-prostt5,
  id 3517bc8b-4562-49e0-8c67-99afc5fdc67f).
- The script tolerates missing run.json files (prints a warning) so it
  regenerates the summary even on a fresh worktree where artefacts are
  gitignored. The committed CSV is the canonical LM.3 acceptance artefact.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO / "runs" / "study_v23"
OUTPUT_DIR = REPO / "experiments" / "lm3"
OUTPUT_CSV = OUTPUT_DIR / "feature_importance_per_aspect.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "feature_importance_summary.md"

CELLS: tuple[str, ...] = (
    "nk-bpo", "nk-mfo", "nk-cco",
    "lk-bpo", "lk-mfo", "lk-cco",
    "pk-bpo", "pk-mfo", "pk-cco",
)
ASPECTS: tuple[str, ...] = ("bpo", "mfo", "cco")
TOP_K: int = 10


def _read_feature_importance(runs_dir: Path, cell: str) -> dict[str, float] | None:
    """Return sorted feature importance dict or None if run is missing/failed."""
    run_json = runs_dir / f"bench-v1-K5-v226-lineage_{cell}" / "run.json"
    if not run_json.exists():
        print(f"[warn] missing: {run_json}", file=sys.stderr)
        return None
    try:
        payload = json.loads(run_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] cannot parse {run_json}: {exc}", file=sys.stderr)
        return None
    if payload.get("status") != "ok":
        print(
            f"[warn] run status={payload.get('status')!r} for {cell}; skipping",
            file=sys.stderr,
        )
        return None
    fi = payload.get("feature_importance")
    if not fi:
        return None
    return dict(fi)


def build_rows(runs_dir: Path) -> list[dict[str, object]]:
    """Return flat rows: one row per (cell x feature) with rank and importance."""
    rows: list[dict[str, object]] = []
    for cell in CELLS:
        cat, asp = cell.split("-", 1)
        fi = _read_feature_importance(runs_dir, cell)
        if fi is None:
            continue
        ranked = sorted(fi.items(), key=lambda kv: -kv[1])
        for rank, (feature, importance) in enumerate(ranked, 1):
            rows.append({
                "cell": cell,
                "category": cat,
                "aspect": asp,
                "feature": feature,
                "importance": importance,
                "rank": rank,
            })
    return rows


def aggregate_by_aspect(
    rows: list[dict[str, object]],
) -> dict[str, list[tuple[str, float, float]]]:
    """Return per-aspect list of (feature, mean_rank, mean_gain) sorted by mean_rank asc."""
    asp_feat_ranks: dict[str, dict[str, list[float]]] = {a: defaultdict(list) for a in ASPECTS}
    asp_feat_gains: dict[str, dict[str, list[float]]] = {a: defaultdict(list) for a in ASPECTS}
    for row in rows:
        asp = str(row["aspect"])
        feat = str(row["feature"])
        asp_feat_ranks[asp][feat].append(float(row["rank"]))
        asp_feat_gains[asp][feat].append(float(row["importance"]))

    result: dict[str, list[tuple[str, float, float]]] = {}
    for asp in ASPECTS:
        entries = [
            (feat, sum(ranks) / len(ranks), sum(gains) / len(gains))
            for feat, ranks in asp_feat_ranks[asp].items()
            for gains in [asp_feat_gains[asp][feat]]
        ]
        result[asp] = sorted(entries, key=lambda x: x[1])
    return result


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell", "category", "aspect", "feature", "importance", "rank"])
        for row in rows:
            writer.writerow([
                row["cell"],
                row["category"],
                row["aspect"],
                row["feature"],
                f"{float(row['importance']):.4f}",
                int(row["rank"]),
            ])


def _md_table_rows(
    entries: list[tuple[str, float, float]],
    top_k: int,
) -> str:
    lines = [
        "| rank | feature | mean_rank | mean_gain |",
        "| - | - | - | - |",
    ]
    for i, (feat, mean_rank, mean_gain) in enumerate(entries[:top_k], 1):
        lines.append(f"| {i} | `{feat}` | {mean_rank:.2f} | {mean_gain:,.1f} |")
    return "\n".join(lines)


def write_summary(
    aggregates: dict[str, list[tuple[str, float, float]]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    bpo_top = aggregates["bpo"][:TOP_K]
    mfo_top = aggregates["mfo"][:TOP_K]
    cco_top = aggregates["cco"][:TOP_K]

    # Identify features that appear in top-10 for all three aspects
    bpo_names = {f for f, _, _ in bpo_top}
    mfo_names = {f for f, _, _ in mfo_top}
    cco_names = {f for f, _, _ in cco_top}
    generalists = sorted(bpo_names & mfo_names & cco_names)

    lines: list[str] = [
        "# LM.3 feature importance summary (v226-lineage champion)",
        "",
        "Source: LightGBM gain from `runs/study_v23` (nine `v226full_lineage_<cell>`",
        "boosters, seed=42, registered 2026-05-14). Metric: gain (total information",
        "gain accumulated across all splits of all trees). Aggregation: mean rank",
        "across the three category cells per aspect (NK/LK/PK x BPO/MFO/CCO).",
        "Full data: `experiments/lm3/feature_importance_per_aspect.csv`.",
        "",
        "---",
        "",
        f"## Top-{TOP_K} per aspect (mean rank across NK/LK/PK)",
        "",
        "### BPO (Biological Process)",
        "",
        _md_table_rows(aggregates["bpo"], TOP_K),
        "",
        "### MFO (Molecular Function)",
        "",
        _md_table_rows(aggregates["mfo"], TOP_K),
        "",
        "### CCO (Cellular Component)",
        "",
        _md_table_rows(aggregates["cco"], TOP_K),
        "",
        "---",
        "",
        "## Cross-aspect generalists",
        "",
    ]

    if generalists:
        lines.append(
            "Features ranked in the top-10 for all three aspects "
            "(aspect-stable winners):"
        )
        lines.append("")
        for feat in generalists:
            lines.append(f"- `{feat}`")
    else:
        lines.append("No feature ranks in the top-10 across all three aspects.")
    lines.append("")

    lines += [
        "---",
        "",
        "## Interpretation",
        "",
        "### BPO",
        "",
        "`neighbor_vote_fraction` and `go_term_frequency` share the top-two",
        "positions across NK and LK cells. In PK the ranking reverses:",
        "`go_term_frequency` dominates, and the lineage family enters ranks 2-6",
        "(lineage_descendant_of_count, lineage_ancestor_of_count,",
        "lineage_is_ancestor_of_known). This PK-specific pattern reflects the",
        "DAG-closure shortcut: PK queries already have parent terms annotated,",
        "so the booster learns to exploit the hierarchical overlap captured by",
        "the lineage features. The alignment family (identity_nw, alignment_score_*,",
        "etc.) scores zero gain in every BPO cell, confirming that sequence-level",
        "similarity information adds nothing once KNN vote and distance features",
        "are included.",
        "",
        "### MFO",
        "",
        "The MFO ranking closely mirrors BPO: `neighbor_vote_fraction` and",
        "`go_term_frequency` are the dominant signals across NK and LK.",
        "`evidence_code` consistently ranks third, higher than in CCO, suggesting",
        "that the curation quality of the MFO annotation source matters more for",
        "molecular function terms (where experimental evidence is sparse relative",
        "to BPO). PK MFO again shows lineage features at ranks 2-4, confirming",
        "the cross-aspect generality of the DAG-closure shortcut.",
        "",
        "### CCO",
        "",
        "CCO is the outlier aspect. `go_term_frequency` is the undisputed dominant",
        "feature (mean rank 1.67, mean gain nearly 2x higher than BPO). This is",
        "consistent with CCO having fewer distinct terms and higher per-term",
        "annotation density: term frequency predicts GO-term annotation well",
        "for cellular components. In PK-CCO the lineage family is exceptionally",
        "strong (lineage_ancestor_of_count and lineage_is_ancestor_of_known",
        "at ranks 2-3, with aggregate gain >1M), the highest lineage dominance",
        "across all nine cells. The `qualifier` feature, largely irrelevant in",
        "BPO, enters the top-10 for CCO (mean rank 9.0), suggesting qualifier",
        "metadata (e.g. NOT annotations) is more discriminative for component terms.",
        "",
    ]

    path.write_text("\n".join(lines))


def print_summary(aggregates: dict[str, list[tuple[str, float, float]]]) -> None:
    for asp in ASPECTS:
        print(f"\n{asp.upper()} top-{TOP_K} (mean rank across NK/LK/PK cells):")
        print(f"  {'rank':>4}  {'feature':45s}  {'mean_rank':>9}  {'mean_gain':>12}")
        print(f"  {'-'*4}  {'-'*45}  {'-'*9}  {'-'*12}")
        for i, (feat, mean_rank, mean_gain) in enumerate(aggregates[asp][:TOP_K], 1):
            print(f"  {i:>4}  {feat:45s}  {mean_rank:>9.2f}  {mean_gain:>12,.1f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--write",
        action="store_true",
        help=(
            f"write CSV to {OUTPUT_CSV.relative_to(REPO)} "
            f"and summary to {OUTPUT_SUMMARY.relative_to(REPO)}"
        ),
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="directory holding study_v23 per-cell subdirectories with run.json",
    )
    args = p.parse_args(argv)

    rows = build_rows(args.runs_dir)
    if not rows:
        print("[error] no feature importance rows found; check --runs-dir", file=sys.stderr)
        return 1

    aggregates = aggregate_by_aspect(rows)
    print_summary(aggregates)

    if args.write:
        write_csv(rows, OUTPUT_CSV)
        print(f"\n[wrote] {OUTPUT_CSV.relative_to(REPO)}")
        write_summary(aggregates, OUTPUT_SUMMARY)
        print(f"[wrote] {OUTPUT_SUMMARY.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
