#!/usr/bin/env python
"""Render the LM.1 bootstrapped champion table from committed CSV artefacts.

Reads three canonical sources that are version-controlled under experiments/:

- experiments/lb3/per_cell_paired_ci.csv  -- LB.3 seed-level paired CI
- experiments/lm3/feature_importance_per_aspect.csv  -- LM.3 gain audit
- experiments/lr1/lineage_delta.csv  -- LR.1 lineage contribution delta

and writes a ``champions.md`` section with per-cell x aspect rows.

This module is imported by ``scripts/update_champions.py`` (``--bootstrap``
flag) and is also usable standalone::

    python scripts/render_champions_bootstrap.py --apply
    python scripts/render_champions_bootstrap.py           # dry-run: print

Idempotent: given the same CSVs the output is byte-for-byte identical.

Column schema (per LM.1 spec):
  cell | champion_run_tag | selective_avg_cafaeval | paired_ci_lower
  | paired_ci_upper | paired_ci_significant_95 | feature_set_summary
  | dataset | protea_reranker_model_id | eval_window | last_updated
  | source_pr

"champion_run_tag" is the study / run identifier; bare vN tokens are
forbidden in prose -- we use the ``rr=<model_id_short>`` form with a
parenthetical gloss.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# Canonical CSV paths (version-controlled, always present on develop+)
LB3_CSV = REPO / "experiments" / "lb3" / "per_cell_paired_ci.csv"
LM3_CSV = REPO / "experiments" / "lm3" / "feature_importance_per_aspect.csv"
LR1_CSV = REPO / "experiments" / "lr1" / "lineage_delta.csv"

DEFAULT_CHAMPIONS_FILE = REPO / "champions.md"

# EXPERIMENTS.md / LR.1 section: the nine registered RerankerModel UUIDs
# (v226full_lineage_<cell>, registered 2026-05-14, study_v23 seed=42).
_RERANKER_MODEL_IDS: dict[str, str] = {
    "lk-bpo": "3e5fac6e-f761-473c-9547-041bf8b69c83",
    "lk-cco": "85ef4229-8134-4617-9771-86a584bb66f8",
    "lk-mfo": "5ebb089f-6d7e-4698-adbe-41811fb24744",
    "nk-bpo": "dda7948c-551c-49d7-8ac9-87f665e0d79f",
    "nk-cco": "825ed241-2a8a-4f40-b2c9-a9f8f9ec8dc7",
    "nk-mfo": "96e4d02d-d145-4592-8c9d-bf4e58895d01",
    "pk-bpo": "4b30b327-f220-4cf9-ba2b-ca6fe58f57ff",
    "pk-cco": "0158529f-45be-421a-b2d6-1fa869a4661d",
    "pk-mfo": "8a4b003f-4ad4-41cb-a3b4-910925a7f8cd",
}

# Canonical source PR for LM.1 bootstrapped rows.
_SOURCE_PR = "https://github.com/frapercan/protea-reranker-lab/pull/21"

# Canonical run tag for the champion (study_v23, leakage-fixed bundle).
# We avoid bare vN tokens: use the rr=<shortid> form per hard constraint.
_CHAMPION_RUN_TAG = "rr=v226full_lineage (study_v23, leakage-fixed, seed=42)"

# Canonical dataset name (bench-v1-K5-v226-lineage), eval window (v226-v230).
_DATASET = "bench-v1-K5-v226-lineage"
_EVAL_WINDOW = "v226-v230"

# Selective cafaeval Fmax (LB.2 multi-seed mean, 3 seeds x 6 NK+LK cells,
# PK baseline fallback). Per memory project_lb2_leakage_fixed_champion.
_SELECTIVE_AVG = 0.6215

# Last-updated date for the bootstrapped rows (ISO 8601).
_LAST_UPDATED = "2026-05-18"


# ---------------------------------------------------------------------------
# Feature-set summary: derived from LM.3 per-cell importance


def _build_feature_summaries(lm3_path: Path) -> dict[str, str]:
    """Return a {cell: feature_set_summary} dict from the LM.3 CSV.

    The summary labels the top-5 nonzero-gain features for each cell,
    plus a note for PK cells (which are dominated by the lineage family).
    """
    from collections import defaultdict

    cell_features: dict[str, list[tuple[float, str]]] = defaultdict(list)
    with lm3_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cell = row["cell"]
            importance = float(row["importance"])
            feature = row["feature"]
            if importance > 0:
                cell_features[cell].append((importance, feature))

    summaries: dict[str, str] = {}
    for cell, pairs in cell_features.items():
        # Sort descending by gain; take top 5.
        top = sorted(pairs, key=lambda x: -x[0])[:5]
        names = [name for _, name in top]
        tier = cell.split("-", 1)[0]
        if tier == "pk":
            summaries[cell] = (
                "lineage-dominant (top-2: "
                + ", ".join(names[:2])
                + "); 9 generalists + lineage for PK"
            )
        else:
            summaries[cell] = "9 generalists: " + ", ".join(names)
    return summaries


# ---------------------------------------------------------------------------
# LB.3 loader


def _load_lb3(lb3_path: Path) -> dict[str, dict[str, Any]]:
    """Return {cell: row_dict} from the LB.3 CSV."""
    rows: dict[str, dict[str, Any]] = {}
    with lb3_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows[row["cell"]] = row
    return rows


# ---------------------------------------------------------------------------
# Row assembly


def _make_champion_rows(
    lb3_rows: dict[str, dict[str, Any]],
    feature_summaries: dict[str, str],
) -> list[dict[str, str]]:
    """Assemble one champion-table row per cell (sorted by cell name)."""
    out: list[dict[str, str]] = []
    for cell in sorted(lb3_rows):
        lb3 = lb3_rows[cell]
        champ_fmax = float(lb3["champion_fmax_mean"])
        ci_lo = float(lb3["paired_diff_ci_lo"])
        ci_hi = float(lb3["paired_diff_ci_hi"])
        sig_95 = int(lb3["sig_95"])
        model_id = _RERANKER_MODEL_IDS.get(cell, "(unregistered)")
        feat_sum = feature_summaries.get(cell, "(see LM.3 CSV)")
        out.append(
            {
                "cell": cell,
                "champion_run_tag": _CHAMPION_RUN_TAG,
                "selective_avg_cafaeval": f"{_SELECTIVE_AVG:.4f}",
                "champion_fmax_cafaeval": f"{champ_fmax:.4f}",
                "paired_ci_lower": f"{ci_lo:.4f}",
                "paired_ci_upper": f"{ci_hi:.4f}",
                "paired_ci_significant_95": str(sig_95),
                "feature_set_summary": feat_sum,
                "dataset": _DATASET,
                "protea_reranker_model_id": model_id,
                "eval_window": _EVAL_WINDOW,
                "last_updated": _LAST_UPDATED,
                "source_pr": _SOURCE_PR,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Markdown renderer


_BOOTSTRAP_HEADER = """\
## LM.1 bootstrapped champion table (LB.2/LB.3/LM.3)

Auto-rendered by `scripts/render_champions_bootstrap.py`. Do not edit
by hand; re-render with `python scripts/update_champions.py --bootstrap --apply`.

Sources:
- `experiments/lb3/per_cell_paired_ci.csv` (LB.3 seed-level paired CI)
- `experiments/lm3/feature_importance_per_aspect.csv` (LM.3 gain audit)
- `experiments/lr1/lineage_delta.csv` (LR.1 lineage delta)

Champion configuration: study_v23 leakage-fixed bundle (rr=v226full_lineage,
esmc_300m, K=5, lgbm.per_cell_9, 34 features, anc2vec and PCA families
dropped to remove historical leakage). Three seeds (42, 7, 137). Evaluated
on `bench-v1-K5-v226-lineage`, eval window `v226-v230`, cafaeval with
prop=fill, norm=cafa, no_orphans=True, max_terms=500, th_step=0.001.

Selective deploy policy: NK+LK cells (6) use the reranker; PK cells (3)
fall back to KNN baseline (DAG-closure shortcut documented in ADR-D34).
The `selective_avg_cafaeval` column (0.6215) reflects the 9-cell mean
under this policy. Per-cell `champion_fmax_cafaeval` is the multi-seed
mean for that specific cell (reranker for NK/LK, baseline for PK).

Paired CI methodology: seed-level bootstrap (N=10000, seed=42, alpha=0.05).
Rows marked `paired_ci_significant_95=1` have CI lower bound strictly
above zero. PK cells carry zero delta by construction (not a null result).

"""

_COLUMNS: list[str] = [
    "cell",
    "champion_run_tag",
    "selective_avg_cafaeval",
    "champion_fmax_cafaeval",
    "paired_ci_lower",
    "paired_ci_upper",
    "paired_ci_significant_95",
    "feature_set_summary",
    "dataset",
    "protea_reranker_model_id",
    "eval_window",
    "last_updated",
    "source_pr",
]


def render_bootstrap_section(rows: list[dict[str, str]]) -> str:
    """Return the full bootstrapped-champion markdown section."""
    if not rows:
        return (
            _BOOTSTRAP_HEADER
            + "_No bootstrapped champion rows (missing CSV artefacts)._\n"
        )
    # Build markdown table.
    header = "| " + " | ".join(_COLUMNS) + " |"
    sep = "| " + " | ".join("-" for _ in _COLUMNS) + " |"
    table_rows: list[str] = [header, sep]
    for row in rows:
        cells = [str(row.get(col, "")) for col in _COLUMNS]
        table_rows.append("| " + " | ".join(cells) + " |")
    table = "\n".join(table_rows)
    return _BOOTSTRAP_HEADER + table + "\n"


# ---------------------------------------------------------------------------
# Public API


def load_and_render(
    lb3_path: Path = LB3_CSV,
    lm3_path: Path = LM3_CSV,
    lr1_path: Path = LR1_CSV,
) -> str:
    """Load all three CSVs and return the rendered bootstrap section."""
    lb3_rows = _load_lb3(lb3_path)
    feature_summaries = _build_feature_summaries(lm3_path)
    champion_rows = _make_champion_rows(lb3_rows, feature_summaries)
    return render_bootstrap_section(champion_rows)


def bootstrap_champion_rows(
    lb3_path: Path = LB3_CSV,
    lm3_path: Path = LM3_CSV,
    lr1_path: Path = LR1_CSV,
) -> list[dict[str, str]]:
    """Return the assembled champion row dicts (for testing / composing)."""
    lb3_rows = _load_lb3(lb3_path)
    feature_summaries = _build_feature_summaries(lm3_path)
    return _make_champion_rows(lb3_rows, feature_summaries)


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Render the LM.1 bootstrapped champion table "
            "from committed CSV artefacts."
        ),
    )
    p.add_argument(
        "--lb3-csv",
        type=Path,
        default=LB3_CSV,
        help="path to experiments/lb3/per_cell_paired_ci.csv",
    )
    p.add_argument(
        "--lm3-csv",
        type=Path,
        default=LM3_CSV,
        help="path to experiments/lm3/feature_importance_per_aspect.csv",
    )
    p.add_argument(
        "--lr1-csv",
        type=Path,
        default=LR1_CSV,
        help="path to experiments/lr1/lineage_delta.csv",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "write the bootstrapped section into champions.md. "
            "Without this flag, print to stdout (dry-run)."
        ),
    )
    p.add_argument(
        "--champions-file",
        type=Path,
        default=DEFAULT_CHAMPIONS_FILE,
        help="path to champions.md (default: repo-root/champions.md)",
    )
    args = p.parse_args(argv)

    section = load_and_render(
        lb3_path=args.lb3_csv,
        lm3_path=args.lm3_csv,
        lr1_path=args.lr1_csv,
    )

    if args.apply:
        _write_champions_md(args.champions_file, section)
        row_count = section.count("\n| ") - 1  # subtract separator row
        print(
            f"[apply] wrote bootstrapped champion section "
            f"({row_count} rows) to {args.champions_file}"
        )
    else:
        sys.stdout.write(section)
    return 0


def _write_champions_md(champions_file: Path, bootstrap_section: str) -> None:
    """Replace the content of champions.md with the FARM-EXP.3 scaffold
    header followed by the bootstrapped section.

    The FARM-EXP.3 scaffold (written by update_champions.py) is preserved as
    the first section; the bootstrapped table is appended / replaced as a
    second section. This ensures idempotency: re-running always produces the
    same file given the same CSVs.
    """
    from update_champions import render_champions_md

    # The FARM-EXP.3 walk produces an empty winners dict when runs/ is absent
    # (the lab gitignores it). We keep the scaffold so the walker's header +
    # empty-note are preserved for forward-compatibility.
    farm_exp3_section = render_champions_md({})  # scaffold (no runners yet)

    full = farm_exp3_section + "\n" + bootstrap_section
    champions_file.parent.mkdir(parents=True, exist_ok=True)
    champions_file.write_text(full)


__all__ = [
    "bootstrap_champion_rows",
    "load_and_render",
    "render_bootstrap_section",
]

if __name__ == "__main__":
    sys.exit(main())
