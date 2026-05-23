#!/usr/bin/env python
"""Aggregate v9-study CSVs/JSONs into ``runs/study_v9/SUMMARY.md``.

Idempotent: reads whatever phases are complete and produces a markdown file
with one section per phase. Missing phases are flagged with a ``(pending)``
note instead of erroring.
"""

from __future__ import annotations

import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "runs" / "study_v9"
SUMMARY = ROOT / "SUMMARY.md"

CELLS = [
    "nk-bpo", "nk-mfo", "nk-cco",
    "lk-bpo", "lk-mfo", "lk-cco",
    "pk-bpo", "pk-mfo", "pk-cco",
]
CELLS_REPRESENTATIVE = ["nk-bpo", "lk-cco", "pk-mfo"]


def _read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return list(csv.DictReader(p.open()))


def _f(x: object) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _section_replication() -> str:
    rows = _read_csv(ROOT / "replication" / "results.csv")
    if not rows:
        return "## 1. Replication (3 seeds × 9 cells)\n\n_(pending)_\n"
    by_cell: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        spec = r["spec"]
        cell = spec.rsplit("_seed", 1)[0]
        v = _f(r.get("fmax"))
        if v is not None:
            by_cell[cell].append(v)

    lines = [
        "## 1. Replication (3 seeds × 9 cells)",
        "",
        "| cell | seed42 | seed7 | seed137 | mean | std |",
        "|---|---|---|---|---|---|",
    ]
    cell_order = [c for c in CELLS if c in by_cell] + sorted(set(by_cell) - set(CELLS))

    seed_lookup: dict[tuple[str, int], float] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        spec = r["spec"]
        cell = spec.rsplit("_seed", 1)[0]
        seed = int(spec.rsplit("_seed", 1)[1])
        v = _f(r.get("fmax"))
        if v is not None:
            seed_lookup[(cell, seed)] = v

    cell_means: list[float] = []
    for cell in cell_order:
        vals = by_cell[cell]
        s42 = seed_lookup.get((cell, 42))
        s7 = seed_lookup.get((cell, 7))
        s137 = seed_lookup.get((cell, 137))
        m = stats.mean(vals)
        cell_means.append(m)
        sd = stats.pstdev(vals) if len(vals) > 1 else 0.0
        lines.append(
            f"| {cell} | "
            f"{f'{s42:.4f}' if s42 is not None else '—'} | "
            f"{f'{s7:.4f}' if s7 is not None else '—'} | "
            f"{f'{s137:.4f}' if s137 is not None else '—'} | "
            f"**{m:.4f}** | {sd:.4f} |"
        )
    if cell_means:
        avg = stats.mean(cell_means)
        lines.append("")
        lines.append(f"**avg_fmax across cells: {avg:.4f}** "
                     f"(historical hybrid ceiling: 0.4400 cafaeval — *not directly comparable, "
                     f"see Fase 5*)")
    return "\n".join(lines) + "\n"


def _section_ablation() -> str:
    rows = _read_csv(ROOT / "ablation" / "results.csv")
    if not rows:
        return "## 2. Ablation (leave-one-family-out, 3 representative cells)\n\n_(pending)_\n"
    families: list[str] = []
    by_cell_fam: dict[tuple[str, str], float] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        spec = r["spec"]
        cell, _, family = spec.partition("_drop_")
        if not family:
            continue
        v = _f(r.get("fmax"))
        if v is None:
            continue
        by_cell_fam[(cell, family)] = v
        if family not in families:
            families.append(family)
    families = sorted(families)

    full_by_cell: dict[str, float] = {}
    rep_csv = ROOT / "replication" / "results.csv"
    for r in _read_csv(rep_csv):
        if r.get("status") != "ok":
            continue
        spec = r["spec"]
        cell = spec.rsplit("_seed", 1)[0]
        seed = int(spec.rsplit("_seed", 1)[1])
        if seed != 42 or cell not in CELLS_REPRESENTATIVE:
            continue
        v = _f(r.get("fmax"))
        if v is not None:
            full_by_cell[cell] = v

    lines = [
        "## 2. Ablation (leave-one-family-out, 3 representative cells)",
        "",
        "_Δfmax = fmax(full) − fmax(drop). Larger positive Δ = family carries more signal._",
        "",
        f"| family | {' | '.join(CELLS_REPRESENTATIVE)} | mean Δ |",
        "|---|" + "|".join("---" for _ in CELLS_REPRESENTATIVE) + "|---|",
    ]
    for fam in families:
        deltas: list[float] = []
        cell_cells = []
        for cell in CELLS_REPRESENTATIVE:
            full = full_by_cell.get(cell)
            drop = by_cell_fam.get((cell, fam))
            if full is None or drop is None:
                cell_cells.append("—")
                continue
            d = full - drop
            deltas.append(d)
            cell_cells.append(f"{d:+.4f}")
        mean_delta = stats.mean(deltas) if deltas else None
        cell_cells.append(f"{mean_delta:+.4f}" if mean_delta is not None else "—")
        lines.append(f"| {fam} | {' | '.join(cell_cells)} |")
    return "\n".join(lines) + "\n"


def _section_bootstrap() -> str:
    rows = _read_csv(ROOT / "bootstrap" / "results.csv")
    if not rows:
        return "## 3. Bootstrap CIs vs KNN baseline\n\n_(pending)_\n"
    lines = [
        "## 3. Bootstrap CIs vs KNN baseline",
        "",
        "| cell | fmax_v9 | CI_v9 | fmax_baseline | CI_baseline | Δ | CI_Δ | p (one-sided) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cell = r.get("cell", "?")
        f9 = _f(r.get("fmax_v9"))
        f9_lo = _f(r.get("fmax_v9_ci_lo"))
        f9_hi = _f(r.get("fmax_v9_ci_hi"))
        fb = _f(r.get("fmax_baseline"))
        fb_lo = _f(r.get("fmax_baseline_ci_lo"))
        fb_hi = _f(r.get("fmax_baseline_ci_hi"))
        d = _f(r.get("paired_diff_mean"))
        d_lo = _f(r.get("paired_diff_ci_lo"))
        d_hi = _f(r.get("paired_diff_ci_hi"))
        p = _f(r.get("p_value_one_sided"))
        lines.append(
            f"| {cell} | "
            f"{f9:.4f} | [{f9_lo:.4f}, {f9_hi:.4f}] | "
            f"{fb:.4f} | [{fb_lo:.4f}, {fb_hi:.4f}] | "
            f"{d:+.4f} | [{d_lo:+.4f}, {d_hi:+.4f}] | "
            f"{p:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _section_hparam() -> str:
    rows = _read_csv(ROOT / "hparam" / "results.csv")
    if not rows:
        return "## 4. Hparam robustness (nk-bpo, 3³ grid)\n\n_(pending)_\n"
    grid_rows = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        spec = r["spec"]
        leaves = lr = npr = "?"
        for token in spec.split("_"):
            if token.startswith("L"):
                leaves = token[1:]
            elif token.startswith("lr"):
                lr = token[2:]
            elif token.startswith("npr"):
                npr = token[3:]
        v = _f(r.get("fmax"))
        if v is None:
            continue
        grid_rows.append((leaves, lr, npr, v))
    if not grid_rows:
        return "## 4. Hparam robustness (nk-bpo, 3³ grid)\n\n_(no completed rows)_\n"
    fmaxes = [v for *_, v in grid_rows]
    lo, hi = min(fmaxes), max(fmaxes)
    lines = [
        "## 4. Hparam robustness (nk-bpo, 3³ grid)",
        "",
        f"Grid range: **{lo:.4f} – {hi:.4f}** (delta {hi - lo:.4f}; threshold for 'robust': ≤0.02)",
        "",
        "| leaves | lr | neg_pos_ratio | fmax |",
        "|---|---|---|---|",
    ]
    for leaves, lr, npr, v in grid_rows:
        lines.append(f"| {leaves} | {lr} | {npr} | {v:.4f} |")
    return "\n".join(lines) + "\n"


def _section_cafaeval() -> str:
    out_dir = ROOT / "cafaeval"
    csv_path = out_dir / "results.csv"
    if not csv_path.exists():
        return "## 5. Cafaeval re-validation (winners)\n\n_(pending)_\n"
    rows = _read_csv(csv_path)
    if not rows:
        return "## 5. Cafaeval re-validation (winners)\n\n_(pending)_\n"
    lines = [
        "## 5. Cafaeval re-validation (winners)",
        "",
        "_Lab Fmax = lab evaluator (no prediction-side propagation). Cafaeval = "
        "PROTEA fork (CAFA `prop=fill`, `norm=cafa`). Cafaeval propagates "
        "predictions and ground truth through the GO DAG; lab Fmax does not._",
        "",
        "| cell | lab Fmax | cafaeval Fmax | ratio | Δ (cafa − lab) |",
        "|---|---|---|---|---|",
    ]
    cell_order = [c for c in CELLS if any(r.get("cell") == c for r in rows)]
    cell_order += [r["cell"] for r in rows if r.get("cell") not in cell_order]
    by_cell = {r.get("cell"): r for r in rows}
    cafas: list[float] = []
    labs: list[float] = []
    for c in cell_order:
        r = by_cell.get(c)
        if r is None:
            continue
        lab = _f(r.get("lab_fmax"))
        cafa = _f(r.get("cafaeval_fmax"))
        if lab is None or cafa is None:
            continue
        labs.append(lab)
        cafas.append(cafa)
        ratio = cafa / lab if lab else float("nan")
        delta = cafa - lab
        lines.append(f"| {c} | {lab:.4f} | **{cafa:.4f}** | {ratio:.2f}× | {delta:+.4f} |")
    if cafas:
        avg_lab = sum(labs) / len(labs)
        avg_cafa = sum(cafas) / len(cafas)
        lines.append("")
        lines.append(
            f"**avg lab Fmax: {avg_lab:.4f}**  ·  "
            f"**avg cafaeval Fmax: {avg_cafa:.4f}**  "
            f"(historical hybrid ceiling: 0.4400; v8-full record: 0.4251)"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Reranker v9 study — summary",
        "",
        "Auto-generated by `scripts/summarise_study.py`. Plan: memory `project_study_v9_plan.md`.",
        "",
        _section_replication(),
        _section_ablation(),
        _section_bootstrap(),
        _section_hparam(),
        _section_cafaeval(),
    ]
    SUMMARY.write_text("\n".join(parts))
    print(f"[summarise_study] wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
