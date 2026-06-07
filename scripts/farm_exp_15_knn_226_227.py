"""FARM-EXP.15 (reframed): KNN-only baseline on the 226->227 validation band.

Produces IA-weighted micro-F (f_micro_w) for the full grid of
8 PLM x K{3,5,10} x 9 cells (NK/LK/PK x MFO/BPO/CCO) = 216 cells, on the
226->227 validation window. No KNN re-inference: the KNN-only score is
``1 - distance`` read directly from the frozen v226-lineage eval.parquet.

Ground truth methodology mirrors phase3d_val227_champion_select.py exactly:
  1. The 226->227 delta proteins (proteins gaining a non-IEA annotation at
     v227) come from bench-v1-K3-v227-lineage-prostt5/train.parquet,
     snapshot_pair == 'v226-v227'. This split is PLM-independent.
  2. For each (PLM, K, cell) the eval.parquet candidate rows are filtered to
     those delta proteins.
  3. pred.tsv = every candidate (protein, go, 1 - distance).
     gt.tsv = candidate rows whose eval.parquet label > 0.
  4. cafaeval is run with the v227 LAFA-aligned IA table; f_micro_w is the
     headline (LAFA-comparable) metric per aspect namespace.

The script is idempotent: a per-cell metrics json gates recompute, so a
killed run resumes. cafaeval runs in the PROTEA venv (where cafaeval-protea is
installed) via a subprocess driver; everything else is the lab venv.

Outputs:
  <out-root>/<plm>_K<k>_<cell>/  metrics.json + pred.tsv + gt.tsv
  <csv>                          aggregated benchmark table
  <csv>.md                       majority-winner summary
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

# Frozen datasets live only in the canonical lab checkout, never in an
# ephemeral worktree. Resolve against it (override with --datasets-root).
CANONICAL_LAB = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab")
DATASETS = CANONICAL_LAB / "datasets"
OBO_PATH = DATASETS / "bench-v1-K5" / "go.obo"

# v227 LAFA-aligned IA (authoritative for the 226->227 band; matches the
# deployed LAFA endpoint). See IA_PROVENANCE_v227.md in protea-deploy.
IA_PATH = Path(
    "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
)
# cafaeval-protea lives in the PROTEA venv, not the lab venv.
PROTEA_PYTHON = Path("/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python")

PLMS = [
    "ankh_base",
    "ankh_large",
    "esm2_150m",
    "esm2_3b",
    "esm2_650m",
    "esmc_600m",
    "prostt5",
    "prot_t5",
]
KS = [3, 5, 10]
CELLS = [
    "nk-mfo", "nk-bpo", "nk-cco",
    "lk-mfo", "lk-bpo", "lk-cco",
    "pk-mfo", "pk-bpo", "pk-cco",
]
NK_LK = {c for c in CELLS if not c.startswith("pk-")}

ASPECT_TO_NS = {
    "bpo": "biological_process",
    "mfo": "molecular_function",
    "cco": "cellular_component",
}

_DRIVER_SRC = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGINT, signal.SIG_DFL)
df, dfs_best = cafa_eval(
    "{obo}", "{pred_dir}", "{gt}",
    ia="{ia}",
    prop="fill", norm="cafa", no_orphans=True,
    max_terms=500, th_step=0.001, n_cpu=1, weighted_only=False,
)
out = {{}}
for kind, df_best in dfs_best.items():
    out[kind] = df_best.reset_index().to_dict(orient="records")
with open("{out_json}", "w") as f:
    json.dump(out, f, indent=2, default=str)
'''


def load_v227_delta(train_pq: Path) -> dict[str, dict]:
    """Return {cell: {"proteins": set, "pos_pairs": set}} from the v226->v227 delta.

    The delta set is the proteins that appear in the v226-v227 snapshot pair
    for the cell's (category, aspect). PLM-independent. ``pos_pairs`` are the
    true (protein, go) annotations gained at v227, used as a GT fallback for
    legacy exports whose eval.parquet label column was never populated.
    """
    t = pq.read_table(
        str(train_pq),
        columns=["protein_accession", "go_term_id", "label",
                 "snapshot_pair", "category", "aspect"],
    )
    delta = t.filter(pc.equal(t.column("snapshot_pair"), "v226-v227"))
    per_cell: dict[str, dict] = {}
    for cell in CELLS:
        cat, asp = cell.split("-", 1)
        m = pc.and_(
            pc.equal(delta.column("category"), cat),
            pc.equal(delta.column("aspect"), asp),
        )
        rows = delta.filter(m)
        prots = rows.column("protein_accession").to_pylist()
        gos = rows.column("go_term_id").to_pylist()
        labs = rows.column("label").to_pylist()
        per_cell[cell] = {
            "proteins": set(prots),
            "pos_pairs": {
                (p, g) for p, g, lab in zip(prots, gos, labs) if lab > 0
            },
        }
    return per_cell


def _eval_path(plm: str, k: int) -> Path:
    return DATASETS / f"bench-v1-K{k}-v226-lineage-{plm}" / "eval.parquet"


def build_cell_tsvs(
    eval_pq: Path,
    cell: str,
    delta_proteins: set[str],
    delta_pos_pairs: set[tuple[str, str]],
    out_dir: Path,
) -> dict | None:
    """Write pred.tsv (1-distance) + gt.tsv for the 226->227 delta slice.

    GT = eval candidate rows whose eval.parquet label > 0 (canonical, matches
    phase3d_val227). If the eval label column is unpopulated (legacy export),
    fall back to marking candidates positive when their (protein, go) pair is a
    true v227 annotation from the delta. Returns counts dict, or None if empty.
    """
    cat, asp = cell.split("-", 1)
    schema_names = set(pq.ParquetFile(str(eval_pq)).schema_arrow.names)
    go_col = "go_term_id" if "go_term_id" in schema_names else "go_id"
    cols = ["protein_accession", go_col, "distance", "label", "aspect"]
    # Older exports (e.g. ankh_large K3) lack the category column and encode
    # aspect with single-letter GO codes (F/P/C) instead of mfo/bpo/cco. The
    # cell's category is already encoded in the delta protein set (partitioned
    # by category x aspect), so aspect + delta-membership uniquely selects the
    # cell; an explicit category pushdown is only an optimisation when present.
    asp_letter = {"mfo": "F", "bpo": "P", "cco": "C"}[asp]
    filters = [("aspect", "in", [asp, asp_letter])]
    if "category" in schema_names:
        filters.insert(0, ("category", "=", cat))
        cols.append("category")
    t = pq.read_table(str(eval_pq), columns=cols, filters=filters)
    if t.num_rows == 0:
        return None

    prots = t.column("protein_accession").to_numpy(zero_copy_only=False)
    keep = np.fromiter(
        (p in delta_proteins for p in prots), count=len(prots), dtype=bool
    )
    if not keep.any():
        return None

    gos = t.column(go_col).to_numpy(zero_copy_only=False)[keep]
    dist = t.column("distance").to_numpy(zero_copy_only=False).astype(np.float64)[keep]
    labels = t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[keep]
    prots = prots[keep]
    # KNN-only score = 1 - cosine distance (embedding_only formula).
    scores = 1.0 - dist

    pos = labels > 0
    gt_source = "eval_label"
    if not pos.any():
        # Legacy export with an unpopulated label column: reconstruct GT from
        # the v227 delta positive pairs.
        pos = np.fromiter(
            ((p, g) in delta_pos_pairs for p, g in zip(prots, gos)),
            count=len(prots), dtype=bool,
        )
        gt_source = "delta_pos_pairs"

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "pred_dir"
    pred_dir.mkdir(exist_ok=True)
    pred_tsv = out_dir / "pred.tsv"
    gt_tsv = out_dir / "gt.tsv"

    with pred_tsv.open("w") as fh:
        for p, g, s in zip(prots, gos, scores):
            fh.write(f"{p}\t{g}\t{s:.6f}\n")
    with gt_tsv.open("w") as fh:
        for p, g in zip(prots[pos], gos[pos]):
            fh.write(f"{p}\t{g}\n")

    (pred_dir / f"{cell}.tsv").write_bytes(pred_tsv.read_bytes())

    return {
        "gt_source": gt_source,
        "n_rows": int(len(prots)),
        "n_positives": int(pos.sum()),
        "n_proteins": int(len(set(prots.tolist()))),
        "n_delta_proteins": len(delta_proteins),
        "n_gt_proteins": int(len(set(prots[pos].tolist()))),
    }


def run_cafaeval(out_dir: Path, cell: str) -> dict:
    """Invoke cafaeval (PROTEA venv) and parse f_micro_w / f_micro for the ns."""
    raw_json = out_dir / "_raw_metrics.json"
    driver = out_dir / "_driver.py"
    driver.write_text(_DRIVER_SRC.format(
        obo=str(OBO_PATH),
        pred_dir=str(out_dir / "pred_dir"),
        gt=str(out_dir / "gt.tsv"),
        ia=str(IA_PATH),
        out_json=str(raw_json),
    ))
    proc = subprocess.run(
        [str(PROTEA_PYTHON), str(driver)],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr[-600:]}

    raw = json.loads(raw_json.read_text())
    asp = cell.split("-", 1)[1]
    target_ns = ASPECT_TO_NS[asp]

    def _pick(kind: str, col: str) -> float | None:
        for rec in raw.get(kind, []):
            ns = rec.get("ns") or rec.get("namespace") or ""
            if ns == target_ns and rec.get(col) is not None:
                try:
                    return float(rec[col])
                except (TypeError, ValueError):
                    return None
        return None

    return {
        "f_micro_w": _pick("f_micro_w", "f_micro_w"),
        "f_micro": _pick("f_micro", "f_micro"),
        "f_w": _pick("f_w", "f_w"),
        "fmax": _pick("f", "f"),
    }


def run_one(
    plm: str, k: int, cell: str,
    delta: dict[str, set[str]],
    out_root: Path,
) -> dict:
    cell_dir = out_root / f"{plm}_K{k}_{cell}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = cell_dir / "metrics.json"

    if metrics_json.exists():
        try:
            m = json.loads(metrics_json.read_text())
            if m.get("status") == "ok":
                return m
        except Exception:
            pass

    eval_pq = _eval_path(plm, k)
    base = {"plm": plm, "k": k, "cell": cell, "regime": cell.split("-")[0],
            "aspect": cell.split("-")[1]}

    if not eval_pq.exists():
        m = {**base, "status": "missing_eval_parquet", "f_micro_w": None,
             "n_proteins": 0}
        metrics_json.write_text(json.dumps(m, indent=2))
        return m

    counts = build_cell_tsvs(
        eval_pq, cell, delta[cell]["proteins"], delta[cell]["pos_pairs"], cell_dir
    )
    if counts is None:
        m = {**base, "status": "empty_slice", "f_micro_w": None, "n_proteins": 0}
        metrics_json.write_text(json.dumps(m, indent=2))
        return m
    if counts["n_positives"] == 0:
        m = {**base, "status": "no_positives", "f_micro_w": None, **counts}
        metrics_json.write_text(json.dumps(m, indent=2))
        return m

    scores = run_cafaeval(cell_dir, cell)
    if "error" in scores:
        m = {**base, "status": "cafaeval_error", "f_micro_w": None,
             "error": scores["error"], **counts}
        metrics_json.write_text(json.dumps(m, indent=2))
        return m

    m = {**base, "status": "ok", **scores, **counts}
    metrics_json.write_text(json.dumps(m, indent=2))
    return m


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["plm", "k", "regime", "aspect", "cell", "f_micro_w",
            "n_proteins", "status"]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["plm"], x["k"], x["cell"])):
            w.writerow({c: r.get(c) for c in cols})


def majority_winner(rows: list[dict]) -> tuple[tuple[str, int] | None, dict]:
    """Per NK+LK cell, find the (plm,k) with the highest f_micro_w, tally wins.

    The winner is the (plm,k) that wins the most NK+LK cells (PK excluded).
    """
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        if r["cell"] in NK_LK and r["status"] == "ok" and r["f_micro_w"] is not None:
            by_cell.setdefault(r["cell"], []).append(r)

    wins: Counter = Counter()
    per_cell_winner: dict[str, str] = {}
    for cell, recs in by_cell.items():
        best = max(recs, key=lambda x: x["f_micro_w"])
        key = (best["plm"], best["k"])
        wins[key] += 1
        per_cell_winner[cell] = f"{best['plm']} K{best['k']} ({best['f_micro_w']:.4f})"

    if not wins:
        return None, {"per_cell_winner": per_cell_winner, "win_counts": {}}
    winner = wins.most_common(1)[0][0]
    return winner, {
        "per_cell_winner": per_cell_winner,
        "win_counts": {f"{p} K{k}": n for (p, k), n in wins.most_common()},
    }


def write_summary_md(rows: list[dict], md_path: Path) -> dict:
    winner, info = majority_winner(rows)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    lines = [
        "# FARM-EXP.15 KNN-only baseline (226->227 validation, f_micro_w)",
        "",
        f"Grid: {len(PLMS)} PLM x K{KS} x {len(CELLS)} cells = {len(rows)} cells "
        f"({n_ok} scored OK).",
        "",
        "Metric: IA-weighted micro-F (f_micro_w), v227 LAFA-aligned IA. "
        "KNN-only score = 1 - cosine distance.",
        "",
        "## Majority winner over NK+LK cells (PK excluded)",
        "",
    ]
    if winner is None:
        lines.append("No NK+LK cells scored; winner undetermined.")
    else:
        p, k = winner
        lines.append(f"**Winner: {p} K{k}** "
                     f"({info['win_counts'].get(f'{p} K{k}', 0)} of 6 NK+LK cells)")
        lines.append("")
        lines.append("Win counts (plm K):")
        for key, n in info["win_counts"].items():
            lines.append(f"- {key}: {n}")
        lines.append("")
        lines.append("Per-cell best (plm K, f_micro_w):")
        for cell in sorted(info["per_cell_winner"]):
            lines.append(f"- {cell}: {info['per_cell_winner'][cell]}")
    md_path.write_text("\n".join(lines) + "\n")
    return {"winner": winner, **info}


def main() -> int:
    global DATASETS, OBO_PATH
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--v227-train",
        required=True,
        help="path to bench-v1-K3-v227-lineage-prostt5/train.parquet",
    )
    ap.add_argument(
        "--out-root",
        default=str(CANONICAL_LAB / "runs" / "knn_226_227"),
        help="per-cell artefact root",
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="aggregated benchmark CSV output path",
    )
    ap.add_argument(
        "--datasets-root",
        default=str(DATASETS),
        help="root holding bench-v1-K*-v226-lineage-* dataset dirs",
    )
    args = ap.parse_args()

    DATASETS = Path(args.datasets_root)
    OBO_PATH = DATASETS / "bench-v1-K5" / "go.obo"

    for path, label in [(OBO_PATH, "OBO"), (IA_PATH, "IA"),
                        (PROTEA_PYTHON, "PROTEA python"),
                        (Path(args.v227_train), "v227 train.parquet")]:
        if not Path(path).exists():
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            return 1

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading v226->v227 delta protein sets ...", flush=True)
    delta = load_v227_delta(Path(args.v227_train))
    for cell in CELLS:
        print(f"  {cell}: {len(delta[cell]['proteins'])} delta proteins, "
              f"{len(delta[cell]['pos_pairs'])} positive pairs", flush=True)

    rows: list[dict] = []
    total = len(PLMS) * len(KS) * len(CELLS)
    i = 0
    for plm in PLMS:
        for k in KS:
            for cell in CELLS:
                i += 1
                m = run_one(plm, k, cell, delta, out_root)
                rows.append(m)
                fmw = m.get("f_micro_w")
                fmw_s = f"{fmw:.4f}" if isinstance(fmw, float) else str(fmw)
                print(f"  [{i}/{total}] {plm} K{k} {cell}: "
                      f"f_micro_w={fmw_s} ({m['status']})", flush=True)

    csv_path = Path(args.csv)
    write_csv(rows, csv_path)
    md_path = csv_path.with_suffix(".md")
    summary = write_summary_md(rows, md_path)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")
    if summary["winner"]:
        p, k = summary["winner"]
        print(f"Majority winner (NK+LK): {p} K{k}")
        print(f"Win counts: {summary['win_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
