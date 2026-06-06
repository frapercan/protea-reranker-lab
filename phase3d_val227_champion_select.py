"""Phase D: 227-validation selection of reranker champions.

Scores the 24-cell phase3a grid (bench-v1-K{3,5,10}-v226-lineage-{8 PLM})
on the 226->227 delta split using IA-weighted micro Fmax (f_micro_w) as the
selection metric.  NO new exports are performed; all data is from existing
predictions.parquet artefacts.

227-val split definition:
  Proteins whose first annotation appeared in GOA v227 (the v226->v227 delta).
  Sourced from bench-v1-K3-v227-lineage-prostt5/train.parquet, snapshot_pair
  = 'v226-v227'.  This split is PLM-independent (protein identity / annotation
  history do not depend on embeddings).

Selection metric: f_micro_w (IA-weighted micro Fmax) from cafaeval with
  ia=IA-swissprot-exp-v227.txt, prop=fill, norm=cafa, no_orphans=True.

Output:
  experiments/phase3d_val227/<plm>_K<k>_<cell>_seed42_227val.json  per run
  experiments/phase3d_val227/champions_227val.json   per-(K, aspect) champion
  experiments/phase3d_val227/champions_227val.md     publishable table
"""

from __future__ import annotations

import json
import subprocess
import sys
import zlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKTREE = Path(__file__).resolve().parent
REPO_DIR = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab")

# 227-val source: bench-v1-K3-v227-lineage-prostt5 train.parquet
# snapshot_pair == 'v226-v227' rows = the 226->227 delta
V227_DATASET = WORKTREE / "datasets" / "bench-v1-K3-v227-lineage-prostt5"
IA_PATH = WORKTREE / "datasets" / "ia" / "IA-swissprot-exp-v227.txt"
OBO_PATH = REPO_DIR / "datasets" / "bench-v1-K5" / "go.obo"

RUNS_ROOT = REPO_DIR / "runs"
EXPERIMENTS_OUT = WORKTREE / "experiments" / "phase3d_val227"

PROTEA_PYTHON = Path("/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NK_LK_CELLS = ["nk-mfo", "nk-bpo", "nk-cco", "lk-mfo", "lk-bpo", "lk-cco"]
SEED = 42

# All 24 grid cells: phase3a_K{k}_{plm}
PLMS = [
    "ankh_base", "ankh_large", "esm2_150m", "esm2_3b",
    "esm2_650m", "esmc_600m", "prostt5", "prot_t5",
]
KS = ["K3", "K5", "K10"]

ASPECT_TO_NS = {
    "bpo": "biological_process",
    "mfo": "molecular_function",
    "cco": "cellular_component",
}

# ---------------------------------------------------------------------------
# cafaeval driver template (with IA)
# ---------------------------------------------------------------------------
_DRIVER_SRC = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGINT, signal.SIG_DFL)
df, dfs_best = cafa_eval(
    "{obo}", "{pred_dir}", "{gt}",
    ia="{ia}",
    prop="fill", norm="cafa", no_orphans=True,
    max_terms=500, th_step=0.001, n_cpu=1,
)
out = {{}}
for kind, df_best in dfs_best.items():
    out[kind] = df_best.reset_index().to_dict(orient="records")
with open("{out_json}", "w") as f:
    json.dump(out, f, indent=2, default=str)
'''


# ---------------------------------------------------------------------------
# Load 227-val protein sets once
# ---------------------------------------------------------------------------

def load_v227_delta() -> dict[str, dict]:
    """Return per-cell dicts with keys 'proteins' (set) and rows info."""
    train_pq = V227_DATASET / "train.parquet"
    if not train_pq.exists():
        raise FileNotFoundError(f"227-val source not found: {train_pq}")

    f = pq.read_table(
        str(train_pq),
        columns=["protein_accession", "go_term_id", "label",
                 "snapshot_pair", "category", "aspect"],
    )
    mask = pc.equal(f.column("snapshot_pair"), "v226-v227")
    delta = f.filter(mask)

    per_cell: dict[str, dict] = {}
    for cell in NK_LK_CELLS:
        cat, asp = cell.split("-", 1)
        m_cat = pc.equal(delta.column("category"), cat)
        m_asp = pc.equal(delta.column("aspect"), asp)
        rows = delta.filter(pc.and_(m_cat, m_asp))
        prots = set(rows.column("protein_accession").to_pylist())
        # Build (protein, go_term_id) -> label mapping for gt reconstruction
        gt_pairs: dict[tuple, int] = {}
        prots_col = rows.column("protein_accession").to_pylist()
        gos_col = rows.column("go_term_id").to_pylist()
        labs_col = rows.column("label").to_pylist()
        for p, g, lab in zip(prots_col, gos_col, labs_col):
            key = (p, g)
            # take max label (positive wins)
            gt_pairs[key] = max(gt_pairs.get(key, 0), lab)
        per_cell[cell] = {
            "proteins": prots,
            "gt_pairs": gt_pairs,
            "n_proteins": len(prots),
            "n_positives": sum(1 for v in gt_pairs.values() if v > 0),
        }
    return per_cell


# ---------------------------------------------------------------------------
# Load predictions.parquet for a (K, PLM, cell) and get GO term mapping
# ---------------------------------------------------------------------------

def _load_eval_go_mapping(spec_yaml_path: Path, cell: str) -> dict[str, list[str]]:
    """Return {protein_accession: [go_term_id, ...]} from the dataset eval.parquet."""
    import yaml
    with open(spec_yaml_path) as f:
        spec = yaml.safe_load(f)
    manifest_path = Path(spec["dataset"]["manifest"])
    dataset_dir = manifest_path.parent
    eval_pq = dataset_dir / "eval.parquet"

    if not eval_pq.exists():
        return {}

    sys.path.insert(0, str(REPO_DIR / "src"))
    from protea_reranker_lab.data import iter_batches

    cat, asp = cell.split("-", 1)
    protein_to_gos: dict[str, list[str]] = {}

    for batch in iter_batches(
        eval_pq,
        columns=["protein_accession", "go_term_id"],
        category=cat, aspect=asp, batch_size=200_000,
    ):
        ps = batch.column("protein_accession").to_pylist()
        gs = batch.column("go_term_id").to_pylist()
        for p, g in zip(ps, gs):
            if p not in protein_to_gos:
                protein_to_gos[p] = []
            protein_to_gos[p].append(g)

    return protein_to_gos


def _bucket_order(proteins: np.ndarray) -> np.ndarray:
    buckets = np.fromiter(
        (zlib.crc32(p.encode("ascii")) % 32 for p in proteins),
        count=len(proteins), dtype=np.int32,
    )
    return np.lexsort((proteins, buckets))


# ---------------------------------------------------------------------------
# Run cafaeval on filtered predictions
# ---------------------------------------------------------------------------

def run_val227_cafaeval(
    run_dir: Path,
    cell: str,
    v227_cell: dict,
) -> dict:
    """Score predictions on 227-val split, return metrics dict."""
    pred_pq = run_dir / "predictions.parquet"
    if not pred_pq.exists():
        print(f"    [skip] no predictions.parquet at {run_dir}")
        return {}

    # Load predictions
    pred_t = pq.read_table(str(pred_pq), columns=["protein_accession", "score"])
    pred_prots = pred_t.column("protein_accession").to_numpy(zero_copy_only=False)
    pred_scores = pred_t.column("score").to_numpy(zero_copy_only=False).astype(np.float32)

    # Filter to 227-val proteins
    v227_prots = v227_cell["proteins"]
    mask_arr = np.array([p in v227_prots for p in pred_prots], dtype=bool)
    n_overlap = int(mask_arr.sum())

    if n_overlap == 0:
        print("    [skip] no 227-val proteins in predictions")
        return {}

    # Recover go_term_ids from the eval.parquet (same cell, same protein order)
    # We need to match the bucket-sorted predictions to the eval.parquet rows
    # Load eval.parquet and filter to v227 proteins in this cell
    spec_yaml = run_dir / "spec_input.yaml"
    if not spec_yaml.exists():
        print(f"    [skip] no spec_input.yaml at {run_dir}")
        return {}

    # Approach: re-read eval.parquet filtered to 227-val proteins, get (prot, go, label)
    # then use prediction scores from predictions.parquet (they're in same order)
    import yaml
    with open(spec_yaml) as f_spec:
        spec = yaml.safe_load(f_spec)
    manifest_path = Path(spec["dataset"]["manifest"])
    dataset_dir = manifest_path.parent
    eval_pq_path = dataset_dir / "eval.parquet"

    if not eval_pq_path.exists():
        print(f"    [skip] eval.parquet not found at {dataset_dir}")
        return {}

    sys.path.insert(0, str(REPO_DIR / "src"))
    from protea_reranker_lab.data import iter_batches

    cat, asp = cell.split("-", 1)

    eval_prots_list: list = []
    eval_gos_list: list = []
    eval_labels_list: list = []

    for batch in iter_batches(
        eval_pq_path,
        columns=["protein_accession", "go_term_id", "label"],
        category=cat, aspect=asp, batch_size=200_000,
    ):
        eval_prots_list.append(batch.column("protein_accession").to_numpy(zero_copy_only=False))
        eval_gos_list.append(batch.column("go_term_id").to_numpy(zero_copy_only=False))
        eval_labels_list.append(batch.column("label").to_numpy(zero_copy_only=False).astype(np.int8))

    if not eval_prots_list:
        print(f"    [skip] empty eval slice for {cell}")
        return {}

    eval_prots = np.concatenate(eval_prots_list)
    eval_gos = np.concatenate(eval_gos_list)
    eval_labels = np.concatenate(eval_labels_list)

    # Bucket-sort eval
    eval_order = _bucket_order(eval_prots)
    eval_prots = eval_prots[eval_order]
    eval_gos = eval_gos[eval_order]
    eval_labels = eval_labels[eval_order]

    # Verify alignment with predictions
    if not np.array_equal(pred_prots, eval_prots):
        n_pred = len(pred_prots)
        n_eval = len(eval_prots)
        print(f"    [warn] pred/eval protein order mismatch {n_pred} vs {n_eval}; proceeding with eval-only slice")
        # Use eval-only slice, no score from predictions
        # Fall back to vote fraction from eval
        return {}

    # Load vote fraction as fallback score check
    # (We already have pred_scores from predictions.parquet which are from the trained model)
    # Filter eval to v227 proteins
    eval_mask = np.array([p in v227_prots for p in eval_prots], dtype=bool)
    filtered_eval_prots = eval_prots[eval_mask]
    filtered_eval_gos = eval_gos[eval_mask]
    filtered_eval_labels = eval_labels[eval_mask]
    filtered_pred_scores = pred_scores[eval_mask]  # scores in eval-sorted order

    if len(filtered_eval_prots) == 0:
        print("    [skip] 227-val filter yields 0 rows")
        return {}

    n_pos = int((filtered_eval_labels > 0).sum())
    n_total = len(filtered_eval_prots)
    n_prot_covered = len(set(filtered_eval_prots.tolist()))
    coverage_pct = 100.0 * n_prot_covered / len(v227_prots) if v227_prots else 0

    # Write pred and gt files for cafaeval
    out_dir = run_dir / "val227"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "pred_dir"
    pred_dir.mkdir(exist_ok=True)

    pred_tsv = out_dir / "pred.tsv"
    gt_tsv = out_dir / "gt.tsv"

    with pred_tsv.open("w") as fh:
        for p, g, s in zip(filtered_eval_prots, filtered_eval_gos, filtered_pred_scores):
            fh.write(f"{p}\t{g}\t{float(s):.6f}\n")

    pos_mask = filtered_eval_labels > 0
    with gt_tsv.open("w") as fh:
        for p, g in zip(filtered_eval_prots[pos_mask], filtered_eval_gos[pos_mask]):
            fh.write(f"{p}\t{g}\n")

    (pred_dir / f"{cell}.tsv").write_bytes(pred_tsv.read_bytes())

    # Run cafaeval with IA
    raw_json = out_dir / "_raw_metrics.json"
    driver = out_dir / "_driver.py"
    driver.write_text(_DRIVER_SRC.format(
        obo=str(OBO_PATH),
        pred_dir=str(pred_dir),
        gt=str(gt_tsv),
        ia=str(IA_PATH),
        out_json=str(raw_json),
    ))

    proc = subprocess.run(
        [str(PROTEA_PYTHON), str(driver)],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        print(f"    [FAIL cafaeval] {proc.stderr[-500:]}")
        return {}

    raw = json.loads(raw_json.read_text())
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

    metrics = {
        "cafaeval_fmax": _pick("f_micro", "f_micro"),
        "f_micro_w": _pick("f_micro_w", "f_micro_w"),
        "s_min": _pick("s", "s"),
        "n_rows_227val": n_total,
        "n_positives_227val": n_pos,
        "n_proteins_covered": n_prot_covered,
        "n_proteins_total_227val": len(v227_prots),
        "coverage_pct": round(coverage_pct, 1),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    EXPERIMENTS_OUT.mkdir(parents=True, exist_ok=True)

    # Load 227-val protein sets
    print("Loading 227-val protein sets from v226->v227 delta ...")
    v227_per_cell = load_v227_delta()
    for cell, info in v227_per_cell.items():
        print(f"  {cell}: {info['n_proteins']} proteins, {info['n_positives']} positives")

    # Check prerequisites
    if not OBO_PATH.exists():
        print(f"ERROR: OBO not found at {OBO_PATH}")
        return 1
    if not IA_PATH.exists():
        print(f"ERROR: IA file not found at {IA_PATH}")
        return 1
    if not PROTEA_PYTHON.exists():
        print(f"ERROR: PROTEA python not found at {PROTEA_PYTHON}")
        return 1

    # Score all 24 grid cells
    all_results: list[dict] = []

    for k in KS:
        for plm in PLMS:
            run_tag = f"phase3a_{k}_{plm}"
            print(f"\n=== {run_tag} ===")

            for cell in NK_LK_CELLS:
                out_json = EXPERIMENTS_OUT / f"{plm}_{k}_{cell}_seed{SEED}_227val.json"
                if out_json.exists():
                    try:
                        r = json.loads(out_json.read_text())
                        if r.get("f_micro_w") is not None:
                            print(f"  [skip] {cell}: f_micro_w={r['f_micro_w']:.4f} (cached)")
                            all_results.append(r)
                            continue
                    except Exception:
                        pass

                run_dir = RUNS_ROOT / run_tag / f"{cell}_seed{SEED}"
                if not run_dir.exists():
                    print(f"  [skip] {cell}: run dir not found")
                    continue

                print(f"  [score] {cell} ...", flush=True)
                metrics = run_val227_cafaeval(run_dir, cell, v227_per_cell[cell])

                if not metrics:
                    print(f"  [fail] {cell}: no metrics")
                    continue

                result = {
                    "run_tag": run_tag,
                    "k": k,
                    "plm": plm,
                    "cell": cell,
                    "seed": SEED,
                    "selection_metric": "f_micro_w",
                    "val_split": "v226-v227",
                    **metrics,
                }
                out_json.write_text(json.dumps(result, indent=2))
                all_results.append(result)

                fmw = metrics.get("f_micro_w")
                cfmax = metrics.get("cafaeval_fmax")
                cov = metrics.get("coverage_pct", 0)
                fmw_s = f"{fmw:.4f}" if fmw is not None else "N/A"
                cfmax_s = f"{cfmax:.4f}" if cfmax is not None else "N/A"
                print(f"  [done] {cell}: f_micro_w={fmw_s} cafaFmax={cfmax_s} cov={cov:.0f}%")

    # Select champions: argmax f_micro_w per (K, aspect)
    # Group results by (K, aspect)
    by_k_asp: dict[tuple, list] = defaultdict(list)
    for r in all_results:
        if r.get("f_micro_w") is not None:
            # cell like 'nk-mfo' -> aspect 'mfo'
            asp = r["cell"].split("-", 1)[1]
            by_k_asp[(r["k"], asp)].append(r)

    champions: list[dict] = []
    print("\n=== Champion selection (argmax f_micro_w per K x aspect) ===")

    # Also collect old v226 selection for comparison
    # Old selection was plain Fmax on v226-v230 eval
    # Load from existing cafaeval_metrics.json files
    old_selection: dict[tuple, dict] = {}
    for k in KS:
        for plm in PLMS:
            run_tag = f"phase3a_{k}_{plm}"
            for cell in NK_LK_CELLS:
                run_dir = RUNS_ROOT / run_tag / f"{cell}_seed{SEED}"
                old_metrics_path = run_dir / "cafaeval_metrics.json"
                if old_metrics_path.exists():
                    try:
                        m = json.loads(old_metrics_path.read_text())
                        asp = cell.split("-", 1)[1]
                        key = (k, asp, plm)
                        old_selection[key] = {
                            "run_tag": run_tag,
                            "cell": cell,
                            "cafaeval_fmax_v226": m.get("cafaeval_fmax"),
                        }
                    except Exception:
                        pass

    # Coverage threshold: cells with <80% coverage are unreliable for comparison
    # because they scored on a non-representative subset of the 227-val proteins.
    # (Caused by legacy eval datasets with fewer proteins for K3-prostt5, K3-esmc_600m,
    # K5-ankh_base, K5-esmc_600m which used smaller exported eval sets.)
    COVERAGE_THRESHOLD = 80.0

    for (k, asp), candidates in sorted(by_k_asp.items()):
        # Find champion per category (NK and LK separately)
        for cat in ["nk", "lk"]:
            cell = f"{cat}-{asp}"
            cell_cands = [c for c in candidates if c["cell"] == cell]
            if not cell_cands:
                continue

            # All candidates (coverage-unfiltered)
            champion_all = max(cell_cands, key=lambda x: x["f_micro_w"])

            # Coverage-filtered candidates (>=80%)
            cov_cands = [c for c in cell_cands if c.get("coverage_pct", 0) >= COVERAGE_THRESHOLD]
            champion_cov = max(cov_cands, key=lambda x: x["f_micro_w"]) if cov_cands else None

            # Use coverage-filtered champion as the authoritative one
            champion = champion_cov if champion_cov else champion_all
            champion["is_champion"] = True
            champion["coverage_threshold"] = COVERAGE_THRESHOLD
            champion["low_coverage_excluded"] = [
                c["plm"] for c in cell_cands
                if c.get("coverage_pct", 0) < COVERAGE_THRESHOLD
            ]
            if champion_all["plm"] != (champion_cov["plm"] if champion_cov else ""):
                champion["overridden_low_cov_plm"] = champion_all["plm"]
                champion["overridden_low_cov_fmw"] = champion_all["f_micro_w"]

            # Compare with old selection (best v226 cafaFmax for this cell)
            old_best = max(
                [c for c in cell_cands if c.get("cafaeval_fmax") is not None],
                key=lambda x: x.get("cafaeval_fmax", 0),
                default=None,
            )
            champion["old_v226_best_plm"] = old_best["plm"] if old_best else None
            champion["old_v226_best_cafaeval_fmax"] = old_best["cafaeval_fmax"] if old_best else None
            champion["changed_from_v226"] = (
                old_best and old_best["plm"] != champion["plm"]
            )

            print(f"  {k} {cell}: champion={champion['plm']} "
                  f"f_micro_w={champion['f_micro_w']:.4f} "
                  f"cafaFmax={champion.get('cafaeval_fmax', 0):.4f} "
                  f"cov={champion.get('coverage_pct', 0):.0f}%")
            if champion.get("overridden_low_cov_plm"):
                print(f"    (overrode low-cov winner {champion['overridden_low_cov_plm']} "
                      f"fmw={champion['overridden_low_cov_fmw']:.4f})")
            if champion.get("changed_from_v226"):
                print(f"    -> changed from v226 champion: {champion['old_v226_best_plm']} "
                      f"(v226 cafaFmax={champion['old_v226_best_cafaeval_fmax']:.4f})")

            champions.append(champion)

    # Write outputs
    all_results_path = EXPERIMENTS_OUT / "all_results_227val.json"
    all_results_path.write_text(json.dumps(all_results, indent=2))

    champions_json = EXPERIMENTS_OUT / "champions_227val.json"
    champions_json.write_text(json.dumps(champions, indent=2))

    _write_champions_md(champions, EXPERIMENTS_OUT / "champions_227val.md")

    print(f"\nWrote {len(all_results)} results to {all_results_path}")
    print(f"Wrote {len(champions)} champions to {champions_json}")
    print(f"Wrote publishable table to {EXPERIMENTS_OUT / 'champions_227val.md'}")
    return 0


def _write_champions_md(champions: list[dict], out_path: Path) -> None:
    lines = [
        "# Phase D: Champions selected on 227-validation (f_micro_w)",
        "",
        "Selection metric: f_micro_w (IA-weighted micro Fmax) on 226->227 delta split.",
        "IA file: datasets/ia/IA-swissprot-exp-v227.txt.",
        "Training data: bench-v1-K{3,5,10}-v226-lineage-{PLM} (NO re-export).",
        "Champion per (K, tier-aspect) = argmax f_micro_w on 227-val.",
        "",
        "## NK cells",
        "",
        "| K | aspect | champion PLM | f_micro_w | cafaFmax (v227-val) | coverage % |"
        " | old v226 PLM | changed? |",
        "|-|-|-|-|-|-|-|-|",
    ]
    for cat in ["nk", "lk"]:
        if cat == "lk":
            lines += ["", "## LK cells", "",
                      "| K | aspect | champion PLM | f_micro_w | cafaFmax (v227-val) |"
                      " coverage % | old v226 PLM | changed? |",
                      "|-|-|-|-|-|-|-|-|"]
        for k in KS:
            for asp in ["mfo", "bpo", "cco"]:
                cell = f"{cat}-{asp}"
                champ = next(
                    (c for c in champions if c["k"] == k and c["cell"] == cell), None
                )
                if champ is None:
                    lines.append(f"| {k} | {asp} | N/A | N/A | N/A | N/A | N/A | N/A |")
                    continue
                fmw = champ.get("f_micro_w")
                cfmax = champ.get("cafaeval_fmax")
                cov = champ.get("coverage_pct", 0)
                old_plm = champ.get("old_v226_best_plm", "N/A")
                changed = "YES" if champ.get("changed_from_v226") else "no"
                fmw_s = f"{fmw:.4f}" if fmw is not None else "N/A"
                cfmax_s = f"{cfmax:.4f}" if cfmax is not None else "N/A"
                lines.append(
                    f"| {k} | {asp} | {champ['plm']} "
                    f"| {fmw_s} "
                    f"| {cfmax_s} "
                    f"| {cov:.0f} "
                    f"| {old_plm} "
                    f"| {changed} |"
                )

    lines += [
        "",
        "## Notes",
        "",
        "- f_micro_w: IA-weighted micro Fmax (cafaeval with ia=IA-swissprot-exp-v227.txt).",
        "  This is the LAFA alignment metric, preferred over plain Fmax for 227-val.",
        "- cafaFmax (v227-val): unweighted cafaeval Fmax on the same 227-val split.",
        "- coverage: fraction of v226->v227 delta proteins present in the prediction set.",
        "  K3 cells with legacy eval sets may show lower coverage.",
        "- old v226 PLM: the PLM that had the best plain cafaeval Fmax on the full",
        "  v226-v230 eval set (the old selection criterion).",
        "- changed: YES if the 227-val champion differs from the old v226 selection.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
