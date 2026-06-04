"""F-LAFA-IA.1b: IA-weighted eval of the v227 probe cell (prostt5 K3).

Evaluates the freshly re-exported ``bench-v1-K3-v227-lineage-prostt5``
dataset (train cutoff v227, eval band v227->v230) against the LAFA
protocol pinned in ``experiments/lafa_ia_v227_protocol/``:

  prop=fill, norm=cafa, no_orphans, ia=IA-swissprot-exp-v227.txt,
  headline = IA-weighted micro Fmax (f_micro_w) + S_min (s).

Two arms per cell:

  * reranker: v26-binary recipe (binary objective, lean+lin+emb,
    neg_pos_ratio=10), seed 42, trained via ``scripts/run.py``.
  * knn baseline: no training; the prediction score is the raw KNN
    ``neighbor_vote_fraction`` carried in eval.parquet. cafaeval Fmax is
    rank-invariant to a monotone score scale, so the unnormalised vote
    fraction yields a valid KNN-only baseline.

For every (cell, arm) it records four numbers:

  internal_fmax  -- lab threshold-grid micro Fmax (reranker run.json;
                    for the KNN arm, recomputed on the vote score).
  cafaeval_fmax  -- cafaeval unweighted micro Fmax (f_micro).
  wfmax          -- cafaeval IA-weighted micro Fmax (f_micro_w); the
                    LAFA headline.
  s_min          -- cafaeval minimum semantic distance (s).

Writes:
  runs/lafa_ia_v227_probe/<cell>_<arm>/cafaeval_metrics.json
  experiments/lafa_ia_v227_probe/results.json
  experiments/lafa_ia_v227_probe/summary.md
"""

from __future__ import annotations

import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

WORKTREE = Path(__file__).resolve().parents[2]
REPO_DIR = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab")
DATASET_NAME = "bench-v1-K3-v227-lineage-prostt5"
DATASET_DIR = WORKTREE / "datasets" / DATASET_NAME
# go.obo is a large local artefact; it is the releases/2026-01-23 snapshot
# that the IA table was propagated against (protocol item 2).
OBO_PATH = REPO_DIR / "datasets" / "bench-v1-K5" / "go.obo"
IA_PATH = WORKTREE / "datasets" / "ia" / "IA-swissprot-exp-v227.txt"
RUNS_ROOT = WORKTREE / "runs" / "lafa_ia_v227_probe"
OUT_DIR = WORKTREE / "experiments" / "lafa_ia_v227_probe"

LAB_PYTHON = REPO_DIR / ".venv" / "bin" / "python"
# cafaeval lives in the PROTEA venv (same as the v27 sweep).
PROTEA_PYTHON = Path("/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python")

CELLS = ["nk-mfo", "nk-bpo", "nk-cco", "lk-mfo", "lk-bpo", "lk-cco"]
SEED = 42

V26_HPARAMS = {
    "objective": "binary",
    "num_boost_round": 10000,
    "early_stopping_rounds": 100,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "drop_features": [],
}

ASPECT_TO_NS = {
    "bpo": "biological_process",
    "mfo": "molecular_function",
    "cco": "cellular_component",
}

# cafaeval driver: prop=fill norm=cafa no_orphans ia=<IA>. Emits every
# best-metric table so we can pull f_micro, f_micro_w and s.
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


def _run_dir(cell: str, arm: str) -> Path:
    return RUNS_ROOT / f"{cell}_{arm}"


def _make_spec_yaml(cell: str, out_dir: Path) -> Path:
    spec = {
        "schema_version": "v1",
        "name": f"lafa_ia_v227_probe_{cell}_seed{SEED}",
        "dataset": {"manifest": str(DATASET_DIR / "manifest.json")},
        "model": {"kind": "lgbm_reranker", "defaults": {**V26_HPARAMS}},
        "training": {
            "cell": cell,
            "val_strategy": "protein_group",
            "val_fraction": 0.2,
            "seed": SEED,
            "propagate_labels": False,
            "neg_pos_ratio": 10,
        },
        "sweep": {"backend": "none"},
        "output_dir": str(out_dir),
        "tags": [DATASET_NAME, cell, f"seed{SEED}", "lafa_ia_v227_probe"],
        "keep_staging": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "spec_input.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return spec_path


def _bucket_order(proteins: np.ndarray) -> np.ndarray:
    buckets = np.fromiter(
        (zlib.crc32(p.encode("ascii")) % 32 for p in proteins),
        count=len(proteins), dtype=np.int32,
    )
    return np.lexsort((proteins, buckets))


def _eval_slice(cell: str, extra_cols: list[str]):
    """Return (proteins, gos, labels, extra...) in bucket-sort order."""
    sys.path.insert(0, str(REPO_DIR / "src"))
    from protea_reranker_lab.data import iter_batches

    cat, asp = cell.split("-", 1)
    cols = ["protein_accession", "go_term_id", "label", *extra_cols]
    chunks: dict[str, list] = {c: [] for c in cols}
    for batch in iter_batches(
        DATASET_DIR / "eval.parquet", columns=cols,
        category=cat, aspect=asp, batch_size=200_000,
    ):
        for c in cols:
            chunks[c].append(batch.column(c).to_numpy(zero_copy_only=False))
    arrs = {c: np.concatenate(chunks[c]) if chunks[c] else np.array([]) for c in cols}
    order = _bucket_order(arrs["protein_accession"])
    return {c: arrs[c][order] for c in cols}


def train_reranker(cell: str) -> dict:
    out_dir = _run_dir(cell, "reranker")
    run_json = out_dir / "run.json"
    if run_json.exists():
        try:
            r = json.loads(run_json.read_text())
            if r.get("status") == "ok":
                print(f"  [skip train] {cell} reranker")
                return r
        except Exception:
            pass
    spec_path = _make_spec_yaml(cell, out_dir)
    print(f"  [train] {cell} reranker -> {out_dir}", flush=True)
    proc = subprocess.run(
        [str(LAB_PYTHON), str(REPO_DIR / "scripts" / "run.py"), str(spec_path),
         "--datasets-root", str(WORKTREE / "datasets")],
        cwd=str(REPO_DIR), capture_output=True, text=True, timeout=5400,
    )
    if proc.returncode != 0:
        print(f"  [FAIL train] {cell}: {proc.stderr[-600:]}")
        return {"status": "fail"}
    return json.loads(run_json.read_text())


def _write_pred_gt(pred_dir: Path, gt_tsv: Path, proteins, gos, scores, labels):
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_tsv = pred_dir / "pred.tsv"
    with pred_tsv.open("w") as f:
        for p, g, s in zip(proteins, gos, scores):
            f.write(f"{p}\t{g}\t{s:.6f}\n")
    pos = labels > 0
    with gt_tsv.open("w") as f:
        for p, g in zip(proteins[pos], gos[pos]):
            f.write(f"{p}\t{g}\n")
    return pred_tsv


def _internal_fmax(scores: np.ndarray, labels: np.ndarray) -> float:
    """Micro Fmax on the candidate rows over a fine threshold grid.

    Rank-based, candidate-restricted; matches the lab's internal number
    convention (the propagation-free Fmax the trainer reports)."""
    pos = labels > 0
    n_pos = int(pos.sum())
    if n_pos == 0:
        return 0.0
    best = 0.0
    for tau in np.linspace(scores.min(), scores.max(), 200):
        pred = scores >= tau
        tp = int((pred & pos).sum())
        if tp == 0:
            continue
        pr = tp / int(pred.sum())
        rc = tp / n_pos
        f = 2 * pr * rc / (pr + rc)
        best = max(best, f)
    return best


def run_cafaeval(cell: str, arm: str, proteins, gos, scores, labels) -> dict:
    out_dir = _run_dir(cell, arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = out_dir / "cafaeval_metrics.json"

    cell_dir = out_dir / "cafaeval"
    pred_dir = cell_dir / "pred_dir"
    gt_tsv = cell_dir / "gt.tsv"
    pred_tsv = _write_pred_gt(pred_dir, gt_tsv, proteins, gos, scores, labels)
    (pred_dir / f"{cell}.tsv").write_bytes(pred_tsv.read_bytes())

    raw_json = cell_dir / "_raw_metrics.json"
    driver = cell_dir / "_driver.py"
    driver.write_text(_DRIVER_SRC.format(
        obo=str(OBO_PATH), pred_dir=str(pred_dir), gt=str(gt_tsv),
        ia=str(IA_PATH), out_json=str(raw_json),
    ))
    proc = subprocess.run(
        [str(PROTEA_PYTHON), str(driver)],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        print(f"  [FAIL cafaeval] {cell}/{arm}: {proc.stderr[-500:]}")
        return {}

    raw = json.loads(raw_json.read_text())
    target_ns = ASPECT_TO_NS[cell.split("-", 1)[1]]

    def _pick(kind: str, col: str) -> float | None:
        for rec in raw.get(kind, []):
            ns = rec.get("ns") or rec.get("namespace") or ""
            if ns == target_ns and rec.get(col) is not None:
                try:
                    return float(rec[col])
                except (TypeError, ValueError):
                    return None
        return None

    m = {
        "cell": cell,
        "arm": arm,
        "internal_fmax": _internal_fmax(scores, labels),
        "cafaeval_fmax": _pick("f_micro", "f_micro"),
        "wfmax": _pick("f_micro_w", "f_micro_w"),
        "s_min": _pick("s", "s"),
        "n_rows": int(len(proteins)),
        "n_pos": int((labels > 0).sum()),
    }
    metrics_json.write_text(json.dumps(m, indent=2))
    print(f"  [eval] {cell}/{arm}: wFmax={m['wfmax']} "
          f"cafaFmax={m['cafaeval_fmax']} Smin={m['s_min']}")
    return m


def eval_reranker(cell: str) -> dict:
    out_dir = _run_dir(cell, "reranker")
    pred_pq = out_dir / "predictions.parquet"
    if not pred_pq.exists():
        print(f"  [skip] {cell} reranker: no predictions.parquet")
        return {}
    # The trainer's predictions.parquet carries protein_accession/label/score
    # but NOT go_term_id. Re-stage the eval slice in the same bucket-sort
    # order (verified byte-identical on protein + label) to recover go_term_id.
    pred_t = pq.read_table(str(pred_pq),
                           columns=["protein_accession", "label", "score"])
    proteins = pred_t.column("protein_accession").to_numpy(zero_copy_only=False)
    labels = pred_t.column("label").to_numpy(zero_copy_only=False).astype(np.int8)
    scores = pred_t.column("score").to_numpy(zero_copy_only=False).astype(np.float32)

    sl = _eval_slice(cell, [])
    if not np.array_equal(sl["protein_accession"], proteins):
        print(f"  [skip] {cell} reranker: eval-slice protein order drifted "
              f"vs predictions ({len(sl['protein_accession'])} vs {len(proteins)})")
        return {}
    gos = sl["go_term_id"]
    return run_cafaeval(cell, "reranker", proteins, gos, scores, labels)


def eval_knn_baseline(cell: str) -> dict:
    sl = _eval_slice(cell, ["neighbor_vote_fraction"])
    return run_cafaeval(
        cell, "knn",
        sl["protein_accession"], sl["go_term_id"],
        sl["neighbor_vote_fraction"].astype(np.float32), sl["label"].astype(np.int8),
    )


def main() -> int:
    if not (DATASET_DIR / "eval.parquet").exists():
        print(f"ERROR: {DATASET_DIR}/eval.parquet missing; download it first.")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for cell in CELLS:
        print(f"=== {cell} ===", flush=True)
        rr_run = train_reranker(cell)
        if rr_run.get("status") == "ok":
            rr = eval_reranker(cell)
            if rr:
                rr["internal_fmax_train"] = rr_run.get("metrics", {}).get("test_fmax")
                results.append(rr)
        knn = eval_knn_baseline(cell)
        if knn:
            results.append(knn)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    _write_summary(results)
    print(f"\nwrote {OUT_DIR / 'results.json'}")
    return 0


def _write_summary(results: list[dict]) -> None:
    lines = [
        "# F-LAFA-IA.1b v227 probe: IA-weighted eval (prostt5 K3)",
        "",
        "Dataset: bench-v1-K3-v227-lineage-prostt5 (train cutoff v227, "
        "eval band v227->v230).",
        "Protocol: prop=fill norm=cafa no_orphans "
        "ia=IA-swissprot-exp-v227.txt. Headline wFmax = IA-weighted micro "
        "Fmax (f_micro_w); S_min = min semantic distance (s).",
        "",
        "| cell | arm | internal Fmax | cafaeval Fmax | wFmax | S_min |",
        "|---|---|---|---|---|---|",
    ]

    def _fmt(v: object) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"

    for r in results:
        lines.append(
            f"| {r['cell']} | {r['arm']} | {_fmt(r.get('internal_fmax'))} | "
            f"{_fmt(r.get('cafaeval_fmax'))} | {_fmt(r.get('wfmax'))} | "
            f"{_fmt(r.get('s_min'))} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
