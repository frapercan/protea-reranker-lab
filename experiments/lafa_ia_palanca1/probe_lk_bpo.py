"""F-LAFA-IA.2 palanca-1 probe: IA sample weighting on lk-bpo (seed 42).

Trains the v26-binary recipe on ``bench-v1-K5-v226-lineage`` lk-bpo three
ways, all else equal:

  * ``unweighted``   -- ia_weighting=none (the F-LAFA-IA.0 baseline recipe).
  * ``ia_positives`` -- ia_weighting=positives (palanca 1, positives only).
  * ``ia_all``       -- ia_weighting=all (palanca 1, every row by IA(go)).

plus the KNN baseline arm (no training; the ``neighbor_vote_fraction``
score carried in eval.parquet). For each arm it records the four reference
numbers under the LAFA protocol (prop=fill, norm=cafa, no_orphans,
ia=IA-swissprot-exp-v227.txt):

  internal_fmax : lab candidate-restricted micro Fmax (no propagation).
  cafaeval_fmax : cafaeval unweighted micro Fmax (f_micro).
  wfmax         : cafaeval IA-weighted micro Fmax (f_micro_w); headline.
  s_min         : cafaeval minimum semantic distance (s).

GATE (brief section 7.3): does the IA-weighted reranker lift wFmax over the
KNN baseline by more than the F-LAFA-IA.0 reference delta (lk-bpo reranker
wFmax was +0.0785 over KNN)? If not, stop and report.

cafaeval lives in the PROTEA venv; training runs in the lab venv. Both
parquet artefacts are read-only inputs that live in the dev workspace
(gitignored), so this driver resolves them by absolute path.

Run::

    .venv/bin/python experiments/lafa_ia_palanca1/probe_lk_bpo.py
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
DATASET_NAME = "bench-v1-K5-v226-lineage"
DATASET_DIR = REPO_DIR / "datasets" / DATASET_NAME
OBO_PATH = REPO_DIR / "datasets" / "bench-v1-K5" / "go.obo"
IA_PATH = WORKTREE / "datasets" / "ia" / "IA-swissprot-exp-v227.txt"
# Generated artefacts (run dirs, results.json, SUMMARY.md) land under runs/,
# which the dataset-name linter skips; only this driver lives in experiments/.
RUNS_ROOT = WORKTREE / "runs" / "lafa_ia_palanca1"
OUT_DIR = RUNS_ROOT

LAB_PYTHON = REPO_DIR / ".venv" / "bin" / "python"
PROTEA_PYTHON = Path("/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python")

CELL = "lk-bpo"
SEED = 42
# F-LAFA-IA.0 reference (PR #57): lk-bpo reranker wFmax was +0.0785 over KNN.
BASELINE_DELTA_WFMAX = 0.0785

# v26-binary recipe (binary objective, lean+lin+emb, neg_pos_ratio=10).
V26_HPARAMS = {
    "objective": "binary",
    "num_boost_round": 10000,
    "early_stopping_rounds": 100,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "drop_features": [],
}

# reranker arms: (arm name, ia_weighting mode).
ARMS = [
    ("unweighted", "none"),
    ("ia_positives", "positives"),
    ("ia_all", "all"),
]

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
    max_terms=500, th_step=0.001, n_cpu=1,
)
out = {{}}
for kind, df_best in dfs_best.items():
    out[kind] = df_best.reset_index().to_dict(orient="records")
with open("{out_json}", "w") as f:
    json.dump(out, f, indent=2, default=str)
'''


def _run_dir(arm: str) -> Path:
    return RUNS_ROOT / arm


def _make_spec_yaml(arm: str, ia_mode: str, out_dir: Path) -> Path:
    defaults = {**V26_HPARAMS, "ia_weighting": ia_mode, "ia_path": str(IA_PATH)}
    spec = {
        "schema_version": "v1",
        "name": f"lafa_ia_palanca1_{CELL}_{arm}_seed{SEED}",
        "dataset": {"manifest": str(DATASET_DIR / "manifest.json")},
        "model": {"kind": "lgbm_reranker", "defaults": defaults},
        "training": {
            "cell": CELL,
            "val_strategy": "protein_group",
            "val_fraction": 0.2,
            "seed": SEED,
            "propagate_labels": False,
            "neg_pos_ratio": 10,
        },
        "sweep": {"backend": "none"},
        "output_dir": str(out_dir),
        "tags": [DATASET_NAME, CELL, f"seed{SEED}", "lafa_ia_palanca1", arm],
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


def _eval_slice(extra_cols: list[str]) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(REPO_DIR / "src"))
    from protea_reranker_lab.data import iter_batches

    cat, asp = CELL.split("-", 1)
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


def train_reranker(arm: str, ia_mode: str) -> dict:
    out_dir = _run_dir(arm)
    run_json = out_dir / "run.json"
    if run_json.exists():
        try:
            r = json.loads(run_json.read_text())
            if r.get("status") == "ok":
                print(f"  [skip train] {arm}")
                return r
        except json.JSONDecodeError:
            pass
    spec_path = _make_spec_yaml(arm, ia_mode, out_dir)
    print(f"  [train] {arm} (ia_weighting={ia_mode}) -> {out_dir}", flush=True)
    proc = subprocess.run(
        [str(LAB_PYTHON), str(REPO_DIR / "scripts" / "run.py"), str(spec_path),
         "--datasets-root", str(DATASET_DIR.parent)],
        cwd=str(REPO_DIR), capture_output=True, text=True, timeout=7200,
    )
    if proc.returncode != 0:
        print(f"  [FAIL train] {arm}: {proc.stderr[-800:]}")
        return {"status": "fail"}
    return json.loads(run_json.read_text())


def _write_pred_gt(pred_dir, gt_tsv, proteins, gos, scores, labels) -> Path:
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
        best = max(best, 2 * pr * rc / (pr + rc))
    return best


def run_cafaeval(arm: str, proteins, gos, scores, labels) -> dict:
    out_dir = _run_dir(arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_dir = out_dir / "cafaeval"
    pred_dir = cell_dir / "pred_dir"
    gt_tsv = cell_dir / "gt.tsv"
    pred_tsv = _write_pred_gt(pred_dir, gt_tsv, proteins, gos, scores, labels)
    (pred_dir / f"{CELL}.tsv").write_bytes(pred_tsv.read_bytes())

    raw_json = cell_dir / "_raw_metrics.json"
    driver = cell_dir / "_driver.py"
    driver.write_text(_DRIVER_SRC.format(
        obo=str(OBO_PATH), pred_dir=str(pred_dir), gt=str(gt_tsv),
        ia=str(IA_PATH), out_json=str(raw_json),
    ))
    proc = subprocess.run(
        [str(PROTEA_PYTHON), str(driver)],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        print(f"  [FAIL cafaeval] {arm}: {proc.stderr[-700:]}")
        return {}

    raw = json.loads(raw_json.read_text())
    target_ns = ASPECT_TO_NS[CELL.split("-", 1)[1]]

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
        "arm": arm,
        "internal_fmax": _internal_fmax(scores, labels),
        "cafaeval_fmax": _pick("f_micro", "f_micro"),
        "wfmax": _pick("f_micro_w", "f_micro_w"),
        "s_min": _pick("s", "s"),
        "n_rows": int(len(proteins)),
        "n_pos": int((labels > 0).sum()),
    }
    (out_dir / "cafaeval_metrics.json").write_text(json.dumps(m, indent=2))
    print(f"  [eval] {arm}: wFmax={m['wfmax']} cafaFmax={m['cafaeval_fmax']} "
          f"Smin={m['s_min']}")
    return m


def eval_reranker(arm: str) -> dict:
    pred_pq = _run_dir(arm) / "predictions.parquet"
    if not pred_pq.exists():
        print(f"  [skip] {arm}: no predictions.parquet")
        return {}
    pred_t = pq.read_table(
        str(pred_pq), columns=["protein_accession", "label", "score"]
    )
    proteins = pred_t.column("protein_accession").to_numpy(zero_copy_only=False)
    labels = pred_t.column("label").to_numpy(zero_copy_only=False).astype(np.int8)
    scores = pred_t.column("score").to_numpy(zero_copy_only=False).astype(np.float32)

    sl = _eval_slice([])
    if not np.array_equal(sl["protein_accession"], proteins):
        print(f"  [skip] {arm}: eval-slice protein order drifted "
              f"({len(sl['protein_accession'])} vs {len(proteins)})")
        return {}
    return run_cafaeval(arm, proteins, sl["go_term_id"], scores, labels)


def eval_knn_baseline() -> dict:
    sl = _eval_slice(["neighbor_vote_fraction"])
    return run_cafaeval(
        "knn",
        sl["protein_accession"], sl["go_term_id"],
        sl["neighbor_vote_fraction"].astype(np.float32),
        sl["label"].astype(np.int8),
    )


def main() -> int:
    if not (DATASET_DIR / "eval.parquet").exists():
        print(f"ERROR: {DATASET_DIR}/eval.parquet missing.")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for arm, ia_mode in ARMS:
        print(f"=== reranker arm: {arm} ===", flush=True)
        run = train_reranker(arm, ia_mode)
        if run.get("status") != "ok":
            continue
        m = eval_reranker(arm)
        if m:
            m["test_fmax_train"] = run.get("metrics", {}).get("test_fmax")
            m["ia_weighting"] = run.get("ia_weighting")
            results[arm] = m

    print("=== knn baseline ===", flush=True)
    knn = eval_knn_baseline()
    if knn:
        results["knn"] = knn

    verdict = _gate(results)
    payload = {
        "cell": CELL,
        "seed": SEED,
        "dataset": DATASET_NAME,
        "obo": "datasets/bench-v1-K5/go.obo",
        "ia_table": str(IA_PATH.relative_to(WORKTREE)),
        "baseline_delta_wfmax": BASELINE_DELTA_WFMAX,
        "cafaeval_call": {
            "prop": "fill", "norm": "cafa", "no_orphans": True,
            "max_terms": 500, "th_step": 0.001, "n_cpu": 1, "ia": True,
        },
        "results": results,
        "gate": verdict,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2))
    _write_summary(payload)
    print(f"\nwrote {OUT_DIR / 'results.json'}")
    print(f"\nGATE: {verdict['verdict']}\n{verdict['explanation']}")
    return 0


def _gate(results: dict[str, dict]) -> dict:
    knn = results.get("knn", {}).get("wfmax")
    deltas: dict[str, float | None] = {}
    best_arm, best_delta = None, None
    for arm, _ in ARMS:
        w = results.get(arm, {}).get("wfmax")
        d = (w - knn) if (w is not None and knn is not None) else None
        deltas[arm] = d
        if d is not None and (best_delta is None or d > best_delta):
            best_arm, best_delta = arm, d

    base_unw = deltas.get("unweighted")
    lifts_over_knn = best_delta is not None and best_delta > BASELINE_DELTA_WFMAX
    best_ia = None
    best_ia_delta = None
    for arm in ("ia_positives", "ia_all"):
        d = deltas.get(arm)
        if d is not None and (best_ia_delta is None or d > best_ia_delta):
            best_ia, best_ia_delta = arm, d
    lifts_over_unweighted = (
        best_ia_delta is not None and base_unw is not None
        and best_ia_delta > base_unw
    )

    passed = bool(lifts_over_knn and lifts_over_unweighted)
    explanation = (
        f"best wFmax delta over KNN = {best_delta} ({best_arm}); "
        f"F-LAFA-IA.0 reference = +{BASELINE_DELTA_WFMAX}. "
        f"best IA arm = {best_ia} (delta vs KNN {best_ia_delta}); "
        f"unweighted reranker delta vs KNN = {base_unw}. "
        f"IA lifts over baseline-over-KNN: {lifts_over_knn}; "
        f"IA lifts over unweighted reranker: {lifts_over_unweighted}."
    )
    return {
        "verdict": "PASS" if passed else "STOP",
        "deltas_wfmax_vs_knn": deltas,
        "best_arm": best_arm,
        "best_ia_arm": best_ia,
        "recommend_palanca2": passed,
        "explanation": explanation,
    }


def _write_summary(payload: dict) -> None:
    res = payload["results"]
    lines = [
        "# F-LAFA-IA.2 palanca-1 probe: lk-bpo seed 42",
        "",
        f"Dataset: {payload['dataset']} (v26-binary recipe, binary objective, "
        "neg_pos_ratio=10).",
        "Protocol: prop=fill norm=cafa no_orphans "
        "ia=IA-swissprot-exp-v227.txt. Headline wFmax = IA-weighted micro "
        "Fmax (f_micro_w); S_min = min semantic distance (s, lower is better).",
        "",
        "| arm | internal Fmax | cafaeval Fmax | wFmax | S_min | d wFmax vs KNN |",
        "|---|---|---|---|---|---|",
    ]
    knn_w = res.get("knn", {}).get("wfmax")

    def _fmt(v: object) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"

    order = [a for a, _ in ARMS] + ["knn"]
    for arm in order:
        r = res.get(arm)
        if not r:
            continue
        w = r.get("wfmax")
        d = (w - knn_w) if (w is not None and knn_w is not None and arm != "knn") else None
        lines.append(
            f"| {arm} | {_fmt(r.get('internal_fmax'))} "
            f"| {_fmt(r.get('cafaeval_fmax'))} | {_fmt(w)} "
            f"| {_fmt(r.get('s_min'))} | {_fmt(d)} |"
        )
    g = payload["gate"]
    lines += [
        "",
        f"GATE: **{g['verdict']}** (reference delta +{payload['baseline_delta_wfmax']}).",
        "",
        g["explanation"],
        "",
        "S_min is a semantic distance: lower is better. wFmax is the honest "
        "headline; the unweighted cafaeval Fmax is propagation-dominated and "
        "may stay flat or dip (brief section 5).",
    ]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
