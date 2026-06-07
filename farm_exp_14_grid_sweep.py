"""FARM-EXP.14. Per-PLM reranker sweep (full binary x lambdarank grid).

Single-seed grid: 9 cells x 8 PLMs x 3 K x 2 objectives = 432 reranker runs.

  - binary arm    = v27-binary recipe (objective=binary, neg_pos_ratio=10,
                    num_boost_round=10000, early_stopping_rounds=100,
                    learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                    val_strategy=protein_group, val_fraction=0.2, no drops).
  - lambdarank arm = v22-lambdarank recipe (objective=lambdarank, drop the
                    anc2vec_* + emb_pca_* families, otherwise same hparams).

Each run lands at::

    runs/transversal/<shortid>/<cell>/seed<seed>/
        spec_input.yaml
        run.json                 (status=ok + metrics from scripts/run.py)
        model.txt
        predictions.parquet
        cafaeval_metrics.json    (cafaeval Fmax, run via PROTEA venv)

The <shortid> is the canonical ``protea_contracts.axis_tuple_shortid`` digest
of the (plm, k, reranker_spec_id, ..., propagation, ensemble_spec) axis tuple,
so FARM-EXP.3 bootstrap_cis and FARM-EXP.4 champion tracking treat these runs
uniformly with the rest of the transversal grid.

Idempotency / resume
--------------------
A run whose ``run.json`` reports ``status=ok`` is skipped. cafaeval is skipped
when ``cafaeval_metrics.json`` already holds a non-null fmax. Re-invoking the
script after a crash resumes from the first incomplete cell. Safe to launch
under nohup and re-launch on top of a partial grid.

Grid log
--------
After every cell the manifest CSV is rewritten at
``agent-farm/plans/farm-platform/artefacts/farm_exp_14_grid_log.csv`` with:
plm, k, cell, objective, seed, fmax, auprc, smin, wall_seconds, status.

CLI
---
::

    # full 432-run grid, default seed 42
    python farm_exp_14_grid_sweep.py

    # dry-run plan (no training): print skip/todo classification
    python farm_exp_14_grid_sweep.py --dry-run

    # restrict the grid (handy for batched executor passes)
    python farm_exp_14_grid_sweep.py --plms prostt5,esm2_650m --ks 5 \\
        --objectives binary --cells nk-mfo,lk-cco
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from protea_contracts import axis_tuple_shortid

# ----------------------------------------------------------------- paths
#
# datasets/ and runs/ are gitignored; the parquet artefacts + LightGBM
# boosters live ONLY in the canonical main-repo checkout, never in an
# ephemeral worktree. So while this script is version-controlled (and may
# run from a worktree), at runtime it always reads datasets and writes runs
# under the canonical repo, matching v27_binary_multiseed_sweep.py.

LAB_REPO = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab")
REPO_DATASETS = LAB_REPO / "datasets"
RUNS_ROOT = LAB_REPO / "runs" / "transversal"
RUN_SCRIPT = LAB_REPO / "scripts" / "run.py"

# obo is shared (PLM-blind), under the legacy bench-v1-K5 dataset.
OBO_PATH = REPO_DATASETS / "bench-v1-K5" / "go.obo"

LAB_PYTHON = LAB_REPO / ".venv" / "bin" / "python"
# cafaeval ships in the PROTEA venv, not the lab venv (see run_cafaeval_phase.py).
PROTEA_PYTHON = Path("/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python")

GRID_LOG = (
    Path("/home/frapercan/Thesis2/agent-farm/plans/farm-platform")
    / "artefacts" / "farm_exp_14_grid_log.csv"
)

# ----------------------------------------------------------------- grid

PLMS: tuple[str, ...] = (
    "ankh_base", "ankh_large", "esm2_150m", "esm2_3b",
    "esm2_650m", "esmc_600m", "prostt5", "prot_t5",
)
KS: tuple[int, ...] = (3, 5, 10)
CELLS: tuple[str, ...] = (
    "nk-bpo", "nk-cco", "nk-mfo",
    "lk-bpo", "lk-cco", "lk-mfo",
    "pk-bpo", "pk-cco", "pk-mfo",
)
OBJECTIVES: tuple[str, ...] = ("binary", "lambdarank")

ASPECT_TO_NS = {
    "bpo": "biological_process",
    "mfo": "molecular_function",
    "cco": "cellular_component",
}

# Shared hparams (acceptance FARM-EXP.14): both arms.
_BASE_HPARAMS: dict[str, Any] = {
    "num_boost_round": 10000,
    "early_stopping_rounds": 100,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
}

# v22-lambdarank arm drops anc2vec_* + emb_pca_* (acceptance), nothing else.
_LAMBDARANK_DROP_FEATURES: list[str] = [
    "anc2vec_neighbor_cos", "anc2vec_neighbor_maxcos", "anc2vec_has_emb",
    "anc2vec_query_known_cos", "anc2vec_query_known_maxcos",
    "anc2vec_query_known_count",
    *[f"emb_pca_query_{i}" for i in range(16)],
]

# axis-tuple reranker_spec_id tag per objective arm (matches champions.md
# "rr=reranker:..." convention).
_RERANKER_SPEC_ID = {
    "binary": "reranker:v27-binary",
    "lambdarank": "reranker:v22-lambdarank",
}

# Per-run cafaeval hard limit (cafa_eval over ~500k rows).
_CAFAEVAL_TIMEOUT_S = 900
# Per-run training hard limit. lambdarank converges slower at full rows.
_TRAIN_TIMEOUT_S = 5400


# ------------------------------------------------------------- axis / paths


def _dataset_dir(plm: str, k: int) -> Path:
    return REPO_DATASETS / f"bench-v1-K{k}-v226-lineage-{plm}"


def _axis_tuple(plm: str, k: int, objective: str) -> dict[str, Any]:
    """Deterministic axis payload -> stable shortid per (plm, k, objective)."""
    return {
        "plm": plm,
        "k": k,
        "reranker_spec_id": _RERANKER_SPEC_ID[objective],
        "feature_schema_sha": "lean_lin_emb" if objective == "binary" else "lean_lin",
        "eval_set_name": f"bench-v1-K{k}-v226-lineage-{plm}",
        "eval_set_manifest_sha": "v226-lineage",
        "propagation": "tpr_pred",
        "ensemble_spec": "none",
    }


def _shortid(plm: str, k: int, objective: str) -> str:
    return axis_tuple_shortid(_axis_tuple(plm, k, objective))


def _run_dir(plm: str, k: int, objective: str, cell: str, seed: int) -> Path:
    return RUNS_ROOT / _shortid(plm, k, objective) / cell / f"seed{seed}"


# ------------------------------------------------------------- spec build


def _make_spec_yaml(
    plm: str, k: int, objective: str, cell: str, seed: int, out_dir: Path,
) -> Path:
    drop = _LAMBDARANK_DROP_FEATURES if objective == "lambdarank" else []
    spec = {
        "schema_version": "v1",
        "name": f"farm_exp_14_{plm}_K{k}_{objective}_{cell}_seed{seed}",
        "dataset": {
            "manifest": str(_dataset_dir(plm, k) / "manifest.json"),
        },
        "model": {
            "kind": "lgbm_reranker",
            "defaults": {
                "objective": objective,
                **_BASE_HPARAMS,
                "drop_features": drop,
            },
        },
        "training": {
            "cell": cell,
            "val_strategy": "protein_group",
            "val_fraction": 0.2,
            "seed": seed,
            "propagate_labels": False,
            "neg_pos_ratio": 10,
        },
        "sweep": {"backend": "none"},
        "output_dir": str(out_dir),
        "tags": [
            f"bench-v1-K{k}-v226-lineage-{plm}",
            cell,
            f"seed{seed}",
            "farm_exp_14",
            _RERANKER_SPEC_ID[objective],
            f"K{k}",
            plm,
        ],
        "keep_staging": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "spec_input.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return spec_path


# ------------------------------------------------------------- training


def _train_done(out_dir: Path) -> bool:
    run_json = out_dir / "run.json"
    if not run_json.exists():
        return False
    try:
        report = json.loads(run_json.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(report, dict) and report.get("status") == "ok"


def run_training(
    plm: str, k: int, objective: str, cell: str, seed: int,
) -> tuple[dict, float]:
    out_dir = _run_dir(plm, k, objective, cell, seed)
    tag = f"{plm} K{k} {objective} {cell} seed={seed}"

    if _train_done(out_dir):
        report = json.loads((out_dir / "run.json").read_text())
        fmax = report.get("metrics", {}).get("test_fmax", 0.0)
        print(f"  [skip train] {tag} (test_fmax={fmax:.4f})", flush=True)
        return report, 0.0

    spec_path = _make_spec_yaml(plm, k, objective, cell, seed, out_dir)
    print(f"  [train] {tag} -> {out_dir}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(
        [str(LAB_PYTHON), str(RUN_SCRIPT), str(spec_path),
         "--datasets-root", str(REPO_DATASETS)],
        cwd=str(LAB_REPO), capture_output=True, text=True,
        timeout=_TRAIN_TIMEOUT_S,
    )
    dur = time.monotonic() - t0

    if proc.returncode != 0 or not _train_done(out_dir):
        print(f"  [FAIL train] {tag} exit={proc.returncode} in {dur:.0f}s")
        print(f"  stderr: {proc.stderr[-600:]}")
        return {"status": "fail", "duration_s": dur}, dur

    report = json.loads((out_dir / "run.json").read_text())
    fmax = report.get("metrics", {}).get("test_fmax", 0.0)
    print(f"  [done train] {tag} test_fmax={fmax:.4f} dur={dur / 60:.1f}min",
          flush=True)
    return report, dur


# ------------------------------------------------------------- cafaeval

_DRIVER_SRC = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGINT, signal.SIG_DFL)
df, dfs_best = cafa_eval(
    "{obo}", "{pred_dir}", "{gt}",
    prop="fill", norm="cafa", no_orphans=True,
    max_terms=500, th_step=0.001, n_cpu=1,
)
out = {{}}
for kind, df_best in dfs_best.items():
    out[kind] = df_best.reset_index().to_dict(orient="records")
with open("{out_json}", "w") as f:
    json.dump(out, f, indent=2, default=str)
'''


def _extract_alignment(
    plm: str, k: int, cell: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-derive (protein, go, label) in the trainer's bucket-sort order."""
    sys.path.insert(0, str(LAB_REPO / "src"))
    from protea_reranker_lab.data import iter_batches

    cat, asp = cell.split("-", 1)
    prots, gos, labs = [], [], []
    for batch in iter_batches(
        _dataset_dir(plm, k) / "eval.parquet",
        columns=["protein_accession", "go_term_id", "label"],
        category=cat, aspect=asp, batch_size=200_000,
    ):
        prots.append(batch.column("protein_accession").to_numpy(zero_copy_only=False))
        gos.append(batch.column("go_term_id").to_numpy(zero_copy_only=False))
        labs.append(
            batch.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
        )
    proteins = np.concatenate(prots)
    go_terms = np.concatenate(gos)
    labels = np.concatenate(labs)
    buckets = np.fromiter(
        (zlib.crc32(p.encode("ascii")) % 32 for p in proteins),
        count=len(proteins), dtype=np.int32,
    )
    order = np.lexsort((proteins, buckets))
    return proteins[order], go_terms[order], labels[order]


def _parse_cafaeval_fmax(raw: dict, aspect: str) -> float | None:
    target_ns = ASPECT_TO_NS.get(aspect)
    for records in raw.values():
        for rec in records:
            ns = rec.get("ns") or rec.get("namespace") or ""
            if ns == target_ns:
                for key in ("f", "Fmax", "fmax"):
                    if rec.get(key) is not None:
                        try:
                            return float(rec[key])
                        except (TypeError, ValueError):
                            pass
    # fall back to the first parseable fmax across all namespaces
    for records in raw.values():
        for rec in records:
            for key in ("f", "Fmax", "fmax"):
                if rec.get(key) is not None:
                    try:
                        return float(rec[key])
                    except (TypeError, ValueError):
                        pass
    return None


def run_cafaeval(plm: str, k: int, objective: str, cell: str, seed: int) -> dict:
    out_dir = _run_dir(plm, k, objective, cell, seed)
    pred_pq = out_dir / "predictions.parquet"
    metrics_json = out_dir / "cafaeval_metrics.json"
    tag = f"{plm} K{k} {objective} {cell} seed={seed}"

    if metrics_json.exists():
        try:
            m = json.loads(metrics_json.read_text())
            if m.get("cafaeval_fmax") is not None:
                print(f"  [skip cafaeval] {tag} fmax={m['cafaeval_fmax']:.4f}")
                return m
        except (OSError, json.JSONDecodeError):
            pass

    if not pred_pq.exists():
        print(f"  [skip cafaeval] {tag}: no predictions.parquet")
        return {}

    print(f"  [cafaeval] {tag}", flush=True)
    pred_t = pq.read_table(
        str(pred_pq), columns=["protein_accession", "label", "score"]
    )
    proteins_p = pred_t.column("protein_accession").to_numpy(zero_copy_only=False)
    labels_p = pred_t.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    scores = pred_t.column("score").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    proteins_a, gos, labels_a = _extract_alignment(plm, k, cell)
    if not np.array_equal(proteins_p, proteins_a):
        print(f"  [err cafaeval] {tag}: protein alignment mismatch")
        return {}
    if not np.array_equal(labels_p, labels_a):
        print(f"  [err cafaeval] {tag}: label mismatch")
        return {}

    cell_dir = out_dir / "cafaeval"
    pred_dir = cell_dir / "pred_dir"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_tsv = cell_dir / "pred.tsv"
    gt_tsv = cell_dir / "gt.tsv"
    with pred_tsv.open("w") as fh:
        for p, g, s in zip(proteins_p, gos, scores):
            fh.write(f"{p}\t{g}\t{s:.6f}\n")
    pos = labels_p > 0
    with gt_tsv.open("w") as fh:
        for p, g in zip(proteins_p[pos], gos[pos]):
            fh.write(f"{p}\t{g}\n")
    (pred_dir / f"{cell}.tsv").write_bytes(pred_tsv.read_bytes())

    raw_json = cell_dir / "_raw_metrics.json"
    driver = cell_dir / "_driver.py"
    driver.write_text(_DRIVER_SRC.format(
        obo=str(OBO_PATH), pred_dir=str(pred_dir),
        gt=str(gt_tsv), out_json=str(raw_json),
    ))
    proc = subprocess.run(
        [str(PROTEA_PYTHON), str(driver)],
        capture_output=True, text=True, timeout=_CAFAEVAL_TIMEOUT_S,
    )
    if proc.returncode != 0:
        print(f"  [FAIL cafaeval] {tag}: {proc.stderr[-400:]}")
        return {}

    raw = json.loads(raw_json.read_text())
    aspect = cell.split("-", 1)[1]
    cafaeval_fmax = _parse_cafaeval_fmax(raw, aspect)
    metrics = {
        "plm": plm, "k": k, "objective": objective, "cell": cell, "seed": seed,
        "cafaeval_fmax": cafaeval_fmax,
        "n_rows": int(len(proteins_p)),
        "n_positives": int(pos.sum()),
    }
    metrics_json.write_text(json.dumps(metrics, indent=2))
    fmax_str = f"{cafaeval_fmax:.4f}" if cafaeval_fmax is not None else "None"
    print(f"  [done cafaeval] {tag} cafaeval_fmax={fmax_str}", flush=True)
    return metrics


# ------------------------------------------------------------- grid log


def _collect_log_rows(
    plms: list[str], ks: list[int], objectives: list[str],
    cells: list[str], seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plm in plms:
        for k in ks:
            for objective in objectives:
                for cell in cells:
                    out_dir = _run_dir(plm, k, objective, cell, seed)
                    row: dict[str, Any] = {
                        "plm": plm, "k": k, "cell": cell,
                        "objective": objective, "seed": seed,
                        "fmax": "", "auprc": "", "smin": "",
                        "wall_seconds": "", "status": "todo",
                    }
                    run_json = out_dir / "run.json"
                    if run_json.exists():
                        try:
                            rep = json.loads(run_json.read_text())
                        except (OSError, json.JSONDecodeError):
                            rep = {}
                        row["status"] = rep.get("status", "todo")
                        metrics = rep.get("metrics", {}) or {}
                        if metrics.get("test_auprc") is not None:
                            row["auprc"] = metrics["test_auprc"]
                        if metrics.get("test_smin") is not None:
                            row["smin"] = metrics["test_smin"]
                        if rep.get("duration_s") is not None:
                            row["wall_seconds"] = round(float(rep["duration_s"]), 1)
                    caf = out_dir / "cafaeval_metrics.json"
                    if caf.exists():
                        try:
                            m = json.loads(caf.read_text())
                            if m.get("cafaeval_fmax") is not None:
                                row["fmax"] = m["cafaeval_fmax"]
                        except (OSError, json.JSONDecodeError):
                            pass
                    rows.append(row)
    return rows


def write_grid_log(rows: list[dict[str, Any]]) -> None:
    GRID_LOG.parent.mkdir(parents=True, exist_ok=True)
    fields = ["plm", "k", "cell", "objective", "seed",
              "fmax", "auprc", "smin", "wall_seconds", "status"]
    with GRID_LOG.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_full_grid_log(seed: int) -> None:
    """Rewrite the manifest CSV over the full canonical 432-cell grid.

    Always logs every (plm, k, objective, cell) regardless of the run
    filter, so a scoped re-run never shrinks the campaign manifest.
    """
    write_grid_log(
        _collect_log_rows(list(PLMS), list(KS), list(OBJECTIVES), list(CELLS), seed)
    )


# ------------------------------------------------------------- driver


def _csv_list(raw: str | None, default: tuple) -> list:
    if not raw:
        return list(default)
    return [v.strip() for v in raw.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FARM-EXP.14 reranker sweep")
    parser.add_argument("--plms", default=None, help="comma list (default all 8)")
    parser.add_argument("--ks", default=None, help="comma list (default 3,5,10)")
    parser.add_argument("--objectives", default=None,
                        help="comma list (default binary,lambdarank)")
    parser.add_argument("--cells", default=None, help="comma list (default 9 cells)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    plms = _csv_list(args.plms, PLMS)
    ks = [int(v) for v in _csv_list(args.ks, tuple(str(x) for x in KS))]
    objectives = _csv_list(args.objectives, OBJECTIVES)
    cells = _csv_list(args.cells, CELLS)
    seed = args.seed

    total = len(plms) * len(ks) * len(objectives) * len(cells)
    print(f"=== FARM-EXP.14 grid: {total} runs "
          f"({len(plms)} PLM x {len(ks)} K x {len(objectives)} obj x "
          f"{len(cells)} cell) seed={seed} ===", flush=True)

    if args.dry_run:
        done = todo = blocked = 0
        for plm in plms:
            for k in ks:
                ds = _dataset_dir(plm, k)
                ds_ok = (ds / "manifest.json").exists()
                for objective in objectives:
                    for cell in cells:
                        out_dir = _run_dir(plm, k, objective, cell, seed)
                        if not ds_ok:
                            blocked += 1
                            print(f"  [blocked] {plm} K{k} {objective} {cell} "
                                  f"(missing dataset {ds.name})")
                        elif _train_done(out_dir) and (
                            out_dir / "cafaeval_metrics.json").exists():
                            done += 1
                        else:
                            todo += 1
        print(f"=== dry-run: done={done} todo={todo} blocked={blocked} ===")
        write_full_grid_log(seed)
        return 0

    n_ok = n_fail = n_skip = 0
    for plm in plms:
        for k in ks:
            ds = _dataset_dir(plm, k)
            if not (ds / "manifest.json").exists():
                print(f"  [blocked] dataset missing: {ds.name}; skipping (plm={plm}, K={k})")
                continue
            for objective in objectives:
                for cell in cells:
                    out_dir = _run_dir(plm, k, objective, cell, seed)
                    already = _train_done(out_dir) and (
                        out_dir / "cafaeval_metrics.json").exists()
                    report, _ = run_training(plm, k, objective, cell, seed)
                    if report.get("status") == "ok":
                        run_cafaeval(plm, k, objective, cell, seed)
                        n_skip += 1 if already else 0
                        n_ok += 0 if already else 1
                    else:
                        n_fail += 1
                    # rewrite the manifest after every cell so progress is
                    # observable mid-run and survives a crash.
                    write_full_grid_log(seed)

    print(f"=== FARM-EXP.14 grid pass complete: "
          f"new_ok={n_ok} skipped={n_skip} failed={n_fail} ===", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
