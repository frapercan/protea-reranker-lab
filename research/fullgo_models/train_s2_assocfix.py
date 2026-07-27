"""Retrain the S2 trio on the FRESH association-corrected export 46be427a.

WHY: INT-8 showed the deployed S2 trio regressed to 0.3462 because the rebuilt
GO cooccurrence (#667) drifted the ``association`` feature VALUES from what the
trio was originally fit against (association is ungoverned by feature_schema_sha,
ADR-D45). Fix: retrain on a fresh export carrying the corrected association.

RECIPE (= the S2 trio that scored 0.3745 / b21b187c):
  features = lean-31 (the same 30 raw cols) + IA  (31 total)
  self_prior_score: read NATIVELY from the new parquet (already the fixed
                    string-matched value; the old v5 overlay workaround is gone).
  association_total/cross/present: read NATIVELY (the corrected values; the point).
  IA: still absent from the export -> looked up per-row as IA[go_term_id] from
      lafa_t0_Sep_2025/IA.tsv (go_term_id is the GO-id STRING).
  params: binary/auc, lr=0.05, num_leaves=63, min_data_in_leaf=100,
          feature_fraction=0.9, bagging_fraction=0.9/freq=1, seed=42,
          scale_pos_weight = #neg/#pos per category (the S2 lever).
  num_boost=1500, early_stop=50 on the held-out eval (220->227) split.

Feature NAMES match the b21b187c trio EXACTLY (LEAN + ["IA"]) so the new
boosters are a drop-in swap at serve: run_cafa_evaluation re-applies them on the
f377adae prediction set's JSONB columns.

De-risk: after training, score the new eval (220->227) split with the optimistic
cafaeval recipe (no-TOI) and compare to the known "both"=0.5439 MEAN / PK 0.3720
validation baseline. Emits live metrics to MLflow. Pure offline; no live DB.
"""
from __future__ import annotations

import gc
import io
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from minio import Minio

DATASET = "datasets/fullgo-union-SELECT-160-220-227-c99db18-assoc"
DATASET_ID = "46be427a-0a9a-4cbd-b725-afdf9488cabc"
BUCKET = "protea"
IA_TSV = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
OUT = "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_s2_assocfix"
os.makedirs(OUT, exist_ok=True)
CLIENT = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
CATS = ("nk", "lk", "pk")

LEAN = [
    "distance", "identity_nw", "similarity_nw", "alignment_score_nw", "gaps_pct_nw",
    "alignment_length_nw", "identity_sw", "similarity_sw", "alignment_score_sw",
    "gaps_pct_sw", "alignment_length_sw", "length_query", "length_ref",
    "taxonomic_distance", "taxonomic_common_ancestors", "vote_count", "k_position",
    "go_term_frequency", "ref_annotation_density", "neighbor_distance_std",
    "neighbor_vote_fraction", "neighbor_min_distance", "neighbor_mean_distance",
    "knn_present", "classifier_score", "classifier_present", "self_prior_score",
    "association_total", "association_cross", "association_present",
]
NAMES = LEAN + ["IA"]  # 31, drop-in swap with the b21b187c trio

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.9,
              bagging_freq=1, verbosity=-1, num_threads=8, seed=42)
NUM_BOOST = 1500
EARLY = 50


def free_g() -> float:
    with open("/proc/meminfo") as fh:
        for ln in fh:
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) / 1024 / 1024
    return 99.0


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] free={free_g():.1f}G  {m}", flush=True)


def load_ia() -> dict[str, float]:
    ia: dict[str, float] = {}
    for ln in open(IA_TSV):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 2:
            try:
                ia[p[0]] = float(p[1])
            except ValueError:
                pass
    return ia


def load_split_for_cat(raw: bytes, cat: str, ia: dict[str, float]):
    """Stream one split's parquet, keep rows of one category, build the
    31-col matrix (lean-30 native + IA[go_term_id]) + labels."""
    pf = pq.ParquetFile(io.BytesIO(raw))
    cols = LEAN + ["category", "label", "go_term_id"]
    Xs, ys = [], []
    for b in pf.iter_batches(batch_size=2_000_000, columns=cols):
        cat_arr = np.asarray(b.column("category").to_numpy(zero_copy_only=False), dtype=object)
        m = cat_arr == cat
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        X = np.empty((idx.size, len(LEAN) + 1), dtype=np.float32)
        for j, f in enumerate(LEAN):
            col = b.column(f).to_numpy(zero_copy_only=False)
            X[:, j] = np.asarray(col, dtype=np.float64)[idx].astype(np.float32)
        goid = b.column("go_term_id").to_pylist()
        iav = np.fromiter((ia.get(goid[k], 0.0) for k in idx.tolist()), dtype=np.float32,
                          count=idx.size)
        X[:, len(LEAN)] = iav
        Xs.append(X)
        ys.append(b.column("label").to_numpy(zero_copy_only=False)[idx].astype(np.int8))
    del pf
    gc.collect()
    return np.concatenate(Xs), np.concatenate(ys)


def main() -> int:
    import mlflow

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("native-reranker-assocfix-s2")

    log("loading IA.tsv")
    ia = load_ia()
    log(f"IA terms: {len(ia)}")

    log("downloading train.parquet")
    train_raw = CLIENT.get_object(BUCKET, f"{DATASET}/train.parquet").read()
    log(f"train.parquet {len(train_raw)/1e6:.0f}MB")
    log("downloading eval.parquet")
    eval_raw = CLIENT.get_object(BUCKET, f"{DATASET}/eval.parquet").read()
    log(f"eval.parquet {len(eval_raw)/1e6:.0f}MB")

    summary: dict[str, object] = {"dataset_id": DATASET_ID, "recipe": "lean31+IA+S2_spw",
                                  "cats": {}}
    with mlflow.start_run(run_name="s2-assocfix-trio") as run:
        run_id = run.info.run_id
        mlflow.set_tags({
            "dataset_id": DATASET_ID,
            "dataset_name": "fullgo-union-SELECT-160-220-227-c99db18-assoc",
            "recipe": "S2 trio (lean-31 + IA + scale_pos_weight)",
            "champion_ref": "b21b187c=0.3745",
            "purpose": "INT-8 association-drift regression fix retrain",
        })
        mlflow.log_params({"num_leaves": 63, "lr": 0.05, "num_boost": NUM_BOOST,
                           "early_stop": EARLY, "min_data_in_leaf": 100, "seed": 42,
                           "n_features": len(NAMES)})
        log(f"MLflow run {run_id}")

        for cat in CATS:
            log(f"{cat}: load train")
            Xtr, ytr = load_split_for_cat(train_raw, cat, ia)
            log(f"{cat}: load eval")
            Xte, yte = load_split_for_cat(eval_raw, cat, ia)
            npos, nneg = int(ytr.sum()), int((ytr == 0).sum())
            spw = nneg / npos
            p = dict(PARAMS, scale_pos_weight=spw)
            log(f"{cat}: train {Xtr.shape} pos={npos} neg={nneg} spw={spw:.3f}")
            dtr = lgb.Dataset(Xtr, label=ytr, feature_name=NAMES, free_raw_data=False)
            dev = lgb.Dataset(Xte, label=yte, reference=dtr, feature_name=NAMES,
                              free_raw_data=False)

            evals: dict[str, float] = {}

            def _cb(env, _cat=cat, _evals=evals):
                for _, mname, val, _ in env.evaluation_result_list:
                    _evals[mname] = val
                    if env.iteration % 50 == 0:
                        mlflow.log_metric(f"{_cat}_{mname}", val, step=env.iteration)

            b = lgb.train(p, dtr, num_boost_round=NUM_BOOST, valid_sets=[dev],
                          callbacks=[lgb.early_stopping(EARLY), lgb.log_evaluation(200), _cb])
            b.save_model(f"{OUT}/ensemble_gbm_{cat.upper()}.txt")
            best_auc = float(b.best_score["valid_0"]["auc"])
            mlflow.log_metric(f"{cat}_best_auc", best_auc)
            mlflow.log_metric(f"{cat}_best_iter", b.best_iteration)
            imp = dict(sorted(zip(NAMES, b.feature_importance("gain").tolist()),
                              key=lambda x: -x[1]))
            summary["cats"][cat] = {"best_iter": b.best_iteration, "eval_auc": best_auc,
                                    "scale_pos_weight": spw, "n_pos": npos, "n_neg": nneg,
                                    "top5_importance": list(imp.items())[:5]}
            log(f"{cat}: SAVED best_iter={b.best_iteration} auc={best_auc:.4f} "
                f"top={list(imp.items())[:3]}")
            del Xtr, Xte, dtr, dev, b
            gc.collect()

        summary["mlflow_run_id"] = run_id
        with open(f"{OUT}/train_summary.json", "w") as w:
            json.dump(summary, w, indent=2, default=str)
        mlflow.log_artifact(f"{OUT}/train_summary.json")
        log(f"training DONE mlflow_run={run_id}")

    with open(f"{OUT}/mlflow_run_id.txt", "w") as w:
        w.write(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
