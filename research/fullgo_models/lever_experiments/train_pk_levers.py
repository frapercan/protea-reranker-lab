"""PK-only SCORE-LEVER trainings for the native reranker, OFFLINE.

Loads the PK rows of dataset ``fullgo-union-SELECT-160-220-227-v5`` (train + eval)
ONCE into memory (streamed row-group batches, float32, 69 numeric features in the
v5 summary.json order), then trains ONE lever booster per invocation so only a
single LightGBM run is ever in flight (RAM-conscious box).

Baseline PK booster (comparison point, validation 220->227, no-TOI/no-PK-known
recipe): f_micro_w PK = 0.4142, config = binary/auc, lr=0.05, num_leaves=63,
min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
early-stopped on eval, best_iter=769.

Levers (one per --lever):
  s2   : class-imbalance via scale_pos_weight=(#neg/#pos) [+ optional is_unbalance]
  s13  : boosting_type='dart', drop_rate=0.1, fixed num_boost_round (DART ignores
         early stopping), budget ~= baseline best_iter 769.

Saves: ensemble_gbm_PK.txt under the lever out-dir + a small train_meta.json.
Does NOT score; scoring is done by apply_score_pk.py (same recipe as baseline).
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from minio import Minio

HERE = "/home/frapercan/Thesis2/storage/fullgo_models"
V5 = f"{HERE}/native_boosters_v5"
BUCKET = "protea"
BASE = "datasets/fullgo-union-SELECT-160-220-227-v5"

with open(f"{V5}/summary.json") as fh:
    FEATURES = json.load(fh)["features"]
assert len(FEATURES) == 69, len(FEATURES)

CLIENT = Minio("localhost:9000", access_key="minioadmin",
               secret_key="minioadmin", secure=False)

BASE_PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05,
    "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9,
    "bagging_fraction": 0.9, "bagging_freq": 1, "verbosity": -1,
}


def _to_float(col) -> np.ndarray:
    a = np.asarray(col, dtype=object)
    out = np.empty(len(a), dtype=np.float32)
    for i, v in enumerate(a):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def load_pk(key: str):
    """Stream a parquet, keep only PK rows -> (X float32, y int8)."""
    raw = CLIENT.get_object(BUCKET, key).read()
    pf = pq.ParquetFile(io.BytesIO(raw))
    cols = FEATURES + ["label", "category"]
    parts, ys = [], []
    n = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        d = batch.to_pydict()
        cat = np.asarray([str(x) for x in d["category"]])
        lab = np.asarray(d["label"], dtype=np.int8)
        m = cat == "pk"
        if m.any():
            xb = np.empty((int(m.sum()), len(FEATURES)), dtype=np.float32)
            idx = np.nonzero(m)[0]
            for j, f in enumerate(FEATURES):
                col = d[f]
                try:
                    full = np.asarray(col, dtype=np.float32)
                except (TypeError, ValueError):
                    full = _to_float(col)
                xb[:, j] = full[idx]
            parts.append(xb)
            ys.append(lab[m])
        n += len(cat)
        del d
        gc.collect()
        print(f"    ...{n} rows streamed ({key.split('/')[-1]})", flush=True)
    del raw, pf
    gc.collect()
    X = np.vstack(parts)
    y = np.concatenate(ys)
    del parts, ys
    gc.collect()
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", required=True, choices=["s2", "s2_unbal", "s13"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("loading PK train...", flush=True)
    Xtr, ytr = load_pk(f"{BASE}/train.parquet")
    print("loading PK eval...", flush=True)
    Xev, yev = load_pk(f"{BASE}/eval.parquet")
    npos, nneg = int(ytr.sum()), int(len(ytr) - ytr.sum())
    spw = nneg / npos
    print(f"PK train rows={len(ytr)} pos={npos} neg={nneg} spw={spw:.3f}", flush=True)
    print(f"PK eval  rows={len(yev)} pos={int(yev.sum())}", flush=True)

    params = dict(BASE_PARAMS)
    meta = {"lever": args.lever, "train_rows": len(ytr), "pos": npos,
            "neg": nneg, "scale_pos_weight_value": spw}

    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURES, free_raw_data=False)
    dev = lgb.Dataset(Xev, label=yev, feature_name=FEATURES, reference=dtr,
                      free_raw_data=False)

    if args.lever == "s2":
        params["scale_pos_weight"] = spw
        nrounds, callbacks, valid = 5000, [
            lgb.log_evaluation(period=50),
            lgb.early_stopping(50, verbose=True),
        ], [dev]
    elif args.lever == "s2_unbal":
        params["is_unbalance"] = True
        nrounds, callbacks, valid = 5000, [
            lgb.log_evaluation(period=50),
            lgb.early_stopping(50, verbose=True),
        ], [dev]
    else:  # s13 DART
        params["boosting_type"] = "dart"
        params["drop_rate"] = 0.1
        # DART ignores early stopping; fix budget ~= baseline PK best_iter.
        nrounds = 769
        callbacks, valid = [lgb.log_evaluation(period=50)], [dev]

    meta["params"] = {k: v for k, v in params.items()}
    meta["num_boost_round"] = nrounds

    booster = lgb.train(params, dtr, num_boost_round=nrounds,
                        valid_sets=valid, valid_names=["eval"],
                        callbacks=callbacks)
    best_iter = booster.best_iteration or nrounds
    path = f"{args.out}/ensemble_gbm_PK.txt"
    booster.save_model(path)  # DART: saves all trees (no best_iteration truncation)
    best_auc = None
    try:
        best_auc = booster.best_score.get("eval", {}).get("auc")
    except Exception:
        pass
    meta["best_iteration"] = int(best_iter)
    meta["eval_auc"] = best_auc
    with open(f"{args.out}/train_meta.json", "w") as w:
        json.dump(meta, w, indent=2)
    print(f"DONE lever={args.lever} best_iter={best_iter} eval_auc={best_auc} -> {path}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
