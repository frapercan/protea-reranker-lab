"""A1 attribution ablation (lafa-levers PLAN, Phase A).

Why is the native baseline 0.315 < KNN 0.324, and what subtracts on LK? Stream
the parity export parquet ONCE, hold per-category float32 matrices with ALL
features (numeric + the 4 categoricals factorized, vocabulary persisted for B1),
subsample pk for the screen, and train a booster per cumulative feature subset.
Report held-out (227 eval split) AUC per category per subset. The marginal
deltas attribute the regression.

Leakage-clean: trains on the 220->227 SELECT frame, evaluates on the 227 split,
never touches the 7401 TEST frame.
"""

from __future__ import annotations

import gc
import io
import hashlib
import json
import sys

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from minio import Minio

sys.path.insert(0, "/home/frapercan/Thesis2/repositories/PROTEA")
from protea_contracts.feature_schema import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_FAMILIES,
)

BUCKET = "protea"
BASE = "datasets/fullgo-native-parity-SELECT-220-227"
CATS = ("nk", "lk", "pk")
PK_KEEP = 0.10  # subsample pk to ~5M for the screen
OUT = "/home/frapercan/Thesis2/storage/fullgo_models/ablation_a1.json"
PARAMS = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
          "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9,
          "bagging_fraction": 0.9, "bagging_freq": 1, "verbosity": -1}
_C = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)

# Cumulative feature subsets (families). knn -> +base -> +self_prior -> +assoc
# -> +classifier (=current numeric) -> +categoricals (full).
KNN_FAMS = ["knn", "knn_distance", "knn_vote", "go_context"]
BASE_FAMS = KNN_FAMS + ["alignment_nw", "alignment_sw", "length", "taxonomy_pair",
                        "taxonomy_voters", "anc2vec_neighbor", "anc2vec_query", "emb_pca"]


def _fams_to_cols(fams, present):
    cols, seen = [], set()
    for f in fams:
        for c in FEATURE_FAMILIES.get(f, []):
            if c in present and c not in seen and c not in CATEGORICAL_FEATURES:
                seen.add(c)
                cols.append(c)
    return cols


def _accumulate(key, feat, codemaps, subsample_pk):
    raw = _C.get_object(BUCKET, key).read()
    pf = pq.ParquetFile(io.BytesIO(raw))
    cat_set = set(CATEGORICAL_FEATURES)
    parts = {c: [] for c in CATS}
    ys = {c: [] for c in CATS}
    read_cols = feat + ["label", "category"]
    if subsample_pk:
        read_cols = read_cols + ["protein_accession"]
    for batch in pf.iter_batches(batch_size=500_000, columns=read_cols):
        d = batch.to_pydict()
        cat = np.asarray([str(x) for x in d["category"]])
        lab = np.asarray(d["label"], dtype=np.int8)
        xb = np.empty((len(cat), len(feat)), dtype=np.float32)
        for j, f in enumerate(feat):
            col = d[f]
            if f in cat_set:
                cm = codemaps.setdefault(f, {})
                xb[:, j] = np.asarray(
                    [-1 if (v is None or str(v) == "") else cm.setdefault(str(v), len(cm)) for v in col],
                    dtype=np.float32)
            else:
                try:
                    xb[:, j] = np.asarray(col, dtype=np.float32)
                except (TypeError, ValueError):
                    xb[:, j] = np.asarray([float(v) if v not in (None, "") else np.nan for v in col], dtype=np.float32)
        for c in CATS:
            m = cat == c
            if c == "pk" and subsample_pk:
                keep = np.array([int(hashlib.md5(str(a).encode()).hexdigest(), 16) % 100 < int(PK_KEEP * 100)
                                 for a in d["protein_accession"]])
                m = m & keep
            if m.any():
                parts[c].append(xb[m])
                ys[c].append(lab[m])
        del d, xb
        gc.collect()
    out = {}
    for c in CATS:
        if parts[c]:
            out[c] = (np.vstack(parts[c]), np.concatenate(ys[c]))
        parts[c] = []
    del raw, pf
    gc.collect()
    return out


def main() -> int:
    present = set()
    schema = pq.read_schema(io.BytesIO(_C.get_object(BUCKET, f"{BASE}/train.parquet").read())).names
    present = set(schema)
    numeric = [f for f in ALL_FEATURES if f in present and f not in set(CATEGORICAL_FEATURES)]
    cats_present = [f for f in CATEGORICAL_FEATURES if f in present]
    feat = numeric + cats_present  # categoricals last
    cidx = {f: i for i, f in enumerate(feat)}
    print(f"numeric={len(numeric)} categorical={cats_present}", flush=True)

    subsets = {
        "knn": _fams_to_cols(KNN_FAMS, present),
        "base": _fams_to_cols(BASE_FAMS, present),
        "base+selfprior": _fams_to_cols(BASE_FAMS + ["self_prior"], present),
        "base+sp+assoc": _fams_to_cols(BASE_FAMS + ["self_prior", "association"], present),
        "full_numeric": numeric,
        "full+categoricals": numeric + cats_present,
    }
    for k, v in subsets.items():
        print(f"  subset {k}: {len(v)} cols", flush=True)

    codemaps: dict = {}
    print("streaming train (pk subsampled)...", flush=True)
    tr = _accumulate(f"{BASE}/train.parquet", feat, codemaps, subsample_pk=True)
    print("streaming eval...", flush=True)
    ev = _accumulate(f"{BASE}/eval.parquet", feat, codemaps, subsample_pk=False)

    results = {"subsets": {k: len(v) for k, v in subsets.items()}, "auc": {}}
    for cat in CATS:
        if cat not in tr or cat not in ev:
            continue
        Xtr, ytr = tr[cat]
        Xev, yev = ev[cat]
        results["auc"][cat] = {}
        for sname, scols in subsets.items():
            idx = [cidx[c] for c in scols if c in cidx]
            catidx = [k for k, c in enumerate(scols) if c in CATEGORICAL_FEATURES]
            dtr = lgb.Dataset(Xtr[:, idx], label=ytr, categorical_feature=catidx or "auto", free_raw_data=False)
            dev = lgb.Dataset(Xev[:, idx], label=yev, reference=dtr, free_raw_data=False)
            b = lgb.train(PARAMS, dtr, num_boost_round=1500, valid_sets=[dev], valid_names=["eval"],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            auc = b.best_score["eval"]["auc"]
            results["auc"][cat][sname] = round(float(auc), 4)
            print(f"  {cat} {sname}: AUC={auc:.4f} (iter {b.best_iteration})", flush=True)
            del dtr, dev, b
            gc.collect()
        del Xtr, Xev
        tr[cat] = None
        gc.collect()

    results["categorical_vocab"] = {f: sorted(codemaps.get(f, {}), key=lambda k: codemaps[f][k]) for f in cats_present}
    with open(OUT, "w") as w:
        json.dump(results, w, indent=2)
    print("\n=== A1 ablation (held-out 227 AUC) ===", flush=True)
    for cat in results["auc"]:
        print(f" {cat}: " + "  ".join(f"{k}={v}" for k, v in results["auc"][cat].items()), flush=True)
    print("DONE ->", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
