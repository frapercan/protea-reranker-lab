"""S2 lever: retrain the PK booster on the lean-31 + IA recipe (= the "both"
recipe that gave 0.3722) BUT with scale_pos_weight = #neg/#pos to upweight the
rare PK positives. Validation-measured +0.0020 PK; this seals it on TEST.
NK/LK are kept from native_boosters_both_serve; only PK is swapped.
Feature names match the "both" PK booster exactly so it is a drop-in swap."""
import gc
import io
import os
import time

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from minio import Minio

BASE = "datasets/fullgo-union-SELECT-160-220-227-v5"
BUCKET = "protea"
D = "/home/frapercan/Thesis2/storage/fullgo_models/selfprior_ia_experiment"
OUT = "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_pk_s2"
os.makedirs(OUT, exist_ok=True)
CLIENT = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)

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
SP_IDX = LEAN.index("self_prior_score")
NAMES = LEAN + ["IA"]  # 31, self_prior_score column carries the fixed values

# "both" config (num_leaves=63) + scale_pos_weight for PK imbalance.
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.9,
              bagging_freq=1, verbosity=-1, num_threads=8, seed=42)
NUM_BOOST = 1500
EARLY = 50
CAT = "pk"


def free_g():
    with open("/proc/meminfo") as fh:
        for ln in fh:
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) / 1024 / 1024
    return 99.0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] free={free_g():.1f}G  {m}", flush=True)


def load_split_for_cat(split, cat):
    key = f"{BASE}/{split}.parquet"
    raw = CLIENT.get_object(BUCKET, key).read()
    pf = pq.ParquetFile(io.BytesIO(raw))
    ov = np.load(f"{D}/{split}_overlay.npz")
    spf_all, ia_all = ov["self_prior_fixed"], ov["IA"]
    cols = LEAN + ["category", "label"]
    Xs, ys, spfs, ias = [], [], [], []
    off = 0
    for b in pf.iter_batches(batch_size=2_000_000, columns=cols):
        cat_arr = np.asarray(b.column("category").to_numpy(zero_copy_only=False), dtype=object)
        bn = len(cat_arr)
        m = cat_arr == cat
        if m.any():
            idx = np.nonzero(m)[0]
            X = np.empty((idx.size, len(LEAN)), dtype=np.float32)
            for j, f in enumerate(LEAN):
                X[:, j] = b.column(f).to_numpy(zero_copy_only=False)[idx].astype(np.float32)
            Xs.append(X)
            ys.append(b.column("label").to_numpy(zero_copy_only=False)[idx].astype(np.int8))
            spfs.append(spf_all[off:off + bn][idx])
            ias.append(ia_all[off:off + bn][idx])
        off += bn
    del raw, pf, ov
    gc.collect()
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(spfs), np.concatenate(ias))


def build(X, spf, ia):
    X = X.copy()
    X[:, SP_IDX] = spf  # self_prior fix
    return np.hstack([X, ia.reshape(-1, 1).astype(np.float32)])


log(f"{CAT}: loading train")
Xtr, ytr, spftr, iatr = load_split_for_cat("train", CAT)
log(f"{CAT}: loading eval (dev)")
Xte, yte, spfte, iate = load_split_for_cat("eval", CAT)
Xtr2, Xte2 = build(Xtr, spftr, iatr), build(Xte, spfte, iate)
del Xtr, Xte
gc.collect()
npos, nneg = int(ytr.sum()), int((ytr == 0).sum())
spw = nneg / npos
PARAMS["scale_pos_weight"] = spw
log(f"{CAT}: train {Xtr2.shape} pos={npos} neg={nneg} scale_pos_weight={spw:.3f}")
dtr = lgb.Dataset(Xtr2, label=ytr, feature_name=NAMES, free_raw_data=False)
dev = lgb.Dataset(Xte2, label=yte, reference=dtr, feature_name=NAMES, free_raw_data=False)
b = lgb.train(PARAMS, dtr, num_boost_round=NUM_BOOST, valid_sets=[dev],
              callbacks=[lgb.early_stopping(EARLY), lgb.log_evaluation(200)])
b.save_model(f"{OUT}/ensemble_gbm_PK.txt")
log(f"{CAT}: SAVED best_iter={b.best_iteration} eval_auc={b.best_score['valid_0']['auc']:.4f}")
log("pk_s2 training DONE")
