"""De-risk the descendant_prior PK scorer on the VALIDATION frame (220->227).
Train PK on [lean-31 + IA + descendant_prior] (= "both" + descprior), score on the
eval split with the de-risk cafaeval recipe (no-TOI, MAX-collapse), compare PK to the
"both" PK validation baseline 0.3720. If it helps, do the TEST cycle. Optimistic
validation numbers; only the RELATIVE delta vs 0.3720 is meaningful."""
import gc
import io
import os
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from cafaeval.evaluation import cafa_eval
from minio import Minio

BASE = "datasets/fullgo-union-SELECT-160-220-227-v5"
BUCKET = "protea"
D = "/home/frapercan/Thesis2/storage/fullgo_models/selfprior_ia_experiment"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_TSV = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
CLIENT = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
CAT = "pk"
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
NAMES = LEAN + ["IA", "descendant_prior"]
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.9,
              bagging_freq=1, verbosity=-1, num_threads=8, seed=42)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(split):
    raw = CLIENT.get_object(BUCKET, f"{BASE}/{split}.parquet").read()
    pf = pq.ParquetFile(io.BytesIO(raw))
    ov = np.load(f"{D}/{split}_overlay.npz")
    dp_all = np.load(f"{D}/{split}_descprior.npz")["descendant_prior"]
    spf_all, ia_all = ov["self_prior_fixed"], ov["IA"]
    cols = LEAN + ["category", "label", "protein_accession", "go_term_id"]
    Xs, ys, spfs, ias, dps, prots, gids = [], [], [], [], [], [], []
    off = 0
    for b in pf.iter_batches(batch_size=2_000_000, columns=cols):
        cat = np.asarray(b.column("category").to_numpy(zero_copy_only=False), dtype=object)
        bn = len(cat)
        m = cat == CAT
        if m.any():
            idx = np.nonzero(m)[0]
            X = np.empty((idx.size, len(LEAN)), dtype=np.float32)
            for j, f in enumerate(LEAN):
                X[:, j] = b.column(f).to_numpy(zero_copy_only=False)[idx].astype(np.float32)
            X[:, SP_IDX] = spf_all[off:off + bn][idx]
            Xs.append(np.hstack([X, ia_all[off:off + bn][idx].reshape(-1, 1).astype(np.float32),
                                 dp_all[off:off + bn][idx].reshape(-1, 1).astype(np.float32)]))
            ys.append(b.column("label").to_numpy(zero_copy_only=False)[idx].astype(np.int8))
            prots.append(np.asarray(b.column("protein_accession").to_pylist(), dtype=object)[idx])
            gids.append(np.asarray(b.column("go_term_id").to_pylist(), dtype=object)[idx])
        off += bn
    del raw, pf, ov
    gc.collect()
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(prots), np.concatenate(gids))


log("load train")
Xtr, ytr, _, _ = load("train")
log(f"train {Xtr.shape} pos={int(ytr.sum())} descprior_nz={int((Xtr[:,-1]>0).sum())}")
log("load eval")
Xte, yte, prot_te, gid_te = load("eval")
dtr = lgb.Dataset(Xtr, label=ytr, feature_name=NAMES, free_raw_data=False)
dev = lgb.Dataset(Xte, label=yte, reference=dtr, feature_name=NAMES, free_raw_data=False)
b = lgb.train(PARAMS, dtr, num_boost_round=1500, valid_sets=[dev],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(300)])
imp = dict(sorted(zip(NAMES, b.feature_importance("gain").tolist()), key=lambda x: -x[1]))
log(f"best_iter={b.best_iteration} descprior_imp_rank={list(imp).index('descendant_prior')+1} "
    f"descprior_gain={imp['descendant_prior']:.0f} top5={list(imp.items())[:5]}")
pred = b.predict(Xte)

# MAX-collapse per (protein, go_id), GT = label==1, cafaeval (de-risk recipe)
best = {}
gt = set()
for p, g, s, y in zip(prot_te, gid_te, pred.tolist(), yte.tolist()):
    k = (p, g)
    if k not in best or s > best[k]:
        best[k] = s
    if y == 1:
        gt.add(k)
with tempfile.TemporaryDirectory() as td:
    pd_ = Path(td) / "pred"
    pd_.mkdir()
    with open(pd_ / "m.tsv", "w") as w:
        for (p, g), s in best.items():
            w.write(f"{p}\t{g}\t{s:.6f}\n")
    gtf = Path(td) / "gt.tsv"
    with open(gtf, "w") as w:
        for (p, g) in sorted(gt):
            w.write(f"{p}\t{g}\n")
    df, _ = cafa_eval(OBO, str(pd_), str(gtf), ia=IA_TSV, prop="fill", norm="cafa",
                      no_orphans=True, max_terms=None, th_step=0.01, n_cpu=1)
    sub = df.reset_index()
    col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
    pk = float(sub.groupby("ns")[col].max().mean())
log(f"PK f_micro_w (validation, both+descprior) = {pk:.4f}   vs both 0.3720   delta {pk-0.3720:+.4f}")
log("DONE descprior de-risk")
