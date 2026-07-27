"""LEAN vs LEAN+IA+sp_present de-risk trainer (offline, lab).

Treatment-1 tests the CHAMPION'S GENUINELY-MISSING features against the native
lean 30-feature set. KEY FINDING (verified against ensemble_seal.py): the offline
champion's `_p` suffix means PRESENCE flag (1.0 if t in stream), NOT DAG-propagation.
The native lean set already carries knn_present / classifier_present /
association_present. The only champion features the native lean set LACKS are:
    - IA   = information accretion weight IA[go_term_id]   (per-candidate)
    - lfreq vs go_term_frequency (native has a frequency feature already)
    - sp_present = (self_prior_score > 0)  -- the one missing presence flag

So treatment-1 = LEAN(30) + IA + sp_present  (pure per-row lookups, memory-trivial).
A separate script tests literal DAG-propagated clf_p/sp_p/knn_p.

Trains one per-category LightGBM at a time (PK is 67.5M rows). Same params as the
lean baseline. Reports eval AUC + gain importances. Memory-guarded: aborts if
free RAM < 5G before a category build.
"""
from __future__ import annotations

import gc
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

D = "/home/frapercan/Thesis2/storage/fullgo_models/lean_ia_p_experiment"
DATA = f"{D}/data"
IA_TSV = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
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
# derived treatment columns appended after the parquet LEAN columns:
#   IA          (lookup on go_term_id)
#   sp_present  (self_prior_score > 0)
DERIVED = ["IA", "sp_present"]

PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05,
    "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9,
    "bagging_fraction": 0.9, "bagging_freq": 1, "verbosity": -1, "num_threads": 8,
    "seed": 42,
}
NUM_BOOST = 1500
EARLY = 50


def load_ia():
    ia = {}
    for line in open(IA_TSV):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            try:
                ia[p[0]] = float(p[1])
            except ValueError:
                pass
    return ia


def free_g():
    with open("/proc/meminfo") as fh:
        for ln in fh:
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) / 1024 / 1024
    return 99.0


def build_cat(path, cat, feats, ia, want_label):
    """Stream one parquet, return (X float32, y int8) for a single category."""
    pf = pq.ParquetFile(path)
    read_cols = LEAN + ["go_term_id", "self_prior_score", "category", "label"]
    read_cols = list(dict.fromkeys(read_cols))
    xs, ys = [], []
    for b in pf.iter_batches(batch_size=1_000_000, columns=read_cols):
        d = b.to_pydict()
        c = np.asarray([str(x) for x in d["category"]])
        m = c == cat
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        n = idx.size
        X = np.empty((n, len(feats)), dtype=np.float32)
        for j, f in enumerate(feats):
            if f == "IA":
                terms = [d["go_term_id"][k] for k in idx]
                X[:, j] = np.asarray([ia.get(t, 0.0) for t in terms], dtype=np.float32)
            elif f == "sp_present":
                sp = np.asarray(d["self_prior_score"], dtype=np.float32)[idx]
                X[:, j] = (sp > 0).astype(np.float32)
            else:
                arr = np.asarray(d[f])
                if arr.dtype == object or arr.dtype == bool:
                    X[:, j] = np.asarray(
                        [float(v) if v is not None else np.nan for v in arr[idx]],
                        dtype=np.float32)
                else:
                    X[:, j] = arr[idx].astype(np.float32)
        xs.append(X)
        if want_label:
            ys.append(np.asarray(d["label"], dtype=np.int8)[idx])
        del d
    X = np.concatenate(xs); del xs; gc.collect()
    y = np.concatenate(ys) if want_label else None
    return X, y


def main():
    variant = sys.argv[1]  # "lean" or "lean_ia"
    feats = LEAN if variant == "lean" else (LEAN + DERIVED)
    outdir = f"{D}/{variant}"
    os.makedirs(outdir, exist_ok=True)
    ia = load_ia()
    print(f"variant={variant} n_features={len(feats)} IA_terms={len(ia)}", flush=True)
    summary = {"variant": variant, "features": feats, "boosters": {}}
    for cat in CATS:
        if free_g() < 5.0:
            print(f"ABORT: free RAM {free_g():.1f}G < 5G before {cat}", flush=True)
            break
        print(f"[{cat}] building train... free={free_g():.1f}G", flush=True)
        Xtr, ytr = build_cat(f"{DATA}/train.parquet", cat, feats, ia, True)
        print(f"[{cat}] train {Xtr.shape} pos={int(ytr.sum())} free={free_g():.1f}G", flush=True)
        Xev, yev = build_cat(f"{DATA}/eval.parquet", cat, feats, ia, True)
        print(f"[{cat}] eval  {Xev.shape} pos={int(yev.sum())} free={free_g():.1f}G", flush=True)
        dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=True)
        dev = lgb.Dataset(Xev, label=yev, reference=dtr, free_raw_data=True)
        evals = {}
        bst = lgb.train(
            PARAMS, dtr, num_boost_round=NUM_BOOST, valid_sets=[dev],
            valid_names=["eval"],
            callbacks=[lgb.early_stopping(EARLY, verbose=False),
                       lgb.record_evaluation(evals),
                       lgb.log_evaluation(period=100)],
        )
        auc = float(bst.best_score["eval"]["auc"])
        bst.save_model(f"{outdir}/ensemble_gbm_{cat.upper()}.txt")
        gains = bst.feature_importance(importance_type="gain")
        imp = sorted(zip(feats, gains.tolist()), key=lambda x: -x[1])
        summary["boosters"][cat] = {
            "train_rows": int(Xtr.shape[0]), "pos": int(ytr.sum()),
            "best_iter": int(bst.best_iteration), "eval_auc": auc,
            "top_importance_gain": [[f, round(g, 1)] for f, g in imp[:12]],
            "derived_importance": {f: round(float(dict(imp).get(f, 0.0)), 1)
                                   for f in DERIVED if f in feats},
        }
        print(f"[{cat}] AUC={auc:.4f} best_iter={bst.best_iteration}", flush=True)
        for f in DERIVED:
            if f in feats:
                print(f"    {f} gain={dict(imp)[f]:.1f}  rank={[x[0] for x in imp].index(f)+1}/{len(feats)}", flush=True)
        del Xtr, ytr, Xev, yev, dtr, dev, bst
        gc.collect()
    with open(f"{outdir}/summary.json", "w") as w:
        json.dump(summary, w, indent=2)
    print(f"WROTE {outdir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
