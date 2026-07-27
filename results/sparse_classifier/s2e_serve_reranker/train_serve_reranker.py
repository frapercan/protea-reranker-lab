"""S2e: per-category reranker on the SERVE-PRODUCIBLE feature set.

Champion recipe (percut_rerank/train_rerank.py) restricted to the features
the deployed predict path actually computes end-to-end (post PR #693 lineage):
the interpro feature family is DROPPED because the serve predict pipeline has
no interpro producer (it enters PROTEA only as a separate prediction source /
EBI InterPro2GO graft, never as a reranker feature; and it is all-zero in the
parquet anyway).

Composite champion trio (the SUMMARY net recommendation):
  nk, lk -> baseline reranker (M2 anc2vec candidate pool)
  pk     -> percut reranker (per-cut sparse-classifier / d8979601 pool)
Each trained per-category, early-stop on v225-v227, pminmax lever on LK-bpo,
PK pool = knn_present only. aspect is a feature (aspect_code); models are not
split per aspect.

Models carry feature_schema_sha = 851849df48e2 (governed families
align+tax+v6+lineage). The blob families alignment_sw / classifier /
self_prior / association ride OUTSIDE the sha (D45 seam) and must be switched
on value-wise at serve via their compute_* flags.
"""
import os, json, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
PC = os.path.join(HERE, "..", "percut_rerank")
BASELINE = os.path.join(PC, "baseline")
PRED_DIR = os.path.join(HERE, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

# Per-category data source (composite champion).
SRC = {
    "nk": {"train": os.path.join(BASELINE, "train.parquet"), "eval": os.path.join(BASELINE, "eval.parquet")},
    "lk": {"train": os.path.join(BASELINE, "train.parquet"), "eval": os.path.join(BASELINE, "eval.parquet")},
    "pk": {"train": os.path.join(PC, "train.parquet"), "eval": os.path.join(PC, "eval.parquet")},
}

VALID_PAIR = "v225-v227"
ASPECTS = ["mfo", "bpo", "cco"]
ASPECT_CODE = {"mfo": 0, "bpo": 1, "cco": 2}
SEED = 42
FEATURE_SCHEMA_SHA = "851849df48e2"  # align+tax+v6+lineage governed families

META = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
        "qualifier", "evidence_code", "taxonomic_relation", "aspect"}

# interpro family: NOT serve-producible as reranker features (no predict-path
# producer; all-zero in parquet). Dropping it is the serve/train alignment.
INTERPRO_FAMILY = [
    "interpro_hit", "interpro_score", "interpro_n_signatures",
    "interpro_db_pfam", "interpro_db_panther", "interpro_db_superfamily",
    "interpro_db_smart", "interpro_db_cdd", "interpro_db_prosite",
    "interpro_present",
]

_schema = pq.ParquetFile(SRC["pk"]["train"]).schema_arrow
ALL_COLS = list(_schema.names)
BOOL_COLS = [n for n in ALL_COLS if str(_schema.field(n).type) == "bool"]
ALL_FEATURES_69 = [c for c in ALL_COLS if c not in META]
# serve-producible reranker features = champion 69 minus interpro family
SERVE_FEATURES = [c for c in ALL_FEATURES_69 if c not in INTERPRO_FAMILY]
FEATURES = SERVE_FEATURES + ["aspect_code"]
READ_COLS = list(META) + [c for c in FEATURES if c != "aspect_code"]
print(f"[cols] champion_69={len(ALL_FEATURES_69)} serve_producible={len(SERVE_FEATURES)} "
      f"(+aspect_code -> {len(FEATURES)}) dropped_interpro={len(INTERPRO_FAMILY)}")


def load_category(parquet_path, category, pk_knn_only):
    filt = [("category", "=", category)]
    if category == "pk" and pk_knn_only:
        filt.append(("knn_present", "=", True))
    t = pq.read_table(parquet_path, columns=READ_COLS, filters=filt)
    df = t.to_pandas()
    for b in BOOL_COLS:
        if b in df.columns:
            df[b] = df[b].astype("int8")
    df["aspect_code"] = df["aspect"].map(ASPECT_CODE).astype("int8")
    for c in FEATURES:
        if c in df.columns and df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


def train_one(category):
    t0 = time.time()
    df = load_category(SRC[category]["train"], category, pk_knn_only=True)
    tr = df[df["snapshot_pair"] != VALID_PAIR]
    va = df[df["snapshot_pair"] == VALID_PAIR]
    print(f"[{category}] train rows={len(tr)} pos={int(tr.label.sum())} | "
          f"valid rows={len(va)} pos={int(va.label.sum())} (load {time.time()-t0:.1f}s)")
    Xtr, ytr = tr[FEATURES], tr["label"].astype("int8").values
    Xva, yva = va[FEATURES], va["label"].astype("int8").values
    dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr, free_raw_data=False)
    params = dict(objective="binary", metric=["auc", "average_precision"],
                  learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  max_depth=-1, seed=SEED, verbosity=-1, num_threads=12)
    booster = lgb.train(params, dtr, num_boost_round=3000,
                        valid_sets=[dtr, dva], valid_names=["train", "valid"],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)])
    best_it = booster.best_iteration
    print(f"[{category}] best_iter={best_it} valid_auc={booster.best_score['valid']['auc']:.5f} "
          f"valid_ap={booster.best_score['valid']['average_precision']:.5f}")
    return booster, best_it


def apply_pminmax_lk_bpo(df):
    mask = (df["aspect"] == "bpo")
    if not mask.any():
        return df
    sub = df.loc[mask, ["protein_accession", "score"]].copy()
    g = sub.groupby("protein_accession")["score"]
    mn = g.transform("min")
    mx = g.transform("max")
    rng = (mx - mn)
    norm = np.where(rng > 0, (sub["score"] - mn) / rng, 1.0)
    df.loc[mask, "score"] = norm
    return df


def main():
    boosters = {}
    for cat in ["nk", "lk", "pk"]:
        booster, best_it = train_one(cat)
        boosters[cat] = (booster, best_it)
        booster.save_model(os.path.join(PRED_DIR, f"model_{cat}.txt"), num_iteration=best_it)
    # TEST predictions
    for cat in ["nk", "lk", "pk"]:
        booster, best_it = boosters[cat]
        te = load_category(SRC[cat]["eval"], cat, pk_knn_only=True)
        te["score"] = booster.predict(te[FEATURES], num_iteration=best_it)
        out = te[["protein_accession", "go_term_id", "aspect", "score"]].copy()
        if cat == "lk":
            out = apply_pminmax_lk_bpo(out)
        d = os.path.join(PRED_DIR, cat)
        os.makedirs(d, exist_ok=True)
        out[["protein_accession", "go_term_id", "score"]].to_csv(
            os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False,
            float_format="%.6f")
        print(f"[{cat}] TEST predictions written: {len(out)} rows")
    # feature list for the manifest
    with open(os.path.join(HERE, "_features_used.json"), "w") as f:
        json.dump({"feature_schema_sha": FEATURE_SCHEMA_SHA,
                   "features": FEATURES, "n": len(FEATURES),
                   "dropped_interpro": INTERPRO_FAMILY}, f, indent=2)
    print("[done] models + predictions written")


if __name__ == "__main__":
    main()
