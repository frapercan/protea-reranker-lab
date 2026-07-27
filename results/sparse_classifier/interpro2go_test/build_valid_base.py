"""Build validation (v225-v227) base-reranker scores + truth sets per category.

NK/LK base = baseline M2 reranker (baseline/train.parquet + baseline model).
PK base = per-cut sparse reranker (percut train.parquet + percut model, knn_present).
Applies the LK-bpo pminmax lever exactly as the graft does.

Writes valid_base.parquet: protein_accession, go_term_id, aspect, category,
base_score, label  (label = raw pool label; truth built separately by propagation).
"""
import os, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
PERC = os.path.join(SC, "percut_rerank")
BASE = os.path.join(PERC, "baseline")
HERE = os.path.dirname(os.path.abspath(__file__))
VALID_PAIR = "v225-v227"
ASPECT_CODE = {"mfo": 0, "bpo": 1, "cco": 2}

SRC = {  # category -> (train_parquet, model_path, pk_knn_only)
    "nk": (os.path.join(BASE, "train.parquet"), os.path.join(BASE, "predictions/model_nk.txt"), False),
    "lk": (os.path.join(BASE, "train.parquet"), os.path.join(BASE, "predictions/model_lk.txt"), False),
    "pk": (os.path.join(PERC, "train.parquet"), os.path.join(PERC, "predictions/model_pk.txt"), True),
}


def feature_cols(parquet_path):
    sch = pq.ParquetFile(parquet_path).schema_arrow
    allc = list(sch.names)
    meta = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
            "qualifier", "evidence_code", "taxonomic_relation", "aspect"}
    boolc = [n for n in allc if str(sch.field(n).type) == "bool"]
    feats = [c for c in allc if c not in meta] + ["aspect_code"]
    readc = list(meta) + [c for c in feats if c != "aspect_code"]
    return feats, boolc, readc


def apply_pminmax_lk_bpo(df):
    mask = (df["aspect"] == "bpo")
    if not mask.any():
        return df
    sub = df.loc[mask, ["protein_accession", "base_score"]].copy()
    g = sub.groupby("protein_accession")["base_score"]
    mn, mx = g.transform("min"), g.transform("max")
    rng = mx - mn
    df.loc[mask, "base_score"] = np.where(rng > 0, (sub["base_score"] - mn) / rng, 1.0)
    return df


parts = []
for cat, (tp, mp, knn) in SRC.items():
    feats, boolc, readc = feature_cols(tp)
    filt = [("snapshot_pair", "=", VALID_PAIR), ("category", "=", cat)]
    if knn:
        filt.append(("knn_present", "=", True))
    df = pq.read_table(tp, columns=readc, filters=filt).to_pandas()
    for b in boolc:
        if b in df.columns:
            df[b] = df[b].astype("int8")
    df["aspect_code"] = df["aspect"].map(ASPECT_CODE).astype("int8")
    for c in feats:
        if c in df.columns and df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    booster = lgb.Booster(model_file=mp)
    df["base_score"] = booster.predict(df[feats])
    if cat == "lk":
        df = apply_pminmax_lk_bpo(df)
    out = df[["protein_accession", "go_term_id", "aspect", "label"]].copy()
    out["category"] = cat
    out["base_score"] = df["base_score"].astype("float32")
    parts.append(out)
    print(f"[{cat}] valid rows {len(out)} pos {int(out.label.sum())} prots {out.protein_accession.nunique()}")

allv = pd.concat(parts, ignore_index=True)
allv.to_parquet(os.path.join(HERE, "valid_base.parquet"))
print("written valid_base.parquet", len(allv))
