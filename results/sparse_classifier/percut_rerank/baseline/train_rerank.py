"""Per-category LightGBM reranker over the frozen per-cut sparse-classifier dataset.

Protocol (D40 rolling-origin, leakage-clean):
  TRAIN  = train.parquet rows with snapshot_pair in {v160-v165 .. v220-v225}
  VALID  = train.parquet rows with snapshot_pair == v225-v227  (early stopping + selection)
  TEST   = eval.parquet (v227-v230 LAFA frame)  -- never tuned on.

Per-category models (nk/lk/pk). aspect is a FEATURE, not a split.
PK pool = knn_present only; NK/LK = full pool.
Lever: pminmax on LK-bpo only (per-protein min-max of the score before thresholding).
"""
import os, json, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PARQUET = os.path.join(HERE, "train.parquet")
EVAL_PARQUET = os.path.join(HERE, "eval.parquet")
PRED_DIR = os.path.join(HERE, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

VALID_PAIR = "v225-v227"
ASPECTS = ["mfo", "bpo", "cco"]
ASPECT_CODE = {"mfo": 0, "bpo": 1, "cco": 2}
SEED = 42

# ---- columns ----
_schema = pq.ParquetFile(TRAIN_PARQUET).schema_arrow
ALL_COLS = list(_schema.names)
META = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
        "qualifier", "evidence_code", "taxonomic_relation", "aspect"}
BOOL_COLS = [n for n in ALL_COLS if str(_schema.field(n).type) == "bool"]
FEATURES = [c for c in ALL_COLS if c not in META]  # numeric + bool feature cols
FEATURES = FEATURES + ["aspect_code"]
print(f"[cols] n_features={len(FEATURES)} bool_cols={len(BOOL_COLS)}")

READ_COLS = list(META) + [c for c in FEATURES if c != "aspect_code"]


def load_category(parquet_path, category, pk_knn_only):
    filt = [("category", "=", category)]
    if category == "pk" and pk_knn_only:
        filt.append(("knn_present", "=", True))
    t = pq.read_table(parquet_path, columns=READ_COLS, filters=filt)
    df = t.to_pandas()
    # cast bool -> int8
    for b in BOOL_COLS:
        if b in df.columns:
            df[b] = df[b].astype("int8")
    df["aspect_code"] = df["aspect"].map(ASPECT_CODE).astype("int8")
    # downcast floats
    for c in FEATURES:
        if c in df.columns and df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


def train_one(category):
    t0 = time.time()
    df = load_category(TRAIN_PARQUET, category, pk_knn_only=True)
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

    # validation predictions for selection proxy
    va = va.copy()
    va["score"] = booster.predict(Xva, num_iteration=best_it)
    return booster, best_it, va[["protein_accession", "go_term_id", "aspect", "label", "score"]]


def apply_pminmax_lk_bpo(df):
    """Per-protein min-max of score for LK category + BPO aspect rows."""
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


def ia_weighted_micro_fmax(df, ia_map, th_step=0.01):
    """Flat IA-weighted micro-Fmax over candidate rows (validation selection proxy)."""
    w = df["go_term_id"].map(ia_map).fillna(0.0).values
    s = df["score"].values
    y = df["label"].values.astype(bool)
    taus = np.arange(0.0, 1.0 + 1e-9, th_step)
    best = 0.0
    wy = w * y
    wny = w * (~y)
    tot_pos = wy.sum()
    for tau in taus:
        sel = s >= tau
        tp = wy[sel].sum()
        fp = wny[sel].sum()
        fn = tot_pos - tp
        if tp <= 0:
            continue
        pr = tp / (tp + fp)
        rc = tp / (tp + fn)
        if pr + rc > 0:
            f = 2 * pr * rc / (pr + rc)
            if f > best:
                best = f
    return float(best)


def main():
    # IA map
    ia_map = {}
    with open("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                try:
                    ia_map[p[0]] = float(p[1])
                except ValueError:
                    pass

    val_proxy = {}      # category -> aspect -> fmax
    boosters = {}
    val_frames = {}
    for cat in ["nk", "lk", "pk"]:
        booster, best_it, va = train_one(cat)
        boosters[cat] = (booster, best_it)
        # apply lever to LK-bpo on validation too
        if cat == "lk":
            va = apply_pminmax_lk_bpo(va)
        val_frames[cat] = va
        cell = {}
        for asp in ASPECTS:
            sub = va[va["aspect"] == asp]
            cell[asp] = ia_weighted_micro_fmax(sub, ia_map) if len(sub) else None
        val_proxy[cat] = cell
        print(f"[{cat}] VALIDATION proxy 9-cell: {cell}")
        booster.save_model(os.path.join(PRED_DIR, f"model_{cat}.txt"), num_iteration=best_it)

    # ---- TEST predictions ----
    for cat in ["nk", "lk", "pk"]:
        booster, best_it = boosters[cat]
        te = load_category(EVAL_PARQUET, cat, pk_knn_only=True)
        te = te[["protein_accession", "go_term_id", "aspect"] + FEATURES].copy() \
            if False else te  # keep all
        te["score"] = booster.predict(te[FEATURES], num_iteration=best_it)
        out = te[["protein_accession", "go_term_id", "aspect", "score"]].copy()
        if cat == "lk":
            out = apply_pminmax_lk_bpo(out)
        # write per-category pred dir (single TSV, no header)
        d = os.path.join(PRED_DIR, cat)
        os.makedirs(d, exist_ok=True)
        out[["protein_accession", "go_term_id", "score"]].to_csv(
            os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False,
            float_format="%.6f")
        print(f"[{cat}] TEST predictions written: {len(out)} rows -> {d}/{cat}.tsv")

    with open(os.path.join(HERE, "validation_proxy_9cell.json"), "w") as f:
        json.dump(val_proxy, f, indent=2)
    print("[done] validation proxy ->", os.path.join(HERE, "validation_proxy_9cell.json"))


if __name__ == "__main__":
    main()
