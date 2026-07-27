"""MODE B -- SSE AS A RERANKER FEATURE. Retrain the deployed per-category LightGBM reranker with the
SSE entailment score (MIN consensus) added as ONE MORE feature, temporal split identical to the
deployed recipe (train <= v225, valid v225-v227, test eval.parquet v227-v230).

Variants (paired, same data/seed, only the feature set differs):
  baseline       : deployed FEATURES (reproduces the anchor)
  sse            : FEATURES + sse_score
  sse_shuffled   : FEATURES + sse_score permuted within aspect (signal-destroyed control)

Writes per variant: modeB/{variant}/predictions/{cat}/{cat}.tsv (true-frame ready) + models + valproxy.
Leakage: SSE was trained with eval proteins EXCLUDED, so test-set sse_score is generalization; train
rows' sse_score is partially in-fit (standard for a model-derived feature; TEST is clean). Run under
the reranker-lab venv (lightgbm).
"""
import os, json, time, collections
import numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = "/home/frapercan/Thesis2"
RR = f"{ROOT}/repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank"
TRAIN_PARQUET = f"{RR}/train.parquet"; EVAL_PARQUET = f"{RR}/eval.parquet"
SSE = f"{ROOT}/storage/sse_full"; OUTB = f"{SSE}/modeB"
OBO = f"{ROOT}/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = f"{ROOT}/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
os.makedirs(OUTB, exist_ok=True)
VALID_PAIR = "v225-v227"; ASPECTS = ["mfo", "bpo", "cco"]; ASPECT_CODE = {"mfo": 0, "bpo": 1, "cco": 2}
SEED = 42; VARIANTS = ["baseline", "sse", "sse_shuffled"]

# alt-id map (align go ids to SSE vocab canonical ids)
alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur

# SSE score lookups
covmin = {}; uni_pos = None; term_col = {}
for asp in ASPECTS:
    cm = f"{SSE}/covmin_{asp}.npy"
    if not os.path.exists(cm): log(f"WARN covmin {asp} missing"); continue
    covmin[asp] = np.load(cm)
    U = np.load(f"{SSE}/covuniverse_{asp}.npz", allow_pickle=True)
    prots = U["proteins"].tolist(); terms = U["terms"].tolist()
    if uni_pos is None: uni_pos = {a: i for i, a in enumerate(prots)}
    term_col[asp] = {g: j for j, g in enumerate(terms)}
    log(f"loaded covmin {asp} {covmin[asp].shape}")

def add_sse(df, shuffle=False, rng=None):
    sse = np.full(len(df), np.nan, dtype=np.float32)
    prot_row = df["protein_accession"].map(uni_pos)                       # NaN if not in universe
    go_can = df["go_term_id"].map(lambda g: alt.get(g, g))
    aspv = df["aspect"].values
    for asp in ASPECTS:
        if asp not in covmin: continue
        m = np.where(aspv == asp)[0]
        if len(m) == 0: continue
        ri = prot_row.values[m]
        ci = go_can.iloc[m].map(term_col[asp]).values                    # NaN if not in vocab
        ok = (~pd.isna(ri)) & (~pd.isna(ci))
        vals = np.full(len(m), np.nan, dtype=np.float32)
        rr = ri[ok].astype(np.int64); cc = ci[ok].astype(np.int64)
        vals[ok] = covmin[asp][rr, cc].astype(np.float32)
        if shuffle:
            good = np.where(~np.isnan(vals))[0]
            vals[good] = vals[rng.permutation(good)]
        sse[m] = vals
    return sse

# columns
_schema = pq.ParquetFile(TRAIN_PARQUET).schema_arrow
ALL_COLS = list(_schema.names)
META = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
        "qualifier", "evidence_code", "taxonomic_relation", "aspect"}
BOOL_COLS = [n for n in ALL_COLS if str(_schema.field(n).type) == "bool"]
BASE_FEATURES = [c for c in ALL_COLS if c not in META] + ["aspect_code"]
READ_COLS = list(META) + [c for c in BASE_FEATURES if c != "aspect_code"]

def load_category(path, category):
    filt = [("category", "=", category)]
    if category == "pk": filt.append(("knn_present", "=", True))
    df = pq.read_table(path, columns=READ_COLS, filters=filt).to_pandas()
    for b in BOOL_COLS:
        if b in df.columns: df[b] = df[b].astype("int8")
    df["aspect_code"] = df["aspect"].map(ASPECT_CODE).astype("int8")
    for c in BASE_FEATURES:
        if c in df.columns and df[c].dtype == "float64": df[c] = df[c].astype("float32")
    return df

def apply_pminmax_lk_bpo(df):
    mask = (df["aspect"] == "bpo")
    if not mask.any(): return df
    sub = df.loc[mask, ["protein_accession", "score"]].copy()
    g = sub.groupby("protein_accession")["score"]; mn = g.transform("min"); mx = g.transform("max")
    rng = (mx - mn); df.loc[mask, "score"] = np.where(rng > 0, (sub["score"] - mn) / rng, 1.0)
    return df

def ia_wmf(df, ia_map, th_step=0.01):
    w = df["go_term_id"].map(ia_map).fillna(0.0).values; s = df["score"].values
    y = df["label"].values.astype(bool); wy = w * y; wny = w * (~y); tot = wy.sum(); best = 0.0
    for tau in np.arange(0.0, 1.0 + 1e-9, th_step):
        sel = s >= tau; tp = wy[sel].sum(); fp = wny[sel].sum(); fn = tot - tp
        if tp <= 0: continue
        pr = tp / (tp + fp); rc = tp / (tp + fn)
        if pr + rc > 0: best = max(best, 2 * pr * rc / (pr + rc))
    return float(best)

ia_map = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: ia_map[p[0]] = float(p[1])
        except ValueError: pass

def feats(variant): return BASE_FEATURES + (["sse_score"] if variant != "baseline" else [])
PARAMS = dict(objective="binary", metric=["auc", "average_precision"], learning_rate=0.05,
              num_leaves=63, min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, max_depth=-1, seed=SEED, verbosity=-1, num_threads=12)

summary = {"variants": VARIANTS, "cats": {}, "valproxy": {v: {} for v in VARIANTS},
           "sse_gain": {v: {} for v in VARIANTS}}
for cat in ["nk", "lk", "pk"]:
    log(f"===== category {cat}: load train =====")
    df = load_category(TRAIN_PARQUET, cat)
    df["sse_score"] = add_sse(df)
    cov = float(np.mean(~np.isnan(df["sse_score"].values)))
    summary["cats"][cat] = {"train_rows": len(df), "sse_coverage": round(cov, 4)}
    log(f"  {cat} train rows {len(df)} sse-coverage {cov:.3f}")
    te = load_category(EVAL_PARQUET, cat)
    te["sse_score"] = add_sse(te)
    log(f"  {cat} test rows {len(te)}")
    rng = np.random.default_rng(SEED)
    for variant in VARIANTS:
        F = feats(variant)
        if variant == "sse_shuffled":
            df["sse_score"] = add_sse(df, shuffle=True, rng=rng)
        elif variant == "sse":
            df["sse_score"] = add_sse(df)  # restore true
        tr = df[df["snapshot_pair"] != VALID_PAIR]; va = df[df["snapshot_pair"] == VALID_PAIR]
        dtr = lgb.Dataset(tr[F], label=tr["label"].astype("int8").values, free_raw_data=False)
        dva = lgb.Dataset(va[F], label=va["label"].astype("int8").values, reference=dtr, free_raw_data=False)
        booster = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva], valid_names=["valid"],
                            callbacks=[lgb.early_stopping(80)])
        bi = booster.best_iteration
        vp = va.copy(); vp["score"] = booster.predict(va[F], num_iteration=bi)
        if cat == "lk": vp = apply_pminmax_lk_bpo(vp)
        cell = {asp: (ia_wmf(vp[vp.aspect == asp], ia_map) if (vp.aspect == asp).any() else None)
                for asp in ASPECTS}
        summary["valproxy"][variant][cat] = cell
        # test predictions
        te2 = te.copy(); te2["score"] = booster.predict(te[F], num_iteration=bi)
        if cat == "lk": te2 = apply_pminmax_lk_bpo(te2)
        d = f"{OUTB}/{variant}/predictions/{cat}"; os.makedirs(d, exist_ok=True)
        te2[["protein_accession", "go_term_id", "score"]].to_csv(
            f"{d}/{cat}.tsv", sep="\t", header=False, index=False, float_format="%.6f")
        if variant == "sse":
            imp = dict(zip(F, booster.feature_importance(importance_type="gain")))
            summary["sse_gain"]["sse"][cat] = {"best_iter": bi,
                "sse_score_gain": float(imp.get("sse_score", 0.0)),
                "sse_gain_rank": int(sorted(imp.values(), reverse=True).index(imp.get("sse_score", 0.0)) + 1),
                "n_features": len(F)}
        log(f"  [{cat}/{variant}] best_iter {bi} valproxy {cell}")
    del df, te
    json.dump(summary, open(f"{SSE}/modeB_retrain.json", "w"), indent=1, default=float)

json.dump(summary, open(f"{SSE}/modeB_retrain.json", "w"), indent=1, default=float)
log("MODE B RETRAIN DONE"); print(json.dumps(summary, indent=1, default=float))
