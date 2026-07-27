"""Exp 10 done right: a UNIVERSAL reranker with all signals on the correct dataset.

One LightGBM booster trained over ALL three categories pooled (nk+lk+pk), FULL
feature set (the champion 69, InterPro included) + aspect_code + a new category_code
conditioning feature, on the same clean v227-lineage 227->230 frame and composite
candidate pool the per-category champion uses (nk/lk from the M2 baseline pool, pk
from the d8979601 per-cut pool). Board-faithful 9-cell f_micro_w vs the per-category
champion, cell by cell.

Big-parquet handling: the train parquets are 42M (baseline) and 26M (percut) rows
and their category filter does not push down, so a plain read materialises the whole
table (~29GB). We instead STREAM row-batches and keep ALL positives + a fraction of
negatives for training (memory-safe, fast). Scoring uses the FULL eval set (no
subsampling), so the reported 9-cell numbers are exact; only training negatives are
sampled, which a gradient-boosted ranker is robust to. NEG_FRAC is recorded.
"""
import os, json, time
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.compute as pc
import pyarrow.parquet as pq
import lightgbm as lgb
from cafaeval.evaluation import cafa_eval

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
PC = os.path.join(SC, "percut_rerank")
BASELINE = os.path.join(PC, "baseline")
PRED_DIR = os.path.join(HERE, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

VALID_PAIR = "v225-v227"
ASPECT_CODE = {"mfo": 0, "bpo": 1, "cco": 2}
CATEGORY_CODE = {"nk": 0, "lk": 1, "pk": 2}
SEED = 42
NEG_FRAC = 0.15  # keep all positives + this fraction of negatives for TRAINING only
META = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
        "qualifier", "evidence_code", "taxonomic_relation", "aspect"}

_schema = pq.ParquetFile(os.path.join(PC, "train.parquet")).schema_arrow
ALL_COLS = list(_schema.names)
BOOL_COLS = [n for n in ALL_COLS if str(_schema.field(n).type) == "bool"]
ALL_FEATURES_69 = [c for c in ALL_COLS if c not in META]
FEATURES = ALL_FEATURES_69 + ["aspect_code", "category_code"]
READ_COLS = list(META) + ALL_FEATURES_69

GT_DIR = os.path.join(SC, "lafa_gt")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = os.path.join(GT_DIR, "groundtruth_terms_of_interest.txt")
NS2ASP = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
GT = {"nk": (os.path.join(GT_DIR, "groundtruth_NK.tsv"), None),
      "lk": (os.path.join(GT_DIR, "groundtruth_LK.tsv"), None),
      "pk": (os.path.join(GT_DIR, "groundtruth_PK.tsv"),
             os.path.join(GT_DIR, "groundtruth_PK_known.tsv"))}
CHAMP = json.load(open(os.path.join(SC, "s2e_serve_reranker/result_9cell.json")))
CHAMP_CELL = {c: {a: CHAMP["serve_producible_9cell"][c][a]["f_micro_w"]
                  for a in ("mfo", "bpo", "cco")} for c in ("nk", "lk", "pk")}


def _finalise(out):
    for b in BOOL_COLS:
        if b in out.columns:
            out[b] = out[b].astype("int8")
    out["aspect_code"] = out["aspect"].map(ASPECT_CODE).astype("int8")
    out["category_code"] = out["category"].map(CATEGORY_CODE).astype("int8")
    for c in FEATURES:
        if c in out.columns and out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    return out


def load_stream(parquet_path, categories, neg_frac=1.0):
    """Stream row-batches (never materialise the full table). pk -> knn_present
    only. neg_frac<1.0 keeps all positives + a sample of negatives (train only)."""
    dset = pads.dataset(parquet_path, format="parquet")
    scanner = dset.scanner(columns=READ_COLS,
                           filter=pc.field("category").isin(list(categories)),
                           batch_size=400_000)
    rng = np.random.default_rng(SEED)
    parts = []
    for rb in scanner.to_batches():
        if rb.num_rows == 0:
            continue
        df = rb.to_pandas()
        if "pk" in categories and "knn_present" in df.columns:
            df = df[(df["category"] != "pk") | (df["knn_present"].astype(bool))]
        if neg_frac < 1.0 and len(df):
            neg = df[df["label"] == 0]
            keep_neg = neg.sample(frac=neg_frac, random_state=int(rng.integers(1 << 30))) if len(neg) else neg
            df = pd.concat([df[df["label"] == 1], keep_neg])
        if len(df):
            parts.append(df)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=READ_COLS)
    return _finalise(out)


def apply_pminmax_lk_bpo(df):
    mask = (df["aspect"] == "bpo")
    if not mask.any():
        return df
    sub = df.loc[mask, ["protein_accession", "score"]].copy()
    g = sub.groupby("protein_accession")["score"]
    mn, mx = g.transform("min"), g.transform("max")
    rng = mx - mn
    df.loc[mask, "score"] = np.where(rng > 0, (sub["score"] - mn) / rng, 1.0)
    return df


def main():
    t0 = time.time()
    base = load_stream(os.path.join(BASELINE, "train.parquet"), ["nk", "lk"], NEG_FRAC)
    print(f"[load] baseline nk+lk rows={len(base)} ({time.time()-t0:.1f}s)", flush=True)
    pkt = load_stream(os.path.join(PC, "train.parquet"), ["pk"], NEG_FRAC)
    print(f"[load] percut pk rows={len(pkt)} ({time.time()-t0:.1f}s)", flush=True)
    full = pd.concat([base, pkt], ignore_index=True); del base, pkt
    tr = full[full["snapshot_pair"] != VALID_PAIR].copy()
    va = full[full["snapshot_pair"] == VALID_PAIR].copy()
    del full
    print(f"[universal] train={len(tr)} pos={int(tr.label.sum())} | valid={len(va)} "
          f"pos={int(va.label.sum())} neg_frac={NEG_FRAC} (load {time.time()-t0:.1f}s)", flush=True)

    dtr = lgb.Dataset(tr[FEATURES], label=tr["label"].astype("int8").values, free_raw_data=False)
    dva = lgb.Dataset(va[FEATURES], label=va["label"].astype("int8").values, reference=dtr, free_raw_data=False)
    params = dict(objective="binary", metric=["auc", "average_precision"],
                  learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  max_depth=-1, seed=SEED, verbosity=-1, num_threads=12)
    booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dtr, dva],
                        valid_names=["train", "valid"],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)])
    bi = booster.best_iteration
    booster.save_model(os.path.join(PRED_DIR, "model_universal.txt"), num_iteration=bi)
    print(f"[universal] best_iter={bi} valid_auc={booster.best_score['valid']['auc']:.5f}", flush=True)

    # FULL eval (no subsampling) -> exact board-faithful numbers
    base_ev = load_stream(os.path.join(BASELINE, "eval.parquet"), ["nk", "lk"], 1.0)
    pk_ev = load_stream(os.path.join(PC, "eval.parquet"), ["pk"], 1.0)
    eval_all = pd.concat([base_ev, pk_ev], ignore_index=True); del base_ev, pk_ev
    res, delta = {}, {}
    for cat in ("nk", "lk", "pk"):
        te = eval_all[eval_all["category"] == cat].copy()
        te["score"] = booster.predict(te[FEATURES], num_iteration=bi)
        out = te[["protein_accession", "go_term_id", "aspect", "score"]].copy()
        if cat == "lk":
            out = apply_pminmax_lk_bpo(out)
        d = os.path.join(PRED_DIR, cat); os.makedirs(d, exist_ok=True)
        out[["protein_accession", "go_term_id", "score"]].to_csv(
            os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False, float_format="%.6f")
        gt_file, known = GT[cat]
        _, best = cafa_eval(OBO, d, gt_file, ia=IA, no_orphans=True, norm="cafa", prop="fill",
                            exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
        cells = {}
        for _, r in best["f_micro_w"].reset_index().iterrows():
            a = NS2ASP.get(r["ns"])
            if a:
                cells[a] = round(float(r["f_micro_w"]), 5)
        res[cat] = cells
        delta[cat] = {a: round(cells[a] - CHAMP_CELL[cat][a], 5) for a in ("mfo", "bpo", "cco")}
        print(f"{cat}: universal={cells} vs champion={CHAMP_CELL[cat]} delta={delta[cat]}", flush=True)

    mu = sum(res[c][a] for c in res for a in res[c]) / 9
    mc = sum(CHAMP_CELL[c][a] for c in CHAMP_CELL for a in CHAMP_CELL[c]) / 9
    out = {"note": "universal reranker (one model, all signals incl interpro, aspect+category "
                   "conditioning, composite pool, clean 227->230 frame, train negatives sampled "
                   f"at neg_frac={NEG_FRAC}, full eval) vs per-category champion",
           "neg_frac": NEG_FRAC, "universal_9cell": res, "champion_9cell": CHAMP_CELL,
           "delta_universal_minus_champion": delta, "mean_universal": round(mu, 5),
           "mean_champion": round(mc, 5), "mean_delta": round(mu - mc, 5), "best_iteration": bi}
    json.dump(out, open(os.path.join(HERE, "result_9cell.json"), "w"), indent=2)
    print(f"\nMEAN universal {mu:.5f} vs per-category champion {mc:.5f} ({mu-mc:+.5f}). "
          f"wrote result_9cell.json", flush=True)


if __name__ == "__main__":
    main()
