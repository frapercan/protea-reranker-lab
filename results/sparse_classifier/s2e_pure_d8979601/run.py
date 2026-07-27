"""Option C (serve<->offline pool reconciliation): retrain NK/LK rerankers on the
d8979601 (per-cut) candidate pool instead of the M2 anc2vec baseline pool, so the
PURE-d8979601 config the serve path already delivers (0.3536) matches the offline
number end-to-end -- and, per the validated +40% learned-encoder lever, may EXCEED
the composite champion's serve-producible 0.3884.

Only NK/LK change (they moved from baseline -> percut). PK is UNCHANGED (the champion
already trains PK on percut/d8979601), so we REUSE the champion PK model + predictions
rather than burn the 26M-row PK retrain. Everything else identical to the composite
champion recipe (train_serve_reranker.py): same serve-producible 69-interpro features
+ aspect_code, same LightGBM params, early-stop on v225-v227, pminmax on LK-bpo.

Board-faithful 9-cell (cafa_eval, OBO/IA/TOI, prop=fill, norm=cafa, no_orphans, PK
excludes PK_known, f_micro_w). Compares to the composite champion serve-producible
9-cell (result reused from s2e_serve_reranker/result_9cell.json).
"""
import os, json, time, shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from cafaeval.evaluation import cafa_eval

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
PC = os.path.join(SC, "percut_rerank")
CHAMP = os.path.join(SC, "s2e_serve_reranker")
PRED_DIR = os.path.join(HERE, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

# NK/LK now from percut (d8979601); PK reused from champion.
SRC = {
    "nk": {"train": os.path.join(PC, "train.parquet"), "eval": os.path.join(PC, "eval.parquet")},
    "lk": {"train": os.path.join(PC, "train.parquet"), "eval": os.path.join(PC, "eval.parquet")},
}
VALID_PAIR = "v225-v227"
ASPECT_CODE = {"mfo": 0, "bpo": 1, "cco": 2}
SEED = 42
META = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
        "qualifier", "evidence_code", "taxonomic_relation", "aspect"}
INTERPRO_FAMILY = [
    "interpro_hit", "interpro_score", "interpro_n_signatures",
    "interpro_db_pfam", "interpro_db_panther", "interpro_db_superfamily",
    "interpro_db_smart", "interpro_db_cdd", "interpro_db_prosite", "interpro_present",
]
_schema = pq.ParquetFile(SRC["nk"]["train"]).schema_arrow
ALL_COLS = list(_schema.names)
BOOL_COLS = [n for n in ALL_COLS if str(_schema.field(n).type) == "bool"]
ALL_FEATURES = [c for c in ALL_COLS if c not in META]
SERVE_FEATURES = [c for c in ALL_FEATURES if c not in INTERPRO_FAMILY]
FEATURES = SERVE_FEATURES + ["aspect_code"]
READ_COLS = list(META) + [c for c in FEATURES if c != "aspect_code"]

GT_DIR = os.path.join(SC, "lafa_gt")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = os.path.join(GT_DIR, "groundtruth_terms_of_interest.txt")
NS2ASP = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
GT = {"nk": (os.path.join(GT_DIR, "groundtruth_NK.tsv"), None),
      "lk": (os.path.join(GT_DIR, "groundtruth_LK.tsv"), None),
      "pk": (os.path.join(GT_DIR, "groundtruth_PK.tsv"),
             os.path.join(GT_DIR, "groundtruth_PK_known.tsv"))}


def load_category(parquet_path, category):
    filt = [("category", "=", category)]
    df = pq.read_table(parquet_path, columns=READ_COLS, filters=filt).to_pandas()
    for b in BOOL_COLS:
        if b in df.columns:
            df[b] = df[b].astype("int8")
    df["aspect_code"] = df["aspect"].map(ASPECT_CODE).astype("int8")
    for c in FEATURES:
        if c in df.columns and df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


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


def train_one(cat):
    t0 = time.time()
    df = load_category(SRC[cat]["train"], cat)
    tr = df[df["snapshot_pair"] != VALID_PAIR]
    va = df[df["snapshot_pair"] == VALID_PAIR]
    print(f"[{cat}] train={len(tr)} pos={int(tr.label.sum())} | valid={len(va)} "
          f"pos={int(va.label.sum())} (load {time.time()-t0:.1f}s)", flush=True)
    dtr = lgb.Dataset(tr[FEATURES], label=tr["label"].astype("int8").values, free_raw_data=False)
    dva = lgb.Dataset(va[FEATURES], label=va["label"].astype("int8").values,
                      reference=dtr, free_raw_data=False)
    params = dict(objective="binary", metric=["auc", "average_precision"],
                  learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  max_depth=-1, seed=SEED, verbosity=-1, num_threads=12)
    booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dtr, dva],
                        valid_names=["train", "valid"],
                        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)])
    bi = booster.best_iteration
    print(f"[{cat}] best_iter={bi} valid_auc={booster.best_score['valid']['auc']:.5f} "
          f"valid_ap={booster.best_score['valid']['average_precision']:.5f}", flush=True)
    booster.save_model(os.path.join(PRED_DIR, f"model_{cat}.txt"), num_iteration=bi)
    return booster, bi


def predict_test(cat, booster, bi):
    te = load_category(SRC[cat]["eval"], cat)
    te["score"] = booster.predict(te[FEATURES], num_iteration=bi)
    out = te[["protein_accession", "go_term_id", "aspect", "score"]].copy()
    if cat == "lk":
        out = apply_pminmax_lk_bpo(out)
    d = os.path.join(PRED_DIR, cat)
    os.makedirs(d, exist_ok=True)
    out[["protein_accession", "go_term_id", "score"]].to_csv(
        os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False,
        float_format="%.6f")
    print(f"[{cat}] TEST preds written: {len(out)} rows", flush=True)


def score_category(cat):
    gt_file, known = GT[cat]
    _, best = cafa_eval(OBO, os.path.join(PRED_DIR, cat), gt_file, ia=IA,
                        no_orphans=True, norm="cafa", prop="fill", exclude=known,
                        toi_file=TOI, th_step=0.01, n_cpu=8)
    cells = {}
    for _, r in best["f_micro_w"].reset_index().iterrows():
        a = NS2ASP.get(r["ns"])
        if a:
            cells[a] = round(float(r["f_micro_w"]), 5)
    return cells


def main():
    # NK/LK retrain on d8979601
    for cat in ("nk", "lk"):
        booster, bi = train_one(cat)
        predict_test(cat, booster, bi)
    # reuse champion PK (already d8979601)
    os.makedirs(os.path.join(PRED_DIR, "pk"), exist_ok=True)
    shutil.copy(os.path.join(CHAMP, "predictions/pk/pk.tsv"), os.path.join(PRED_DIR, "pk/pk.tsv"))
    shutil.copy(os.path.join(CHAMP, "predictions/model_pk.txt"), os.path.join(PRED_DIR, "model_pk.txt"))
    print("[pk] reused champion PK model + predictions", flush=True)

    champ = json.load(open(os.path.join(CHAMP, "result_9cell.json")))
    comp = {c: {a: champ["serve_producible_9cell"][c][a]["f_micro_w"]
                for a in ("mfo", "bpo", "cco")} for c in ("nk", "lk", "pk")}
    res, delta = {}, {}
    for cat in ("nk", "lk", "pk"):
        res[cat] = score_category(cat)
        delta[cat] = {a: round(res[cat][a] - comp[cat][a], 5) for a in ("mfo", "bpo", "cco")}
        print(f"{cat}: pure={res[cat]} vs composite={ {a:round(comp[cat][a],5) for a in comp[cat]} } "
              f"delta={delta[cat]}", flush=True)
    mean_pure = sum(res[c][a] for c in res for a in res[c]) / 9
    mean_comp = sum(comp[c][a] for c in comp for a in comp[c]) / 9
    out = {"note": "pure-d8979601 NK/LK (option C) vs composite champion serve-producible",
           "pure_d8979601_9cell": res, "composite_champion_9cell": comp,
           "delta_pure_minus_composite": delta,
           "mean_pure_d8979601": round(mean_pure, 5),
           "mean_composite_champion": round(mean_comp, 5),
           "mean_delta": round(mean_pure - mean_comp, 5)}
    json.dump(out, open(os.path.join(HERE, "result_9cell.json"), "w"), indent=2)
    print(f"\nMEAN pure-d8979601 {mean_pure:.5f} vs composite {mean_comp:.5f} "
          f"({mean_pure-mean_comp:+.5f}). wrote result_9cell.json", flush=True)


if __name__ == "__main__":
    main()
