"""DEPLOYABLE temporal check: does the known-conditioned rescoring gain survive when the model is
trained on the PAST only (v160..v225, early-stop v225-v227) and applied to the BLIND eval window
(v227->v230)? The Phase-1/2 GBDT was grouped-CV OOF *on the eval window* and could have adapted to
eval-period label statistics the deployed reranker (trained on the past) never saw. This mirrors the
deployed reranker's temporal training and is the honest go/no-go number.
"""
import json, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from cafaeval.evaluation import cafa_eval

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
PC = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank"
GTDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
OUT = ROOT / "storage/regen_headline"
GT = {"lk": (str(GTDIR / "groundtruth_LK.tsv"), None),
      "pk": (str(GTDIR / "groundtruth_PK.tsv"), str(GTDIR / "groundtruth_PK_known.tsv"))}
KF = ["association_total", "association_cross", "association_present", "anc2vec_query_known_cos",
      "anc2vec_query_known_maxcos", "anc2vec_query_known_count", "self_prior_score",
      "lineage_is_ancestor_of_known", "lineage_is_descendant_of_known", "lineage_ancestor_of_count",
      "lineage_descendant_of_count", "go_term_frequency"]
VAL_PAIR = "v225-v227"
ns = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
BP = {t for t, n in ns.items() if n == "biological_process"}


def score_bpo(pred_df, cat):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "pd"; d.mkdir()
        pred_df[["prot", "term", "score"]].to_csv(d / "p.tsv", sep="\t", header=False, index=False,
                                                  float_format="%.6f")
        gt, known = GT[cat]
        _, dfs = cafa_eval(OBO, str(d), gt, ia=IA, no_orphans=True, norm="cafa", prop="fill",
                           exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
    b = dfs["f_micro_w"].reset_index(); r = b[b.ns == "biological_process"]
    return (float(r.iloc[0]["f_micro_w"]), float(r.iloc[0]["tau"])) if len(r) else (None, None)


def minmax(x):
    x = np.nan_to_num(np.asarray(x, float)); lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


print(f"[{time.time()-t0:.0f}s] loading train.parquet (past folds)...", flush=True)
cols = ["category", "aspect", "snapshot_pair", "label"] + KF
tr = pq.read_table(PC / "train.parquet", columns=cols).to_pandas()
print(f"[{time.time()-t0:.0f}s] train rows {len(tr):,}; snapshot pairs {sorted(tr.snapshot_pair.unique())}", flush=True)

results = {}
for cat in ["pk", "lk"]:
    print(f"\n[{time.time()-t0:.0f}s] ===== {cat.upper()}-BPO deployable =====", flush=True)
    d = tr[(tr.category == cat) & (tr.aspect == "bpo")]
    past = d[d.snapshot_pair != VAL_PAIR]; val = d[d.snapshot_pair == VAL_PAIR]
    print(f"  past {len(past):,}/{past.label.sum():,} pos | val {len(val):,}/{val.label.sum():,} pos", flush=True)
    Xtr = past[KF].values.astype(np.float32); ytr = (past.label.values > 0).astype(int)
    Xva = val[KF].values.astype(np.float32); yva = (val.label.values > 0).astype(int)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, seed=42, verbose=-1,
                  num_threads=8, metric="average_precision",
                  scale_pos_weight=(ytr == 0).sum() / max(1, (ytr == 1).sum()))
    m = lgb.train(params, lgb.Dataset(Xtr, label=ytr), num_boost_round=1000,
                  valid_sets=[lgb.Dataset(Xva, label=yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    vauc = roc_auc_score(yva, m.predict(Xva)) if yva.sum() else float("nan")
    print(f"  trained best_iter {m.best_iteration}; val ROC-AUC {vauc:.4f}", flush=True)

    # apply to eval window
    ev = pq.read_table(PC / "eval.parquet",
                       columns=["protein_accession", "go_term_id", "category", "aspect", "label"] + KF).to_pandas()
    ev = ev[(ev.category == cat) & (ev.aspect == "bpo")].copy()
    ev["dep_model"] = m.predict(ev[KF].values.astype(np.float32))
    evu = ev.drop_duplicates(["protein_accession", "go_term_id"])[
        ["protein_accession", "go_term_id", "dep_model"]].rename(
        columns={"protein_accession": "prot", "go_term_id": "term"})
    # per-protein AUC of the temporal model on eval (info)
    evd = ev.drop_duplicates(["protein_accession", "go_term_id"])
    aucs = []
    for p, g in evd.groupby("protein_accession"):
        if g.label.nunique() == 2:
            aucs.append(roc_auc_score(g.label > 0, g.dep_model))
    ppauc = float(np.mean(aucs))
    print(f"  temporal-model per-protein AUC on eval: {ppauc:.4f} (n={len(aucs)})", flush=True)

    full = pd.read_csv(PC / "predictions" / cat / f"{cat}.tsv", sep="\t", header=None,
                       names=["prot", "term", "score"])
    full["isbp"] = full.term.isin(BP)
    full = full.merge(evu, on=["prot", "term"], how="left")
    bp = full.isbp.values; dep = full.score.values.astype(float)
    mvec = np.zeros(len(full)); mk = bp & full.dep_model.notna().values
    mvec[mk] = minmax(full.loc[mk, "dep_model"].values)

    f_inc, t_inc = score_bpo(full.assign(score=dep), cat)
    cell = {"val_roc_auc": round(vauc, 4), "temporal_per_protein_auc_eval": round(ppauc, 4),
            "incumbent": round(f_inc, 5), "arms": {}}
    best = {"tag": None, "f": -1.0}
    for w in [0.25, 0.5, 1.0, 2.0]:
        s = dep.copy(); s[bp] = np.clip(dep[bp] + w * mvec[bp], 0, 1)
        f, tau = score_bpo(full.assign(score=s), cat)
        cell["arms"][f"add_w{w}"] = {"f": round(f, 5), "tau": tau, "delta": round(f - f_inc, 5)}
        print(f"    add_w{w:<4} f={f:.5f} (tau {tau}) delta {f-f_inc:+.5f}", flush=True)
        if f > best["f"]:
            best = {"tag": f"add_w{w}", "f": round(f, 5), "delta": round(f - f_inc, 5)}
    # full rerank on temporal model alone
    s = dep.copy(); s[bp] = mvec[bp]
    f, tau = score_bpo(full.assign(score=s), cat)
    cell["arms"]["rerank"] = {"f": round(f, 5), "tau": tau, "delta": round(f - f_inc, 5)}
    print(f"    rerank    f={f:.5f} (tau {tau}) delta {f-f_inc:+.5f}", flush=True)
    if f > best["f"]:
        best = {"tag": "rerank", "f": round(f, 5), "delta": round(f - f_inc, 5)}
    cell["best_deployable"] = best
    results[cat] = cell
    json.dump(results, open(OUT / "temporal_deployable.json", "w"), indent=2)

print(f"\n[{time.time()-t0:.0f}s] DONE"); print(json.dumps(results, indent=2))
