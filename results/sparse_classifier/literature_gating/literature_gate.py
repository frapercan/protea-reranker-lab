"""Learned gating of literature features into the BP meta-reranker (ADR-D43).

Union evidence pool per category (LK, PK), BP only:
  candidates = reranker BP pool  UNION  InterPro2GO BP pool
  evidence features:
    BASE: s_rerank, s_interpro, s_max(=naivemax baseline score), has_rerank,
          has_interpro, ia, depth, prot_max_rerank, prot_n_cand
    LIT : text_score, text_cos, text_rank_pct, rerank_rank_pct, rank_gap,
          text_top1/3/5, has_text   (literature -> GO max-cosine, S-PubMedBERT)

Decision (include literature?) by board-faithful-style validation Fmax on the
225->227 split (protein-grouped 3-fold CV, fixed-denominator IA-weighted micro-Fmax,
TOI+namespace filtered). NO test peeking for tuning. Selected model refit on all
validation, applied to 227->230 TEST, scored board-faithful with cafa_eval (exact
score_cafaeval.py setup). Literature routed into BP only; MF/CC/NK untouched.
"""
import os, sys, json, collections, tempfile
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
sys.path.insert(0, os.path.join(SC, "interpro2go_test"))
import interpro_lib as L  # noqa: E402
from cafaeval.evaluation import cafa_eval  # noqa: E402

OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_PATH = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
GT_DIR = os.path.join(SC, "lafa_gt")
TOI_FILE = os.path.join(GT_DIR, "groundtruth_terms_of_interest.txt")
VALID_BASE = os.path.join(SC, "interpro2go_test", "valid_base.parquet")
NS_JSON = os.path.join(SC, "interpro2go_test", "go_namespace.json")
PARENTS = os.path.join(SC, "go_parents.json")

RERANK_TEST = {
    "lk": os.path.join(SC, "percut_rerank/baseline/predictions/lk/lk.tsv"),
    "pk": os.path.join(SC, "percut_rerank/predictions/pk/pk.tsv"),
}
GT = {
    "lk": (os.path.join(GT_DIR, "groundtruth_LK.tsv"), None),
    "pk": (os.path.join(GT_DIR, "groundtruth_PK.tsv"),
           os.path.join(GT_DIR, "groundtruth_PK_known.tsv")),
}
SEED = 42
BASE_FEATS = ["s_rerank", "s_interpro", "s_max", "has_rerank", "has_interpro",
              "ia", "depth", "prot_max_rerank", "prot_n_cand"]
LIT_FEATS = ["text_score", "text_cos", "text_rank_pct", "rerank_rank_pct",
             "rank_gap", "text_top1", "text_top3", "text_top5", "has_text"]

# ---- shared maps ----
nm = json.load(open(NS_JSON))                       # go -> bpo/mfo/cco
parents = json.load(open(PARENTS))
toi = set(x.strip() for x in open(TOI_FILE) if x.strip())
ia = {}
for line in open(IA_PATH):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try:
            ia[p[0]] = float(p[1])
        except ValueError:
            pass


def ancestors(go):
    seen, stack = set(), [go]
    while stack:
        g = stack.pop()
        for q in parents.get(g, []):
            if q not in seen:
                seen.add(q)
                stack.append(q)
    return seen


def depth_map(go_ids):
    d = {}

    def dep(g):
        if g in d:
            return d[g]
        ps = parents.get(g, [])
        if not ps:
            d[g] = 0
            return 0
        d[g] = 0
        best = 0
        for p in ps:
            best = max(best, 1 + dep(p))
        d[g] = best
        return best
    sys.setrecursionlimit(100000)
    for g in go_ids:
        dep(g)
    return d


def add_lit_and_rank_feats(df, text):
    """df has acc,go,s_rerank,s_interpro,s_max. text: dict (acc,go)->(cos,text_score)."""
    df = df.copy()
    df["text_cos"] = [text.get((a, g), (0.0, 0.0))[0] for a, g in zip(df.acc, df.go)]
    df["text_score"] = [text.get((a, g), (0.0, 0.0))[1] for a, g in zip(df.acc, df.go)]
    acc_has_text = df.groupby("acc")["text_cos"].transform(lambda s: float((s > 0).any()))
    df["has_text"] = acc_has_text
    # within-protein percentile ranks (1.0 = top of that protein's candidates)
    df["text_rank_pct"] = df.groupby("acc")["text_score"].rank(pct=True, method="average")
    df["rerank_rank_pct"] = df.groupby("acc")["s_rerank"].rank(pct=True, method="average")
    df["rank_gap"] = df["text_rank_pct"] - df["rerank_rank_pct"]
    # text top-k indicators within protein (only where protein has text)
    txt_rank_desc = df.groupby("acc")["text_score"].rank(ascending=False, method="first")
    for k in (1, 3, 5):
        df[f"text_top{k}"] = ((txt_rank_desc <= k) & (df["has_text"] > 0)).astype("float32")
    # zero-out rank features for no-text proteins so they don't masquerade as signal
    notext = df["has_text"] <= 0
    df.loc[notext, ["text_rank_pct", "rank_gap"]] = 0.0
    return df


def add_prot_feats(df):
    df = df.copy()
    df["ia"] = df["go"].map(ia).fillna(0.0).astype("float32")
    dmap = depth_map(df["go"].unique())
    df["depth"] = df["go"].map(dmap).fillna(0).astype("float32")
    df["prot_max_rerank"] = df.groupby("acc")["s_rerank"].transform("max").astype("float32")
    df["prot_n_cand"] = df.groupby("acc")["go"].transform("count").astype("float32")
    return df


def build_union(rerank_df, ip_map, text_df, proteins, truth=None):
    """rerank_df: acc,go,s_rerank (BP). ip_map: {acc:{go:graded}}. text_df: acc,go,cos,text_score.
    proteins: category protein universe (set). truth: {acc:set(true bp terms)} or None."""
    text = {(a, g): (c, t) for a, g, c, t in
            zip(text_df.acc, text_df.go, text_df.cos, text_df.text_score)}
    rr_df = rerank_df[["acc", "go", "s_rerank"]].copy()
    ip_rows = []
    pset = set(proteins)
    for a, gd in ip_map.items():
        if a not in pset:
            continue
        for g, s in gd.items():
            if nm.get(g) == "bpo":
                ip_rows.append((a, g, float(s)))
    ip_df = pd.DataFrame(ip_rows, columns=["acc", "go", "s_interpro"]) if ip_rows \
        else pd.DataFrame(columns=["acc", "go", "s_interpro"])
    u = pd.merge(rr_df, ip_df, on=["acc", "go"], how="outer")
    u["s_rerank"] = u["s_rerank"].fillna(0.0).astype("float32")
    u["s_interpro"] = u["s_interpro"].fillna(0.0).astype("float32")
    u = u[u["acc"].isin(pset)].copy()
    u["has_rerank"] = (u["s_rerank"] > 0).astype("float32")
    u["has_interpro"] = (u["s_interpro"] > 0).astype("float32")
    u["s_max"] = np.maximum(u["s_rerank"], u["s_interpro"]).astype("float32")
    u = add_prot_feats(u)
    u = add_lit_and_rank_feats(u, text)
    if truth is not None:
        u["label"] = [1 if g in truth.get(a, ()) else 0 for a, g in zip(u.acc, u.go)]
    return u.reset_index(drop=True)


def fmax_fixed(score, w, y, total_pos_ia, th_step=0.01):
    if len(score) == 0 or total_pos_ia <= 0:
        return 0.0, 0.0
    s = np.asarray(score); w = np.asarray(w); y = np.asarray(y, dtype=bool)
    wt, wf = w * y, w * (~y)
    best_f, best_t = 0.0, 0.0
    for tau in np.arange(th_step, 1.0 + 1e-9, th_step):
        sel = s >= tau
        tp = wt[sel].sum(); fp = wf[sel].sum()
        if tp <= 0:
            continue
        pr = tp / (tp + fp); rc = tp / total_pos_ia
        if pr + rc > 0:
            f = 2 * pr * rc / (pr + rc)
            if f > best_f:
                best_f, best_t = f, float(tau)
    return float(best_f), best_t


def train_gate(tr, feats):
    rng = np.random.RandomState(SEED)
    prots = tr["acc"].unique().copy()
    rng.shuffle(prots)
    cut = int(0.8 * len(prots))
    es = set(prots[cut:])
    a = tr[~tr.acc.isin(es)]; b = tr[tr.acc.isin(es)]
    dtr = lgb.Dataset(a[feats], label=a["label"].astype("int8").values)
    dva = lgb.Dataset(b[feats], label=b["label"].astype("int8").values, reference=dtr)
    params = dict(objective="binary", metric="average_precision", learning_rate=0.05,
                  num_leaves=31, min_child_samples=50, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, seed=SEED, verbosity=-1,
                  num_threads=12)
    return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                     valid_names=["v"], callbacks=[lgb.early_stopping(60, verbose=False)])


def valid_truth(cat):
    vb = pd.read_parquet(VALID_BASE)
    dfc = vb[(vb.category == cat)]
    pos = dfc[dfc.label == 1]
    truth = collections.defaultdict(set)
    for acc, sub in pos.groupby("protein_accession"):
        t = set()
        for term in sub.go_term_id:
            t.add(term); t |= ancestors(term)
        truth[acc] = {x for x in t if nm.get(x) == "bpo" and x in toi}
    # rerank BP frame for valid
    bp = dfc[dfc.aspect == "bpo"][["protein_accession", "go_term_id", "base_score"]]
    bp = bp.rename(columns={"protein_accession": "acc", "go_term_id": "go",
                            "base_score": "s_rerank"})
    return truth, bp


def cell_from_cafaeval(pred_rows, cat):
    """pred_rows: list of (acc,go,score). Returns bpo cell dict via board-faithful cafa_eval."""
    gt_file, known = GT[cat]
    d = tempfile.mkdtemp(prefix=f"lit_{cat}_")
    pd.DataFrame(pred_rows, columns=["acc", "go", "score"]).to_csv(
        os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False,
        float_format="%.6f")
    _, dfs_best = cafa_eval(OBO, d, gt_file, ia=IA_PATH, no_orphans=True, norm="cafa",
                            prop="fill", exclude=known, toi_file=TOI_FILE,
                            th_step=0.01, n_cpu=8)
    best = dfs_best["f_micro_w"].reset_index()
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    for _, r in best.iterrows():
        if r["ns"] == "biological_process":
            return {"f_micro_w": round(float(r["f_micro_w"]), 5),
                    "tau": round(float(r["tau"]), 3),
                    "cov": round(float(r.get("cov", float("nan"))), 4)}
    return {"f_micro_w": 0.0, "tau": 0.0, "cov": 0.0}


def main():
    # ---- TEST literature scores: LK from existing precut parquet, PK from new ----
    txt_lk = pd.read_parquet(os.path.join(SC, "goretriever_lite",
                                           "text_scores_bpo_spubmed_precut.parquet"))
    txt_lk = txt_lk[txt_lk.category == "lk"][["acc", "go", "cos", "text_score"]]
    pk_txt_fn = os.path.join(HERE, "text_scores_test_pk_bpo.parquet")
    txt_pk = pd.read_parquet(pk_txt_fn)[["acc", "go", "cos", "text_score"]]
    test_text = {"lk": txt_lk, "pk": txt_pk}

    # ---- VALIDATION literature scores (earlier cut) ----
    vtxt = pd.read_parquet(os.path.join(HERE, "text_scores_valid_bpo.parquet"))

    # ---- InterPro maps ----
    vip = L.interpro_preds(json.load(open(os.path.join(SC, "interpro2go_test",
                                                        "valid_protein2ipr.json"))))
    tip = L.interpro_preds(json.load(open(os.path.join(SC, "interpro2go_test",
                                                        "protein2ipr.json"))))

    summary = {"baseline_canonical": {"LK_bpo": 0.42807, "PK_bpo": 0.21797},
               "transfew": {"LK_bpo": 0.512, "PK_bpo": 0.294},
               "validation": {}, "test": {}}

    for cat in ["lk", "pk"]:
        print("\n" + "=" * 60, "CATEGORY", cat.upper())
        # ----- build VALIDATION union -----
        truth, vbp = valid_truth(cat)
        vprots = set(vbp.acc.unique())
        vtxt_c = vtxt[vtxt.category == cat][["acc", "go", "cos", "text_score"]]
        vu = build_union(vbp, vip, vtxt_c, vprots, truth=truth)
        cov_text = float((vu.has_text > 0).mean())
        print(f"[{cat}] VALID union rows={len(vu)} prots={vu.acc.nunique()} "
              f"pos={int(vu.label.sum())} text_cov={cov_text:.3f}")
        vu.to_parquet(os.path.join(HERE, f"valid_union_{cat}.parquet"))

        # ----- 3-fold protein-grouped CV for selection -----
        rng = np.random.RandomState(SEED)
        allp = np.array(sorted(vprots)); rng.shuffle(allp)
        folds = np.array_split(allp, 3)
        variant_f = {"naivemax": [], "learned_base": [], "learned_lit": []}
        for fi in range(3):
            held = set(folds[fi])
            tr = vu[~vu.acc.isin(held)]
            te = vu[vu.acc.isin(held)]
            # fixed denom from held-out truth
            tot = sum(ia.get(t, 0.0) for a in held for t in truth.get(a, ()))
            w = te["go"].map(ia).fillna(0.0).values
            y = te["label"].values
            # naivemax
            f, _ = fmax_fixed(te["s_max"].values, w, y, tot)
            variant_f["naivemax"].append(f)
            for name, feats in [("learned_base", BASE_FEATS),
                                ("learned_lit", BASE_FEATS + LIT_FEATS)]:
                g = train_gate(tr, feats)
                pr = g.predict(te[feats], num_iteration=g.best_iteration)
                f, _ = fmax_fixed(pr, w, y, tot)
                variant_f[name].append(f)
        vmean = {k: round(float(np.mean(v)), 5) for k, v in variant_f.items()}
        print(f"[{cat}] VALID 3-fold mean Fmax: {vmean}")
        summary["validation"][cat] = {"per_fold": {k: [round(x, 5) for x in v]
                                                    for k, v in variant_f.items()},
                                      "mean": vmean, "text_coverage": round(cov_text, 4)}

        # ----- selection: include literature only if it beats learned_base AND naivemax -----
        chosen = max(["naivemax", "learned_base", "learned_lit"], key=lambda k: vmean[k])
        lit_helps = (vmean["learned_lit"] > vmean["learned_base"] and
                     vmean["learned_lit"] >= vmean["naivemax"])
        summary["validation"][cat]["chosen_variant"] = chosen
        summary["validation"][cat]["literature_helps_on_valid"] = bool(lit_helps)
        print(f"[{cat}] chosen={chosen}  literature_helps_on_valid={lit_helps}")

        # ----- build TEST union -----
        rr = pd.read_csv(RERANK_TEST[cat], sep="\t", header=None,
                         names=["acc", "go", "s_rerank"])
        rr = rr[rr.go.map(nm) == "bpo"][["acc", "go", "s_rerank"]]
        rr = rr.groupby(["acc", "go"], as_index=False)["s_rerank"].max()
        tprots = set(rr.acc.unique())
        tu = build_union(rr, tip, test_text[cat], tprots, truth=None)
        tcov = float((tu.has_text > 0).mean())
        print(f"[{cat}] TEST union rows={len(tu)} prots={tu.acc.nunique()} text_cov={tcov:.3f}")
        tu.to_parquet(os.path.join(HERE, f"test_union_{cat}.parquet"))

        # ----- refit each variant on ALL validation, apply to test, board-faithful score -----
        test_cells = {}
        # naivemax baseline (recomputed on my union -> sanity vs 0.428/0.218)
        test_cells["naivemax"] = cell_from_cafaeval(
            list(zip(tu.acc, tu.go, tu.s_max)), cat)
        models = {}
        for name, feats in [("learned_base", BASE_FEATS),
                            ("learned_lit", BASE_FEATS + LIT_FEATS)]:
            g = train_gate(vu, feats)
            models[name] = g
            pr = g.predict(tu[feats], num_iteration=g.best_iteration)
            test_cells[name] = cell_from_cafaeval(list(zip(tu.acc, tu.go, pr)), cat)
            imp = dict(zip(feats, g.feature_importance(importance_type="gain")))
            summary.setdefault("feature_gain", {})[f"{cat}_{name}"] = {
                k: round(float(v), 1) for k, v in sorted(imp.items(),
                                                         key=lambda x: -x[1])[:12]}
        # save chosen model
        if chosen in models:
            models[chosen].save_model(os.path.join(HERE, f"model_{cat}_{chosen}.txt"),
                                      num_iteration=models[chosen].best_iteration)
        summary["test"][cat] = {k: v for k, v in test_cells.items()}
        summary["test"][cat]["text_coverage"] = round(tcov, 4)
        print(f"[{cat}] TEST bpo Fmax: " +
              ", ".join(f"{k}={v['f_micro_w']}" for k, v in test_cells.items()))

    json.dump(summary, open(os.path.join(HERE, "result_9cell.json"), "w"), indent=2)
    print("\nwritten result_9cell.json")


if __name__ == "__main__":
    main()
