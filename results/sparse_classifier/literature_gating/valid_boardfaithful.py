"""Board-faithful VALIDATION selection: hold out 25% of v225->v227 proteins, train
the gate variants on the other 75%, score the held-out fold with the SAME cafa_eval
machinery used for test (OBO, IA, TOI, prop=fill, norm=cafa, no_orphans), against a
validation GT built from the v225->v227 pool positives. This replaces the
fixed-denominator proxy so the include-literature decision is made on the real board
metric. No test peeking.
"""
import os, sys, json, tempfile, shutil, collections
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
from cafaeval.evaluation import cafa_eval  # noqa: E402

OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_PATH = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI_FILE = os.path.join(SC, "lafa_gt", "groundtruth_terms_of_interest.txt")
VALID_BASE = os.path.join(SC, "interpro2go_test", "valid_base.parquet")
NS_JSON = os.path.join(SC, "interpro2go_test", "go_namespace.json")
nm = json.load(open(NS_JSON))
ASPLET = {"bpo": "P", "mfo": "F", "cco": "C"}
SEED = 42
BASE_FEATS = ["s_rerank", "s_interpro", "s_max", "has_rerank", "has_interpro",
              "ia", "depth", "prot_max_rerank", "prot_n_cand"]
LIT_FEATS = ["text_score", "text_cos", "text_rank_pct", "rerank_rank_pct",
             "rank_gap", "text_top1", "text_top3", "text_top5", "has_text"]


def train_gate(tr, feats):
    rng = np.random.RandomState(SEED)
    prots = tr["acc"].unique().copy(); rng.shuffle(prots)
    cut = int(0.8 * len(prots)); es = set(prots[cut:])
    a = tr[~tr.acc.isin(es)]; b = tr[tr.acc.isin(es)]
    dtr = lgb.Dataset(a[feats], label=a["label"].astype("int8").values)
    dva = lgb.Dataset(b[feats], label=b["label"].astype("int8").values, reference=dtr)
    params = dict(objective="binary", metric="average_precision", learning_rate=0.05,
                  num_leaves=31, min_child_samples=50, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, seed=SEED, verbosity=-1,
                  num_threads=12)
    return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                     valid_names=["v"], callbacks=[lgb.early_stopping(60, verbose=False)])


def bpo_cell(pred_rows, gt_path, exclude=None):
    d = tempfile.mkdtemp(prefix="vbf_")
    pd.DataFrame(pred_rows, columns=["acc", "go", "score"]).to_csv(
        os.path.join(d, "p.tsv"), sep="\t", header=False, index=False, float_format="%.6f")
    _, best = cafa_eval(OBO, d, gt_path, ia=IA_PATH, no_orphans=True, norm="cafa",
                        prop="fill", exclude=exclude, toi_file=TOI_FILE,
                        th_step=0.01, n_cpu=8)
    shutil.rmtree(d, ignore_errors=True)
    b = best["f_micro_w"].reset_index()
    for _, r in b.iterrows():
        if r["ns"] == "biological_process":
            return round(float(r["f_micro_w"]), 5)
    return 0.0


def main():
    vb = pd.read_parquet(VALID_BASE)
    out = {}
    for cat in ["lk", "pk"]:
        vu = pd.read_parquet(os.path.join(HERE, f"valid_union_{cat}.parquet"))
        prots = np.array(sorted(vu.acc.unique()))
        rng = np.random.RandomState(SEED); rng.shuffle(prots)
        cut = int(0.75 * len(prots))
        held = set(prots[cut:]); train = vu[~vu.acc.isin(held)]
        te = vu[vu.acc.isin(held)]
        # validation GT for held-out proteins (all aspects, from pool positives)
        pos = vb[(vb.category == cat) & (vb.label == 1) & (vb.protein_accession.isin(held))]
        gt = pos[["protein_accession", "go_term_id"]].copy()
        gt["asp"] = gt["go_term_id"].map(nm).map(ASPLET)
        gt = gt.dropna(subset=["asp"])
        gtf = os.path.join(HERE, f"valid_gt_held_{cat}.tsv")
        with open(gtf, "w") as f:
            f.write("EntryID\tterm\taspect\n")
            for a, g, asp in zip(gt.protein_accession, gt.go_term_id, gt.asp):
                f.write(f"{a}\t{g}\t{asp}\n")
        cells = {}
        cells["naivemax"] = bpo_cell(list(zip(te.acc, te.go, te.s_max)), gtf)
        for name, feats in [("learned_base", BASE_FEATS),
                            ("learned_lit", BASE_FEATS + LIT_FEATS)]:
            g = train_gate(train, feats)
            pr = g.predict(te[feats], num_iteration=g.best_iteration)
            cells[name] = bpo_cell(list(zip(te.acc, te.go, pr)), gtf)
        chosen = max(cells, key=cells.get)
        lit_helps = cells["learned_lit"] > cells["learned_base"] and cells["learned_lit"] >= cells["naivemax"]
        out[cat] = {"board_faithful_valid_bpo": cells, "chosen": chosen,
                    "literature_helps": bool(lit_helps), "n_held_prots": len(held)}
        print(f"[{cat}] board-faithful VALID bpo: {cells} -> chosen={chosen} lit_helps={lit_helps}")
    json.dump(out, open(os.path.join(HERE, "valid_boardfaithful.json"), "w"), indent=2)
    print("written valid_boardfaithful.json")


if __name__ == "__main__":
    main()
