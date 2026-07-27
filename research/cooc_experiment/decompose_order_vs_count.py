"""Split the oracle gap into ORDERING and COUNTING. Which one do we actually lack?

The oracle (0.6077) scores the pool by the true label, so it gets TWO gifts at once:
a perfect within-protein ORDER, and a perfect per-protein COUNT (exactly the true
terms land above the threshold). We deliver 0.2222 with our order and ONE global
threshold. Nobody has ever separated the two.

The distinction matters because a global threshold sweep CANNOT express a per-protein
count: every protein is cut at the same tau, whether it deserves 3 terms or 30. That
is not fixable by any objective change and not by any globally monotone calibration
(best-F is invariant to those). It is only fixable by a per-protein decision.

Arms, all on the SAME lambdarank per-cell model, same eval rows, same cafaeval, same
FULL gt:

  G  our scores, one global threshold          -> control, must reproduce ~0.2222
  K  our ORDER frozen, PERFECT per-protein count: emit each protein's top-k_p at
     score 1.0, where k_p = |true terms of p in the full gt|, capped at how many
     candidates we hold for p. Constant across tau, so best-F == that F.
  R  our ORDER frozen, a REALISTIC count proxy: k_p = the count a cheap predictor
     could plausibly know, here the protein's number of t0-known BP terms (a feature
     we already carry). Tells us whether the K headroom is reachable at all.

Read it as: K - G = what perfect COUNTING is worth on our current ordering.
            0.6077 - K = what is left that only better ORDERING can buy.

WHY k_p IS CAPPED AND NEVER 0: 33.9% of PK proteins have zero true BP terms in the
pool. If k_p were 0 they would emit nothing, and no_orphans=True would DROP them from
the evaluation, inflating the score by shrinking the denominator. That is exactly the
artifact that had to be ruled out for arm E earlier today. Every protein that appears
in G must appear in K and R.
"""
import json, subprocess, tempfile, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, lightgbm as lgb

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
VAL = "v225-v227"
EX = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair", "qualifier",
      "evidence_code", "taxonomic_relation", "aspect"}
FEATS = [c for c in pq.ParquetFile(DS / "train.parquet").schema_arrow.names if c not in EX]
P = {"objective": "lambdarank", "metric": ["ndcg", "map"], "ndcg_eval_at": [5, 10],
     "label_gain": [0, 1], "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
     "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5, "seed": 42,
     "verbose": -1, "num_threads": 12, "max_bin": 63, "two_round": True, "force_col_wise": True}


def load(p):
    cols = FEATS + ["category", "aspect", "snapshot_pair", "protein_accession", "go_term_id", "label"]
    t = pq.read_table(p, columns=list(dict.fromkeys(cols)))
    cat = np.asarray(t.column("category").to_pylist())
    asp = np.asarray(t.column("aspect").to_pylist())
    m = (cat == "pk") & (asp == "bpo")
    X = np.empty((int(m.sum()), len(FEATS)), dtype=np.float32)
    for j, c in enumerate(FEATS):
        X[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)[m]
    md = {k: np.asarray(t.column(k).to_pylist())[m]
          for k in ("snapshot_pair", "protein_accession", "go_term_id", "aspect")}
    md["label"] = (t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m] > 0).astype(np.int32)
    return X, md


DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pred_dir}","{gt}",ia="{ia}",prop="fill",norm="cafa",
    no_orphans=True,max_terms=500,th_step=0.001,n_cpu=1,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{out_json}","w"),default=str)
'''
TRUTH = [tuple(l.rstrip("\n").split("\t")) for l in open(DS / "gt_pk_bp.tsv")]
GT = {}
for p_, g_ in TRUTH:
    GT.setdefault(p_, set()).add(g_)


def score(name, prot, go, s):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pred_dir"
        d.mkdir(parents=True)
        with (d / f"{name}.tsv").open("w") as fh:
            for p_, g_, v in zip(prot, go, s):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_, g_ in TRUTH:
                fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pred_dir=str(d), gt=str(gt), ia=IA, out_json=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=5400)
        if r.returncode != 0:
            print(f"  cafaeval FAILED {name}: {r.stderr[-300:]}", flush=True)
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return {k: (round(float(best[k]), 4) if isinstance(best.get(k), (int, float)) else best.get(k))
                for k in ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_max")} if best else None


def topk_mask(prot, s, kmap):
    """Frozen order, per-protein cut: keep each protein's top k_p by our score s."""
    out = np.zeros(len(s), dtype=np.float64)
    order = np.argsort(prot, kind="stable")
    bounds = np.flatnonzero(np.r_[True, prot[order][1:] != prot[order][:-1]])
    for a, b in zip(np.r_[bounds], np.r_[bounds[1:], len(order)]):
        g = order[a:b]
        p_ = prot[g[0]]
        k = int(min(max(kmap.get(p_, 1), 1), len(g)))   # never 0: see docstring
        top = g[np.argsort(-s[g], kind="stable")[:k]]
        out[top] = 1.0
    return out


t0 = time.time()
Xtr, mtr = load(DS / "train.parquet")
Xev, mev = load(DS / "eval.parquet")
iv = mtr["snapshot_pair"] == VAL

# deployed grouping key
def key(md, sel):
    return np.char.add(np.char.add(np.char.add(md["snapshot_pair"][sel], "|"),
                                   np.char.add(md["protein_accession"][sel], "|")), md["aspect"][sel])


parts = {}
for tag, sel in (("tr", ~iv), ("va", iv)):
    k = key(mtr, sel)
    o = np.argsort(k, kind="stable")
    _, sizes = np.unique(k[o], return_counts=True)
    parts[tag] = (Xtr[sel][o], mtr["label"][sel][o], sizes)
dtr = lgb.Dataset(parts["tr"][0], label=parts["tr"][1], group=parts["tr"][2], feature_name=FEATS)
dva = lgb.Dataset(parts["va"][0], label=parts["va"][1], group=parts["va"][2], reference=dtr,
                  feature_name=FEATS)
b = lgb.train(P, dtr, num_boost_round=3000, valid_sets=[dva],
              callbacks=[lgb.early_stopping(50, verbose=False)])
s = b.predict(Xev, num_iteration=b.best_iteration)
prot, go = mev["protein_accession"], mev["go_term_id"]
res = {"note": "same model, same order, same eval rows, same gt. Only the CUT varies.",
       "oracle_perfect_order_and_count": 0.6077, "deployed_pooled_recipe": 0.1255}

res["G_global_threshold"] = score("G", prot, go, s)
print(f"[G] global threshold (control) {res['G_global_threshold']}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(W / "decompose_order_vs_count.json", "w"), indent=1)

# --- K: perfect per-protein count, our order --------------------------------
k_true = {p_: len(v) for p_, v in GT.items()}
sK = topk_mask(prot, s, k_true)
res["K_perfect_count_our_order"] = score("K", prot, go, sK)
res["mean_k_true"] = round(float(np.mean([k_true.get(p_, 1) for p_ in set(prot)])), 2)
print(f"[K] perfect count, our order  {res['K_perfect_count_our_order']}  (mean k={res['mean_k_true']})", flush=True)
json.dump(res, open(W / "decompose_order_vs_count.json", "w"), indent=1)

# --- R: realistic count proxy = the protein's t0-known BP term count ----------
# Must be a COUNT, not a similarity: anc2vec_query_known_cos also matches "known"
# and is a cosine in [-1,1], which would silently make every k_p = 1.
kn = next((c for c in ("anc2vec_query_known_count",) if c in FEATS), None)
if kn:
    v = Xev[:, FEATS.index(kn)]
    k_proxy = {}
    for p_ in set(prot):
        k_proxy[p_] = int(max(round(float(np.nanmax(v[prot == p_]))), 1))
    sR = topk_mask(prot, s, k_proxy)
    res["R_proxy_count_feature"] = kn
    res["R_proxy_count_our_order"] = score("R", prot, go, sR)
    res["mean_k_proxy"] = round(float(np.mean(list(k_proxy.values()))), 2)
    print(f"[R] proxy count ({kn}) {res['R_proxy_count_our_order']}  (mean k={res['mean_k_proxy']})", flush=True)
else:
    res["R_proxy_count_our_order"] = "no known-count feature in the export"
    print("[R] skipped: no known-count feature", flush=True)

fG = (res.get("G_global_threshold") or {}).get("f_micro_w")
fK = (res.get("K_perfect_count_our_order") or {}).get("f_micro_w")
if fG and fK:
    res["counting_is_worth"] = round(fK - fG, 4)
    res["ordering_still_missing"] = round(0.6077 - fK, 4)
json.dump(res, open(W / "decompose_order_vs_count.json", "w"), indent=1)
print("\n=== PK-BPO: is the gap ORDERING or COUNTING? ===", flush=True)
print(f"  deployed pooled recipe            0.1255", flush=True)
print(f"  G our order + ONE global threshold {fG}", flush=True)
print(f"  K our order + PERFECT count        {fK}", flush=True)
print(f"  oracle: perfect order AND count    0.6077", flush=True)
if fG and fK:
    print(f"  -> perfect COUNTING is worth  {fK-fG:+.4f}", flush=True)
    print(f"  -> what only ORDERING can buy {0.6077-fK:+.4f}", flush=True)
print("DONE", flush=True)
