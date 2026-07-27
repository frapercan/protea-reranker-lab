"""Cheap complementarity check: does ProTrek's CCO signal stack on ProtST's BP signal?
Triple kNN combine cos(protst-text)+cos(protrek-text)+cos(d8979601) vs the best pair,
9-cell f_micro_w. Informs whether the reranker should carry 1 or 2 text scorers.
"""
import os, json, numpy as np
from pathlib import Path
from collections import defaultdict
from cafaeval.evaluation import cafa_eval

W = "/home/frapercan/Thesis2/storage/text_scorer"
LA = "/home/frapercan/Thesis2/storage/layer_ablation"
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
NS2A = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
CATS = {"NK": (str(REL / "groundtruth_NK.tsv"), None), "LK": (str(REL / "groundtruth_LK.tsv"), None),
        "PK": (str(REL / "groundtruth_PK.tsv"), str(REL / "groundtruth_PK_known.tsv"))}
K = 30

qaccs_pst = json.load(open(os.path.join(W, "meta.json")))["accs"]
raccs_pst = json.load(open(os.path.join(W, "ref_meta.json")))["accs"]
qaccs_ptk = json.load(open(os.path.join(W, "query_meta.json")))["accs"]
qaccs_d89 = json.load(open(os.path.join(LA, "emb_ankh_base/meta.json")))["accs"]
raccs_d89 = json.load(open(os.path.join(LA, "ref_emb/meta.json")))["accs"]
ref_go = json.load(open(os.path.join(LA, "ref_go.json")))
qset = set(qaccs_pst) & set(qaccs_ptk) & set(qaccs_d89)
rset = set(raccs_pst) & set(raccs_d89) & set(ref_go)
Q = [a for a in qaccs_pst if a in qset]; R = [a for a in raccs_pst if a in rset]
ref_go_list = [ref_go.get(a, []) for a in R]
qi_pst = {a: i for i, a in enumerate(qaccs_pst)}; ri_pst = {a: i for i, a in enumerate(raccs_pst)}
qi_ptk = {a: i for i, a in enumerate(qaccs_ptk)}
qi_d89 = {a: i for i, a in enumerate(qaccs_d89)}; ri_d89 = {a: i for i, a in enumerate(raccs_d89)}
print(f"common query {len(Q)} | common ref {len(R)}", flush=True)

def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1; return X / n
def take(Qe, Re, qi, ri):
    return l2(np.stack([Qe[qi[a]] for a in Q])), l2(np.stack([Re[ri[a]] for a in R]))
PST = take(np.load(f"{W}/query_protst.npy").astype("f4"), np.load(f"{W}/ref_protst.npy").astype("f4"), qi_pst, ri_pst)
PTK = take(np.load(f"{W}/query_protrek.npy").astype("f4"), np.load(f"{W}/ref_protrek.npy").astype("f4"), qi_ptk, ri_pst)
D89 = take(np.load(f"{LA}/query_d8979601.npy").astype("f4"), np.load(f"{LA}/ref_d8979601.npy").astype("f4"), qi_d89, ri_d89)

def knn(parts, outdir):
    os.makedirs(outdir, exist_ok=True); w = open(f"{outdir}/p.tsv", "w")
    for i in range(0, len(Q), 500):
        sims = sum(Qn[i:i+500] @ Rn.T for Qn, Rn in parts)
        for r, srow in enumerate(sims):
            nn = np.argpartition(-srow, K)[:K]; votes = defaultdict(float)
            for n in nn:
                s = float(srow[n])
                if s <= 0: continue
                for go in ref_go_list[n]: votes[go] += s
            if votes:
                mx = max(votes.values()); acc = Q[i+r]
                for go, v in votes.items(): w.write(f"{acc}\t{go}\t{v/mx:.6f}\n")
    w.close()
def nine(outdir):
    cells = {}
    for cat, (gt, known) in CATS.items():
        _, best = cafa_eval(OBO, outdir, gt, ia=IA, no_orphans=True, norm="cafa", prop="fill",
                            exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
        for _, row in best["f_micro_w"].reset_index().iterrows():
            a = NS2A.get(row["ns"])
            if a: cells[f"{cat.lower()}-{a}"] = round(float(row["f_micro_w"]), 5)
    return cells

combos = {"pst+d89": [PST, D89], "pst+ptk+d89": [PST, PTK, D89], "pst+ptk": [PST, PTK]}
res = {}
for name, parts in combos.items():
    d = f"{W}/preds_triple/{name.replace('+','_')}"; knn(parts, d); res[name] = nine(d)
    print(f"[{name}] mean={sum(res[name].values())/9:.4f} {res[name]}", flush=True)
order = [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]
print("\n=== triple-combine (f_micro_w): does ProTrek-CCO stack on ProtST-BP? ===")
print(f"{'cell':10s}" + "".join(f"{c:>14s}" for c in combos) + "   ptk_adds")
for k in order:
    print(f"{k:10s}" + "".join(f"{res[c].get(k,0):>14.4f}" for c in combos) +
          f"   {res['pst+ptk+d89'].get(k,0)-res['pst+d89'].get(k,0):+.4f}")
print(f"{'MEAN':10s}" + "".join(f"{sum(res[c].values())/9:>14.4f}" for c in combos))
json.dump(res, open(f"{W}/knn_triple_result.json", "w"), indent=2)
print("\nptk_adds = (pst+ptk+d89) - (pst+d89): does adding ProTrek help over ProtST+champion?")
