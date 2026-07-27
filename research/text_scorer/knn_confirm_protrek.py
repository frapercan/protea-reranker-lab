"""ProtST vs ProTrek cross-model confirm: does a SECOND text-aligned PLM with a
DIFFERENT base (ProTrek=ESM2) reproduce ProtST's (ESM-1b) BP lift over the champion?

Same board-faithful kNN protocol as knn_confirm_text.py (query 7401 -> 15k ref,
cosine top-30, cosine-weighted GO vote, cafaeval f_micro_w 9-cell). Reps:
  - d8979601    (champion learned k-WTA, ankh)      [reference point]
  - protst-text (ESM-1b + PubMedBERT, 512-d)        [1st text model]
  - protrek-text(ESM2 + PubMedBERT, 1024-d)         [2nd text model, DIFFERENT base]
  - combined    (cos(protrek-text)+cos(d8979601))
The datapoint: if protrek-text ALSO beats the champion on BP (nk-BP leakage-free,
lk-BP the wall), the text->GO lift is NOT a ProtST/ESM-1b artifact. NK = leakage-free.
Preds saved to preds/<cfg>/ for neighbor-identity stratification reuse.
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

# accession orders per representation
qaccs_pst = json.load(open(os.path.join(W, "meta.json")))["accs"]          # ProtST query (7401)
raccs_pst = json.load(open(os.path.join(W, "ref_meta.json")))["accs"]      # ProtST/ProTrek ref (15k)
qaccs_ptk = json.load(open(os.path.join(W, "query_meta.json")))["accs"]    # ProTrek query (7401)
qaccs_d89 = json.load(open(os.path.join(LA, "emb_ankh_base/meta.json")))["accs"]
raccs_d89 = json.load(open(os.path.join(LA, "ref_emb/meta.json")))["accs"]
ref_go = json.load(open(os.path.join(LA, "ref_go.json")))

# common query + reference accessions across ALL three representations
qset = set(qaccs_pst) & set(qaccs_ptk) & set(qaccs_d89)
rset = set(raccs_pst) & set(raccs_d89) & set(ref_go)
Q = [a for a in qaccs_pst if a in qset]
R = [a for a in raccs_pst if a in rset]
print(f"common query {len(Q)} | common ref {len(R)}", flush=True)
ref_go_list = [ref_go.get(a, []) for a in R]

qi_pst = {a: i for i, a in enumerate(qaccs_pst)}; ri_pst = {a: i for i, a in enumerate(raccs_pst)}
qi_ptk = {a: i for i, a in enumerate(qaccs_ptk)}
qi_d89 = {a: i for i, a in enumerate(qaccs_d89)}; ri_d89 = {a: i for i, a in enumerate(raccs_d89)}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1; return X / n

def load(name):
    if name == "d8979601":
        Qe = np.load(os.path.join(LA, "query_d8979601.npy")).astype(np.float32)
        Re = np.load(os.path.join(LA, "ref_d8979601.npy")).astype(np.float32)
        return (l2(np.stack([Qe[qi_d89[a]] for a in Q])), l2(np.stack([Re[ri_d89[a]] for a in R])))
    if name == "protst-text":
        Qe = np.load(os.path.join(W, "query_protst.npy")).astype(np.float32)
        Re = np.load(os.path.join(W, "ref_protst.npy")).astype(np.float32)
        return (l2(np.stack([Qe[qi_pst[a]] for a in Q])), l2(np.stack([Re[ri_pst[a]] for a in R])))
    if name == "protrek-text":
        Qe = np.load(os.path.join(W, "query_protrek.npy")).astype(np.float32)
        Re = np.load(os.path.join(W, "ref_protrek.npy")).astype(np.float32)
        return (l2(np.stack([Qe[qi_ptk[a]] for a in Q])), l2(np.stack([Re[ri_pst[a]] for a in R])))
    raise ValueError(name)


def knn_pred(Qn, Rn, outdir, second=None):
    os.makedirs(outdir, exist_ok=True); w = open(os.path.join(outdir, "p.tsv"), "w")
    Q2 = second[0] if second else None; R2 = second[1] if second else None
    for i in range(0, len(Q), 500):
        sims = Qn[i:i + 500] @ Rn.T
        if second is not None:
            sims = sims + (Q2[i:i + 500] @ R2.T)
        for r, srow in enumerate(sims):
            nn = np.argpartition(-srow, K)[:K]; votes = defaultdict(float)
            for n in nn:
                s = float(srow[n])
                if s <= 0: continue
                for go in ref_go_list[n]: votes[go] += s
            if votes:
                mx = max(votes.values()); acc = Q[i + r]
                for go, v in votes.items(): w.write(f"{acc}\t{go}\t{v / mx:.6f}\n")
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

PRED = os.path.join(W, "preds_protrek"); os.makedirs(PRED, exist_ok=True)
emb = {n: load(n) for n in ("d8979601", "protst-text", "protrek-text")}
results = {}
for name in ("d8979601", "protst-text", "protrek-text"):
    d = os.path.join(PRED, name); knn_pred(*emb[name], d); results[name] = nine(d)
    print(f"[{name}] mean={sum(results[name].values())/9:.4f} {results[name]}", flush=True)
# combined = protrek-text + d8979601 (cosine sum)
d = os.path.join(PRED, "combined"); knn_pred(*emb["protrek-text"], d, second=emb["d8979601"])
results["combined"] = nine(d)
print(f"[combined] mean={sum(results['combined'].values())/9:.4f} {results['combined']}", flush=True)

order = [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]
cfgs = ("d8979601", "protst-text", "protrek-text", "combined")
print("\n=== ProTrek 2nd-datapoint confirm (f_micro_w) ===")
print(f"{'cell':10s}" + "".join(f"{c:>14s}" for c in cfgs) + "  PTK-d89")
for k in order:
    row = f"{k:10s}" + "".join(f"{results[c].get(k, 0):>14.4f}" for c in cfgs)
    ptk = results["protrek-text"].get(k, 0) - results["d8979601"].get(k, 0)
    star = "  <== BP" if k.endswith("bpo") else ""
    print(row + f"  {ptk:+.4f}{star}")
print(f"{'MEAN':10s}" + "".join(f"{sum(results[c].values())/9:>14.4f}" for c in cfgs))
json.dump(results, open(os.path.join(W, "knn_confirm_protrek_result.json"), "w"), indent=2)
print("\nPTK-d89 = protrek-text - champion d8979601. Does a 2nd text model (ESM2 base) "
      "reproduce ProtST's (ESM-1b) BP lift? NK = leakage-free.")
