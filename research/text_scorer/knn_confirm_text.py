"""Board-faithful kNN confirm: does ProtST text-alignment add for GO transfer?

Four representations, same kNN protocol (query 7401 -> 15k reference, cosine top-30,
cosine-weighted GO vote, cafaeval f_micro_w 9-cell, stratified by category):
  - d8979601  (champion learned k-WTA, ankh)         [reference point]
  - ProtST-text  (protein_feature, text-aligned)     [the candidate]
  - ESM1b-raw    (mean of residue_feature, same base) [the same-base control -> isolates TEXT]
  - combined     (cos(ProtST-text)+cos(d8979601))
Text contribution = ProtST-text vs ESM1b-raw (same ESM-1b base, +/- text alignment).
NK is the leakage-free anchor. Numbers are low (15k ref, raw kNN, no reranker); the
DELTA is the finding.
"""
import os, json, tempfile, shutil, numpy as np
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

qaccs_txt = json.load(open(os.path.join(W, "meta.json")))["accs"]          # ProtST query order (7401)
raccs_txt = json.load(open(os.path.join(W, "ref_meta.json")))["accs"]      # ProtST ref order (15k)
qaccs_d89 = json.load(open(os.path.join(LA, "emb_ankh_base/meta.json")))["accs"]
raccs_d89 = json.load(open(os.path.join(LA, "ref_emb/meta.json")))["accs"]
ref_go = json.load(open(os.path.join(LA, "ref_go.json")))

# common query + reference accessions across all representations
Q = [a for a in qaccs_txt if a in set(qaccs_d89)]
R = [a for a in raccs_txt if a in set(raccs_d89) and a in ref_go]
qi_txt = {a: i for i, a in enumerate(qaccs_txt)}; qi_d89 = {a: i for i, a in enumerate(qaccs_d89)}
ri_txt = {a: i for i, a in enumerate(raccs_txt)}; ri_d89 = {a: i for i, a in enumerate(raccs_d89)}
print(f"common query {len(Q)} | common ref {len(R)}", flush=True)
ref_go_list = [ref_go.get(a, []) for a in R]


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1; return X / n

def load(name):
    if name == "d8979601":
        Qe = np.load(os.path.join(LA, "query_d8979601.npy")).astype(np.float32)
        Re = np.load(os.path.join(LA, "ref_d8979601.npy")).astype(np.float32)
        return (l2(np.stack([Qe[qi_d89[a]] for a in Q])), l2(np.stack([Re[ri_d89[a]] for a in R])))
    fn = {"protst-text": "protst", "esm1b-raw": "esm1braw"}[name]
    Qe = np.load(os.path.join(W, f"query_{fn}.npy")).astype(np.float32) if name == "protst-text" \
        else np.load(os.path.join(W, f"query_{fn}.npy")).astype(np.float32)
    Re = np.load(os.path.join(W, f"ref_{fn}.npy")).astype(np.float32)
    return (l2(np.stack([Qe[qi_txt[a]] for a in Q])), l2(np.stack([Re[ri_txt[a]] for a in R])))


def knn_pred(Qn, Rn, outdir):
    os.makedirs(outdir, exist_ok=True); w = open(os.path.join(outdir, "p.tsv"), "w")
    for i in range(0, len(Q), 500):
        sims = Qn[i:i + 500] @ Rn.T
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

PRED = os.path.join(W, "preds"); os.makedirs(PRED, exist_ok=True)
emb = {n: load(n) for n in ("d8979601", "protst-text", "esm1b-raw")}
results = {}
for name in ("d8979601", "esm1b-raw", "protst-text"):
    d = os.path.join(PRED, name); knn_pred(*emb[name], d); results[name] = nine(d)  # preds kept for neighbor-identity stratification
    print(f"[{name}] mean={sum(results[name].values())/9:.4f} {results[name]}", flush=True)
# combined = ProtST-text + d8979601 (cosine sum) -- reuse precomputed Qn/Rn
Qc = emb["protst-text"][0]; Rc = emb["protst-text"][1]; Qd = emb["d8979601"][0]; Rd = emb["d8979601"][1]
def knn_pred_combined(outdir):
    os.makedirs(outdir, exist_ok=True); w = open(os.path.join(outdir, "p.tsv"), "w")
    for i in range(0, len(Q), 500):
        sims = (Qc[i:i+500] @ Rc.T) + (Qd[i:i+500] @ Rd.T)
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
d = os.path.join(PRED, "combined"); knn_pred_combined(d); results["combined"] = nine(d)
print(f"[combined] mean={sum(results['combined'].values())/9:.4f} {results['combined']}", flush=True)

order = [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]
cfgs = ("d8979601", "esm1b-raw", "protst-text", "combined")
print("\n=== ProtST text-scorer board-faithful confirm (f_micro_w) ===")
print(f"{'cell':10s}" + "".join(f"{c:>13s}" for c in cfgs) + "   TEXT(pt-esm)")
for k in order:
    row = f"{k:10s}" + "".join(f"{results[c].get(k, 0):>13.4f}" for c in cfgs)
    txt = results["protst-text"].get(k, 0) - results["esm1b-raw"].get(k, 0)
    star = "  <== BP" if k.endswith("bpo") else ""
    print(row + f"   {txt:+.4f}{star}")
print(f"{'MEAN':10s}" + "".join(f"{sum(results[c].values())/9:>13.4f}" for c in cfgs))
json.dump(results, open(os.path.join(W, "knn_confirm_text_result.json"), "w"), indent=2)
print("\nTEXT contribution = protst-text - esm1b-raw (same ESM-1b base). NK = leakage-free.")
