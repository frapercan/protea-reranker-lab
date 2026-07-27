"""Deliverable 1 intrinsic quality: does the learned k-WTA GO code give co-occurring / same-process
terms OVERLAPPING codes while text-close-but-non-co-occurring terms are SEPARATED (hard-neg objective)?
Compares each learned variant against the FIXED-recipe two-tower on a COMMON held-out pair set:
  POS  = co-occurring term pairs (share proteins, t0 corpus)
  HARD = text-close (BioBERT kNN) but NEVER co-occurring
Reports mean overlap(POS), mean overlap(HARD), separation AUC, and named smoke pairs.
Read-only w.r.t. repos.
"""
import json, time
from pathlib import Path
import numpy as np
t0 = time.time()
OUT = Path("/home/frapercan/Thesis2/storage/kwta_go_encoder")
tf = np.load(OUT / "term_features.npz", allow_pickle=True)
vocab = tf["vocab"].tolist(); knn_idx = tf["knn_idx"]
cg = np.load(OUT / "cooc_graph.npz", allow_pickle=True)
indptr, indices = cg["indptr"], cg["indices"]
N = len(vocab)
neigh = [indices[indptr[i]:indptr[i+1]] for i in range(N)]
cooc_set = [set(x.tolist()) for x in neigh]
anchors = [i for i in range(N) if len(neigh[i]) > 0]

rng = np.random.default_rng(123)          # fresh seed: held-out from training's seed-0 split
sel = rng.choice(anchors, size=min(4000, len(anchors)), replace=False)
pos_pairs, hard_pairs = [], []
for i in sel.tolist():
    j = int(rng.choice(neigh[i])); pos_pairs.append((vocab[i], vocab[j]))
    hn = [int(x) for x in knn_idx[i] if int(x) not in cooc_set[i] and int(x) != i]
    if hn: hard_pairs.append((vocab[i], vocab[int(rng.choice(hn))]))

SMOKE = [("glycolysis~gluconeogenesis", "GO:0006096", "GO:0006094"),
         ("glycolysis~translation", "GO:0006096", "GO:0006412"),
         ("glycolysis~plasma_membrane", "GO:0006096", "GO:0005886"),
         ("DNA_repair~DNA_replication", "GO:0006281", "GO:0006260"),
         ("apoptosis~cell_cycle", "GO:0006915", "GO:0007049")]

def auc(pos, hard):
    s = np.concatenate([pos, hard]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(hard))])
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(len(s))
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))

def eval_codes(path):
    d = np.load(path, allow_pickle=True)
    ids = {g: k for k, g in enumerate(d["go_ids"].tolist())}
    C = d["codes"].astype(np.float32)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    def ov(a, b): return float(C[ids[a]] @ C[ids[b]]) if a in ids and b in ids else None
    po = np.array([ov(a, b) for a, b in pos_pairs if a in ids and b in ids])
    ho = np.array([ov(a, b) for a, b in hard_pairs if a in ids and b in ids])
    return {"pos_overlap_mean": round(float(po.mean()), 4), "hard_overlap_mean": round(float(ho.mean()), 4),
            "separation_auc": round(auc(po, ho), 4), "n_pos": len(po), "n_hard": len(ho),
            "smoke": {name: (round(ov(a, b), 4) if ov(a, b) is not None else None) for name, a, b in SMOKE}}

rep = {"pairs": {"pos": len(pos_pairs), "hard": len(hard_pairs)}, "codes": {}}
for tag in ["coann", "coann_struct", "coann_struct_text", "twotower"]:
    p = OUT / (f"go_codes_{tag}.npz")
    if p.exists():
        rep["codes"][tag] = eval_codes(p)
        print(f"[{time.time()-t0:.0f}s] {tag}: {rep['codes'][tag]}", flush=True)
json.dump(rep, open(OUT / "intrinsic_report.json", "w"), indent=1)
print("DONE intrinsic_report", flush=True)
