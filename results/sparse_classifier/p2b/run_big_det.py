"""FULL 7-seed retrain on the DETERMINISTIC saved SVD basis (prep_big_det.npz).
Same SEEDS + CFG as run_big.py (the champion). Writes heads to heads_det/ and a
TEMPORAL verify-gate: ensemble recall@100 (NK/LK/PK on the 227->230 eval proteins)
vs the champion serve ensemble (0.678/0.716/0.376). If it matches, the reproducible
basis is adopted and per-cut codes become valid. Reversible: new paths only, does
NOT touch the serve artifacts. File -> file, no live DB.
"""
import os, sys, json, glob, time, numpy as np, torch
P2B = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(P2B)
sys.path.insert(0, os.path.join(BASE, "p2"))
from two_tower import Data, train_seed, recall_at_k, recall_tf_baseline, ProjHead, DEV
from generate import generate_top100

OUT = f"{P2B}/heads_det"
os.makedirs(OUT, exist_ok=True)
PREP = f"{P2B}/prep_big_det.npz"
SEEDS = [0, 7, 137, 23, 91, 31, 53]
CFG = dict(hidden=2048, kwta=0, dropout=0.0, lr=2e-3, wd=1e-5, epochs=20, bs=512,
           gn=4.0, gp=1.0, clip=0.05, dag_w=0.2, dag_edges=4096, dag_margin=0.0)
CHAMP = {"NK": 0.678, "LK": 0.716, "PK": 0.376}   # serve-ensemble temporal recall@100

t0 = time.time()
data = Data(PREP, f"{BASE}/clf_protein_codes_big.npz")
print(f"DETRUN vocab V={data.V} train={len(data.train_idx)} test={len(data.test_idx)} dev={DEV}", flush=True)

models = []
for sd in SEEDS:
    m, _ = train_seed(data, sd, data.train_idx, CFG, log_prefix="DETBIG ")
    torch.save({"state_dict": m.state_dict(), "cfg": CFG, "V": data.V, "seed": sd},
               f"{OUT}/head_seed{sd}.pt")
    print(f"  saved seed{sd}  ({time.time()-t0:.0f}s)", flush=True)
    models.append(m)

# in-distribution held-out ensemble recall (matches run_big.py reporting)
rec, ceil, n = recall_at_k(data, models, data.test_idx, 100)
tf100 = recall_tf_baseline(data, data.test_idx, 100)
print(f"in-dist ensemble recall@100={rec:.4f} tf={tf100:.4f} ceiling={ceil:.4f}", flush=True)

# TEMPORAL verify-gate (NK/LK/PK on 227->230) with the det GO_V
parents = json.load(open(f"{BASE}/go_parents.json"))
def clo(ts):
    s = set(ts); st = list(ts)
    while st:
        for p in parents.get(st.pop(), []):
            if p not in s: s.add(p); st.append(p)
    return s
gt = {}
for cat in ["NK", "LK", "PK"]:
    for line in open(f"{BASE}/lafa_gt/groundtruth_{cat}.tsv"):
        if line.startswith("EntryID"): continue
        p, t, a = line.rstrip("\n").split("\t")[:3]
        gt.setdefault(cat, {}).setdefault(p, set()).add(t)
ev = np.load(f"{BASE}/eval_protein_codes.npz", allow_pickle=True)
accs = [str(a) for a in ev["accs"]]; codes = ev["codes"]
det_vocab = [str(x) for x in np.load(PREP, allow_pickle=True)["vocab_go"]]
top = generate_top100(codes, accs, models=models, GO_V=data.GO_V, vocab_go=det_vocab, k=100)
pc = {a: clo([g for g, _ in top[a]]) for a in accs}
temporal = {}
for cat in ["NK", "LK", "PK"]:
    rs = [len(tr & pc[p]) / len(tr) for p, tr in gt[cat].items() if p in pc and tr]
    temporal[cat] = round(float(np.mean(rs)), 4)
delta = {k: round(temporal[k] - CHAMP[k], 4) for k in CHAMP}
verdict = "PASS" if all(delta[k] >= -0.008 for k in delta) else "REVIEW"
metrics = {"corpus": "554K-curated-v227-DETbasis", "seeds": SEEDS, "cfg": CFG,
           "in_dist_recall100": rec, "tf_baseline": tf100, "ceiling": ceil,
           "temporal_recall100": temporal, "champion": CHAMP, "delta_vs_champion": delta,
           "verdict": verdict, "secs": time.time() - t0}
json.dump(metrics, open(f"{OUT}/metrics_det.json", "w"), indent=2)
print(f"TEMPORAL det ensemble: {temporal}  champion {CHAMP}  delta {delta}", flush=True)
print(f"VERDICT {verdict}  ({time.time()-t0:.0f}s)", flush=True)
print("DETRUN_DONE", flush=True)
