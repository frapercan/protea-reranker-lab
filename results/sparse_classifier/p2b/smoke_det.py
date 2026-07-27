"""SMOKE for the deterministic-basis retrain: train ONE seed on prep_big_det.npz
(deterministic saved SVD basis) and compare its TEMPORAL recall@100 (NK/LK/PK on
the 227->230 eval proteins) against ONE existing serve-basis head. If the single
det seed recovers the single serve seed, the deterministic basis is sound and the
full 7-seed retrain is worth launching. File -> file, no live DB.
"""
import os, sys, glob, json, time, numpy as np, torch
P2B = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(P2B)
sys.path.insert(0, os.path.join(BASE, "p2"))
from two_tower import Data, train_seed, ProjHead, DEV
from generate import generate_top100

ASP = {"P": "bpo", "F": "mfo", "C": "cco"}
CFG = dict(hidden=2048, kwta=0, dropout=0.0, lr=2e-3, wd=1e-5, epochs=20, bs=512,
           gn=4.0, gp=1.0, clip=0.05, dag_w=0.2, dag_edges=4096, dag_margin=0.0)

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

def temporal(models, GO_V, vocab):
    top = generate_top100(codes, accs, models=models, GO_V=GO_V, vocab_go=vocab, k=100)
    pc = {a: clo([g for g, _ in top[a]]) for a in accs}
    out = {}
    for cat in ["NK", "LK", "PK"]:
        rs = [len(tr & pc[p]) / len(tr) for p, tr in gt[cat].items() if p in pc and tr]
        out[cat] = round(float(np.mean(rs)), 4)
    return out

# reference: ONE serve-basis head (existing) with serve GO_V
serve_prep = np.load(f"{P2B}/prep_big.npz", allow_pickle=True)
serve_vocab = [str(x) for x in serve_prep["vocab_go"]]
serve_GO_V = torch.from_numpy(serve_prep["GO_V"].astype(np.float32)).to(DEV)
ck = torch.load(sorted(glob.glob(f"{P2B}/head_seed*.pt"))[0], map_location=DEV)
sm = ProjHead(2048, 1024, V=ck["V"], hidden=ck["cfg"]["hidden"], kwta=ck["cfg"]["kwta"], dropout=0.0).to(DEV)
sm.load_state_dict(ck["state_dict"]); sm.eval()
ref = temporal([sm], serve_GO_V, serve_vocab)
print(f"REF serve 1-seed temporal: {ref}", flush=True)

# train ONE det seed on the deterministic prep
t0 = time.time()
data = Data(f"{P2B}/prep_big_det.npz", f"{BASE}/clf_protein_codes_big.npz")
print(f"det data V={data.V} train={len(data.train_idx)} dev={DEV}", flush=True)
m, _ = train_seed(data, 0, data.train_idx, CFG, log_prefix="DET ")
os.makedirs(f"{P2B}/heads_det", exist_ok=True)
torch.save({"state_dict": m.state_dict(), "cfg": CFG, "V": data.V, "seed": 0},
           f"{P2B}/heads_det/head_seed0.pt")
det_vocab = [str(x) for x in np.load(f"{P2B}/prep_big_det.npz", allow_pickle=True)["vocab_go"]]
det = temporal([m], data.GO_V, det_vocab)
print(f"DET   det 1-seed temporal: {det}  ({time.time()-t0:.0f}s)", flush=True)
print(f"delta det-serve (1 seed): {{'NK':{round(det['NK']-ref['NK'],4)},'LK':{round(det['LK']-ref['LK'],4)},'PK':{round(det['PK']-ref['PK'],4)}}}", flush=True)
print("SMOKE_DONE", flush=True)
