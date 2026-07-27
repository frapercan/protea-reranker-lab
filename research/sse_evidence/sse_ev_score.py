"""Score evidence-arm SSE (K=1) over EVAL-target proteins only (lean, small on disk).
usage: sse_ev_score.py <arm> [asp,...]  -> scores_{arm}_{asp}.npz {proteins, terms, score}
Eval targets = union of LK+PK groundtruth proteins for the aspect, intersect ESM corpus.
Run under PROTEA/.venv (torch+CUDA).
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
import torch as th
from torch import nn

ARM = sys.argv[1]
ASPS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["mfo", "bpo", "cco"]
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] [{ARM}] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
GF = ROOT / "storage/cooc_experiment/generator_frames"
GTDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
OUT = ROOT / "storage/sse_evidence"; MDIR = OUT / "models"
DEV = "cuda:0"; N, PKk, TEMP = 2048, 128, 0.3
ASPLET = {"mfo": "F", "bpo": "P", "cco": "C"}

def soft_kwta_scalar(a, k):
    thr = a.topk(k, dim=1).values[:, -1:].detach(); return th.sigmoid((a - thr) / TEMP)
def _thr_perrow(a, ks):
    srt = a.sort(dim=1, descending=True).values
    gi = (ks - 1).clamp(0, a.shape[1] - 1).unsqueeze(1); return srt.gather(1, gi).detach()
def soft_kwta_perrow(a, ks): return th.sigmoid((a - _thr_perrow(a, ks)) / TEMP)
class SSE(nn.Module):
    def __init__(self, in_dim, n_terms, n_rels, MTt):
        super().__init__(); self.MTt = MTt
        self.prot = nn.Sequential(nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(1024, N))
        self.pln = nn.LayerNorm(N)
        self.term = nn.Embedding(n_terms, N); self.tln = nn.LayerNorm(N)
        self.rel = nn.Parameter(th.zeros(n_rels, N))
    def prot_gate(self, x): return soft_kwta_scalar(self.pln(self.prot(x)), PKk)
    def term_gate(self): return soft_kwta_perrow(self.tln(self.term.weight), self.MTt)
    def coverage(self, gp, gt): return (gp @ gt.t()) / (gt.sum(1) + 1e-6)

accs = json.load(open(GF / "accs.json")); acc_idx = {a: i for i, a in enumerate(accs)}
EMB = np.load(GF / "raw_esm2_3b.npy", mmap_mode="r")
gtL = pd.read_csv(GTDIR / "groundtruth_LK.tsv", sep="\t")
gtP = pd.read_csv(GTDIR / "groundtruth_PK.tsv", sep="\t")

for asp in ASPS:
    let = ASPLET[asp]
    tset = set(gtL[gtL.aspect == let].EntryID) | set(gtP[gtP.aspect == let].EntryID)
    prots = sorted(a for a in tset if a in acc_idx)
    rows = np.array([acc_idx[a] for a in prots])
    meta = np.load(MDIR / f"{ARM}_{asp}_meta.npz", allow_pickle=True)
    terms = meta["terms"].tolist(); M_T = meta["M_T"]; mu = meta["mu"]; sd = meta["sd"]
    n_rels = int(meta["n_rels"][0]); n_terms = len(terms)
    MTt = th.tensor(M_T, device=DEV)
    net = SSE(2560, n_terms, n_rels, MTt).to(DEV)
    net.load_state_dict(th.load(MDIR / f"{ARM}_{asp}_0.th")); net.eval()
    mu_t = th.tensor(mu, device=DEV); sd_t = th.tensor(sd, device=DEV)
    P = len(prots); score = np.empty((P, n_terms), dtype=np.float32); CH = 4096
    with th.no_grad():
        gt = net.term_gate()
        for s in range(0, P, CH):
            X = th.tensor(np.ascontiguousarray(EMB[rows[s:s+CH]].astype(np.float32)), device=DEV)
            X = (X - mu_t) / sd_t
            score[s:s+CH] = net.coverage(net.prot_gate(X), gt).cpu().numpy().astype(np.float32)
    np.savez_compressed(OUT / f"scores_{ARM}_{asp}.npz", proteins=np.array(prots),
                        terms=np.array(terms), score=score)
    log(f"{asp}: scored {P} prots x {n_terms} terms")
log(f"SSE EV SCORE DONE arm={ARM}")
