"""Score the trained per-aspect SSE ensembles.

Builds, per aspect, the ensemble coverage over (parquet-protein-universe x aspect-vocab):
  - covmin_{asp}.npy  float16  (P x T)  MIN-over-seeds soft coverage  (primary consensus)
  - covmean_{asp}.npy float16  (P x T)  MEAN-over-seeds soft coverage
  - covuniverse_{asp}.npz  proteins (P,), terms (T,)
From the same tensors it slices Mode-A eval scores:
  - scoresA_{CELL}_{asp}.npz  proteins, terms, avg, min  (eval LK/PK targets)

Protein universe = union of train.parquet + eval.parquet proteins that are in the ESM corpus.
Run under PROTEA/.venv (torch+CUDA).
"""
import json, time
from pathlib import Path
import numpy as np
import torch as th
from torch import nn
import pyarrow.parquet as pq
import pandas as pd

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
GF = ROOT / "storage/cooc_experiment/generator_frames"
OUT = ROOT / "storage/sse_full"; MDIR = OUT / "models"
RR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank"
GTDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
DEV = "cuda:0"
N, PK, TEMP = 2048, 128, 0.3
K_SEEDS = 5
ASPECTS = ["mfo", "bpo", "cco"]

def _thr_perrow(a, ks):
    srt = a.sort(dim=1, descending=True).values
    gi = (ks - 1).clamp(0, a.shape[1] - 1).unsqueeze(1)
    return srt.gather(1, gi).detach()
def soft_kwta_perrow(a, ks): return th.sigmoid((a - _thr_perrow(a, ks)) / TEMP)
def soft_kwta_scalar(a, k):
    thr = a.topk(k, dim=1).values[:, -1:].detach(); return th.sigmoid((a - thr) / TEMP)

class SSE(nn.Module):
    def __init__(self, in_dim, n_terms, n_rels, MTt):
        super().__init__(); self.MTt = MTt
        self.prot = nn.Sequential(nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(1024, N))
        self.pln = nn.LayerNorm(N)
        self.term = nn.Embedding(n_terms, N); self.tln = nn.LayerNorm(N)
        self.rel = nn.Parameter(th.zeros(n_rels, N))
    def prot_gate(self, x): return soft_kwta_scalar(self.pln(self.prot(x)), PK)
    def term_gate(self): return soft_kwta_perrow(self.tln(self.term.weight), self.MTt)
    def coverage(self, gp, gt): return (gp @ gt.t()) / (gt.sum(1) + 1e-6)

# ---- protein universe from parquets ∩ corpus ----
accs = json.load(open(GF / "accs.json")); acc_idx = {a: i for i, a in enumerate(accs)}
EMB = np.load(GF / "raw_esm2_3b.npy", mmap_mode="r")
uni = set()
for pqf in [RR / "train.parquet", RR / "eval.parquet"]:
    tp = pq.read_table(pqf, columns=["protein_accession"]).to_pandas()["protein_accession"]
    uni |= set(tp.unique())
universe = sorted(a for a in uni if a in acc_idx)
log(f"protein universe (parquet ∩ corpus): {len(universe)}")
uni_rows = np.array([acc_idx[a] for a in universe])
uni_pos = {a: i for i, a in enumerate(universe)}

# eval targets per cell (for Mode A slices)
evtargets = {}
for cell, fn in [("LK", "groundtruth_LK.tsv"), ("PK", "groundtruth_PK.tsv")]:
    gt = pd.read_csv(GTDIR / fn, sep="\t")
    evtargets[cell] = {asp: sorted(set(gt[gt.aspect == asp[0].upper()].EntryID.unique()) & set(uni_pos))
                       for asp in [("mfo",), ("bpo",), ("cco",)]}  # placeholder, fixed below
# proper mapping aspect letter
ASPLET = {"mfo": "F", "bpo": "P", "cco": "C"}
evtargets = {}
for cell, fn in [("LK", "groundtruth_LK.tsv"), ("PK", "groundtruth_PK.tsv")]:
    gt = pd.read_csv(GTDIR / fn, sep="\t")
    evtargets[cell] = {asp: sorted(set(gt[gt.aspect == ASPLET[asp]].EntryID.unique()) & set(uni_pos))
                       for asp in ASPECTS}

for asp in ASPECTS:
    mp = OUT / f"{asp}_meta.npz"
    if not mp.exists():
        log(f"SKIP {asp}: meta missing"); continue
    meta = np.load(mp, allow_pickle=True)
    terms = meta["terms"].tolist(); M_T = meta["M_T"]; mu = meta["mu"]; sd = meta["sd"]
    n_rels = int(meta["n_rels"][0]); n_terms = len(terms)
    MTt = th.tensor(M_T, device=DEV)
    log(f"{asp}: universe {len(universe)} x terms {n_terms}")
    nets = []
    for seed in range(K_SEEDS):
        net = SSE(2560, n_terms, n_rels, MTt).to(DEV)
        net.load_state_dict(th.load(MDIR / f"{asp}_{seed}.th")); net.eval(); nets.append(net)
    # precompute term gates per seed
    with th.no_grad():
        gts = [net.term_gate() for net in nets]
    P = len(universe)
    covmin = np.empty((P, n_terms), dtype=np.float16)
    covmean = np.empty((P, n_terms), dtype=np.float16)
    CH = 4096
    mu_t = th.tensor(mu, device=DEV); sd_t = th.tensor(sd, device=DEV)
    with th.no_grad():
        for s in range(0, P, CH):
            rows = uni_rows[s:s + CH]
            X = th.tensor(np.ascontiguousarray(EMB[rows].astype(np.float32)), device=DEV)
            X = (X - mu_t) / sd_t
            per = []
            for net, gt in zip(nets, gts):
                gp = net.prot_gate(X)
                per.append(net.coverage(gp, gt))            # (chunk, T)
            st = th.stack(per)                               # (K, chunk, T)
            covmin[s:s + CH] = st.min(0).values.cpu().numpy().astype(np.float16)
            covmean[s:s + CH] = st.mean(0).cpu().numpy().astype(np.float16)
            if (s // CH) % 5 == 0: log(f"  {asp} chunk {s}/{P}")
    np.save(OUT / f"covmin_{asp}.npy", covmin)
    np.save(OUT / f"covmean_{asp}.npy", covmean)
    np.savez(OUT / f"covuniverse_{asp}.npz", proteins=np.array(universe), terms=np.array(terms))
    log(f"{asp}: saved covmin/covmean {covmin.shape}")

    # Mode A eval slices
    for cell in ["LK", "PK"]:
        tg = evtargets[cell][asp]
        if not tg: continue
        idx = np.array([uni_pos[a] for a in tg])
        np.savez_compressed(OUT / f"scoresA_{cell}_{asp}.npz", proteins=np.array(tg),
                            terms=np.array(terms), min=covmin[idx].astype(np.float32),
                            avg=covmean[idx].astype(np.float32))
        log(f"  {asp} {cell}: Mode-A slice {len(tg)} prots saved")

log("SSE FULL SCORE DONE")
