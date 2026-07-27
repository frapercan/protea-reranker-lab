"""Per-cut temporally-honest joint scorer using the trained GCN label encoder.

score(prot, term @ cut) = mean_seed[ temp_s * <proj_s(prot_code), E_s^cut[term]> + bias_s[term] ]
where E_s^cut = LabelGCN_s(per-cut fused GO codes, DAG) -- the DAG-aware term embeddings
built from THAT cut's co-annotation block (per_cut/go_sparse_codes_v{N}.npz, frozen v227
SVD basis). Protein codes are the frozen d8979601 k-WTA codes (time-independent). So the
only per-cut variation is the GO co-annotation INPUT -- no future-snapshot code touches a
row, the Phase 0 contract.

Public API:
  S = GcnScorer()
  scores = S.score_rows(accs, gos, t0_cuts)      # np arrays, returns float32 (NaN if oov)
  df = S.score_frame(df)                          # df has columns acc, go, snapshot_pair
"""
import os, sys, glob, re
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SC, "p2"))
sys.path.insert(0, HERE)
from two_tower import ProjHead, DEV  # noqa: E402
from label_gcn import LabelGCN, build_norm_adj, load_init_aligned  # noqa: E402

PREP = os.path.join(SC, "p2", "prep.npz")
PROT_CODES = os.path.join(SC, "clf_protein_codes.npz")
PER_CUT = "/home/frapercan/Thesis2/storage/two_tower_sparse/per_cut"


def t0_of(snapshot_pair):
    """'v160-v165' -> 160."""
    return int(re.match(r"v(\d+)-v(\d+)", snapshot_pair).group(1))


class GcnScorer:
    def __init__(self, ckpt_glob=None, chunk=200_000):
        prep = np.load(PREP, allow_pickle=True)
        self.vocab_go = [str(x) for x in prep["vocab_go"]]
        self.go2idx = {g: i for i, g in enumerate(self.vocab_go)}
        self.V = len(self.vocab_go)
        self.A = build_norm_adj(prep["edges"], self.V, DEV)
        self.chunk = chunk

        p = np.load(PROT_CODES, allow_pickle=True)
        self.accs = [str(a) for a in p["accs"]]
        self.acc2idx = {a: i for i, a in enumerate(self.accs)}
        codes = torch.from_numpy(p["codes"].astype(np.float32)).to(DEV)  # N x 2048

        ckpts = sorted(glob.glob(ckpt_glob or os.path.join(HERE, "gcn_seed*.pt")))
        if not ckpts:
            raise FileNotFoundError("no gcn_seed*.pt; train_gcn.py full first")
        self.seeds = []
        for ck in ckpts:
            d = torch.load(ck, map_location=DEV)
            cfg = d["cfg"]
            proj = ProjHead(2048, 1024, V=d["V"], hidden=cfg["hidden"],
                            kwta=cfg["kwta"], dropout=0.0).to(DEV)
            proj.load_state_dict(d["proj"]); proj.eval()
            gcn = LabelGCN(d_in=1024, d_hid=cfg["gcn_hid"], d_out=1024,
                           layers=cfg["gcn_layers"], dropout=0.0).to(DEV)
            gcn.load_state_dict(d["gcn"]); gcn.eval()
            with torch.no_grad():
                proj_code = proj.project(codes)                       # N x 1024
            self.seeds.append({
                "proj_code": proj_code,
                "gcn": gcn,
                "temp": float(proj.log_temp.exp().item()),
                "bias": proj.bias.detach() if proj.bias is not None else None,
            })
        del codes

    @torch.no_grad()
    def _embs_for_cut(self, cut):
        """All-seed DAG-aware term embeddings for one cut (V x 1024 each, on DEV)."""
        init = load_init_aligned(os.path.join(PER_CUT, f"go_sparse_codes_v{cut}.npz"),
                                 self.vocab_go)
        init_t = torch.from_numpy(init).to(DEV)
        return [sd["gcn"](init_t, self.A) for sd in self.seeds]

    @torch.no_grad()
    def score_rows(self, accs, gos, t0_cuts):
        """accs, gos: str arrays; t0_cuts: int array. Returns float32 (NaN where oov)."""
        n = len(accs)
        ai = np.array([self.acc2idx.get(a, -1) for a in accs], dtype=np.int64)
        gi = np.array([self.go2idx.get(g, -1) for g in gos], dtype=np.int64)
        out = np.full(n, np.nan, dtype=np.float32)
        valid = (ai >= 0) & (gi >= 0)
        cuts = np.asarray(t0_cuts)
        for cut in np.unique(cuts):
            m = valid & (cuts == cut)
            idxs = np.nonzero(m)[0]
            if not len(idxs):
                continue
            embs = self._embs_for_cut(int(cut))            # per-seed, this cut only
            ai_c = torch.from_numpy(ai[idxs]).to(DEV)
            gi_c = torch.from_numpy(gi[idxs]).to(DEV)
            acc_s = torch.zeros(len(idxs), device=DEV)
            for ci in range(0, len(idxs), self.chunk):
                a = ai_c[ci:ci + self.chunk]; g = gi_c[ci:ci + self.chunk]
                ssum = torch.zeros(len(a), device=DEV)
                for si, sd in enumerate(self.seeds):
                    pc = sd["proj_code"][a]                 # chunk x 1024
                    ge = embs[si][g]                        # chunk x 1024
                    s = sd["temp"] * (pc * ge).sum(1)
                    if sd["bias"] is not None:
                        s = s + sd["bias"][g]
                    ssum = ssum + s
                acc_s[ci:ci + self.chunk] = ssum / len(self.seeds)
            out[idxs] = acc_s.cpu().numpy().astype(np.float32)
            del embs, ai_c, gi_c, acc_s
            torch.cuda.empty_cache()
        return out

    def score_frame(self, df, acc_col="acc", go_col="go", pair_col="snapshot_pair"):
        cuts = df[pair_col].map(t0_of).to_numpy()
        return self.score_rows(df[acc_col].to_numpy().astype(str),
                               df[go_col].to_numpy().astype(str), cuts)
