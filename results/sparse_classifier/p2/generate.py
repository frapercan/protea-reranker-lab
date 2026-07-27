"""Reusable top-100 GO candidate generator for the sparse functional two-tower model.

generate_top100(protein_codes, accs) -> {acc: [(go_id, score) x 100]}

Loads the 7-seed projection-head ensemble from p2/head_seed*.pt and the frozen GO vocab codes
from p2/prep.npz. Scores = averaged ensemble logits over the GO vocab; top-100 per protein.
No live DB; pure file -> file. This is the candidate generator that feeds the reranker (P4).

CLI:  python generate.py [--codes path.npz] [--out top100.json] [--k 100] [--split test]
  --codes : npz with keys `accs` (str) and `codes` (Nx2048); defaults to the champion clf codes.
  --split : if 'test'/'train', restrict to that protein split from prep.npz (default: all).
"""
import os, sys, json, glob, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from two_tower import ProjHead, DEV

P2 = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(P2)


def load_ensemble(p2=P2):
    prep = np.load(f"{p2}/prep.npz", allow_pickle=True)
    GO_V = torch.from_numpy(prep["GO_V"].astype(np.float32)).to(DEV)   # V x 1024
    vocab_go = [str(x) for x in prep["vocab_go"]]
    models = []
    for ck in sorted(glob.glob(f"{p2}/head_seed*.pt")):
        d = torch.load(ck, map_location=DEV)
        m = ProjHead(2048, 1024, V=d["V"], hidden=d["cfg"]["hidden"],
                     kwta=d["cfg"]["kwta"], dropout=0.0).to(DEV)
        m.load_state_dict(d["state_dict"]); m.eval()
        models.append(m)
    if not models:
        raise FileNotFoundError(f"no head_seed*.pt in {p2}; train first via run_full.py")
    return models, GO_V, vocab_go


@torch.no_grad()
def generate_top100(protein_codes, accs, models=None, GO_V=None, vocab_go=None, k=100, bs=512):
    """protein_codes: (N,2048) np array/tensor; accs: list[str]. Returns {acc:[(go,score)*k]}."""
    if models is None:
        models, GO_V, vocab_go = load_ensemble()
    GO_T = GO_V.t().contiguous()
    X = torch.as_tensor(np.asarray(protein_codes, dtype=np.float32), device=DEV)
    out = {}
    for s in range(0, len(accs), bs):
        x = X[s:s + bs]
        scores = None
        for m in models:
            scores = m(x, GO_T) if scores is None else scores + m(x, GO_T)
        scores = scores / len(models)
        vals, idx = scores.topk(k, dim=1)
        vals = vals.cpu().numpy(); idx = idx.cpu().numpy()
        for b in range(len(x)):
            out[accs[s + b]] = [(vocab_go[idx[b, j]], float(vals[b, j])) for j in range(k)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=f"{BASE}/clf_protein_codes.npz")
    ap.add_argument("--out", default=f"{P2}/top100.json")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--split", default="all", choices=["all", "test", "train"])
    a = ap.parse_args()
    p = np.load(a.codes, allow_pickle=True)
    accs = [str(x) for x in p["accs"]]; codes = p["codes"].astype(np.float32)
    if a.split != "all":
        prep = np.load(f"{P2}/prep.npz", allow_pickle=True)
        sel = prep["test_idx"] if a.split == "test" else prep["train_idx"]
        accs = [accs[i] for i in sel]; codes = codes[sel]
    models, GO_V, vocab_go = load_ensemble()
    res = generate_top100(codes, accs, models, GO_V, vocab_go, k=a.k)
    json.dump(res, open(a.out, "w"))
    print(f"wrote {len(res)} proteins x top-{a.k} -> {a.out}")


if __name__ == "__main__":
    main()
