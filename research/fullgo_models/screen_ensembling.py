"""Cheap, self-contained classifier-level screen: Deep-Ensemble (K seeds,
output-averaged) vs EMA vs SWA, WITHOUT the full export+booster+predict
pipeline and WITHOUT any offline scratch (the old select_cv inputs are gone).

It reuses the from-zero runbook (build_labels / build_matrix / train_seed),
splits the v227 training proteins 50/50 by a stable hash, trains each mode on
split A, and evaluates label-ranking quality on the HELD-OUT split B via
micro-averaged average precision over the propagated TOI labels. This is the
gate that decides whether EMA/SWA are worth the ~3-4h end-to-end re-export to
fold into /benchmark. It is a proxy (classifier-level, not the end f_micro_w),
but it is leakage-clean (B never seen in training) and self-contained.

Run AFTER the parity export frees the GPU:
  cd repositories/PROTEA && .venv/bin/python \
    ../../storage/fullgo_models/screen_ensembling.py --seeds 0,7,137,23,91,31,53
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score

# Load the runbook module by path (it lives in storage/, not on sys.path).
_SPEC = importlib.util.spec_from_file_location(
    "rebuild_seed_checkpoints",
    "/home/frapercan/Thesis2/storage/fullgo_models/rebuild_seed_checkpoints.py",
)
rb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rb)


def _split_mask(accs: list[str], frac_a: float = 0.5) -> np.ndarray:
    """Stable 50/50 protein split by md5 hash (B = held-out)."""
    h = np.array([int(hashlib.md5(a.encode()).hexdigest(), 16) % 1000 for a in accs])
    return h < int(frac_a * 1000)  # True = split A (train)


def _scores_for(ckpt: dict, xn_b: torch.Tensor, lt: torch.Tensor, dev: str) -> np.ndarray:
    model = rb.build_hybrid(rb.IN_DIM, rb.HIDDEN, len(ckpt["vocab"]), int(ckpt["label_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, xn_b.shape[0], 512):
            out.append(torch.sigmoid(model(xn_b[i : i + 512].to(dev), lt)).cpu().numpy())
    return np.vstack(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0,7,137,23,91,31,53")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True

    tr_acc, vocab, rows, cols = rb.build_labels()
    x = rb.build_matrix(tr_acc)
    xt = torch.tensor(x, dtype=torch.float32)
    del x

    mask_a = _split_mask(tr_acc)
    a_idx = np.where(mask_a)[0]
    b_idx = np.where(~mask_a)[0]
    print(f"split: train(A)={len(a_idx)} heldout(B)={len(b_idx)}", flush=True)

    # Standardise with A statistics only (no leakage from B).
    mu = xt[a_idx].mean(0, keepdim=True)
    sd = xt[a_idx].std(0, keepdim=True) + 1e-6
    xn = (xt - mu) / sd
    del xt
    xn_a, xn_b = xn[a_idx], xn[b_idx]

    full = csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(len(tr_acc), len(vocab))
    )
    y_a = full[a_idx]
    y_b = full[b_idx].toarray()
    lm = rb._label_matrix_for_vocab(
        list(vocab), rb.ANC, int(np.load(rb.ANC, allow_pickle=True)["embeddings"].shape[1])
    )
    lt = torch.tensor(lm, dtype=torch.float32, device=dev)

    # keep only label columns that appear in B (AP undefined for all-zero columns)
    col_keep = np.where(y_b.sum(0) > 0)[0]

    def micro_ap(scores: np.ndarray) -> float:
        return float(average_precision_score(y_b[:, col_keep].ravel(), scores[:, col_keep].ravel()))

    results = {}
    # Deep Ensemble: train K members on A, average their B-probabilities.
    ens = np.zeros((len(b_idx), len(vocab)), np.float32)
    for s in seeds:
        ck = rb.train_seed(s, xn_a, y_a, lt, mu, sd, vocab, dev, mode="ensemble")
        ens += _scores_for(ck, xn_b, lt, dev)
    results[f"deep_ensemble_{len(seeds)}seed"] = micro_ap(ens / len(seeds))

    for mode in ("ema", "swa"):
        ck = rb.train_seed(seeds[0], xn_a, y_a, lt, mu, sd, vocab, dev, mode=mode)
        results[mode] = micro_ap(_scores_for(ck, xn_b, lt, dev))

    print("\n=== held-out micro-AP (higher = better) ===", flush=True)
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v:.4f}", flush=True)
    base = results[f"deep_ensemble_{len(seeds)}seed"]
    for m in ("ema", "swa"):
        d = results[m] - base
        print(f"  {m} vs deep_ensemble: {d:+.4f}  -> {'COMPETITIVE' if d > -0.003 else 'WORSE'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
