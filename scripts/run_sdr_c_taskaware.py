"""SDR grid: task-aware objective vs pure-geometry objective.

Both train the same learned sparse-binary encoder (A0, mean-pooled). The GEOMETRY objective
regresses code soft-Tanimoto to Lin GO-sim over random pairs (preserves the whole geometry, what
Spearman measures). The TASK-AWARE objective keeps that Lin regression (geometry stays imprescindible)
but REWEIGHTS it to focus capacity where retrieval/Fmax cares:
  * high-IC POSITIVES up-weighted (high-Lin pairs share specific terms -> the valuable transfers),
  * HARD NEGATIVES up-weighted (high dense-cosine but low GO-sim = false neighbours that kill PK precision).
Both are evaluated on held-out with BOTH metrics: intrinsic Spearman (Resnik/Lin) AND extrinsic
k-NN GO-transfer micro-Fmax. The question: does aligning the objective to the task raise f_micro
beyond the pure-geometry objective? Read-only DB. MLflow.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sdr_chunk_correlation as base  # noqa: E402
from protea_reranker_lab.sdr import (  # noqa: E402
    GoDag,
    propagate,
    information_content,
    resnik_pairwise,
    lin_pairwise,
    cosine_dense,
    tanimoto_dense,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [taskaware] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("taskaware")
torch.manual_seed(42)


def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def topk_bin(Z, k):
    out = np.zeros(Z.shape, dtype=np.uint8)
    idx = np.argpartition(-Z, min(k, Z.shape[1] - 1), axis=1)[:, :k]
    np.put_along_axis(out, idx, 1, axis=1)
    return out


def sample_pairs(n, n_pairs, rng):
    seen, pairs = set(), []
    while len(pairs) < n_pairs:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


# ---- extrinsic: leave-one-out k-NN GO-transfer micro-Fmax on a binary-code matrix ----
def knn_fmicro(Bin, closures, knn=30):
    R = Bin.astype(np.float32)
    rn = np.linalg.norm(R, axis=1, keepdims=True)
    rn[rn == 0] = 1
    R = R / rn
    S = R @ R.T
    np.fill_diagonal(S, -1.0)
    N = R.shape[0]
    nbr = np.argpartition(-S, knn, axis=1)[:, :knn]
    rows = np.repeat(np.arange(N), knn)
    cols = nbr.ravel()
    w = S[rows, cols].copy()
    w[w < 0] = 0
    Sk = sp.csr_matrix((w, (rows, cols)), shape=(N, N), dtype=np.float32)
    terms = sorted({t for c in closures for t in c})
    tix = {t: i for i, t in enumerate(terms)}
    r, c = [], []
    for i, cl in enumerate(closures):
        for t in cl:
            r.append(i)
            c.append(tix[t])
    T = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(N, len(terms)))
    pred = np.asarray((Sk @ T).todense()).ravel()
    truth = np.asarray(T.todense()).astype(bool).ravel()
    npos = int(truth.sum())
    order = np.argsort(-pred, kind="stable")
    tp = np.cumsum(truth[order])
    fp = np.cumsum(~truth[order])
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(npos, 1)
    f = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    valid = pred[order] > 0
    return float(np.max(f[valid])) if valid.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dsn", default="host=localhost dbname=protea user=protea password=protea"
    )
    ap.add_argument(
        "--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
    )
    ap.add_argument("--n-proteins", type=int, default=10000)
    ap.add_argument("--dict", type=int, default=2048)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-pairs", type=int, default=300_000)
    ap.add_argument("--eval-pairs", type=int, default=200_000)
    ap.add_argument("--pos-boost", type=float, default=3.0)
    ap.add_argument("--neg-boost", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    acc, acc_chunks, leaves = base.load_chunked(args.dsn, args.n_proteins, args.seed)
    dag = GoDag.from_obo(Path(args.obo).expanduser())
    closures = [propagate(leaves[a], dag) for a in acc]
    keep = [i for i, c in enumerate(closures) if c]
    acc = [acc[i] for i in keep]
    closures = [closures[i] for i in keep]
    dense = l2n(base.dense_mean(acc_chunks, acc))
    n, d = dense.shape
    perm = rng.permutation(n)
    n_tr = n // 2
    tr, ev = perm[:n_tr], perm[n_tr:]
    Xtr = torch.tensor(dense[tr])
    Xev = torch.tensor(dense[ev])
    clo_tr = [closures[i] for i in tr]
    clo_ev = [closures[i] for i in ev]
    log.info("n=%d d=%d  train=%d eval=%d", n, d, len(tr), len(ev))

    # train pairs: Lin target + dense-cosine + task-aware weights
    ic_tr = information_content(clo_tr, dag)
    bic_tr = [max((ic_tr.get(t, 0.0) for t in c), default=0.0) for c in clo_tr]
    tp = sample_pairs(len(tr), args.train_pairs, rng)
    ti = np.array([p[0] for p in tp])
    tj = np.array([p[1] for p in tp])
    lin = np.asarray(lin_pairwise(clo_tr, ic_tr, tp, bic_tr), dtype=np.float32)
    dcos = (l2n(dense[tr]) @ l2n(dense[tr]).T)[ti, tj].astype(np.float32)
    c_hi = np.quantile(dcos, 0.90)
    l_lo = np.quantile(lin, 0.25)
    hardneg = (dcos > c_hi) & (lin < l_lo)
    w_task = (
        1.0
        + args.pos_boost * (lin / max(lin.max(), 1e-6))
        + args.neg_boost * hardneg.astype(np.float32)
    )
    log.info(
        "hard negatives: %.1f%% of train pairs | weight range %.2f-%.2f",
        100 * hardneg.mean(),
        float(w_task.min()),
        float(w_task.max()),
    )
    Y = torch.tensor(lin)
    TI = torch.tensor(ti)
    TJ = torch.tensor(tj)
    W = {"geometry": torch.ones(len(tp)), "taskaware": torch.tensor(w_task)}

    # held-out eval setup
    ic_ev = information_content(clo_ev, dag)
    bic_ev = [max((ic_ev.get(t, 0.0) for t in c), default=0.0) for c in clo_ev]
    ep = sample_pairs(len(ev), args.eval_pairs, rng)
    ii = np.array([p[0] for p in ep])
    jj = np.array([p[1] for p in ep])
    GO = {
        "resnik": np.asarray(resnik_pairwise(clo_ev, ic_ev, ep), dtype=np.float64),
        "lin": np.asarray(lin_pairwise(clo_ev, ic_ev, ep, bic_ev), dtype=np.float64),
    }
    De = base.dense_mean(acc_chunks, [acc[i] for i in ev])
    dense_rho = {
        m: float(spearmanr(cosine_dense(De)[ii, jj], g)[0]) for m, g in GO.items()
    }
    _dense_fmicro = (
        knn_fmicro(
            (cosine_dense(De) > np.quantile(cosine_dense(De), 0.5)).astype(np.uint8),
            clo_ev,
        )
        if False
        else None
    )  # dense uses real cosine below

    # dense f_micro via real cosine kNN
    def dense_knn_fmicro():
        Rn = l2n(De)
        S = Rn @ Rn.T
        np.fill_diagonal(S, -1)
        return _fmicro_from_S(S, clo_ev)

    log.info(
        "baselines: dense Spearman resnik=%.4f lin=%.4f",
        dense_rho["resnik"],
        dense_rho["lin"],
    )

    def train(obj):
        enc = nn.Linear(d, args.dict)
        opt = torch.optim.Adam(enc.parameters(), lr=args.lr)
        bs = 16384
        ALPHA = 20.0
        w = W[obj]
        sw = float(w.sum())
        for e in range(args.epochs):
            Z = enc(Xtr)
            tau = torch.topk(Z, args.k, dim=1).values[:, -1:].detach()
            S = torch.sigmoid(ALPHA * (Z - tau))
            opt.zero_grad()
            for b in range(0, len(tp), bs):
                sl = slice(b, b + bs)
                si, sj = S[TI[sl]], S[TJ[sl]]
                inter = (si * sj).sum(1)
                sim = inter / (si.sum(1) + sj.sum(1) - inter + 1e-6)
                loss = (w[sl] * (sim - Y[sl]) ** 2).sum() / sw
                loss.backward(retain_graph=True)
            opt.step()
        with torch.no_grad():
            Zev = enc(Xev).numpy().astype(np.float32)
        return Zev

    import mlflow

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("sdr-c-taskaware")
    out = {}
    for obj in ("geometry", "taskaware"):
        log.info("=== training objective: %s ===", obj)
        Zev = train(obj)
        Bin = topk_bin(Zev, args.k)
        rho = {
            m: float(spearmanr(tanimoto_dense(Bin, args.k)[ii, jj], GO[m])[0])
            for m in GO
        }
        fm = knn_fmicro(Bin, clo_ev)
        out[obj] = {
            "spearman_resnik": rho["resnik"],
            "spearman_lin": rho["lin"],
            "fmicro": fm,
        }
        log.info(
            "  %s -> Spearman resnik=%.4f lin=%.4f | k-NN f_micro=%.4f",
            obj,
            rho["resnik"],
            rho["lin"],
            fm,
        )
        with mlflow.start_run(run_name=f"obj-{obj}"):
            mlflow.log_params(
                {
                    "objective": obj,
                    "k": args.k,
                    "dict": args.dict,
                    "epochs": args.epochs,
                    "pos_boost": args.pos_boost,
                    "neg_boost": args.neg_boost,
                }
            )
            for kk, vv in out[obj].items():
                mlflow.log_metric(kk, vv)
            mlflow.log_metric("dense_fmicro", dense_knn_fmicro())
            mlflow.log_metric("dense_spearman_resnik", dense_rho["resnik"])

    g, t = out["geometry"], out["taskaware"]
    log.info("=== VERDICT (held-out) ===")
    log.info("  dense f_micro=%.4f", dense_knn_fmicro())
    log.info(
        "  geometry : Spearman_r=%.4f f_micro=%.4f", g["spearman_resnik"], g["fmicro"]
    )
    log.info(
        "  taskaware: Spearman_r=%.4f f_micro=%.4f", t["spearman_resnik"], t["fmicro"]
    )
    log.info(
        "  >> task-aware vs geometry: dSpearman=%+.4f  dF_MICRO=%+.4f",
        t["spearman_resnik"] - g["spearman_resnik"],
        t["fmicro"] - g["fmicro"],
    )
    return 0


def _fmicro_from_S(S, closures, knn=30):
    N = S.shape[0]
    nbr = np.argpartition(-S, knn, axis=1)[:, :knn]
    rows = np.repeat(np.arange(N), knn)
    cols = nbr.ravel()
    w = S[rows, cols].copy()
    w[w < 0] = 0
    Sk = sp.csr_matrix((w, (rows, cols)), shape=(N, N), dtype=np.float32)
    terms = sorted({t for c in closures for t in c})
    tix = {t: i for i, t in enumerate(terms)}
    r, c = [], []
    for i, cl in enumerate(closures):
        for t in cl:
            r.append(i)
            c.append(tix[t])
    T = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(N, len(terms)))
    pred = np.asarray((Sk @ T).todense()).ravel()
    truth = np.asarray(T.todense()).astype(bool).ravel()
    npos = int(truth.sum())
    order = np.argsort(-pred, kind="stable")
    tp = np.cumsum(truth[order])
    fp = np.cumsum(~truth[order])
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(npos, 1)
    f = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    valid = pred[order] > 0
    return float(np.max(f[valid])) if valid.any() else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
