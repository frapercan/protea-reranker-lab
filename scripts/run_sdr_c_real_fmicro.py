"""Does a LEARNED-REAL sparse code beat dense on the TASK (k-NN GO-transfer f_micro)?

The learned BINARY code won Spearman but LOST f_micro vs dense (binarisation kills top-K
resolution). The real-valued sparse code keeps the magnitudes -> finer top-K -> should retrieve
better. This trains a learned encoder with the cosine-real objective (cosine(z) ~ Lin), then
evaluates k-NN GO-transfer micro-Fmax on the REAL code (full + top-k sparse), vs dense. The only
remaining route to a PRECISION lever on the task. Read-only DB. MLflow.
"""
from __future__ import annotations

import argparse, logging, sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sdr_chunk_correlation as base  # noqa: E402
from protea_reranker_lab.sdr import (  # noqa: E402
    GoDag, propagate, information_content, resnik_pairwise, lin_pairwise, cosine_dense,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [real-fmicro] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("real-fmicro")
torch.manual_seed(42)


def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def topk_real(X, k):
    if k >= X.shape[1]:
        return X.copy()
    out = np.zeros_like(X)
    idx = np.argpartition(-np.abs(X), k, axis=1)[:, :k]
    np.put_along_axis(out, idx, np.take_along_axis(X, idx, axis=1), axis=1)
    return out


def sample_pairs(n, np_, rng):
    seen, pairs = set(), []
    while len(pairs) < np_:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i == j: continue
        key = (i, j) if i < j else (j, i)
        if key in seen: continue
        seen.add(key); pairs.append(key)
    return pairs


def fmicro_from_S(S, closures, knn=30):
    N = S.shape[0]
    nbr = np.argpartition(-S, knn, axis=1)[:, :knn]
    rows = np.repeat(np.arange(N), knn); cols = nbr.ravel()
    w = S[rows, cols].copy(); w[w < 0] = 0
    Sk = sp.csr_matrix((w, (rows, cols)), shape=(N, N), dtype=np.float32)
    terms = sorted({t for c in closures for t in c}); tix = {t: i for i, t in enumerate(terms)}
    r, c = [], []
    for i, cl in enumerate(closures):
        for t in cl:
            r.append(i); c.append(tix[t])
    T = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(N, len(terms)))
    pred = np.asarray((Sk @ T).todense()).ravel(); truth = np.asarray(T.todense()).astype(bool).ravel()
    npos = int(truth.sum()); order = np.argsort(-pred, kind="stable")
    tp = np.cumsum(truth[order]); fp = np.cumsum(~truth[order])
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(npos, 1)
    f = 2 * prec * rec / np.maximum(prec + rec, 1e-12); valid = pred[order] > 0
    return float(np.max(f[valid])) if valid.any() else 0.0


def cos_S(X):
    Xn = l2n(X); S = Xn @ Xn.T; np.fill_diagonal(S, -1.0); return S


def transfer_pred(R, T, knn=30):
    S = cos_S(R); N = R.shape[0]
    nbr = np.argpartition(-S, knn, axis=1)[:, :knn]
    rows = np.repeat(np.arange(N), knn); cols = nbr.ravel()
    w = S[rows, cols].copy(); w[w < 0] = 0
    Sk = sp.csr_matrix((w, (rows, cols)), shape=(N, N), dtype=np.float32)
    return np.asarray((Sk @ T).todense())


def fmicro_rows(pred, Td, mask):
    p = pred[mask].ravel(); t = Td[mask].ravel(); npos = int(t.sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    tp = np.cumsum(t[order]); fp = np.cumsum(~t[order])
    prec = tp / np.maximum(tp + fp, 1); rec = tp / npos
    f = 2 * prec * rec / np.maximum(prec + rec, 1e-12); valid = p[order] > 0
    return float(np.max(f[valid])) if valid.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="host=localhost dbname=protea user=protea password=protea")
    ap.add_argument("--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
    ap.add_argument("--n-proteins", type=int, default=10000)
    ap.add_argument("--dict", type=int, default=2048)
    ap.add_argument("--k", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-pairs", type=int, default=300_000)
    ap.add_argument("--eval-pairs", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    acc, acc_chunks, leaves = base.load_chunked(args.dsn, args.n_proteins, args.seed)
    dag = GoDag.from_obo(Path(args.obo).expanduser())
    closures = [propagate(leaves[a], dag) for a in acc]
    keep = [i for i, c in enumerate(closures) if c]
    acc = [acc[i] for i in keep]; closures = [closures[i] for i in keep]
    dense = l2n(base.dense_mean(acc_chunks, acc)); n, d = dense.shape
    perm = rng.permutation(n); n_tr = n // 2
    tr, ev = perm[:n_tr], perm[n_tr:]
    Xtr = torch.tensor(dense[tr]); Xev = torch.tensor(dense[ev])
    clo_tr = [closures[i] for i in tr]; clo_ev = [closures[i] for i in ev]
    log.info("n=%d train=%d eval=%d", n, len(tr), len(ev))

    ic_tr = information_content(clo_tr, dag)
    bic_tr = [max((ic_tr.get(t, 0.0) for t in c), default=0.0) for c in clo_tr]
    tp = sample_pairs(len(tr), args.train_pairs, rng)
    Y = torch.tensor(np.asarray(lin_pairwise(clo_tr, ic_tr, tp, bic_tr), dtype=np.float32))
    TI = torch.tensor([p[0] for p in tp]); TJ = torch.tensor([p[1] for p in tp])

    enc = nn.Linear(d, args.dict); opt = torch.optim.Adam(enc.parameters(), lr=args.lr)
    bs = 16384; np_ = len(tp)

    # held-out baselines precomputed once
    De = base.dense_mean(acc_chunks, [acc[i] for i in ev])
    f_dense = fmicro_from_S(cos_S(De), clo_ev)
    ep = sample_pairs(len(ev), args.eval_pairs, rng)
    ii = np.array([p[0] for p in ep]); jj = np.array([p[1] for p in ep])
    go_r = np.asarray(resnik_pairwise(clo_ev, information_content(clo_ev, dag), ep))
    sp_dense = float(spearmanr(cosine_dense(De)[ii, jj], go_r)[0])
    eval_k = args.k[0]; EVAL_EVERY = 15; PATIENCE = 6
    log.info("dense baseline f_micro=%.4f Spearman=%.4f | training to CONVERGENCE (early-stop on f_micro, patience %d)",
             f_dense, sp_dense, PATIENCE)

    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("sdr-c-real-fmicro")
    mlflow.start_run(run_name="learned-real-vs-dense-fmicro")
    mlflow.log_params({"n": n, "dict": args.dict, "max_epochs": args.epochs, "seed": args.seed,
                       "eval_k": eval_k, "patience": PATIENCE})
    mlflow.log_metric("f_dense", f_dense)

    best_f, best_ep, since = -1.0, -1, 0
    for e in range(args.epochs):
        Z = enc(Xtr); opt.zero_grad()
        for b in range(0, np_, bs):
            sl = slice(b, b + bs)
            zi, zj = Z[TI[sl]], Z[TJ[sl]]
            cos = (zi * zj).sum(1) / (zi.norm(dim=1) * zj.norm(dim=1) + 1e-8)
            loss = ((cos - Y[sl]) ** 2).sum() / np_
            loss.backward(retain_graph=True)
        opt.step()
        if e % EVAL_EVERY == 0 or e == args.epochs - 1:
            with torch.no_grad():
                Zev = enc(Xev).numpy().astype(np.float32)
            f_sp = fmicro_from_S(cos_S(topk_real(Zev, eval_k)), clo_ev)
            mlflow.log_metric("fmicro_learned_real", f_sp, step=e)
            log.info("  epoch %3d | learned-real(k=%d) f_micro=%.4f vs dense %.4f (%+.4f)",
                     e, eval_k, f_sp, f_dense, f_sp - f_dense)
            if f_sp > best_f + 1e-4:
                best_f, best_ep, since = f_sp, e, 0
            else:
                since += 1
                if since >= PATIENCE:
                    log.info("  PLATEAU at epoch %d (best f_micro=%.4f @ %d)", e, best_f, best_ep)
                    break

    with torch.no_grad():
        Zev = enc(Xev).numpy().astype(np.float32)
    f_lr_full = fmicro_from_S(cos_S(Zev), clo_ev)
    sp_lr = float(spearmanr(cosine_dense(Zev)[ii, jj], go_r)[0])
    log.info("=== f_micro (k-NN GO-transfer, held-out, CONVERGED) ===")
    log.info("  dense                      f_micro=%.4f  (Spearman %.4f)", f_dense, sp_dense)
    log.info("  learned-real FULL (%d-d)   f_micro=%.4f  (Spearman %.4f)", args.dict, f_lr_full, sp_lr)
    res = {"dense": f_dense, "lr_full": f_lr_full}
    for k in args.k:
        f_sp = fmicro_from_S(cos_S(topk_real(Zev, k)), clo_ev)
        res[f"lr_sparse_k{k}"] = f_sp
        log.info("  learned-real SPARSE k=%-4d  f_micro=%.4f  (%+.4f vs dense)", k, f_sp, f_sp - f_dense)
    # H5 (learned): f_micro stratified by protein size, dense vs learned-real
    nch_ev = np.array([acc_chunks[acc[i]].shape[0] for i in ev])
    terms = sorted({t for c in clo_ev for t in c}); tix = {t: i for i, t in enumerate(terms)}
    rr, cc = [], []
    for i, cl in enumerate(clo_ev):
        for t in cl:
            rr.append(i); cc.append(tix[t])
    Tev = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(len(ev), len(terms)))
    Td = np.asarray(Tev.todense()).astype(bool)
    bestk = int(max(args.k, key=lambda k: res[f"lr_sparse_k{k}"]))
    pred_d = transfer_pred(De, Tev); pred_l = transfer_pred(topk_real(Zev, bestk), Tev)
    log.info("=== H5 (LEARNED): f_micro by size, dense vs learned-real(k=%d) ===", bestk)
    for b, m in {"1-chunk": nch_ev == 1, "2-chunk": nch_ev == 2, "3+chunk": nch_ev >= 3}.items():
        fd = fmicro_rows(pred_d, Td, m); fl = fmicro_rows(pred_l, Td, m)
        log.info("  %-9s dense=%.4f learned-real=%.4f delta=%+.4f", b, fd, fl, fl - fd)
        mlflow.log_metric(("strat_dense_" + b).replace("+", "p").replace("-", "_"), fd)
        mlflow.log_metric(("strat_learned_" + b).replace("+", "p").replace("-", "_"), fl)

    for kk, vv in res.items():
        mlflow.log_metric("final_fmicro_" + kk, vv)
    mlflow.log_metric("best_fmicro_learned_real", best_f)
    mlflow.end_run()
    best = max((k for k in res if k != "dense"), key=lambda k: res[k])
    log.info(">> VERDICT: best learned-real = %s f_micro=%.4f vs dense %.4f (%+.4f) | peak-during-train %.4f@%d",
             best, res[best], f_dense, res[best] - f_dense, best_f, best_ep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
