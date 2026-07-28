"""Rung 1 of the SDR lever-validation ladder: sparse-REAL vs dense k-NN GO-transfer.

Validates the EFFICIENCY lever. Leave-one-out over the v227 t0 pool: for each protein,
retrieve its top-K neighbours by cosine, transfer their propagated GO terms (similarity-
weighted vote), and score the transferred prediction against the protein's own propagated
terms with a micro-Fmax. Compare DENSE (full 768) vs SPARSE-REAL (top-k coords + magnitudes)
at several k. The decomposition says sparse-real ~= dense intrinsically; this asks whether
that holds on the actual term-transfer task, at a fraction of the index size.

Reuses the data loading + GO DAG of run_sdr_chunk_correlation.py. Read-only DB. MLflow.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sdr_chunk_correlation as base  # noqa: E402
from protea_reranker_lab.sdr import GoDag, propagate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [knn-transfer] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("knn-transfer")


def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def topk_real(X, k):
    """Keep top-k coords by |magnitude| per row (real values), zero the rest."""
    if k >= X.shape[1]:
        return X.copy()
    out = np.zeros_like(X)
    idx = np.argpartition(-np.abs(X), k, axis=1)[:, :k]
    np.put_along_axis(out, idx, np.take_along_axis(X, idx, axis=1), axis=1)
    return out


def transfer(R, T, knn):
    """R: (N,d) L2-normed. T: (N,n_terms) csr binary truth. Returns (N,n_terms) scores."""
    N = R.shape[0]
    S = (R @ R.T).astype(np.float32)
    np.fill_diagonal(S, -1.0)  # exclude self
    nbr = np.argpartition(-S, knn, axis=1)[:, :knn]
    rows = np.repeat(np.arange(N), knn)
    cols = nbr.ravel()
    w = S[rows, cols].copy()
    w[w < 0] = 0.0  # ignore anti-correlated neighbours
    Sk = sp.csr_matrix((w, (rows, cols)), shape=(N, N), dtype=np.float32)
    return Sk @ T  # similarity-weighted vote, (N, n_terms) sparse->dense on use


def micro_fmax(pred_scores, T):
    """Global micro-Fmax over all (protein, term) pairs across a threshold sweep."""
    pred = np.asarray(pred_scores.todense()).ravel() if sp.issparse(pred_scores) else pred_scores.ravel()
    truth = np.asarray(T.todense()).astype(bool).ravel()
    npos = int(truth.sum())
    order = np.argsort(-pred, kind="stable")
    p_sorted = pred[order]
    t_sorted = truth[order]
    tp = np.cumsum(t_sorted)
    fp = np.cumsum(~t_sorted)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(npos, 1)
    f = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    valid = p_sorted > 0  # do not credit predicting zero-score terms
    return float(np.max(f[valid])) if valid.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="host=localhost dbname=protea user=protea password=protea")
    ap.add_argument("--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
    ap.add_argument("--n-proteins", type=int, default=8000)
    ap.add_argument("--knn", type=int, default=30)
    ap.add_argument("--k-sdr", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log.info("loading %d proteins (v227 t0, chunked ankh)", args.n_proteins)
    acc, acc_chunks, leaves = base.load_chunked(args.dsn, args.n_proteins, args.seed)
    log.info("parsing GO DAG")
    dag = GoDag.from_obo(Path(args.obo).expanduser())
    closures = [propagate(leaves[a], dag) for a in acc]
    keep = [i for i, c in enumerate(closures) if c]
    acc = [acc[i] for i in keep]
    closures = [closures[i] for i in keep]
    log.info("  %d proteins with a non-empty propagated closure", len(acc))

    dense = base.dense_mean(acc_chunks, acc)
    d = dense.shape[1]
    log.info("dense matrix %s (d=%d)", dense.shape, d)

    # protein x term binary truth
    terms = sorted({t for c in closures for t in c})
    tix = {t: i for i, t in enumerate(terms)}
    rows, cols = [], []
    for i, c in enumerate(closures):
        for t in c:
            rows.append(i)
            cols.append(tix[t])
    T = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(len(acc), len(terms)), dtype=np.float32)
    log.info("truth matrix %s, avg terms/protein %.1f", T.shape, T.nnz / len(acc))

    results = {}
    t0 = time.time()
    log.info("ARM dense (full %d)", d)
    f = micro_fmax(transfer(l2norm(dense), T, args.knn), T)
    results["dense_%d" % d] = {"f_micro": f, "k": d, "index_ratio": 1.0}
    log.info("  dense f_micro=%.4f", f)

    for k in args.k_sdr:
        f = micro_fmax(transfer(l2norm(topk_real(dense, k)), T, args.knn), T)
        results["sparse_real_k%d" % k] = {"f_micro": f, "k": k, "index_ratio": k / d}
        log.info("  sparse-real k=%d f_micro=%.4f (index %.1fx smaller)", k, f, d / k)

    log.info("elapsed %.1fs", time.time() - t0)
    dense_f = results["dense_%d" % d]["f_micro"]
    log.info("=== RUNG 1 RESULT (knn=%d, n=%d) ===", args.knn, len(acc))
    for name, r in results.items():
        delta = r["f_micro"] - dense_f
        log.info("  %-18s f_micro=%.4f  delta_vs_dense=%+.4f  index=%.0f%% of dense",
                 name, r["f_micro"], delta, 100 * r["index_ratio"])

    try:
        import mlflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("sdr-knn-transfer")
        with mlflow.start_run(run_name="rung1-sparse-real-vs-dense"):
            mlflow.log_params({"n_proteins": len(acc), "knn": args.knn, "seed": args.seed,
                               "d": d, "k_sdr": str(args.k_sdr)})
            for name, r in results.items():
                mlflow.log_metric("fmicro_" + name, r["f_micro"])
    except Exception as e:  # noqa: BLE001
        log.info("mlflow skipped: %s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
