"""H5 on the TASK metric: is sparse better/worse than dense for LARGE proteins?

Leave-one-out k-NN GO-transfer micro-Fmax, dense vs sparse-REAL (top-k + magnitudes), STRATIFIED
by protein size (chunk count: 1 = small, 2, 3+). No training (cheap, decisive). Answers whether
the SDR advantage/disadvantage depends on protein length on the actual task. Read-only DB.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sdr_chunk_correlation as base  # noqa: E402
from protea_reranker_lab.sdr import GoDag, propagate  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fmicro-len] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fmicro-len")


def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def topk_real(X, k):
    if k >= X.shape[1]:
        return X.copy()
    out = np.zeros_like(X)
    idx = np.argpartition(-np.abs(X), k, axis=1)[:, :k]
    np.put_along_axis(out, idx, np.take_along_axis(X, idx, axis=1), axis=1)
    return out


def transfer_pred(R, T, knn=30):
    R = l2n(R)
    S = R @ R.T
    np.fill_diagonal(S, -1.0)
    N = R.shape[0]
    nbr = np.argpartition(-S, knn, axis=1)[:, :knn]
    rows = np.repeat(np.arange(N), knn)
    cols = nbr.ravel()
    w = S[rows, cols].copy()
    w[w < 0] = 0
    Sk = sp.csr_matrix((w, (rows, cols)), shape=(N, N), dtype=np.float32)
    return np.asarray((Sk @ T).todense())  # (N, n_terms) scores


def fmicro_rows(pred, Td, mask):
    """micro-Fmax over the rows (query proteins) selected by mask."""
    p = pred[mask].ravel()
    t = Td[mask].ravel()
    npos = int(t.sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    tp = np.cumsum(t[order])
    fp = np.cumsum(~t[order])
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / npos
    f = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    valid = p[order] > 0
    return float(np.max(f[valid])) if valid.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dsn", default="host=localhost dbname=protea user=protea password=protea"
    )
    ap.add_argument(
        "--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
    )
    ap.add_argument("--n-proteins", type=int, default=12000)
    ap.add_argument("--k", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--knn", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    acc, acc_chunks, leaves = base.load_chunked(args.dsn, args.n_proteins, args.seed)
    dag = GoDag.from_obo(Path(args.obo).expanduser())
    closures = [propagate(leaves[a], dag) for a in acc]
    keep = [i for i, c in enumerate(closures) if c]
    acc = [acc[i] for i in keep]
    closures = [closures[i] for i in keep]
    dense = base.dense_mean(acc_chunks, acc)
    n = len(acc)
    nch = np.array([acc_chunks[a].shape[0] for a in acc])
    log.info(
        "n=%d  chunk-count dist: 1=%d  2=%d  3+=%d",
        n,
        int((nch == 1).sum()),
        int((nch == 2).sum()),
        int((nch >= 3).sum()),
    )

    terms = sorted({t for c in closures for t in c})
    tix = {t: i for i, t in enumerate(terms)}
    r, c = [], []
    for i, cl in enumerate(closures):
        for t in cl:
            r.append(i)
            c.append(tix[t])
    T = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, len(terms)))
    Td = np.asarray(T.todense()).astype(bool)

    bins = {
        "all": np.ones(n, bool),
        "1-chunk": nch == 1,
        "2-chunk": nch == 2,
        "3+chunk": nch >= 3,
    }
    arms = {"dense": dense}
    for k in args.k:
        arms[f"sparse_real_k{k}"] = topk_real(dense, k)

    log.info("=== micro-Fmax by protein size (chunk count) ===")
    header = "  %-18s " + " ".join(f"{b:>9}" for b in bins)
    log.info(header, "arm")
    rows_out = {}
    for name, R in arms.items():
        pred = transfer_pred(R, T, args.knn)
        vals = {b: fmicro_rows(pred, Td, m) for b, m in bins.items()}
        rows_out[name] = vals
        log.info("  %-18s " + " ".join(f"{vals[b]:9.4f}" for b in bins), name)

    # the H5 verdict: dense-vs-sparse gap per size bin
    log.info("=== H5: (sparse_real_k128 - dense) per size bin ===")
    d = rows_out["dense"]
    s = rows_out.get("sparse_real_k128", {})
    for b in bins:
        if b in s:
            log.info(
                "  %-9s  dense=%.4f  sparse_k128=%.4f  delta=%+.4f",
                b,
                d[b],
                s[b],
                s[b] - d[b],
            )
    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("sdr-fmicro-by-length")
        with mlflow.start_run(run_name="dense-vs-sparse-by-size"):
            for name, vals in rows_out.items():
                for b, v in vals.items():
                    if v == v:  # not nan
                        mlflow.log_metric(
                            f"{name}__{b}".replace("+", "p").replace("-", "_"), v
                        )
    except Exception as e:  # noqa: BLE001
        log.info("mlflow skipped: %s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
