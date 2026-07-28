"""Pooling lever on the TASK metric: mean-pool vs max-pool over residues, f_micro (k-NN GO-transfer).

The naive A2 finding (per-residue amax bundle beats dense on Spearman) was really a POOLING result,
not a sparsity result: aggregating per-residue vectors with a max instead of a mean correlates
better with GO geometry. This script tests whether that survives on the TASK metric (leave-one-out
k-NN GO-transfer micro-Fmax) -- the thing that pays -- on the same per-residue arrays. If max-pool
beats mean-pool on f_micro, it is a free, shippable representation lever independent of SDR.

Poolings compared (per protein, over its residue vectors M of shape (L, 768)):
  mean    : M.mean(0)                       <- what the platform does today (dense_mean)
  max     : M.max(0)                         signed max per dim
  maxabs  : value with largest |.| per dim   (keeps sign of the peak-magnitude residue)

Stratified by length. Read-only DB.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
import psycopg2
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protea_reranker_lab.sdr import GoDag, propagate  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [pool-fmicro] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pool-fmicro")

ANN_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"
RESDIR = "/home/frapercan/Thesis2/storage/fullgo_models/per_residue_v227"


def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


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
    return np.asarray((Sk @ T).todense())


def fmicro_rows(pred, Td, mask):
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


def maxabs_pool(M):
    idx = np.abs(M).argmax(0)
    return M[idx, np.arange(M.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dsn", default="host=localhost dbname=protea user=protea password=protea"
    )
    ap.add_argument(
        "--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
    )
    ap.add_argument("--knn", type=int, default=30)
    args = ap.parse_args()

    files = [f for f in os.listdir(RESDIR) if f.endswith(".npy")]
    accs_all = [f[:-4] for f in files]
    conn = psycopg2.connect(args.dsn)
    cur = conn.cursor()
    cur.execute(
        """SELECT pga.protein_accession, gt.go_id
           FROM protein_go_annotation pga JOIN go_term gt ON gt.id = pga.go_term_id
           WHERE pga.annotation_set_id = %s AND pga.protein_accession = ANY(%s)""",
        (ANN_SET, accs_all),
    )
    leaves: dict[str, list[str]] = {}
    for acc, go in cur.fetchall():
        leaves.setdefault(acc, []).append(go)
    cur.close()
    conn.close()

    dag = GoDag.from_obo(Path(args.obo).expanduser())
    acc, closures = [], []
    for a in accs_all:
        if a in leaves:
            c = propagate(leaves[a], dag)
            if c:
                acc.append(a)
                closures.append(c)
    n = len(acc)
    res = {a: np.load(os.path.join(RESDIR, f"{a}.npy")).astype(np.float32) for a in acc}
    lens = np.array([res[a].shape[0] for a in acc])
    log.info(
        "n=%d  short(<=512)=%d  long(>512)=%d",
        n,
        int((lens <= 512).sum()),
        int((lens > 512).sum()),
    )

    pools = {
        "mean": np.vstack([res[a].mean(0) for a in acc]),
        "max": np.vstack([res[a].max(0) for a in acc]),
        "maxabs": np.vstack([maxabs_pool(res[a]) for a in acc]),
    }

    terms = sorted({t for c in closures for t in c})
    tix = {t: i for i, t in enumerate(terms)}
    r, c = [], []
    for i, cl in enumerate(closures):
        for t in cl:
            r.append(i)
            c.append(tix[t])
    T = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, len(terms)))
    Td = np.asarray(T.todense()).astype(bool)

    bins = {"all": np.ones(n, bool), "short<=512": lens <= 512, "long>512": lens > 512}
    log.info(
        "=== micro-Fmax by pooling (leave-one-out k-NN GO-transfer, knn=%d) ===",
        args.knn,
    )
    log.info("  %-8s " + " ".join(f"{b:>11}" for b in bins), "pool")
    out = {}
    for name, R in pools.items():
        pred = transfer_pred(R.astype(np.float32), T, args.knn)
        vals = {b: fmicro_rows(pred, Td, m) for b, m in bins.items()}
        out[name] = vals
        log.info("  %-8s " + " ".join(f"{vals[b]:11.4f}" for b in bins), name)

    d = out["mean"]
    log.info("=== verdict: (max - mean) and (maxabs - mean) per bin ===")
    for b in bins:
        log.info(
            "  %-10s  mean=%.4f  max=%+.4f  maxabs=%+.4f",
            b,
            d[b],
            out["max"][b] - d[b],
            out["maxabs"][b] - d[b],
        )
    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("pool-fmicro")
        with mlflow.start_run(run_name="mean-vs-max-pool-fmicro"):
            for name, vals in out.items():
                for b, v in vals.items():
                    if v == v:
                        mlflow.log_metric(
                            f"{name}__{b}".replace("<=", "le").replace(">", "gt"), v
                        )
    except Exception as e:  # noqa: BLE001
        log.info("mlflow skipped: %s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
