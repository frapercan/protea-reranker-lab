"""Pooling SWEEP on the task metric (f_micro, leave-one-out k-NN GO-transfer).

Follow-up to run_pool_fmicro.py (maxabs beat mean by +0.036). Questions:
  (1) do COMBINATIONS beat maxabs alone?  concat(mean,max), concat(mean,maxabs), ...
  (2) what is the role of the SIGN?  maxabs keeps the sign of the peak residue; absmax drops it.
  (3) is there anything BETTER than maxabs?  add min, std, and richer concats.

Each pooling maps a protein's residue matrix M (L, 768) to one vector (concat poolings are longer;
transfer_pred L2-normalises the whole vector, so concatenation is a legit richer representation).
Stratified by length. Read-only DB. Pure numpy (cheap burst).
"""
from __future__ import annotations

import argparse, logging, os, sys
from pathlib import Path
import numpy as np
import psycopg2
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protea_reranker_lab.sdr import GoDag, propagate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [pool-sweep] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pool-sweep")

ANN_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"
RESDIR = "/home/frapercan/Thesis2/storage/fullgo_models/per_residue_v227"


def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def transfer_pred(R, T, knn=30):
    R = l2n(R); S = R @ R.T; np.fill_diagonal(S, -1.0)
    N = R.shape[0]
    nbr = np.argpartition(-S, knn, axis=1)[:, :knn]
    rows = np.repeat(np.arange(N), knn); cols = nbr.ravel()
    w = S[rows, cols].copy(); w[w < 0] = 0
    Sk = sp.csr_matrix((w, (rows, cols)), shape=(N, N), dtype=np.float32)
    return np.asarray((Sk @ T).todense())


def fmicro_rows(pred, Td, mask):
    p = pred[mask].ravel(); t = Td[mask].ravel()
    npos = int(t.sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    tp = np.cumsum(t[order]); fp = np.cumsum(~t[order])
    prec = tp / np.maximum(tp + fp, 1); rec = tp / npos
    f = 2 * prec * rec / np.maximum(prec + rec, 1e-12); valid = p[order] > 0
    return float(np.max(f[valid])) if valid.any() else 0.0


# ---- single-protein pooling primitives (M: (L, d) -> (d,) unless noted) ----
def p_mean(M):   return M.mean(0)
def p_max(M):    return M.max(0)                                   # signed max
def p_min(M):    return M.min(0)                                   # signed min
def p_std(M):    return M.std(0)
def p_maxabs(M):                                                   # max by |.|, KEEP sign
    idx = np.abs(M).argmax(0); return M[idx, np.arange(M.shape[1])]
def p_absmax(M): return np.abs(M).max(0)                           # magnitude only, DROP sign
def p_meanabs(M): return np.abs(M).mean(0)                         # mean of magnitudes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="host=localhost dbname=protea user=protea password=protea")
    ap.add_argument("--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
    ap.add_argument("--knn", type=int, default=30)
    args = ap.parse_args()

    files = [f for f in os.listdir(RESDIR) if f.endswith(".npy")]
    accs_all = [f[:-4] for f in files]
    conn = psycopg2.connect(args.dsn); cur = conn.cursor()
    cur.execute(
        """SELECT pga.protein_accession, gt.go_id
           FROM protein_go_annotation pga JOIN go_term gt ON gt.id = pga.go_term_id
           WHERE pga.annotation_set_id = %s AND pga.protein_accession = ANY(%s)""",
        (ANN_SET, accs_all),
    )
    leaves: dict[str, list[str]] = {}
    for acc, go in cur.fetchall():
        leaves.setdefault(acc, []).append(go)
    cur.close(); conn.close()

    dag = GoDag.from_obo(Path(args.obo).expanduser())
    acc, closures = [], []
    for a in accs_all:
        if a in leaves:
            cl = propagate(leaves[a], dag)
            if cl:
                acc.append(a); closures.append(cl)
    n = len(acc)
    res = {a: np.load(os.path.join(RESDIR, f"{a}.npy")).astype(np.float32) for a in acc}
    lens = np.array([res[a].shape[0] for a in acc])
    log.info("n=%d  short(<=512)=%d  long(>512)=%d", n, int((lens <= 512).sum()), int((lens > 512).sum()))

    # precompute base poolings once
    base = {name: np.vstack([fn(res[a]) for a in acc]).astype(np.float32)
            for name, fn in {"mean": p_mean, "max": p_max, "min": p_min, "std": p_std,
                             "maxabs": p_maxabs, "absmax": p_absmax, "meanabs": p_meanabs}.items()}

    def cat(*names):  # L2-normalise each component THEN concat, so no single pool dominates the norm
        return np.hstack([l2n(base[nm]) for nm in names]).astype(np.float32)

    arms = {
        "mean": base["mean"],                       # baseline (platform)
        "max(signed)": base["max"],
        "maxabs(+sign)": base["maxabs"],            # current winner
        "absmax(no sign)": base["absmax"],          # sign probe: magnitude only
        "meanabs": base["meanabs"],
        "min": base["min"],
        "std": base["std"],
        "mean+max": cat("mean", "max"),
        "mean+maxabs": cat("mean", "maxabs"),
        "mean+absmax": cat("mean", "absmax"),
        "maxabs+min": cat("maxabs", "min"),
        "mean+maxabs+min": cat("mean", "maxabs", "min"),
        "mean+maxabs+std": cat("mean", "maxabs", "std"),
        "mean+max+min": cat("mean", "max", "min"),
    }

    terms = sorted({t for c in closures for t in c}); tix = {t: i for i, t in enumerate(terms)}
    r, c = [], []
    for i, cl in enumerate(closures):
        for t in cl:
            r.append(i); c.append(tix[t])
    T = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, len(terms)))
    Td = np.asarray(T.todense()).astype(bool)

    bins = {"all": np.ones(n, bool), "short": lens <= 512, "long": lens > 512}
    log.info("=== micro-Fmax by pooling (knn=%d), sorted by 'all' ===", args.knn)
    out = {}
    for name, R in arms.items():
        pred = transfer_pred(R, T, args.knn)
        out[name] = {b: fmicro_rows(pred, Td, m) for b, m in bins.items()}

    mean_all = out["mean"]["all"]
    log.info("  %-18s %8s %8s %8s %9s", "pool", "all", "short", "long", "d(all)")
    for name in sorted(out, key=lambda k: -out[k]["all"]):
        v = out[name]
        log.info("  %-18s %8.4f %8.4f %8.4f  %+8.4f", name, v["all"], v["short"], v["long"],
                 v["all"] - mean_all)

    try:
        import mlflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("pool-sweep-fmicro")
        with mlflow.start_run(run_name="pooling-sweep"):
            for name, v in out.items():
                key = name.replace("(", "_").replace(")", "").replace("+", "_p_").replace(" ", "")
                for b in bins:
                    if v[b] == v[b]:
                        mlflow.log_metric(f"{key}__{b}", v[b])
    except Exception as e:  # noqa: BLE001
        log.info("mlflow skipped: %s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
