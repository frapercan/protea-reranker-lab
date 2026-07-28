"""Rung 2 of the SDR lever-validation ladder: does a FUNCTION-ALIGNED binary code
recover the binarisation loss?

The decomposition showed binarising the magnitude-top-k code costs ~-0.13 Spearman.
Question: if we choose WHICH bits to fire by FUNCTION instead of by magnitude, does the
binary code recover that loss? Cheap, torch-free, robust test:

  * Build a GO-term prototype = centroid of the (L2-normed) dense vectors of the TRAIN
    proteins carrying that term.
  * A protein's ALIGNED code z = its cosine to every prototype (dim = n_prototypes). The
    active bits of top-k(z) literally mean "close to these GO functions".
  * Measure the intrinsic Spearman (representation similarity vs GO-semantic) on a HELD-OUT
    eval split, for: dense raw cosine (baseline), raw magnitude-binary (the -0.13 floor),
    aligned-real cosine, and aligned-BINARY top-k.

Decision: if aligned-binary >> raw-binary (closes a big part of the -0.13, toward dense),
a learned/aligned bit pattern carries function -> the precision lever exists -> build Rung 3
(deep SAE). If it barely moves, a shallow alignment cannot recover the magnitude.

Reuses run_sdr_chunk_correlation.py (loading, GO DAG, IC, Resnik/Lin) + sdr.py. Read-only DB.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
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
    kwta_binarise,
    tanimoto_dense,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [aligned] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aligned")


def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def topk_binary_general(Z, k):
    """top-k by value (not |.|) per row -> uint8 bitset (aligned scores are >=0-ish cosines)."""
    out = np.zeros(Z.shape, dtype=np.uint8)
    idx = np.argpartition(-Z, min(k, Z.shape[1] - 1), axis=1)[:, :k]
    np.put_along_axis(out, idx, 1, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dsn", default="host=localhost dbname=protea user=protea password=protea"
    )
    ap.add_argument(
        "--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
    )
    ap.add_argument("--n-proteins", type=int, default=10000)
    ap.add_argument("--n-pairs", type=int, default=200_000)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--min-train-per-term", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    acc, acc_chunks, leaves = base.load_chunked(args.dsn, args.n_proteins, args.seed)
    dag = GoDag.from_obo(Path(args.obo).expanduser())
    closures = [propagate(leaves[a], dag) for a in acc]
    keep = [i for i, c in enumerate(closures) if c]
    acc = [acc[i] for i in keep]
    closures = [closures[i] for i in keep]
    dense = base.dense_mean(acc_chunks, acc)
    n = len(acc)
    d = dense.shape[1]
    log.info("n=%d d=%d", n, d)

    # train/eval split (prototypes from train; correlation measured on eval)
    perm = rng.permutation(n)
    n_tr = n // 2
    tr, ev = perm[:n_tr], perm[n_tr:]
    dn = l2n(dense)

    # prototypes from TRAIN: centroid of normed vectors per term (>= min count)
    from collections import defaultdict

    bucket = defaultdict(list)
    for i in tr:
        for t in closures[i]:
            bucket[t].append(i)
    protos = [(t, ii) for t, ii in bucket.items() if len(ii) >= args.min_train_per_term]
    log.info(
        "prototypes (terms with >= %d train proteins): %d",
        args.min_train_per_term,
        len(protos),
    )
    P = np.vstack([dn[ii].mean(axis=0) for _, ii in protos]).astype(np.float32)
    P = l2n(P)

    # eval set
    De = dense[ev]
    Dne = dn[ev]
    closures_e = [closures[i] for i in ev]
    Z = (Dne @ P.T).astype(
        np.float32
    )  # aligned code on eval (cosine to each prototype)
    log.info("aligned code on eval: %s", Z.shape)

    # IC + pairs on eval
    ic = information_content(closures_e, dag)
    best_ic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in closures_e]
    ne = len(ev)
    npairs = min(args.n_pairs, ne * (ne - 1) // 2)
    seen = set()
    pairs = []
    while len(pairs) < npairs:
        i, j = int(rng.integers(ne)), int(rng.integers(ne))
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])

    # representations on eval
    sim = {
        "dense_raw_cosine": cosine_dense(De)[ii, jj],
        "raw_magnitude_binary": tanimoto_dense(kwta_binarise(De, args.k), args.k)[
            ii, jj
        ],
        "aligned_real_cosine": cosine_dense(Z)[ii, jj],
        "aligned_binary_topk": tanimoto_dense(topk_binary_general(Z, args.k), args.k)[
            ii, jj
        ],
    }

    out = {}
    for metric in ("resnik", "lin"):
        go = (
            resnik_pairwise(closures_e, ic, pairs)
            if metric == "resnik"
            else lin_pairwise(closures_e, ic, pairs, best_ic)
        )
        log.info("--- %s (k=%d, eval n=%d, %d pairs) ---", metric, args.k, ne, npairs)
        row = {}
        for name, s in sim.items():
            rho = spearmanr(s, go)[0]
            row[name] = float(rho)
            log.info("  %-22s rho=%.4f", name, rho)
        rec = row["aligned_binary_topk"] - row["raw_magnitude_binary"]
        gap_to_dense = row["aligned_binary_topk"] - row["dense_raw_cosine"]
        log.info(
            "  >> aligned-binary recovers %+.4f over raw-binary;  still %+.4f vs dense",
            rec,
            gap_to_dense,
        )
        out[metric] = row

    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("sdr-aligned-binary")
        with mlflow.start_run(run_name="rung2-go-prototype-alignment"):
            mlflow.log_params(
                {"n": n, "k": args.k, "n_prototypes": len(protos), "seed": args.seed}
            )
            for metric, row in out.items():
                for name, rho in row.items():
                    mlflow.log_metric(f"{metric}_{name}", rho)
    except Exception as e:  # noqa: BLE001
        log.info("mlflow skipped: %s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
