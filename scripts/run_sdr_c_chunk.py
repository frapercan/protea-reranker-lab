"""SDR grid A1: per-CHUNK learned sparse bundle (sparsify-then-bundle, WHEN LEARNED).

The SDR-C (A0) collapsed the protein to a mean vector before learning the sparse code.
This A1 variant does NOT collapse: it encodes EACH chunk, sparsifies it, and BUNDLES the
per-chunk codes (max / OR across chunks) into the protein SDR. Trained contrastively (soft
Tanimoto on the bundled code) vs GO-sim, held-out eval. This is the decisive test of the
sparse.pdf ordering question (sparsify-then-bundle vs pool-then-sparsify) in the LEARNED
regime, and the setup where large / multi-domain proteins (>=2 chunks) could win, since each
domain contributes its own active bits instead of being averaged away.

Compare on held-out vs: dense cosine (~0.22), raw magnitude-binary (~0.09), and A0 SDR-C
learned-binary (~0.49). Read-only DB. MLflow (live per-epoch eval curve).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
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
    kwta_binarise,
    tanimoto_dense,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [sdr-c-chunk] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sdr-c-chunk")
torch.manual_seed(42)


def topk_binary_general(Z, k):
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


def flat_chunks(acc_chunks, accessions):
    """Stack every chunk of every protein + a protein-index for scatter-bundling."""
    mats, pidx = [], []
    for p, a in enumerate(accessions):
        m = acc_chunks[a]
        mats.append(m)
        pidx.extend([p] * m.shape[0])
    X = torch.tensor(np.vstack(mats).astype(np.float32))
    idx = torch.tensor(pidx, dtype=torch.long)
    return X, idx


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
    ap.add_argument("--epochs", type=int, default=150)
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
    acc = [acc[i] for i in keep]
    closures = [closures[i] for i in keep]
    n = len(acc)
    n_chunks_per = np.array([acc_chunks[a].shape[0] for a in acc])
    log.info(
        "n=%d  multi-chunk=%.1f%%  mean chunks=%.2f",
        n,
        100 * (n_chunks_per >= 2).mean(),
        n_chunks_per.mean(),
    )

    perm = rng.permutation(n)
    n_tr = n // 2
    tr, ev = perm[:n_tr], perm[n_tr:]
    acc_tr = [acc[i] for i in tr]
    acc_ev = [acc[i] for i in ev]
    clo_tr = [closures[i] for i in tr]
    clo_ev = [closures[i] for i in ev]
    Xc_tr, pidx_tr = flat_chunks(acc_chunks, acc_tr)
    Xc_ev, pidx_ev = flat_chunks(acc_chunks, acc_ev)
    d = Xc_tr.shape[1]

    # train targets (Lin) over train pairs
    ic_tr = information_content(clo_tr, dag)
    bic_tr = [max((ic_tr.get(t, 0.0) for t in c), default=0.0) for c in clo_tr]
    tpairs = sample_pairs(len(tr), args.train_pairs, rng)
    ytr = torch.tensor(
        np.asarray(lin_pairwise(clo_tr, ic_tr, tpairs, bic_tr), dtype=np.float32)
    )
    ti = torch.tensor([p[0] for p in tpairs])
    tj = torch.tensor([p[1] for p in tpairs])

    enc = nn.Linear(d, args.dict)
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)
    bs = 16384
    ALPHA = 20.0

    def bundle(Xc, pidx, n_prot):
        """encode each chunk -> soft top-k membership -> max-bundle per protein."""
        Z = enc(Xc)
        tau = torch.topk(Z, args.k, dim=1).values[:, -1:].detach()
        S = torch.sigmoid(ALPHA * (Z - tau))  # per-chunk soft membership
        out = torch.zeros(n_prot, args.dict)
        out.scatter_reduce_(
            0,
            pidx.unsqueeze(1).expand(-1, args.dict),
            S,
            reduce="amax",
            include_self=False,
        )  # OR/max across chunks
        return out

    def code_sim(Sp, i, j):
        si, sj = Sp[i], Sp[j]
        inter = (si * sj).sum(1)
        return inter / (si.sum(1) + sj.sum(1) - inter + 1e-6)

    # held-out eval setup
    De = base.dense_mean(acc_chunks, acc_ev)
    ic_ev = information_content(clo_ev, dag)
    bic_ev = [max((ic_ev.get(t, 0.0) for t in c), default=0.0) for c in clo_ev]
    epairs = sample_pairs(len(ev), args.eval_pairs, rng)
    ii = np.array([p[0] for p in epairs])
    jj = np.array([p[1] for p in epairs])
    GO = {
        "resnik": np.asarray(resnik_pairwise(clo_ev, ic_ev, epairs), dtype=np.float64),
        "lin": np.asarray(
            lin_pairwise(clo_ev, ic_ev, epairs, bic_ev), dtype=np.float64
        ),
    }
    base_rho = {
        m: {
            "dense": float(spearmanr(cosine_dense(De)[ii, jj], g)[0]),
            "rawbin": float(
                spearmanr(tanimoto_dense(kwta_binarise(De, args.k), args.k)[ii, jj], g)[
                    0
                ]
            ),
        }
        for m, g in GO.items()
    }
    log.info(
        "baselines dense resnik=%.4f lin=%.4f | rawbin resnik=%.4f",
        base_rho["resnik"]["dense"],
        base_rho["lin"]["dense"],
        base_rho["resnik"]["rawbin"],
    )
    n_chunks_ev = np.array([acc_chunks[a].shape[0] for a in acc_ev])

    def eval_binary(metric, mask_pairs=None):
        with torch.no_grad():
            Sp = bundle(Xc_ev, pidx_ev, len(ev)).numpy().astype(np.float32)
        tan = tanimoto_dense(topk_binary_general(Sp, args.k), args.k)
        s = tan[ii, jj]
        g = GO[metric]
        if mask_pairs is not None:
            s, g = s[mask_pairs], g[mask_pairs]
        return float(spearmanr(s, g)[0]) if len(s) > 100 else float("nan")

    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("sdr-c-chunk-bundle")
        mlflow.start_run(run_name="A1-per-chunk-learned-bundle")
        mlflow.log_params(
            {
                "n": n,
                "dict": args.dict,
                "k": args.k,
                "epochs": args.epochs,
                "granularity": "per-chunk-bundle",
                "seed": args.seed,
            }
        )
        _ml = True
    except Exception as e:  # noqa: BLE001
        log.info("mlflow init skipped: %s", e)
        _ml = False

    log.info(
        "training A1 (per-chunk bundle): dict=%d k=%d epochs=%d",
        args.dict,
        args.k,
        args.epochs,
    )
    np_ = len(tpairs)
    for ep in range(args.epochs):
        Sp = bundle(Xc_tr, pidx_tr, len(tr))
        opt.zero_grad()
        tot = 0.0
        for b in range(0, np_, bs):
            sl = slice(b, b + bs)
            s = code_sim(Sp, ti[sl], tj[sl])
            loss = ((s - ytr[sl]) ** 2).sum() / np_
            loss.backward(retain_graph=True)
            tot += float(loss)
        opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            er, el = eval_binary("resnik"), eval_binary("lin")
            if _ml:
                mlflow.log_metric("train_loss", tot, step=ep)
                mlflow.log_metric("eval_rho_resnik", er, step=ep)
                mlflow.log_metric("eval_rho_lin", el, step=ep)
            log.info(
                "  epoch %3d loss=%.4f | A1 held-out resnik=%.4f lin=%.4f (dense %.4f / A0 ~0.49)",
                ep,
                tot,
                er,
                el,
                base_rho["resnik"]["dense"],
            )

    # final + size-stratified (H5)
    log.info("=== A1 FINAL (per-chunk learned bundle) ===")
    for metric in ("resnik", "lin"):
        full = eval_binary(metric)
        single = eval_binary(
            metric, mask_pairs=(n_chunks_ev[ii] == 1) & (n_chunks_ev[jj] == 1)
        )
        multi = eval_binary(
            metric, mask_pairs=(n_chunks_ev[ii] >= 2) | (n_chunks_ev[jj] >= 2)
        )
        log.info(
            "  %s: A1=%.4f (dense %.4f) | 1-chunk pairs A1=%.4f | multi-chunk pairs A1=%.4f",
            metric,
            full,
            base_rho[metric]["dense"],
            single,
            multi,
        )
        if _ml:
            mlflow.log_metric(f"final_{metric}_A1", full)
            mlflow.log_metric(f"final_{metric}_A1_singlechunk", single)
            mlflow.log_metric(f"final_{metric}_A1_multichunk", multi)
    if _ml:
        mlflow.end_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
