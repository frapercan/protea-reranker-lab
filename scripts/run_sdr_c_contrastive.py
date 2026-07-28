"""Rung 3 of the SDR lever-validation ladder: SDR-C, a LEARNED sparse code.

Rung 2 showed a function-aligned binary code can beat dense, but used GO-term prototypes
(supervised, a ceiling). SDR-C must LEARN the alignment from the embedding with a contrastive
objective and GENERALISE to held-out proteins, without using labels as the code's basis.

Model: z = ReLU(W e) , W: 768 -> D (dictionary). Trained so that cosine(z_i, z_j) tracks GO
semantic similarity (Lin) over TRAIN pairs, with an L1 sparsity pull. Evaluated by binarising
z (top-k active) and measuring Tanimoto-vs-GO Spearman on a HELD-OUT eval split. The active set
of the binary code is the learned SDR.

Compare on eval: dense cosine (~0.22), raw magnitude-binary (~0.09, the floor), aligned-proto
binary (~0.25, the supervised ceiling), and the LEARNED binary (this). Read-only DB. MLflow.
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
    format="%(asctime)s [sdr-c] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sdr-c")
torch.manual_seed(42)


def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


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
    ap.add_argument("--l1", type=float, default=1e-3)
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
    dense = l2n(base.dense_mean(acc_chunks, acc))
    n, d = dense.shape
    log.info("n=%d d=%d", n, d)

    perm = rng.permutation(n)
    n_tr = n // 2
    tr, ev = perm[:n_tr], perm[n_tr:]
    Xtr = torch.tensor(dense[tr])
    Xev = torch.tensor(dense[ev])
    clo_tr = [closures[i] for i in tr]
    clo_ev = [closures[i] for i in ev]

    # train targets: Lin GO-sim over sampled train pairs (normalised to [0,1] already)
    ic_tr = information_content(clo_tr, dag)
    bic_tr = [max((ic_tr.get(t, 0.0) for t in c), default=0.0) for c in clo_tr]
    tpairs = sample_pairs(len(tr), args.train_pairs, rng)
    ytr = torch.tensor(
        np.asarray(lin_pairwise(clo_tr, ic_tr, tpairs, bic_tr), dtype=np.float32)
    )
    ti = torch.tensor([p[0] for p in tpairs])
    tj = torch.tensor([p[1] for p in tpairs])
    log.info("train pairs %d (Lin target mean %.3f)", len(tpairs), float(ytr.mean()))

    # model: linear encoder, NO ReLU (top-k selects by value; ReLU caused dead-unit collapse)
    enc = nn.Linear(d, args.dict)
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)
    bs = 16384

    ALPHA = 20.0

    def soft_members(Z):
        # differentiable soft top-k membership: sigmoid around the k-th largest value
        tau = torch.topk(Z, args.k, dim=1).values[:, -1:].detach()
        return torch.sigmoid(ALPHA * (Z - tau))

    def code_sim(S, i, j):
        # soft Tanimoto on the active SETS (the binary objective): signal must live in
        # WHICH bits are active, not in magnitudes.
        si = S[i]
        sj = S[j]
        inter = (si * sj).sum(1)
        return inter / (si.sum(1) + sj.sum(1) - inter + 1e-6)

    # held-out eval setup (precomputed once -> generalisation curve during training)
    De = base.dense_mean(acc_chunks, [acc[i] for i in ev])
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
    base_rho = {}
    for m, g in GO.items():
        base_rho[(m, "dense_cosine")] = float(spearmanr(cosine_dense(De)[ii, jj], g)[0])
        base_rho[(m, "raw_magnitude_binary")] = float(
            spearmanr(tanimoto_dense(kwta_binarise(De, args.k), args.k)[ii, jj], g)[0]
        )
    log.info(
        "baselines  dense: resnik=%.4f lin=%.4f | raw-binary: resnik=%.4f lin=%.4f",
        base_rho[("resnik", "dense_cosine")],
        base_rho[("lin", "dense_cosine")],
        base_rho[("resnik", "raw_magnitude_binary")],
        base_rho[("lin", "raw_magnitude_binary")],
    )

    def learned_binary_rho(metric):
        with torch.no_grad():
            Zev = enc(Xev).numpy().astype(np.float32)
        tan = tanimoto_dense(topk_binary_general(Zev, args.k), args.k)[ii, jj]
        return float(spearmanr(tan, GO[metric])[0])

    # live MLflow instrumentation (per-epoch loss/density)
    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("sdr-c-binary-objective")
        mlflow.start_run(run_name="rung3b-soft-tanimoto")
        mlflow.log_params(
            {
                "n": n,
                "dict": args.dict,
                "k": args.k,
                "epochs": args.epochs,
                "lr": args.lr,
                "objective": "soft-tanimoto-binary",
                "seed": args.seed,
            }
        )
        _ml = True
    except Exception as e:  # noqa: BLE001
        log.info("mlflow init skipped: %s", e)
        _ml = False

    log.info("training SDR-C: dict=%d k=%d epochs=%d", args.dict, args.k, args.epochs)
    np_ = len(tpairs)
    for ep in range(args.epochs):
        enc.train()
        Ztr = enc(Xtr)  # encode all train proteins once
        Str = soft_members(Ztr)  # soft top-k active sets (the binary objective)
        opt.zero_grad()
        tot = 0.0
        for b in range(0, np_, bs):  # accumulate grad over pair minibatches
            sl = slice(b, b + bs)
            s = code_sim(Str, ti[sl], tj[sl])
            loss = ((s - ytr[sl]) ** 2).sum() / np_
            loss.backward(retain_graph=True)
            tot += float(loss)
        opt.step()
        with torch.no_grad():
            dens = float((enc(Xev) > 0).float().mean())
        if _ml:
            mlflow.log_metric("train_loss", tot, step=ep)
            mlflow.log_metric("eval_density", dens, step=ep)
        if ep % 10 == 0 or ep == args.epochs - 1:
            er, el = learned_binary_rho("resnik"), learned_binary_rho("lin")
            if _ml:
                mlflow.log_metric("eval_rho_resnik", er, step=ep)
                mlflow.log_metric("eval_rho_lin", el, step=ep)
            log.info(
                "  epoch %3d loss=%.4f dens=%.3f | LEARNED-BINARY held-out: resnik=%.4f lin=%.4f "
                "(dense %.4f / raw-bin %.4f)",
                ep,
                tot,
                dens,
                er,
                el,
                base_rho[("resnik", "dense_cosine")],
                base_rho[("resnik", "raw_magnitude_binary")],
            )

    # final eval on held-out (reuse precomputed pairs / GO / baselines)
    with torch.no_grad():
        Zev = enc(Xev).numpy().astype(np.float32)
    out = {}
    for metric in ("resnik", "lin"):
        g = GO[metric]
        row = {
            "dense_cosine": base_rho[(metric, "dense_cosine")],
            "raw_magnitude_binary": base_rho[(metric, "raw_magnitude_binary")],
            "learned_real_cosine": float(spearmanr(cosine_dense(Zev)[ii, jj], g)[0]),
            "learned_binary_topk": float(
                spearmanr(
                    tanimoto_dense(topk_binary_general(Zev, args.k), args.k)[ii, jj], g
                )[0]
            ),
        }
        log.info(
            "--- FINAL %s (held-out n=%d, %d pairs) ---", metric, len(ev), len(epairs)
        )
        for name, rho in row.items():
            log.info("  %-22s rho=%.4f", name, rho)
        log.info(
            "  >> LEARNED binary: %+.4f vs dense, %+.4f vs raw-binary",
            row["learned_binary_topk"] - row["dense_cosine"],
            row["learned_binary_topk"] - row["raw_magnitude_binary"],
        )
        out[metric] = row

    if _ml:
        try:
            import mlflow

            for metric, row in out.items():
                for name, rho in row.items():
                    mlflow.log_metric(f"final_{metric}_{name}", rho)
            mlflow.end_run()
        except Exception as e:  # noqa: BLE001
            log.info("mlflow final-log skipped: %s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
