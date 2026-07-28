"""SDR grid A2: per-RESIDUE sparsify-then-bundle (the sparse.pdf section-2 ordering, literal).

A0 collapsed the protein to a MEAN vector before sparsifying. A1 kept CHUNKS. A2 goes all the
way: it never pools. Each residue vector is sparsified (top-k by magnitude) and the per-residue
codes are BUNDLED (OR / max across residues) into one protein SDR. This is the strongest possible
test of "sparsify-then-bundle beats pool-then-sparsify", because the pooling is removed entirely
(per-residue granularity, no mean anywhere).

Inputs are the per-residue ankh-base arrays in storage/fullgo_models/per_residue_v227/*.npy
(shape (L, 768) float16), produced by extract_per_residue.py. v227 GO annotations pulled for the
SAME accessions (read-only DB).

Three arms, compared to dense cosine (~0.22), naive chunk-SDR (~0.13), A0 (~0.49), A1 (~0.48):
  - naive-real    : per-residue top-k real, amax-magnitude bundle -> cosine          (H1 at residue)
  - naive-binary  : per-residue top-k, OR bundle -> Tanimoto                          (H2/H3 literal)
  - learned       : Linear encoder per residue -> soft top-k -> amax bundle, contrastive (GPU)

MLflow live curve. Read-only DB.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import psycopg2
import torch
import torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    format="%(asctime)s [sdr-c-res] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sdr-c-res")
torch.manual_seed(42)

ANN_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"  # GOA v227, t0
RESDIR = "/home/frapercan/Thesis2/storage/fullgo_models/per_residue_v227"


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


def strided(L, cap):
    """Deterministic uniform residue subsample to <= cap positions (keeps domain spread)."""
    if L <= cap:
        return np.arange(L)
    return np.linspace(0, L - 1, cap).round().astype(int)


def topk_binary_general(Z, k):
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
    ap.add_argument("--dict", type=int, default=2048)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument(
        "--cap-res",
        type=int,
        default=64,
        help="max residues/protein for the LEARNED arm",
    )
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-pairs", type=int, default=300_000)
    ap.add_argument("--eval-pairs", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- accessions present as per-residue arrays
    files = [f for f in os.listdir(RESDIR) if f.endswith(".npy")]
    accs_all = [f[:-4] for f in files]
    log.info("per-residue arrays on disk: %d", len(accs_all))

    # ---- v227 annotations for those accessions
    conn = psycopg2.connect(args.dsn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pga.protein_accession, gt.go_id
        FROM protein_go_annotation pga
        JOIN go_term gt ON gt.id = pga.go_term_id
        WHERE pga.annotation_set_id = %s AND pga.protein_accession = ANY(%s)
        """,
        (ANN_SET, accs_all),
    )
    leaves: dict[str, list[str]] = {}
    for acc, go in cur.fetchall():
        leaves.setdefault(acc, []).append(go)
    cur.close()
    conn.close()
    log.info("accessions with v227 annotations: %d", len(leaves))

    dag = GoDag.from_obo(Path(args.obo).expanduser())
    acc, closures = [], []
    for a in accs_all:
        if a in leaves:
            c = propagate(leaves[a], dag)
            if c:
                acc.append(a)
                closures.append(c)
    n = len(acc)
    log.info("usable proteins (residues + closed annotations): %d", n)

    # ---- load per-residue arrays (float32), record lengths
    res = {a: np.load(os.path.join(RESDIR, f"{a}.npy")).astype(np.float32) for a in acc}
    lens = np.array([res[a].shape[0] for a in acc])
    log.info(
        "residue-length dist: min=%d med=%d max=%d  (>=2 chunks ~ len>512: %.1f%%)",
        int(lens.min()),
        int(np.median(lens)),
        int(lens.max()),
        100 * (lens > 512).mean(),
    )

    # ---- dense reference = mean over residues (the thing A0 learned to beat)
    dense = np.vstack([res[a].mean(0) for a in acc]).astype(np.float32)

    # =====================================================================================
    # NAIVE arms (no training): bundle per-residue top-k codes across ALL residues
    # =====================================================================================
    log.info("=== building NAIVE per-residue bundles (real + binary), k=%d ===", args.k)
    d = dense.shape[1]
    nb_real = np.zeros((n, d), np.float32)  # amax magnitude bundle of top-k-per-residue
    nb_bin = np.zeros((n, d), np.uint8)  # OR bundle of top-k-per-residue
    for i, a in enumerate(acc):
        M = res[a]  # (L, d)
        kk = min(args.k, d - 1)
        idx = np.argpartition(-np.abs(M), kk, axis=1)[:, :kk]  # top-k dims per residue
        sparse = np.zeros_like(M)
        np.put_along_axis(sparse, idx, np.take_along_axis(M, idx, axis=1), axis=1)
        nb_real[i] = np.abs(sparse).max(0)  # amax of |magnitude| across residues
        nb_bin[i] = (sparse != 0).any(0).astype(np.uint8)  # OR of active sets
    log.info(
        "naive bundle active-dim mean: real=%.0f binary=%.0f (of %d)",
        float((nb_real > 0).sum(1).mean()),
        float(nb_bin.sum(1).mean()),
        d,
    )

    # ---- eval pairs + GO targets (single held-out split for naive == the whole set)
    epairs = sample_pairs(n, args.eval_pairs, rng)
    ii = np.array([p[0] for p in epairs])
    jj = np.array([p[1] for p in epairs])
    ic = information_content(closures, dag)
    bic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in closures]
    GO = {
        "resnik": np.asarray(resnik_pairwise(closures, ic, epairs), dtype=np.float64),
        "lin": np.asarray(lin_pairwise(closures, ic, epairs, bic), dtype=np.float64),
    }

    def rho(sim_mat_vals, metric):
        return float(spearmanr(sim_mat_vals, GO[metric])[0])

    dense_cos = cosine_dense(dense)
    nbreal_cos = cosine_dense(nb_real.astype(np.float32))
    nbbin_tan = tanimoto_dense(nb_bin, int(nb_bin.sum(1).mean()) or args.k)
    res_rows = {}
    for m in ("resnik", "lin"):
        res_rows[m] = {
            "dense": rho(dense_cos[ii, jj], m),
            "naive_real": rho(nbreal_cos[ii, jj], m),
            "naive_binary": rho(nbbin_tan[ii, jj], m),
        }
    log.info("=== NAIVE A2 (Spearman vs GO) ===")
    for m in ("resnik", "lin"):
        r = res_rows[m]
        log.info(
            "  %-7s dense=%.4f | naive-real(bundle)=%.4f | naive-binary(OR)=%.4f",
            m,
            r["dense"],
            r["naive_real"],
            r["naive_binary"],
        )

    # =====================================================================================
    # LEARNED arm: encoder per residue -> soft top-k -> amax bundle, contrastive (GPU)
    # =====================================================================================
    perm = rng.permutation(n)
    n_tr = n // 2
    tr, ev = perm[:n_tr], perm[n_tr:]

    def flat_res(indices, cap):
        mats, pidx = [], []
        for p, gi in enumerate(indices):
            a = acc[gi]
            M = res[a]
            sel = strided(M.shape[0], cap)
            mats.append(M[sel])
            pidx.extend([p] * len(sel))
        X = torch.tensor(np.vstack(mats), dtype=torch.float32, device=dev)
        idx = torch.tensor(pidx, dtype=torch.long, device=dev)
        return X, idx

    Xtr, ptr = flat_res(tr, args.cap_res)
    Xev, pev = flat_res(ev, args.cap_res)
    log.info(
        "learned arm units: train=%d residues over %d proteins | eval=%d over %d (cap=%d)",
        Xtr.shape[0],
        len(tr),
        Xev.shape[0],
        len(ev),
        args.cap_res,
    )

    clo_tr = [closures[i] for i in tr]
    clo_ev = [closures[i] for i in ev]
    ic_tr = information_content(clo_tr, dag)
    tpairs = sample_pairs(len(tr), args.train_pairs, rng)
    ytr = torch.tensor(
        np.asarray(
            lin_pairwise(
                clo_tr,
                ic_tr,
                tpairs,
                [max((ic_tr.get(t, 0.0) for t in c), default=0.0) for c in clo_tr],
            ),
            dtype=np.float32,
        ),
        device=dev,
    )
    ti = torch.tensor([p[0] for p in tpairs], device=dev)
    tj = torch.tensor([p[1] for p in tpairs], device=dev)

    ic_ev = information_content(clo_ev, dag)
    bic_ev = [max((ic_ev.get(t, 0.0) for t in c), default=0.0) for c in clo_ev]
    epairs_e = sample_pairs(len(ev), args.eval_pairs, rng)
    iie = np.array([p[0] for p in epairs_e])
    jje = np.array([p[1] for p in epairs_e])
    GOe = {
        "resnik": np.asarray(
            resnik_pairwise(clo_ev, ic_ev, epairs_e), dtype=np.float64
        ),
        "lin": np.asarray(
            lin_pairwise(clo_ev, ic_ev, epairs_e, bic_ev), dtype=np.float64
        ),
    }
    lens_ev = lens[ev]

    enc = nn.Linear(Xtr.shape[1], args.dict).to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)
    ALPHA = 20.0

    def bundle(X, pidx, n_prot):
        Z = enc(X)
        tau = torch.topk(Z, args.k, dim=1).values[:, -1:].detach()
        S = torch.sigmoid(ALPHA * (Z - tau))
        out = torch.zeros(n_prot, args.dict, device=dev)
        out.scatter_reduce_(
            0,
            pidx.unsqueeze(1).expand(-1, args.dict),
            S,
            reduce="amax",
            include_self=False,
        )
        return out

    def code_sim(Sp, i, j):
        si, sj = Sp[i], Sp[j]
        inter = (si * sj).sum(1)
        return inter / (si.sum(1) + sj.sum(1) - inter + 1e-6)

    def eval_binary(metric, mask=None):
        with torch.no_grad():
            Sp = bundle(Xev, pev, len(ev)).cpu().numpy().astype(np.float32)
        tan = tanimoto_dense(topk_binary_general(Sp, args.k), args.k)
        s = tan[iie, jje]
        g = GOe[metric]
        if mask is not None:
            s, g = s[mask], g[mask]
        return float(spearmanr(s, g)[0]) if len(s) > 100 else float("nan")

    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("sdr-c-residue")
        mlflow.start_run(run_name="A2-per-residue-learned-bundle")
        mlflow.log_params(
            {
                "n": n,
                "dict": args.dict,
                "k": args.k,
                "cap_res": args.cap_res,
                "granularity": "per-residue-bundle",
                "seed": args.seed,
            }
        )
        for m in ("resnik", "lin"):
            mlflow.log_metric(f"naive_dense_{m}", res_rows[m]["dense"])
            mlflow.log_metric(f"naive_real_{m}", res_rows[m]["naive_real"])
            mlflow.log_metric(f"naive_binary_{m}", res_rows[m]["naive_binary"])
        _ml = True
    except Exception as e:  # noqa: BLE001
        log.info("mlflow init skipped: %s", e)
        _ml = False

    log.info(
        "training A2 (per-residue learned bundle) on %s: dict=%d k=%d epochs=%d",
        dev,
        args.dict,
        args.k,
        args.epochs,
    )
    bs = 16384
    np_ = len(tpairs)
    for ep in range(args.epochs):
        Sp = bundle(Xtr, ptr, len(tr))
        opt.zero_grad()
        loss = torch.zeros((), device=dev)  # accumulate over pair batches, ONE backward
        for b in range(0, np_, bs):
            sl = slice(b, b + bs)
            s = code_sim(Sp, ti[sl], tj[sl])
            loss = loss + ((s - ytr[sl]) ** 2).sum() / np_
        loss.backward()  # single backward through the bundle graph
        tot = float(loss)
        opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            er, el = eval_binary("resnik"), eval_binary("lin")
            if _ml:
                mlflow.log_metric("train_loss", tot, step=ep)
                mlflow.log_metric("eval_rho_resnik", er, step=ep)
                mlflow.log_metric("eval_rho_lin", el, step=ep)
            log.info(
                "  epoch %3d loss=%.4f | A2 held-out resnik=%.4f lin=%.4f "
                "(dense %.4f / A0 ~0.49 / A1 ~0.48)",
                ep,
                tot,
                er,
                el,
                res_rows["resnik"]["dense"],
            )

    log.info("=== A2 FINAL (per-residue learned bundle) ===")
    for metric in ("resnik", "lin"):
        full = eval_binary(metric)
        small = eval_binary(metric, mask=(lens_ev[iie] <= 512) & (lens_ev[jje] <= 512))
        large = eval_binary(metric, mask=(lens_ev[iie] > 512) | (lens_ev[jje] > 512))
        log.info(
            "  %s: A2=%.4f (dense %.4f) | short pairs=%.4f | long(>512) pairs=%.4f",
            metric,
            full,
            res_rows[metric]["dense"],
            small,
            large,
        )
        if _ml:
            mlflow.log_metric(f"final_{metric}_A2", full)
            mlflow.log_metric(f"final_{metric}_A2_short", small)
            mlflow.log_metric(f"final_{metric}_A2_long", large)
    if _ml:
        mlflow.end_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
