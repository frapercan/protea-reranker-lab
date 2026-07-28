"""Extract the MAX from the signal: a LEARNED pooling that interpolates mean<->maxabs.

Established (same per-residue 5000 sample, held-out f_micro):
  - best aggregation ALONE        = maxabs (max-|.|, keep sign)   (+0.036 vs mean, no learning)
  - best aggregation to ALIGN     = mean   (learned-on-mean +0.072; learned-on-maxabs HURTS)
  - they conflict: maxabs is peaky-but-jagged (bad for gradient), mean is smooth-but-diluted.

This tests the synthesis -- don't pick a fixed operator, LEARN the pooling:

  Exp1 (cheap de-risk):  encoder on input = [mean (+) maxabs]  (concat at the ENCODER INPUT, not
                          the raw-kNN concat that already lost). Can the encoder reweight to use
                          both?
  Exp2 (the real lever): SOFT-POOL with a learnable temperature beta. Per dimension, weights =
                          softmax(beta * |residue|) over residues; pooled = sum(w * residue).
                          beta->0 => mean, beta->inf => maxabs (picks the peak-magnitude residue,
                          KEEPING its sign). Smooth+differentiable => trainable; expressive enough
                          to emphasise peaks. Train beta jointly with the encoder, end to end.

Reports a table vs the known cells (mean/maxabs x dense/learned) and the LEARNED beta (tells us
where on the mean<->maxabs axis the optimum sits). Read-only DB. GPU. MLflow.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sdr_c_real_fmicro as rl  # noqa: E402
from protea_reranker_lab.sdr import GoDag, propagate, information_content, lin_pairwise  # noqa: E402
import psycopg2  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [softpool] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("softpool")
torch.manual_seed(42)

ANN_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"
RESDIR = "/home/frapercan/Thesis2/storage/fullgo_models/per_residue_v227"


def p_mean(M):
    return M.mean(0)


def p_maxabs(M):
    idx = np.abs(M).argmax(0)
    return M[idx, np.arange(M.shape[1])]


def strided(L, cap):
    if L <= cap:
        return np.arange(L)
    return np.linspace(0, L - 1, cap).round().astype(int)


class FixedInputEnc(nn.Module):
    """Encoder over a precomputed pooled vector (Exp1 + mean/maxabs learned cells)."""

    def __init__(self, d, dict_):
        super().__init__()
        self.lin = nn.Linear(d, dict_)

    def forward(self, X):
        return self.lin(X)


class SoftPoolEnc(nn.Module):
    """Learnable soft-pool over residues (softmax on magnitude, temperature beta) -> encoder."""

    def __init__(self, d, dict_):
        super().__init__()
        self.raw_beta = nn.Parameter(
            torch.tensor(0.0)
        )  # softplus(0)=0.69 -> mild peaking at init
        self.lin = nn.Linear(d, dict_)

    def beta(self):
        return F.softplus(self.raw_beta)

    def pool(self, M, mask):
        # M: (B, cap, d)  mask: (B, cap) bool
        logits = self.beta() * M.abs()
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e9)
        w = torch.softmax(logits, dim=1)  # weights over residues, per dim
        return (w * M).sum(1)  # (B, d)

    def forward(self, M, mask):
        return self.lin(self.pool(M, mask))


def train_loop(forward_Z, params, tr, ev, clo_tr, clo_ev, dag, args, dev, label):
    """Generic train: forward_Z(idx_slice, which) -> Z for those proteins. Returns best f_micro."""
    ic_tr = information_content(clo_tr, dag)
    bic_tr = [max((ic_tr.get(t, 0.0) for t in c), default=0.0) for c in clo_tr]
    rng = np.random.default_rng(args.seed)
    tp = rl.sample_pairs(len(tr), args.train_pairs, rng)
    Y = torch.tensor(
        np.asarray(lin_pairwise(clo_tr, ic_tr, tp, bic_tr), dtype=np.float32),
        device=dev,
    )
    TI = torch.tensor([p[0] for p in tp], device=dev)
    TJ = torch.tensor([p[1] for p in tp], device=dev)
    opt = torch.optim.Adam(params, lr=args.lr)
    np_ = len(tp)
    bs = 32768
    EVAL_EVERY = 15
    PATIENCE = 6
    best_f, best_ep, since = -1.0, -1, 0
    try:
        import mlflow as _mlf

        _ml_on = _mlf.active_run() is not None
    except Exception:  # noqa: BLE001
        _ml_on = False
    for e in range(args.epochs):
        Ztr = forward_Z("train")
        opt.zero_grad()
        loss = torch.zeros((), device=dev)
        for b in range(0, np_, bs):
            sl = slice(b, b + bs)
            zi, zj = Ztr[TI[sl]], Ztr[TJ[sl]]
            cos = (zi * zj).sum(1) / (zi.norm(dim=1) * zj.norm(dim=1) + 1e-8)
            loss = loss + ((cos - Y[sl]) ** 2).sum() / np_
        loss.backward()
        opt.step()
        if e % EVAL_EVERY == 0 or e == args.epochs - 1:
            with torch.no_grad():
                Zev = forward_Z("eval").detach().cpu().numpy().astype(np.float32)
            f_sp = rl.fmicro_from_S(rl.cos_S(rl.topk_real(Zev, args.eval_k)), clo_ev)
            extra = ""
            log.info(
                "  [%s] epoch %3d loss=%.4f | f_micro=%.4f%s",
                label,
                e,
                float(loss),
                f_sp,
                extra,
            )
            if _ml_on:  # live per-epoch curve, one metric series per arm/label
                key = (
                    label.replace("(", "_")
                    .replace(")", "")
                    .replace("+", "_p_")
                    .replace(" ", "")
                )
                _mlf.log_metric(f"fmicro_{key}", f_sp, step=e)
                _mlf.log_metric(f"loss_{key}", float(loss), step=e)
            if f_sp > best_f + 1e-4:
                best_f, best_ep, since = f_sp, e, 0
            else:
                since += 1
                if since >= PATIENCE:
                    log.info(
                        "  [%s] PLATEAU @ %d (best=%.4f @ %d)",
                        label,
                        e,
                        best_f,
                        best_ep,
                    )
                    break
    return best_f, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dsn", default="host=localhost dbname=protea user=protea password=protea"
    )
    ap.add_argument(
        "--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
    )
    ap.add_argument("--dict", type=int, default=2048)
    ap.add_argument("--eval-k", type=int, default=128)
    ap.add_argument("--cap-res", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-pairs", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s", dev)

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
            cl = propagate(leaves[a], dag)
            if cl:
                acc.append(a)
                closures.append(cl)
    n = len(acc)
    res = {a: np.load(os.path.join(RESDIR, f"{a}.npy")).astype(np.float32) for a in acc}
    d = 768
    log.info("n=%d", n)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_tr = n // 2
    tr, ev = perm[:n_tr], perm[n_tr:]
    clo_tr = [closures[i] for i in tr]
    clo_ev = [closures[i] for i in ev]

    # precomputed pooled inputs (full residues) for the fixed-input cells
    mean_v = rl.l2n(np.vstack([p_mean(res[a]) for a in acc]).astype(np.float32))
    maxabs_v = rl.l2n(np.vstack([p_maxabs(res[a]) for a in acc]).astype(np.float32))
    concat_v = np.hstack([mean_v, maxabs_v]).astype(np.float32)  # [mean (+) maxabs]
    fixed = {
        "mean": torch.tensor(mean_v, device=dev),
        "maxabs": torch.tensor(maxabs_v, device=dev),
        "concat(mean+maxabs)": torch.tensor(concat_v, device=dev),
    }

    # padded residue tensor for the soft-pool (strided cap)
    cap = args.cap_res
    Xpad = np.zeros((n, cap, d), np.float32)
    mask = np.zeros((n, cap), bool)
    for i, a in enumerate(acc):
        M = res[a]
        sel = strided(M.shape[0], cap)
        Xpad[i, : len(sel)] = M[sel]
        mask[i, : len(sel)] = True
    Xpad_t = torch.tensor(Xpad, device=dev)
    mask_t = torch.tensor(mask, device=dev)

    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("softpool-learned")
        mlflow.start_run(run_name="learned-pooling mean<->maxabs")
        _ml = True
    except Exception as e:  # noqa: BLE001
        log.info("mlflow skipped: %s", e)
        _ml = False

    table = {}
    # dense (no-learn) references on this split
    for nm, X in fixed.items():
        if nm == "concat(mean+maxabs)":
            continue
        Xn = X.cpu().numpy()
        table[f"{nm} dense"] = rl.fmicro_from_S(rl.cos_S(Xn[ev]), clo_ev)

    # Exp1: learned encoder on fixed inputs (mean, maxabs, concat)
    for nm, X in fixed.items():
        enc = FixedInputEnc(X.shape[1], args.dict).to(dev)
        Xtr_idx = torch.tensor(tr, device=dev)
        Xev_idx = torch.tensor(ev, device=dev)

        def fwd(which, enc=enc, X=X, Xtr_idx=Xtr_idx, Xev_idx=Xev_idx):
            return enc(X[Xtr_idx]) if which == "train" else enc(X[Xev_idx])

        log.info("=== Exp1: learned encoder on input=%s ===", nm)
        bf, _ = train_loop(
            fwd, list(enc.parameters()), tr, ev, clo_tr, clo_ev, dag, args, dev, nm
        )
        table[f"{nm} learned"] = bf
        if _ml:
            mlflow.log_metric(
                f"learned_{nm.replace('(', '_').replace(')', '').replace('+', '_p_')}",
                bf,
            )

    # Exp2: learnable soft-pool + encoder, end to end
    sp = SoftPoolEnc(d, args.dict).to(dev)
    tr_t = torch.tensor(tr, device=dev)
    ev_t = torch.tensor(ev, device=dev)

    def fwd_sp(which, sp=sp, tr_t=tr_t, ev_t=ev_t):
        idx = tr_t if which == "train" else ev_t
        return sp(Xpad_t[idx], mask_t[idx])

    log.info(
        "=== Exp2: learnable SOFT-POOL (softmax-magnitude temperature beta) + encoder ==="
    )
    bf_sp, _ = train_loop(
        fwd_sp,
        list(sp.parameters()),
        tr,
        ev,
        clo_tr,
        clo_ev,
        dag,
        args,
        dev,
        "softpool",
    )
    table["softpool learned"] = bf_sp
    learned_beta = float(sp.beta().detach().cpu())
    log.info("  learned beta=%.4f (0=mean, large=maxabs)", learned_beta)
    if _ml:
        mlflow.log_metric("softpool_learned", bf_sp)
        mlflow.log_metric("softpool_beta", learned_beta)

    log.info("=== SUMMARY (held-out f_micro, same split) ===")
    order = [
        "mean dense",
        "maxabs dense",
        "mean learned",
        "maxabs learned",
        "concat(mean+maxabs) learned",
        "softpool learned",
    ]
    base = table.get("mean dense", 0.0)
    for k in order:
        if k in table:
            log.info(
                "  %-28s %.4f  (%+.4f vs mean-dense)", k, table[k], table[k] - base
            )
    best = max(table, key=lambda k: table[k])
    log.info(
        ">> BEST: %s = %.4f (%+.4f vs mean-dense) | learned soft-pool beta=%.4f",
        best,
        table[best],
        table[best] - base,
        learned_beta,
    )
    if _ml:
        mlflow.end_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
