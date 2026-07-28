"""Do the two task levers COMPOSE? maxabs-pool (free, +0.036) x learned-real (learned, +0.05).

learned-real trained on MEAN-pooled features. maxabs-pool beat mean WITHOUT learning. Question:
if we train the SAME learned-real encoder on MAXABS-pooled features (as input, NOT a concat -- the
concat already lost in the sweep), does it climb above both? That would mean the levers stack.

2x2 on the SAME per-residue 5000 sample (leave-one-out k-NN GO-transfer f_micro, held-out eval):
            dense (no learning)      learned-real (encoder, converged)
  mean      ~0.615 (platform)        ?
  maxabs    ~0.651 (free lever)      ?   <- the cell that matters

Reuses the learned-real machinery verbatim (encoder Linear 768->dict, cosine-real ~ Lin objective,
f_micro on topk_real, early-stop). Read-only DB. CPU. MLflow.
"""
from __future__ import annotations

import argparse, logging, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sdr_c_real_fmicro as rl  # noqa: E402  (l2n, topk_real, sample_pairs, fmicro_from_S, cos_S)
from protea_reranker_lab.sdr import (  # noqa: E402
    GoDag, propagate, information_content, lin_pairwise,
)
import psycopg2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [compose] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("compose")
torch.manual_seed(42)

ANN_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"
RESDIR = "/home/frapercan/Thesis2/storage/fullgo_models/per_residue_v227"


def p_mean(M):
    return M.mean(0)


def p_maxabs(M):
    idx = np.abs(M).argmax(0)
    return M[idx, np.arange(M.shape[1])]


def train_learned(Xn, tr, ev, clo_tr, clo_ev, dag, args, rng, label):
    """Train the learned-real encoder on input Xn (L2-normed). Return converged sparse-k f_micro."""
    Xtr = torch.tensor(Xn[tr]); Xev = torch.tensor(Xn[ev]); d = Xn.shape[1]
    ic_tr = information_content(clo_tr, dag)
    bic_tr = [max((ic_tr.get(t, 0.0) for t in c), default=0.0) for c in clo_tr]
    tp = rl.sample_pairs(len(tr), args.train_pairs, rng)
    Y = torch.tensor(np.asarray(lin_pairwise(clo_tr, ic_tr, tp, bic_tr), dtype=np.float32))
    TI = torch.tensor([p[0] for p in tp]); TJ = torch.tensor([p[1] for p in tp])

    enc = nn.Linear(d, args.dict); opt = torch.optim.Adam(enc.parameters(), lr=args.lr)
    bs = 16384; np_ = len(tp); EVAL_EVERY = 15; PATIENCE = 6
    best_f, best_ep, since, _best_Z = -1.0, -1, 0, None
    for e in range(args.epochs):
        Z = enc(Xtr); opt.zero_grad()
        loss = torch.zeros(())
        for b in range(0, np_, bs):
            sl = slice(b, b + bs)
            zi, zj = Z[TI[sl]], Z[TJ[sl]]
            cos = (zi * zj).sum(1) / (zi.norm(dim=1) * zj.norm(dim=1) + 1e-8)
            loss = loss + ((cos - Y[sl]) ** 2).sum() / np_
        loss.backward(); opt.step()
        if e % EVAL_EVERY == 0 or e == args.epochs - 1:
            with torch.no_grad():
                Zev = enc(Xev).numpy().astype(np.float32)
            f_sp = rl.fmicro_from_S(rl.cos_S(rl.topk_real(Zev, args.eval_k)), clo_ev)
            log.info("  [%s] epoch %3d loss=%.4f | learned f_micro=%.4f", label, e, float(loss), f_sp)
            if f_sp > best_f + 1e-4:
                best_f, best_ep, since, _best_Z = f_sp, e, 0, Zev
            else:
                since += 1
                if since >= PATIENCE:
                    log.info("  [%s] PLATEAU @ %d (best=%.4f @ %d)", label, e, best_f, best_ep)
                    break
    return best_f, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="host=localhost dbname=protea user=protea password=protea")
    ap.add_argument("--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
    ap.add_argument("--dict", type=int, default=2048)
    ap.add_argument("--eval-k", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-pairs", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

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
    log.info("n=%d", n)

    inputs = {
        "mean": rl.l2n(np.vstack([p_mean(res[a]) for a in acc]).astype(np.float32)),
        "maxabs": rl.l2n(np.vstack([p_maxabs(res[a]) for a in acc]).astype(np.float32)),
    }

    perm = rng.permutation(n); n_tr = n // 2
    tr, ev = perm[:n_tr], perm[n_tr:]
    clo_tr = [closures[i] for i in tr]; clo_ev = [closures[i] for i in ev]

    try:
        import mlflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("compose-pool-learned")
        mlflow.start_run(run_name="mean-vs-maxabs x dense-vs-learned")
        _ml = True
    except Exception as e:  # noqa: BLE001
        log.info("mlflow skipped: %s", e); _ml = False

    table = {}
    for pool, Xn in inputs.items():
        f_dense = rl.fmicro_from_S(rl.cos_S(Xn[ev]), clo_ev)         # no learning
        log.info("=== input=%s : dense(no-learn) f_micro=%.4f -> training encoder ===", pool, f_dense)
        f_learned, ep = train_learned(Xn, tr, ev, clo_tr, clo_ev, dag, args, rng, pool)
        table[pool] = {"dense": f_dense, "learned": f_learned, "best_ep": ep}
        if _ml:
            mlflow.log_metric(f"{pool}_dense", f_dense)
            mlflow.log_metric(f"{pool}_learned", f_learned)

    log.info("=== 2x2: pooling x learning (held-out f_micro) ===")
    log.info("  %-8s %12s %12s %10s", "input", "dense", "learned", "learn_gain")
    for pool in ("mean", "maxabs"):
        t = table[pool]
        log.info("  %-8s %12.4f %12.4f %+10.4f", pool, t["dense"], t["learned"], t["learned"] - t["dense"])
    base = table["mean"]["dense"]
    log.info("=== deltas vs mean-dense (platform baseline %.4f) ===", base)
    for pool in ("mean", "maxabs"):
        for mode in ("dense", "learned"):
            log.info("  %-8s %-8s  %+.4f", pool, mode, table[pool][mode] - base)
    bestcell = max(((p, m) for p in inputs for m in ("dense", "learned")), key=lambda pm: table[pm[0]][pm[1]])
    log.info(">> BEST CELL: %s + %s = %.4f (%+.4f vs platform). Compose? maxabs-learned vs maxabs-dense = %+.4f, vs mean-learned = %+.4f",
             bestcell[0], bestcell[1], table[bestcell[0]][bestcell[1]], table[bestcell[0]][bestcell[1]] - base,
             table["maxabs"]["learned"] - table["maxabs"]["dense"],
             table["maxabs"]["learned"] - table["mean"]["learned"])
    if _ml:
        mlflow.end_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
