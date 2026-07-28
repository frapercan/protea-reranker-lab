"""Content-based ATTENTION pooling (light-attention style) vs mean-learned, on the task metric.

Literature gap we are closing: our pooling sweep + soft-pool only tested MAGNITUDE-based aggregation
(mean, max, maxabs, GeM-style softmax on |value|). The methods that actually beat mean for protein
FUNCTION in the literature (SPROF-GO self-attention pooling; Light Attention, Staerk et al. 2021)
weight residues by a LEARNED CONTENT score, not by magnitude. This learns *which residues matter*
(active sites, domains) instead of *which have big values*.

Attention pooling here: a small scorer maps each residue e_i -> h head-scores; softmax over residues
(masked); pooled_head = sum_i a_i e_i; the h pooled vectors are concatenated and fed to the encoder.
h=1 is gated single-head (light-attention-like); h=4 is multi-head (Set-Transformer PMA-like).
Trained end to end with the encoder (cosine-real ~ Lin), held-out f_micro, same split as before.

Compare to: mean dense (~0.563), mean learned (~0.633, the cell to beat), maxabs dense (~0.590).
Read-only DB. GPU. MLflow.
"""
from __future__ import annotations

import argparse, logging, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sdr_c_real_fmicro as rl  # noqa: E402
from run_softpool_learned import train_loop, strided, p_mean, FixedInputEnc  # noqa: E402
from protea_reranker_lab.sdr import GoDag, propagate  # noqa: E402
import psycopg2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [attnpool] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("attnpool")
torch.manual_seed(42)

ANN_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"
RESDIR = "/home/frapercan/Thesis2/storage/fullgo_models/per_residue_v227"


class AttnPoolEnc(nn.Module):
    """Content-based multi-head attention pooling over residues, then a linear encoder."""
    def __init__(self, d, dict_, att_dim=128, heads=1):
        super().__init__()
        self.heads = heads
        self.W = nn.Linear(d, att_dim)
        self.v = nn.Linear(att_dim, heads, bias=False)
        self.lin = nn.Linear(d * heads, dict_)

    def forward(self, M, mask):
        s = self.v(torch.tanh(self.W(M)))                  # (B, cap, heads) content scores
        s = s.masked_fill(~mask.unsqueeze(-1), -1e9)
        a = torch.softmax(s, dim=1)                         # attention over residues, per head
        pooled = torch.einsum("bch,bcd->bhd", a, M)         # (B, heads, d) weighted sums
        return self.lin(pooled.reshape(M.shape[0], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="host=localhost dbname=protea user=protea password=protea")
    ap.add_argument("--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
    ap.add_argument("--dict", type=int, default=2048)
    ap.add_argument("--eval-k", type=int, default=128)
    ap.add_argument("--cap-res", type=int, default=256)
    ap.add_argument("--att-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-pairs", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s cap_res=%d", dev, args.cap_res)

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
    n = len(acc); d = 768
    res = {a: np.load(os.path.join(RESDIR, f"{a}.npy")).astype(np.float32) for a in acc}
    log.info("n=%d", n)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n); n_tr = n // 2
    tr, ev = perm[:n_tr], perm[n_tr:]
    clo_tr = [closures[i] for i in tr]; clo_ev = [closures[i] for i in ev]

    # mean-learned reference (full residues), same machinery
    mean_v = rl.l2n(np.vstack([p_mean(res[a]) for a in acc]).astype(np.float32))
    Xmean = torch.tensor(mean_v, device=dev)
    mean_dense = rl.fmicro_from_S(rl.cos_S(mean_v[ev]), clo_ev)

    # padded residue tensor for attention (strided cap)
    cap = args.cap_res
    Xpad = np.zeros((n, cap, d), np.float32); mask = np.zeros((n, cap), bool)
    capped = 0
    for i, a in enumerate(acc):
        M = res[a]; sel = strided(M.shape[0], cap)
        Xpad[i, :len(sel)] = M[sel]; mask[i, :len(sel)] = True
        capped += int(M.shape[0] > cap)
    log.info("residues capped/strided for %d/%d proteins (len>%d)", capped, n, cap)
    Xpad_t = torch.tensor(Xpad, device=dev); mask_t = torch.tensor(mask, device=dev)
    tr_t = torch.tensor(tr, device=dev); ev_t = torch.tensor(ev, device=dev)

    try:
        import mlflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("attn-pool-learned")
        mlflow.start_run(run_name="content-attention-pooling vs mean-learned")
        _ml = True
    except Exception as e:  # noqa: BLE001
        log.info("mlflow init skipped: %s", e); _ml = False

    table = {"mean dense": mean_dense}

    # reference: mean learned (FixedInputEnc on full-residue mean)
    enc = FixedInputEnc(d, args.dict).to(dev)
    def fwd_mean(which, enc=enc):
        idx = tr_t if which == "train" else ev_t
        return enc(Xmean[idx])
    log.info("=== reference: mean learned ===")
    table["mean learned"], _ = train_loop(fwd_mean, list(enc.parameters()), tr, ev, clo_tr, clo_ev,
                                           dag, args, dev, "mean")

    # attention pooling, single-head and multi-head
    for heads in (1, 4):
        ap_mod = AttnPoolEnc(d, args.dict, att_dim=args.att_dim, heads=heads).to(dev)
        def fwd_attn(which, m=ap_mod):
            idx = tr_t if which == "train" else ev_t
            return m(Xpad_t[idx], mask_t[idx])
        log.info("=== attention pooling: heads=%d ===", heads)
        bf, _ = train_loop(fwd_attn, list(ap_mod.parameters()), tr, ev, clo_tr, clo_ev,
                           dag, args, dev, f"attn-h{heads}")
        table[f"attn heads={heads} learned"] = bf
        if _ml:
            mlflow.log_metric(f"attn_h{heads}", bf)

    if _ml:
        mlflow.log_metric("mean_dense", mean_dense); mlflow.log_metric("mean_learned", table["mean learned"])

    log.info("=== SUMMARY (held-out f_micro, same split, cap=%d) ===", cap)
    base = table["mean dense"]
    for k in ("mean dense", "mean learned", "attn heads=1 learned", "attn heads=4 learned"):
        if k in table:
            log.info("  %-26s %.4f  (%+.4f vs mean-dense)", k, table[k], table[k] - base)
    best = max(table, key=lambda k: table[k])
    ml = table["mean learned"]
    log.info(">> BEST: %s = %.4f | vs mean-learned (%.4f): %+.4f",
             best, table[best], ml, table[best] - ml)
    if _ml:
        mlflow.end_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
