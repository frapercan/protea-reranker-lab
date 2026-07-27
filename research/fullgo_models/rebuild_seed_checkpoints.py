"""Reproduce the 7 seed checkpoints of the M2 anc2vec classifier (the #1 LAFA
champion's seed-averaged head) from zero, deterministically.

The offline champion averaged 7 seeds at the prediction-TSV level and never
persisted per-seed checkpoints. PROTEA's native serve path
(``protea.core.classifier_producer.SeedAveragedClassifier``, PR #639) needs N
self-contained checkpoints to reproduce the same average in process. This
script rebuilds the 6-PLM training matrix (v227 experimental proteins) and
trains the EXACT serve module (``build_hybrid`` imported from the producer,
guaranteeing state_dict compatibility) once per fixed seed, saving a
self-contained checkpoint per seed.

Determinism: fixed seed list (no unseeded "base" run), cudnn deterministic,
seeded numpy + torch per seed. Converged seed-averaging (anti winner's curse):
the average over the 7 seeds is the deliverable, not any single lucky seed.

Run with the PROTEA venv python:
  cd repositories/PROTEA && .venv/bin/python \
    ../../storage/fullgo_models/rebuild_seed_checkpoints.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import psycopg
import torch

# Import the EXACT serve module + label-matrix builder so the checkpoints we
# save load byte-for-byte in classifier_producer at inference time.
from protea.core.classifier_producer import (  # noqa: E402
    PLM_CONCAT_ORDER,
    _label_matrix_for_vocab,
    build_hybrid,
)

DB = "postgresql://protea:protea@localhost:5432/protea"
V227 = "c905dffa-a5ce-430b-b17b-503e88666adb"
EXP = (
    "EXP", "IDA", "IMP", "IPI", "IGI", "IEP", "TAS", "IC",
    "HTP", "HDA", "HMP", "HGI", "HEP",
)
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
TOI_PATH = (
    "/home/frapercan/Thesis2/CAFA_forever/data/releases/"
    "Sep_2025_Mar_2026/groundtruth_terms_of_interest.txt"
)
ANC = (
    "/home/frapercan/Thesis2/worktrees/protea-deploy/"
    "artifacts/anc2vec/anc2vec_2020-10.npz"
)
OUT_DIR = "/home/frapercan/Thesis2/storage/fullgo_models/seeds"
SEEDS = [0, 7, 137, 23, 91, 31, 53]
HIDDEN = 1024
EPOCHS = 30
BS = 512
IN_DIM = sum(dim for _, _, dim in PLM_CONCAT_ORDER)  # 8320


def parents_map() -> dict[str, set[str]]:
    parents: dict[str, set[str]] = defaultdict(set)
    cur = None
    for line in open(OBO):
        line = line.strip()
        if line == "[Term]":
            cur = None
        elif line.startswith("id: GO:"):
            cur = line[4:]
        elif line.startswith("is_a:") and cur:
            parents[cur].add(line.split()[1])
        elif line.startswith("relationship: part_of") and cur:
            pr = line.split()
            if len(pr) >= 3:
                parents[cur].add(pr[2])
    return parents


def ancestors(term: str, parents: dict[str, set[str]], cache: dict[str, set[str]]) -> set[str]:
    if term in cache:
        return cache[term]
    out: set[str] = set()
    stack = list(parents.get(term, ()))
    while stack:
        a = stack.pop()
        if a in out:
            continue
        out.add(a)
        stack.extend(parents.get(a, ()))
    cache[term] = out
    return out


def build_labels() -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Return (train_accessions, vocab, label_rows, label_cols)."""
    toi = {l.strip() for l in open(TOI_PATH) if l.strip().startswith("GO:")}
    parents = parents_map()
    cache: dict[str, set[str]] = {}
    conn = psycopg.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "select a.protein_accession, g.go_id from protein_go_annotation a "
        "join go_term g on g.id=a.go_term_id where a.annotation_set_id=%s "
        "and a.evidence_code=any(%s) and coalesce(a.qualifier,'') not like '%%NOT%%' "
        "and g.aspect in ('F','P','C')",
        (V227, list(EXP)),
    )
    leaf: dict[str, set[str]] = defaultdict(set)
    for acc, go in cur:
        leaf[acc].add(go)
    conn.close()
    prop: dict[str, set[str]] = {}
    vc: dict[str, int] = defaultdict(int)
    for acc, terms in leaf.items():
        s = set(terms)
        for t in terms:
            s |= ancestors(t, parents, cache)
        s &= toi
        if s:
            prop[acc] = s
            for t in s:
                vc[t] += 1
    vocab = sorted(vc)
    tidx = {t: i for i, t in enumerate(vocab)}
    tr_acc = sorted(prop)
    acc_idx = {a: i for i, a in enumerate(tr_acc)}
    rows: list[int] = []
    cols: list[int] = []
    for acc, terms in prop.items():
        i = acc_idx[acc]
        for t in terms:
            j = tidx.get(t)
            if j is not None:
                rows.append(i)
                cols.append(j)
    print(f"train={len(tr_acc)} vocab={len(vocab)} nnz={len(rows)}", flush=True)
    return tr_acc, vocab, np.asarray(rows, np.int64), np.asarray(cols, np.int64)


def build_matrix(tr_acc: list[str]) -> np.ndarray:
    """6-PLM concat (in PLM_CONCAT_ORDER) for tr_acc; missing -> zero rows."""
    n = len(tr_acc)
    x = np.zeros((n, IN_DIM), np.float32)
    pos = {a: i for i, a in enumerate(tr_acc)}
    conn = psycopg.connect(DB)
    cur = conn.cursor()
    off = 0
    for name, cfg, dim in PLM_CONCAT_ORDER:
        seen = 0
        block = 5000
        for i in range(0, n, block):
            chunk = tr_acc[i : i + block]
            cur.execute(
                "select p.accession, e.embedding::text from protein p "
                "join sequence s on s.id=p.sequence_id "
                "join sequence_embedding e on e.sequence_id=s.id "
                "where e.embedding_config_id=%s and p.accession=any(%s)",
                (cfg, chunk),
            )
            for acc, vec in cur:
                arr = np.fromstring(vec.strip("[]"), sep=",", dtype=np.float32)
                if len(arr) == dim:
                    x[pos[acc], off : off + dim] = arr
                    seen += 1
        print(f"  {name} ({cfg[:8]}) dim={dim} filled={seen}/{n}", flush=True)
        off += dim
    conn.close()
    assert off == IN_DIM, (off, IN_DIM)
    return x


def asl(logits: torch.Tensor, target: torch.Tensor, gn=4.0, gp=1.0, clip=0.05, eps=1e-8):
    p = torch.sigmoid(logits)
    pm = (p - clip).clamp(min=0)
    pos = target * torch.log(p.clamp(min=eps)) * (1 - p) ** gp
    neg = (1 - target) * torch.log((1 - pm).clamp(min=eps)) * (pm**gn)
    return -(pos + neg).mean()


def _ema_update(ema: dict, model: torch.nn.Module, decay: float) -> None:
    """In-place exponential moving average of the model's parameters/buffers."""
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
            else:
                ema[k].copy_(v)


def train_seed(
    seed, xn, label_csr, lt, mu, sd, vocab, dev,
    mode="ensemble", ema_decay=0.999, swa_start_frac=0.75,
):
    """Train one classifier.

    mode="ensemble": a single Deep-Ensemble member (averaged at SERVE time
    over the seed set). mode="ema": exponential-moving-average of the weights
    along the run (one model, no K-checkpoint dependence). mode="swa":
    Stochastic Weight Averaging of the epoch-end weights over the tail of the
    run. EMA/SWA are valid here because the trunk uses LayerNorm (no batch
    running stats to recompute) so straight weight averaging is well defined.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dev == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_hybrid(IN_DIM, HIDDEN, len(vocab), lt.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    n = xn.shape[0]
    idx = np.arange(n)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if mode == "ema" else None
    swa_acc: dict = {}
    swa_count = 0
    swa_start = int(EPOCHS * swa_start_frac)
    for ep in range(EPOCHS):
        model.train()
        np.random.shuffle(idx)
        for i in range(0, n, BS):
            b = idx[i : i + BS]
            xb = xn[b].to(dev)
            yb = torch.tensor(label_csr[b].toarray(), dtype=torch.float32, device=dev)
            opt.zero_grad()
            loss = asl(model(xb, lt), yb)
            loss.backward()
            opt.step()
            if mode == "ema":
                _ema_update(ema, model, ema_decay)
        if mode == "swa" and ep >= swa_start:
            for k, v in model.state_dict().items():
                swa_acc[k] = v.detach().clone() if k not in swa_acc else swa_acc[k] + v.detach()
            swa_count += 1
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"    {mode} seed {seed} epoch {ep} loss {loss.item():.4f}", flush=True)
    if mode == "ema":
        final = {k: v.cpu() for k, v in ema.items()}
    elif mode == "swa":
        final = {
            k: (v / swa_count).cpu() if v.dtype.is_floating_point else v.cpu()
            for k, v in swa_acc.items()
        }
    else:
        model.eval()
        final = {k: v.cpu() for k, v in model.state_dict().items()}
    return {
        "state_dict": final,
        "mu": mu.cpu().numpy(),
        "sd": sd.cpu().numpy(),
        "in_dim": IN_DIM,
        "hidden": HIDDEN,
        "label_dim": int(lt.shape[1]),
        "vocab": list(vocab),
        "seed": seed,
        "arch": "hybrid_anc2vec_m2",
        "mode": mode,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seeds",
        default=",".join(str(s) for s in SEEDS),
        help="comma-separated seed list (default = canonical 7-seed set). "
        "Convergence draws pass a DIFFERENT set to prove seed-independence.",
    )
    ap.add_argument("--out-dir", default=OUT_DIR, help="checkpoint output dir")
    ap.add_argument(
        "--mode",
        default="ensemble",
        choices=["ensemble", "ema", "swa"],
        help="ensemble = Deep-Ensemble members (serve-time output avg); "
        "ema/swa = single weight-averaged model (one checkpoint, no K dependence).",
    )
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    # EMA/SWA collapse to a single run -> a single checkpoint.
    if args.mode in ("ema", "swa"):
        seeds = seeds[:1]
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} in_dim={IN_DIM} seeds={seeds} out={out_dir}", flush=True)

    tr_acc, vocab, rows, cols = build_labels()
    from scipy.sparse import csr_matrix

    label_csr = csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)),
        shape=(len(tr_acc), len(vocab)),
    )
    t0 = time.time()
    x = build_matrix(tr_acc)
    print(f"matrix {x.shape} built in {time.time()-t0:.0f}s", flush=True)

    xt = torch.tensor(x, dtype=torch.float32)
    del x
    mu = xt.mean(0, keepdim=True)
    sd = xt.std(0, keepdim=True) + 1e-6
    xn = (xt - mu) / sd
    del xt

    lm = _label_matrix_for_vocab(list(vocab), ANC, int(np.load(ANC, allow_pickle=True)["embeddings"].shape[1]))
    lt = torch.tensor(lm, dtype=torch.float32, device=dev)

    for k, seed in enumerate(seeds):
        t1 = time.time()
        ckpt = train_seed(seed, xn, label_csr, lt, mu, sd, vocab, dev, mode=args.mode)
        tag = args.mode if args.mode != "ensemble" else f"seed_{k}_{seed}"
        path = os.path.join(out_dir, f"{tag}.pt")
        torch.save(ckpt, path)
        print(f"  saved {path} ({time.time()-t1:.0f}s)", flush=True)

    spec = {
        "source": "rebuild_seed_checkpoints.py",
        "train_annotation_set": V227,
        "train_proteins": len(tr_acc),
        "vocab": len(vocab),
        "plm_order": [name for name, _, _ in PLM_CONCAT_ORDER],
        "in_dim": IN_DIM,
        "hidden": HIDDEN,
        "epochs": EPOCHS,
        "seeds": seeds,
        "mode": args.mode,
        "averaging": "union per (protein,term), score=sum_present/n_seeds",
        "note": "converged seed-averaging; load all seeds via PROTEA_CLASSIFIER_SEED_DIR",
    }
    with open(os.path.join(out_dir, "feature_spec.json"), "w") as w:
        json.dump(spec, w, indent=2)
    print(f"DONE: {len(seeds)} checkpoints -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
