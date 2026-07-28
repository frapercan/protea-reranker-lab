"""CLOSING TEST: does the learned-real encoder beat dense-mean on the REAL LAFA benchmark?

Upgrades the isolated sample experiments (random leave-one-out, hand-rolled f_micro) to the real
setup, on the OFFICIAL 7401 frame (sidesteps the phantom-gap cross-OBO bug entirely):

  - REAL split: reference pool = v227 t0 annotated proteins (GO-transfer source + encoder train set);
    query = the 7401 official LAFA targets (excluded from the reference -> no leakage).
  - REAL metric: cafaeval IA-weighted f_micro_w (PROTEA venv), official v227 OBO + IA, per NK/LK/PK.
  - Two representations over the SAME ankh-chunked mean embeddings: dense-mean vs learned-real encoder
    (Linear 768->2048, cosine-Lin objective, top-k real). KNN GO-transfer to the queries for each.

Reports NK/LK/PK + NK+LK-mean f_micro_w for dense vs learned. The dense-vs-learned DELTA is the
verdict; it is robust to the phantom-gap (both arms share any framing bias).

Read-only DB. GPU for the encoder. cafaeval shelled to the PROTEA venv.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import psycopg2
import scipy.sparse as sp
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sdr_c_real_fmicro as rl  # noqa: E402  (l2n, topk_real, sample_pairs, cos helpers)
from protea_reranker_lab.sdr import GoDag, propagate, information_content, lin_pairwise  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bench-enc] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench-enc")
torch.manual_seed(42)

ANN_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"  # GOA v227, t0 (reference labels)
ANKH_CFG = "6542db1e-202a-4769-b933-2e0f85aa81e6"  # ankh-base chunked, 768
GT_DIR = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PROTEA_PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
DSN = "host=localhost dbname=protea user=protea password=protea"
ASPECT = {
    "F": "molecular_function",
    "P": "biological_process",
    "C": "cellular_component",
}


def parse_halfvec(text):
    return np.fromstring(text[1:-1], sep=",", dtype=np.float32)


def pull_mean(cur, accs, cfg):
    """accession -> mean-pooled embedding over its chunks, for the given config."""
    cur.execute(
        """SELECT p.accession, se.embedding::text
           FROM protein p JOIN sequence_embedding se ON se.sequence_id = p.sequence_id
           WHERE se.embedding_config_id = %s AND p.accession = ANY(%s)
           ORDER BY p.accession, se.chunk_index_s""",
        (cfg, list(accs)),
    )
    acc_chunks: dict[str, list] = {}
    for acc, emb in cur.fetchall():
        acc_chunks.setdefault(acc, []).append(parse_halfvec(emb))
    return {a: np.vstack(v).mean(0).astype(np.float32) for a, v in acc_chunks.items()}


def load_gt():
    """7401 query GT per category cell. cell -> {proteins:set, pairs:set((acc,go)), aspect}."""
    cells = {}
    for cat in ("NK", "LK", "PK"):
        for ln in open(GT_DIR / f"groundtruth_{cat}.tsv"):
            acc, go, asp = ln.rstrip("\n").split("\t")[:3]
            if acc == "EntryID":
                continue
            key = (cat, asp)
            d = cells.setdefault(
                key, {"proteins": set(), "pairs": set(), "aspect": asp}
            )
            d["proteins"].add(acc)
            d["pairs"].add((acc, go))
    return cells


def knn_transfer(Q, R, ref_closures, ref_terms_ix, knn, batch=1000):
    """Each query row: cosine top-knn in R, similarity-weighted vote over reference closures.
    Returns CSR (n_query, n_terms) of scores."""
    Rn = rl.l2n(R)
    Qn = rl.l2n(Q)
    nq = Qn.shape[0]
    nt = len(ref_terms_ix)
    refT_r, refT_c = [], []
    for i, cl in enumerate(ref_closures):
        for t in cl:
            refT_r.append(i)
            refT_c.append(ref_terms_ix[t])
    refT = sp.csr_matrix(
        (np.ones(len(refT_r), np.float32), (refT_r, refT_c)), shape=(R.shape[0], nt)
    )
    out = sp.lil_matrix((nq, nt), dtype=np.float32)
    for b in range(0, nq, batch):
        sl = slice(b, min(b + batch, nq))
        S = Qn[sl] @ Rn.T  # (bq, nref)
        nbr = np.argpartition(-S, knn, axis=1)[:, :knn]
        bq = S.shape[0]
        w = np.take_along_axis(S, nbr, axis=1)
        w[w < 0] = 0
        rr = np.repeat(np.arange(bq), knn)
        cc = nbr.ravel()
        Sk = sp.csr_matrix(
            (w.ravel(), (rr, cc)), shape=(bq, R.shape[0]), dtype=np.float32
        )
        out[sl] = Sk @ refT
    return out.tocsr()


def write_pred_gt(scores, query_ix, terms, cell, prots, tmpdir, max_terms=500):
    """Write cafaeval pred dir (one tsv) + gt tsv, restricted to this cell's proteins."""
    pred_dir = Path(tmpdir) / f"pred_{cell[0]}_{cell[1]}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_path = Path(tmpdir) / f"gt_{cell[0]}_{cell[1]}.tsv"
    inv = {i: a for a, i in query_ix.items()}
    with open(pred_dir / "model.tsv", "w") as fp:
        sc = scores.tocoo()
        per: dict[int, list] = {}
        for r, c, v in zip(sc.row, sc.col, sc.data):
            if v > 0:
                per.setdefault(r, []).append((v, c))
        for r, lst in per.items():
            acc = inv[r]
            if acc not in prots:
                continue
            lst.sort(reverse=True)
            for v, c in lst[:max_terms]:
                fp.write(f"{acc}\t{terms[c]}\t{v:.6f}\n")
    return pred_dir, gt_path


CAFAEVAL_DRIVER = r"""
import sys, json
from cafaeval.evaluation import cafa_eval
obo, pred_dir, gt, ia = sys.argv[1:5]
df, dfs_best = cafa_eval(obo, pred_dir, gt, ia=ia, prop="fill", norm="cafa",
                         no_orphans=True, max_terms=500, th_step=0.001, n_cpu=1, weighted_only=False)
recs = {}
for kind, dfb in dfs_best.items():
    recs[kind] = dfb.reset_index().to_dict("records")
print(json.dumps(recs))
"""


def run_cafaeval(pred_dir, gt_path, tmpdir):
    drv = Path(tmpdir) / "drv.py"
    drv.write_text(CAFAEVAL_DRIVER)
    r = subprocess.run(
        [PROTEA_PY, str(drv), OBO, str(pred_dir), str(gt_path), IA],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        log.warning("cafaeval failed: %s", r.stderr[-800:])
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref-n", type=int, default=80000, help="reference pool size (v227 t0 sample)"
    )
    ap.add_argument("--knn", type=int, default=30)
    ap.add_argument("--dict", type=int, default=2048)
    ap.add_argument("--eval-k", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--train-pairs", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    cells = load_gt()
    queries = sorted({a for d in cells.values() for a in d["proteins"]})
    log.info(
        "7401 GT: %d query proteins, cells: %s",
        len(queries),
        {f"{c[0]}-{c[1]}": len(d["proteins"]) for c, d in sorted(cells.items())},
    )

    conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    # reference pool: v227 t0 accessions (hash sample), excluding the queries
    qset = set(queries)
    cur.execute(
        """SELECT DISTINCT protein_accession FROM protein_go_annotation
                   WHERE annotation_set_id=%s
                     AND (hashtextextended(protein_accession, 42) %% %s)=0""",
        (ANN_SET, max(2, 556000 // (args.ref_n * 2))),
    )
    ref_accs = [a for (a,) in cur.fetchall() if a not in qset]
    rng.shuffle(ref_accs)
    ref_accs = ref_accs[: args.ref_n]
    log.info("reference candidates: %d (target %d)", len(ref_accs), args.ref_n)

    log.info("pulling ankh-mean embeddings (reference + queries) ...")
    ref_emb = pull_mean(cur, ref_accs, ANKH_CFG)
    q_emb = pull_mean(cur, queries, ANKH_CFG)
    # reference v227 closures
    dag = GoDag.from_obo(Path(OBO))
    cur.execute(
        """SELECT pga.protein_accession, gt.go_id FROM protein_go_annotation pga
                   JOIN go_term gt ON gt.id=pga.go_term_id
                   WHERE pga.annotation_set_id=%s AND pga.protein_accession=ANY(%s)""",
        (ANN_SET, list(ref_emb.keys())),
    )
    leaves: dict[str, list] = {}
    for a, go in cur.fetchall():
        leaves.setdefault(a, []).append(go)
    cur.close()
    conn.close()

    ref_accs = [a for a in ref_emb if a in leaves and propagate(leaves[a], dag)]
    ref_clo = [propagate(leaves[a], dag) for a in ref_accs]
    R = np.vstack([ref_emb[a] for a in ref_accs]).astype(np.float32)
    q_accs = [a for a in queries if a in q_emb]
    Q = np.vstack([q_emb[a] for a in q_accs]).astype(np.float32)
    log.info(
        "reference=%d (with closures) | queries embedded=%d/%d",
        len(ref_accs),
        len(q_accs),
        len(queries),
    )

    terms = sorted({t for c in ref_clo for t in c})
    tix = {t: i for i, t in enumerate(terms)}
    query_ix = {a: i for i, a in enumerate(q_accs)}

    # ---- learned-real encoder trained on the reference pool ----
    log.info("training learned-real encoder on the reference pool (dev=%s) ...", dev)
    d = R.shape[1]
    Rt = torch.tensor(rl.l2n(R), device=dev)
    ic = information_content(ref_clo, dag)
    bic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in ref_clo]
    tp = rl.sample_pairs(len(ref_accs), args.train_pairs, rng)
    Y = torch.tensor(
        np.asarray(lin_pairwise(ref_clo, ic, tp, bic), dtype=np.float32), device=dev
    )
    TI = torch.tensor([p[0] for p in tp], device=dev)
    TJ = torch.tensor([p[1] for p in tp], device=dev)
    enc = nn.Linear(d, args.dict).to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    bs = 32768
    npr = len(tp)
    for e in range(args.epochs):
        Z = enc(Rt)
        opt.zero_grad()
        loss = torch.zeros((), device=dev)
        for b in range(0, npr, bs):
            sl = slice(b, b + bs)
            zi, zj = Z[TI[sl]], Z[TJ[sl]]
            cos = (zi * zj).sum(1) / (zi.norm(dim=1) * zj.norm(dim=1) + 1e-8)
            loss = loss + ((cos - Y[sl]) ** 2).sum() / npr
        loss.backward()
        opt.step()
        if e % 30 == 0:
            log.info("  enc epoch %3d loss=%.4f", e, float(loss))
    with torch.no_grad():
        Rc = rl.topk_real(enc(Rt).cpu().numpy().astype(np.float32), args.eval_k)
        Qc = rl.topk_real(
            enc(torch.tensor(rl.l2n(Q), device=dev)).cpu().numpy().astype(np.float32),
            args.eval_k,
        )

    arms = {"dense": (Q, R), "learned": (Qc, Rc)}
    results = {}
    with tempfile.TemporaryDirectory() as td:
        for arm, (Qx, Rx) in arms.items():
            log.info("=== arm=%s : KNN GO-transfer + cafaeval ===", arm)
            scores = knn_transfer(Qx, Rx, ref_clo, tix, args.knn)
            for cell, dmeta in sorted(cells.items()):
                pred_dir, gt_path = write_pred_gt(
                    scores, query_ix, terms, cell, dmeta["proteins"], td
                )
                with open(gt_path, "w") as g:
                    for a, go in dmeta["pairs"]:
                        g.write(f"{a}\t{go}\n")
                rec = run_cafaeval(pred_dir, gt_path, td)
                fw = None
                if rec and "f_micro_w" in rec:
                    ns = ASPECT[cell[1]]
                    for row in rec["f_micro_w"]:
                        if row.get("ns") == ns:
                            fw = row.get("f_micro_w")
                results[(arm, cell)] = fw
                log.info(
                    "  %-7s %s-%s  f_micro_w=%s",
                    arm,
                    cell[0],
                    cell[1],
                    f"{fw:.4f}" if fw is not None else "NA",
                )

    # summary: NK+LK mean per arm + PK
    log.info("=== SUMMARY: f_micro_w on the official 7401 frame (dense vs learned) ===")
    for arm in ("dense", "learned"):
        nklk = [
            results[(arm, c)]
            for c in cells
            if c[0] in ("NK", "LK") and results.get((arm, c)) is not None
        ]
        pk = [
            results[(arm, c)]
            for c in cells
            if c[0] == "PK" and results.get((arm, c)) is not None
        ]
        mnk = float(np.mean(nklk)) if nklk else float("nan")
        mpk = float(np.mean(pk)) if pk else float("nan")
        log.info("  %-7s  NK+LK mean f_micro_w=%.4f | PK mean=%.4f", arm, mnk, mpk)
    try:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("benchmark-encoder-eval")
        with mlflow.start_run(run_name="dense-vs-learned on 7401 frame"):
            for (arm, cell), v in results.items():
                if v is not None:
                    mlflow.log_metric(f"{arm}__{cell[0]}_{cell[1]}", v)
            for arm in ("dense", "learned"):
                nklk = [
                    results[(arm, c)]
                    for c in cells
                    if c[0] in ("NK", "LK") and results.get((arm, c)) is not None
                ]
                if nklk:
                    mlflow.log_metric(f"{arm}__NKLK_mean", float(np.mean(nklk)))
    except Exception as e:  # noqa: BLE001
        log.info("mlflow skipped: %s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
