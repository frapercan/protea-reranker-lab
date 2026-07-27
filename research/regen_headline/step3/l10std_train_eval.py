"""L10-std champion regeneration, Step 3: train k-WTA hard-neg heads on the
PRODUCTION Ankh-base layer-10 base and evaluate at the KNN-retrieval level (mean9).

Reuses the champion's exact recipe (protea-reranker-lab encoder_ablation._train_encoder:
Linear(768->dict_dim) over l2n(input) + top-k real, hard-neg objective = 300k random
pairs + mined embedding-near negatives, target = Lin GO-similarity, Adam lr=1e-3,
150 epochs, bs=32768, seed=42, float32). Difference vs the served champion: the base is
Ankh layer 10 (config 81436dba), per-dim z-scored (mu/sigma fit on the training pool).

Scoring harness = knn_confirm.py / scale_train.py VERBATIM: query=7,401 LAFA targets,
reference=15,000 t0 subset, cosine top-30 GO-transfer vote over raw reference leaves,
cafaeval f_micro_w over the 9 NK/LK/PK x MFO/BPO/CCO cells (prop=fill, norm=cafa,
no_orphans, th_step=0.01, PK excludes known). mean9 = simple mean of the 9 cells.

Champion baseline (apples): champion head applied to its PRODUCTION L48 base (08234f06),
codes scored in the SAME harness (champ_codes.npz from pull_embeddings.py).

The z-score fit + head training run on PRODUCTION embeddings, resolving the extraction
harness confound (local L48 0.1422 vs served champion 0.2150).

KNN-only mean9 is a CANDIDATE screen number, NOT the sealed 0.4063; the full-pipeline
head-to-head is the conductor's next step.

Author: Francisco Miguel Perez Canales.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from cafaeval.evaluation import cafa_eval

from protea_reranker_lab.encoder_ablation import l2n, sample_pairs, topk_real
from protea_reranker_lab.sdr import GoDag, information_content, lin_pairwise, propagate

# --------------------------------------------------------------------------- config
W = Path("/home/frapercan/Thesis2/storage/layer_ablation")
STEP3 = Path("/home/frapercan/Thesis2/storage/regen_headline/step3")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
DSN = "host=localhost dbname=protea user=protea password=protea"
L10_CFG = "81436dba-1324-4536-bccc-122ac45dd9ba"
ANNOTATION_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"  # GOA v227 t0 (matches pool GO build)

NS2A = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
CATS = {"NK": (str(REL / "groundtruth_NK.tsv"), None),
        "LK": (str(REL / "groundtruth_LK.tsv"), None),
        "PK": (str(REL / "groundtruth_PK.tsv"), str(REL / "groundtruth_PK_known.tsv"))}
ORDER = [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]

KNN_SCORE = 30
KNN_MINE = 30
EPOCHS = 150
TRAIN_PAIRS = 300_000
LR = 1e-3
BS = 32768
SEED = 42
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# arms: screen grid on the 100k pool (top_k x dict_dim) + full-data 2048/128
SCREEN = [(dd, tk) for dd in (2048, 4096) for tk in (64, 128, 256)]  # 6 arms
# PRIMARY (a) = (2048,128) inside SCREEN; FULL-DATA (b) = (2048,128) on full pool.


# --------------------------------------------------------------------------- data
def load_l10():
    d = np.load(STEP3 / "l10_prod.npz", allow_pickle=True)
    accs = list(d["accs"])
    emb = d["emb"].astype(np.float32)
    return {a: i for i, a in enumerate(accs)}, emb, accs


def pull_go(accs: list[str]) -> dict[str, list[str]]:
    import psycopg2
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    cur.execute(
        """SELECT pga.protein_accession, gt.go_id
             FROM protein_go_annotation pga JOIN go_term gt ON gt.id = pga.go_term_id
            WHERE pga.annotation_set_id = %s AND pga.protein_accession = ANY(%s)""",
        (ANNOTATION_SET, list(accs)))
    go: dict[str, list[str]] = {}
    for acc, gid in cur.fetchall():
        go.setdefault(acc, []).append(gid)
    cur.close()
    conn.close()
    return go


# --------------------------------------------------------------------------- head training (champion recipe)
def train_head(pool_base_z: np.ndarray, pool_closures, dag, dict_dim: int, top_k: int,
               seed: int = SEED):
    """Champion hard-neg recipe on an ALREADY z-scored pool base. Returns the trained
    nn.Linear (on CPU) + final loss. Forward is restricted to the unique pair indices,
    which is mathematically identical to the full-batch forward (untouched rows carry
    zero gradient) and keeps memory bounded for the 505k full pool."""
    rng = np.random.default_rng(seed)
    n = pool_base_z.shape[0]
    ic = information_content(pool_closures, dag)
    bic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in pool_closures]

    pairs = sample_pairs(n, TRAIN_PAIRS, rng)
    # mined hard negatives: embedding-near pairs in l2n(z-scored) pool space
    Rn = l2n(pool_base_z)
    anchors = rng.choice(n, size=min(2000, n), replace=False)
    S = Rn[anchors] @ Rn.T
    np.put_along_axis(S, anchors[:, None], -1.0, axis=1)
    nbr = np.argpartition(-S, KNN_MINE, axis=1)[:, :KNN_MINE]
    for a_i, anchor in enumerate(anchors):
        for j in nbr[a_i]:
            pairs.append((int(anchor), int(j)) if anchor < j else (int(j), int(anchor)))
    del S, Rn

    y = np.asarray(lin_pairwise(pool_closures, ic, pairs, bic), dtype=np.float32)
    ti_np = np.array([p[0] for p in pairs], dtype=np.int64)
    tj_np = np.array([p[1] for p in pairs], dtype=np.int64)

    # compact forward over the unique referenced rows
    uniq, remap = np.unique(np.concatenate([ti_np, tj_np]), return_inverse=True)
    npairs = len(pairs)
    ti = torch.tensor(remap[:npairs], device=DEV)
    tj = torch.tensor(remap[npairs:], device=DEV)
    base_c = torch.tensor(l2n(pool_base_z)[uniq], device=DEV)
    y_t = torch.tensor(y, device=DEV)

    torch.manual_seed(seed)
    enc = nn.Linear(pool_base_z.shape[1], dict_dim).to(DEV)
    opt = torch.optim.Adam(enc.parameters(), lr=LR)
    last = float("nan")
    for e in range(EPOCHS):
        opt.zero_grad()
        # per-minibatch forward + backward: gradient of the summed loss = sum of the
        # per-batch gradients (accumulated into enc.grad), identical to a single backward
        # over the full code matrix, but peak activation is one batch (memory-bounded so
        # dict_dim=4096 and the 505k pool fit the 12GB card).
        total = 0.0
        for b in range(0, npairs, BS):
            sl = slice(b, b + BS)
            zi = enc(base_c[ti[sl]])
            zj = enc(base_c[tj[sl]])
            cos = (zi * zj).sum(1) / (zi.norm(dim=1) * zj.norm(dim=1) + 1e-8)
            lb = ((cos - y_t[sl]) ** 2).sum() / npairs
            lb.backward()
            total += float(lb.detach())
        opt.step()
        last = total
    enc_cpu = enc.to("cpu").eval()
    del enc, base_c, y_t, ti, tj
    torch.cuda.empty_cache()
    return enc_cpu, last, len(uniq), npairs


def encode(enc: nn.Linear, base_z: np.ndarray, top_k: int) -> np.ndarray:
    with torch.no_grad():
        z = enc(torch.tensor(l2n(base_z))).numpy().astype(np.float32)
    return topk_real(z, top_k)


# --------------------------------------------------------------------------- scoring (knn_confirm.py verbatim)
def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def knn_predict(Qc, Rc, qaccs, ref_go_list, outdir):
    Qn, Rn = _l2(Qc), _l2(Rc)
    os.makedirs(outdir, exist_ok=True)
    w = open(os.path.join(outdir, "p.tsv"), "w")
    B = 500
    for i in range(0, len(qaccs), B):
        sims = Qn[i:i + B] @ Rn.T
        for r, srow in enumerate(sims):
            nbr = np.argpartition(-srow, KNN_SCORE)[:KNN_SCORE]
            votes = defaultdict(float)
            for nx in nbr:
                s = float(srow[nx])
                if s <= 0:
                    continue
                for go in ref_go_list[nx]:
                    votes[go] += s
            if votes:
                mx = max(votes.values())
                acc = qaccs[i + r]
                for go, v in votes.items():
                    w.write(f"{acc}\t{go}\t{v / mx:.6f}\n")
    w.close()


def nine(outdir):
    cells = {}
    for cat, (gt, known) in CATS.items():
        _, best = cafa_eval(OBO, outdir, gt, ia=IA, no_orphans=True, norm="cafa",
                            prop="fill", exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
        for _, row in best["f_micro_w"].reset_index().iterrows():
            a = NS2A.get(row["ns"])
            if a:
                cells[f"{cat.lower()}-{a}"] = round(float(row["f_micro_w"]), 5)
    return cells


def score_codes(Qc, Rc, qaccs, ref_go_list):
    d = tempfile.mkdtemp(prefix="l10std_")
    try:
        knn_predict(Qc, Rc, qaccs, ref_go_list, d)
        cells = nine(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    m9 = float(np.mean([cells.get(k, 0.0) for k in ORDER]))
    return cells, m9


# --------------------------------------------------------------------------- main
def main() -> None:
    t_start = time.time()
    mlf = None
    if os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            import mlflow
            os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
            os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
            mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "l10std-regen-step3"))
            mlf = mlflow
            print("[mlflow] tracking ->", os.environ["MLFLOW_TRACKING_URI"], flush=True)
        except Exception as exc:
            print("[mlflow] disabled:", exc, flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] load L10 production embeddings", flush=True)
    idx, EMB, all_accs = load_l10()

    qaccs = json.load(open(W / "emb_ankh_base" / "meta.json"))["accs"]
    raccs = json.load(open(W / "ref_emb" / "meta.json"))["accs"]
    pool100 = json.load(open(W / "scale_pool_meta.json"))["accs"]
    ref_go = json.load(open(W / "ref_go.json"))
    ref_go_list = [ref_go.get(a, []) for a in raccs]

    # leakage asserts (scale_train.py:96-97)
    assert not (set(pool100) & set(qaccs)), "pool100 overlaps queries"
    assert not (set(pool100) & set(raccs)), "pool100 overlaps reference"

    # pin pool identity
    json.dump({"pool100k": pool100, "n": len(pool100),
               "provenance": "champion declared 100k v227 pool = scale_pool_meta.json "
                             "(reference_n=100000, seed=42, band=v227); leakage-clean vs "
                             "query(7401) + reference(15000)"},
              open(STEP3 / "pool_accs.json", "w"))

    Q = EMB[[idx[a] for a in qaccs]]
    R = EMB[[idx[a] for a in raccs]]
    P100 = EMB[[idx[a] for a in pool100]]

    # pool GO closures for 100k (from pinned scale_pool_go.json)
    pool_go = json.load(open(W / "scale_pool_go.json"))
    clo100 = [propagate(pool_go.get(a, []), dag_for()) for a in pool100]
    assert all(clo100), "empty closure in 100k pool"

    # ---- full pool (b): all config accs minus query minus ref, v227-annotated ----
    excl = set(qaccs) | set(raccs)
    full_accs = [a for a in all_accs if a not in excl]
    print(f"[{time.strftime('%H:%M:%S')}] full-pool candidate {len(full_accs):,} "
          f"(config {len(all_accs):,} - query - ref)", flush=True)
    full_go = pull_go(full_accs)
    dag = dag_for()
    full_pool, full_clo = [], []
    for a in full_accs:
        leaves = full_go.get(a)
        if not leaves:
            continue
        c = propagate(leaves, dag)
        if c:
            full_pool.append(a)
            full_clo.append(c)
    Pfull = EMB[[idx[a] for a in full_pool]]
    print(f"[{time.strftime('%H:%M:%S')}] full-pool annotated {len(full_pool):,}", flush=True)

    def zfit(base):
        mu = base.mean(0).astype(np.float32)
        sigma = (base.std(0) + 1e-6).astype(np.float32)
        return mu, sigma

    def zapply(base, mu, sigma):
        return ((base - mu) / sigma).astype(np.float32)

    # z-score stats on the two training pools
    mu100, sig100 = zfit(P100)
    muF, sigF = zfit(Pfull)
    P100z = zapply(P100, mu100, sig100)
    Q100z = zapply(Q, mu100, sig100)
    R100z = zapply(R, mu100, sig100)
    Pfullz = zapply(Pfull, muF, sigF)
    QFz = zapply(Q, muF, sigF)
    RFz = zapply(R, muF, sigF)

    results = {}
    heads_dir = STEP3 / "heads"
    heads_dir.mkdir(exist_ok=True)

    def run_arm(name, pool_z, closures, Qz, Rz, dict_dim, top_k, mu, sigma, reference_n, pool_tag):
        t0 = time.time()
        enc, loss, nuniq, npairs = train_head(pool_z, closures, dag, dict_dim, top_k)
        Qc = encode(enc, Qz, top_k)
        Rc = encode(enc, Rz, top_k)
        cells, m9 = score_codes(Qc, Rc, qaccs, ref_go_list)
        dt = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] {name:24s} dict={dict_dim} k={top_k} "
              f"pool={reference_n} loss={loss:.4f} mean9={m9:.4f} ({dt:.0f}s)", flush=True)
        # save head + scaler (same stem)
        stem = heads_dir / name
        torch.save({"state_dict": enc.state_dict(),
                    "meta": {"in_dim": pool_z.shape[1], "dict_dim": dict_dim, "top_k": top_k,
                             "objective": "hard-neg", "seed": SEED, "reference_n": reference_n,
                             "band": "v227", "source_embedding_config_id": L10_CFG,
                             "l2_normalize_input": True, "pool_tag": pool_tag}},
                   f"{stem}.pt")
        np.savez(f"{stem}.scaler.npz", mu=mu, sigma=sigma)
        results[name] = {"dict_dim": dict_dim, "top_k": top_k, "reference_n": reference_n,
                         "pool_tag": pool_tag, "cells": cells, "mean9": m9,
                         "final_loss": loss, "n_unique_train_rows": int(nuniq),
                         "n_pairs": int(npairs), "runtime_s": round(dt, 1),
                         "head_pt": str(f"{stem}.pt"), "scaler_npz": str(f"{stem}.scaler.npz")}
        if mlf is not None:
            try:
                with mlf.start_run(run_name=name):
                    mlf.log_params({"dict_dim": dict_dim, "top_k": top_k, "reference_n": reference_n,
                                    "pool_tag": pool_tag, "objective": "hard-neg", "seed": SEED,
                                    "source_embedding_config_id": L10_CFG, "epochs": EPOCHS,
                                    "train_pairs": TRAIN_PAIRS, "knn_score": KNN_SCORE,
                                    "zscore": True, "base": "ankh-base-L10-prod"})
                    for c, v in cells.items():
                        mlf.log_metric("cell_" + c.replace("-", "_"), v)
                    mlf.log_metric("mean9", m9)
                    mlf.log_metric("final_loss", loss)
            except Exception as exc:
                print("  [mlflow] log failed:", exc, flush=True)
        return m9

    # ---- champion baseline (apples): champion head on production L48, same harness ----
    cc = np.load(STEP3 / "champ_codes.npz")
    champ_cells, champ_m9 = score_codes(cc["q_codes"].astype(np.float32),
                                        cc["r_codes"].astype(np.float32), qaccs, ref_go_list)
    print(f"[{time.strftime('%H:%M:%S')}] CHAMPION (L48 prod, apples) mean9={champ_m9:.4f} "
          f"| pinned knn_confirm=0.2150 | repro_maxdiff={float(cc['repro_maxdiff']):.2e}", flush=True)
    results["champion_L48_apples"] = {
        "dict_dim": 2048, "top_k": 128, "reference_n": 100000, "pool_tag": "mean(L48-raw)",
        "cells": champ_cells, "mean9": champ_m9,
        "note": "champion head (ankh_base_hardneg.pt) over production L48 base 08234f06; "
                "NO z-score (raw base). Pinned knn_confirm value 0.21500."}

    # ---- screen (100k pool) ----
    for dd, tk in SCREEN:
        run_arm(f"L10std_d{dd}_k{tk}_100k", P100z, clo100, Q100z, R100z, dd, tk,
                mu100, sig100, len(pool100), "mean:100k")

    # ---- full-data (b) 2048/128 ----
    run_arm("L10std_d2048_k128_full", Pfullz, full_clo, QFz, RFz, 2048, 128,
            muF, sigF, len(full_pool), "mean:full")

    # ---- leaderboard ----
    board = sorted(((r["mean9"], n) for n, r in results.items()), reverse=True)
    primary = results["L10std_d2048_k128_100k"]["mean9"]
    report = {
        "experiment": "L10-std champion regeneration Step 3 (KNN-retrieval mean9 screen)",
        "harness": "knn_confirm.py verbatim: query=7401, ref=15000, cosine top-30 GO vote, "
                   "cafaeval f_micro_w x 9 cells (prop=fill, norm=cafa, no_orphans, th_step=0.01, "
                   "PK excludes known); mean9 = mean of 9 cells.",
        "base": "Ankh-base layer 10 production config 81436dba (DB-stored L10/32; z-score absorbs "
                "the uniform /32, read as-is), per-dim z-scored (mu/sigma fit on the training pool).",
        "recipe": "champion hard-neg: Linear(768->dict)+topk_real over l2n(z-scored base); 300k "
                  "random pairs + 2000x30 mined near negatives; target Lin GO-sim; Adam lr=1e-3; "
                  "150 epochs; bs=32768; seed=42; float32.",
        "caveat": "KNN-only mean9 is a CANDIDATE screen number, NOT the sealed 0.4063 reranked "
                  "headline. Full-pipeline head-to-head (export->reranker->predict->cafaeval) is "
                  "the conductor's next step. The champion baseline is re-scored in THIS harness "
                  "for apples; it differs from the 0.4063 pipeline measurement.",
        "seed": SEED, "device": DEV,
        "pool_identity": "scale_pool_meta.json 100k (champion declared v227 pool); full pool = "
                         f"{len(full_pool)} v227-annotated config accs minus query/ref.",
        "champion_apples_mean9": champ_m9, "champion_pinned_knn_confirm": 0.21500,
        "primary_L10std_mean9": primary,
        "primary_vs_champion_delta": primary - champ_m9,
        "leaderboard": [{"arm": n, "mean9": m,
                         "vs_champion": m - champ_m9} for m, n in board],
        "arms": results,
    }
    (STEP3 / "l10std_step3_report.json").write_text(json.dumps(report, indent=2, default=float))
    print("\n=== LEADERBOARD (mean9 f_micro_w, KNN-retrieval) ===", flush=True)
    for m, n in board:
        print(f"  {n:26s} mean9={m:.4f}  vs champion {m-champ_m9:+.4f}", flush=True)
    print(f"\nPRIMARY L10-std (d2048/k128/100k) = {primary:.4f}  "
          f"vs champion {champ_m9:.4f}  delta {primary-champ_m9:+.4f}", flush=True)
    print(f"report -> {STEP3/'l10std_step3_report.json'}  ({time.time()-t_start:.0f}s total)", flush=True)


_DAG = None


def dag_for():
    global _DAG
    if _DAG is None:
        _DAG = GoDag.from_obo(OBO)
    return _DAG


if __name__ == "__main__":
    main()
