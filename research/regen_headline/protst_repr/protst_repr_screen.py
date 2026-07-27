"""ProtST representation screen at the KNN-retrieval level (champion knn_confirm harness).

Question (gate for the expensive authoritative export):
  Q1 - does a LEARNED k-WTA head on top of ProtST beat RAW ProtST at kNN GO-transfer
       (especially the BP cells)? The champion retrieval space gets a k-WTA head; ProtST
       has only ever been used RAW (its 512-d text-projected mean-pool) as a reranker feature.
  Q2 - how does ProtST (raw / z-score / k-WTA) compare to the CHAMPION (0.2150) as a
       retrieval space, apples-to-apples on the champion's EXACT knn_confirm harness?

Harness = l10std_train_eval.py VERBATIM (score_codes / knn_predict / nine / train_head /
encode / l2n / topk_real, cafaeval config unchanged): query=7401 LAFA targets, ref=15000,
cosine top-30 GO-transfer vote, cafaeval f_micro_w over the 9 NK/LK/PK x MFO/BPO/CCO cells
(prop=fill, norm=cafa, no_orphans, th_step=0.01, PK excludes known); mean9 = mean of 9 cells.

ONLY change vs the champion harness: the base embedding source is the ProtST bank
(EmbeddingConfig bd3cd470-e384-4f6a-90cf-574704419373, 527,424 vectors, dim 512, single chunk
per protein) instead of Ankh-base. Same accession lists + GO closures as the champion, so the
comparison is apples-to-apples.

Arms:
  champion_L48_apples     - re-score champ_codes.npz (sanity: reproduces pinned 0.2150)
  protst_raw              - raw 512-d ProtST vectors as codes (score_codes l2-normalizes)
  protst_zscore           - per-dim z-score (mu/sigma fit on pool100k ProtST), no head
  protst_kwta_d2048_k128  - champion k-WTA head trained on z-scored ProtST pool100k  (Q1)
  protst_kwta_d4096_k128  - one more dict-dim arm (optional)

Single seed 42 (screen). READ-ONLY DB. Author: Francisco Miguel Perez Canales.
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
import psycopg2
import torch
import torch.nn as nn
from cafaeval.evaluation import cafa_eval

from protea_reranker_lab.encoder_ablation import l2n, sample_pairs, topk_real
from protea_reranker_lab.sdr import GoDag, information_content, lin_pairwise, propagate

# --------------------------------------------------------------------------- config
W = Path("/home/frapercan/Thesis2/storage/layer_ablation")
STEP3 = Path("/home/frapercan/Thesis2/storage/regen_headline/step3")
OUT = Path("/home/frapercan/Thesis2/storage/regen_headline/protst_repr")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
DSN = "host=localhost dbname=protea user=protea password=protea"
PROTST_CFG = "bd3cd470-e384-4f6a-90cf-574704419373"  # ProtST bank, dim 512, 527,424 vectors

NS2A = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
CATS = {"NK": (str(REL / "groundtruth_NK.tsv"), None),
        "LK": (str(REL / "groundtruth_LK.tsv"), None),
        "PK": (str(REL / "groundtruth_PK.tsv"), str(REL / "groundtruth_PK_known.tsv"))}
ORDER = [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]
BP_CELLS = ["nk-bpo", "lk-bpo", "pk-bpo"]

KNN_SCORE = 30
KNN_MINE = 30
EPOCHS = 150
TRAIN_PAIRS = 300_000
LR = 1e-3
BS = 32768
SEED = 42
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- protst bank pull
def parse(text: str) -> np.ndarray:
    return np.fromstring(text[1:-1], sep=",", dtype=np.float32)


def pull_protst(accs: list[str]) -> dict[str, np.ndarray]:
    """Single-chunk (chunk_index_s all 0) ProtST vectors for the given accessions,
    batched ANY() query. Returns acc -> (512,) float32."""
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    out: dict[str, np.ndarray] = {}
    B = 20000
    t0 = time.time()
    for i in range(0, len(accs), B):
        chunk = accs[i:i + B]
        cur.execute(
            """SELECT p.accession, se.embedding::text
                 FROM protein p JOIN sequence_embedding se ON se.sequence_id = p.sequence_id
                WHERE se.embedding_config_id = %s AND p.accession = ANY(%s)""",
            (PROTST_CFG, chunk))
        for acc, emb in cur.fetchall():
            out[acc] = parse(emb)
        print(f"  pulled {len(out):,}/{len(accs):,} ({time.time()-t0:.0f}s)", flush=True)
    cur.close()
    conn.close()
    return out


def load_protst_bank(qaccs, raccs, pool) -> dict[str, np.ndarray]:
    cache = OUT / "protst_prod.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        accs = list(d["accs"])
        emb = d["emb"].astype(np.float32)
        print(f"[skip] {cache} exists ({emb.shape})", flush=True)
        return {a: emb[i] for i, a in enumerate(accs)}
    union = list(dict.fromkeys(list(qaccs) + list(raccs) + list(pool)))
    print(f"[pull] ProtST bank {PROTST_CFG} for union of {len(union):,} accs...", flush=True)
    bank = pull_protst(union)
    accs = list(bank.keys())
    emb = np.vstack([bank[a] for a in accs]).astype(np.float32)
    np.savez(cache, accs=np.array(accs), emb=emb)
    print(f"  saved {cache} : {emb.shape}", flush=True)
    return bank


# ------------------------------------------------------------------ head training (champion recipe)
def train_head(pool_base_z: np.ndarray, pool_closures, dag, dict_dim: int, top_k: int,
               seed: int = SEED):
    """Champion hard-neg recipe on an ALREADY z-scored pool base (VERBATIM from
    l10std_train_eval.py). Returns the trained nn.Linear (on CPU) + final loss."""
    rng = np.random.default_rng(seed)
    n = pool_base_z.shape[0]
    ic = information_content(pool_closures, dag)
    bic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in pool_closures]

    pairs = sample_pairs(n, TRAIN_PAIRS, rng)
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


# --------------------------------------------------------------- scoring (knn_confirm.py verbatim)
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
    d = tempfile.mkdtemp(prefix="protst_repr_")
    try:
        knn_predict(Qc, Rc, qaccs, ref_go_list, d)
        cells = nine(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    m9 = float(np.mean([cells.get(k, 0.0) for k in ORDER]))
    return cells, m9


_DAG = None


def dag_for():
    global _DAG
    if _DAG is None:
        _DAG = GoDag.from_obo(OBO)
    return _DAG


# --------------------------------------------------------------------------- main
def main() -> None:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    mlf = None
    if os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            import mlflow
            os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
            os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
            mlflow.set_experiment("protst-repr-screen")
            mlf = mlflow
            print("[mlflow] tracking ->", os.environ["MLFLOW_TRACKING_URI"], flush=True)
        except Exception as exc:
            print("[mlflow] disabled:", exc, flush=True)

    qaccs = json.load(open(W / "emb_ankh_base" / "meta.json"))["accs"]
    raccs = json.load(open(W / "ref_emb" / "meta.json"))["accs"]
    pool100 = json.load(open(W / "scale_pool_meta.json"))["accs"]
    ref_go = json.load(open(W / "ref_go.json"))
    pool_go = json.load(open(W / "scale_pool_go.json"))

    # leakage asserts (same as champion harness)
    assert not (set(pool100) & set(qaccs)), "pool100 overlaps queries"
    assert not (set(pool100) & set(raccs)), "pool100 overlaps reference"

    bank = load_protst_bank(qaccs, raccs, pool100)

    # coverage (100% expected; drop-missing logic kept defensive)
    q_have = [a for a in qaccs if a in bank]
    r_have = [a for a in raccs if a in bank]
    p_have = [a for a in pool100 if a in bank]
    cov = {"qaccs": [len(q_have), len(qaccs)], "raccs": [len(r_have), len(raccs)],
           "pool100k": [len(p_have), len(pool100)]}
    print(f"[coverage] qaccs {len(q_have)}/{len(qaccs)}  raccs {len(r_have)}/{len(raccs)}  "
          f"pool {len(p_have)}/{len(pool100)}", flush=True)

    dag = dag_for()

    # reference GO list aligned to the covered ref accs
    ref_go_list = [ref_go.get(a, []) for a in r_have]

    # protst matrices
    Q = np.vstack([bank[a] for a in q_have]).astype(np.float32)
    R = np.vstack([bank[a] for a in r_have]).astype(np.float32)
    P100 = np.vstack([bank[a] for a in p_have]).astype(np.float32)

    # pool GO closures (champion scale_pool_go.json), aligned to covered pool accs
    clo100 = [propagate(pool_go.get(a, []), dag) for a in p_have]
    keep = [i for i, c in enumerate(clo100) if c]
    if len(keep) != len(p_have):
        print(f"[pool] dropping {len(p_have)-len(keep)} pool accs with empty closure", flush=True)
        P100 = P100[keep]
        clo100 = [clo100[i] for i in keep]

    # z-score fit on the training pool (champion recipe)
    mu = P100.mean(0).astype(np.float32)
    sigma = (P100.std(0) + 1e-6).astype(np.float32)

    def zapply(base):
        return ((base - mu) / sigma).astype(np.float32)

    P100z, Qz, Rz = zapply(P100), zapply(Q), zapply(R)

    results = {}

    def record(name, cells, m9, extra=None):
        results[name] = {"cells": cells, "mean9": m9, **(extra or {})}
        bp = {c: cells.get(c) for c in BP_CELLS}
        print(f"[{time.strftime('%H:%M:%S')}] {name:24s} mean9={m9:.4f}  "
              f"BP nk={bp['nk-bpo']} lk={bp['lk-bpo']} pk={bp['pk-bpo']}", flush=True)
        if mlf is not None:
            try:
                with mlf.start_run(run_name=name):
                    mlf.log_params({"arm": name, "seed": SEED,
                                    "base": "protst" if name != "champion_L48_apples" else "ankh-L48",
                                    "config_id": PROTST_CFG})
                    for c, v in cells.items():
                        mlf.log_metric("cell_" + c.replace("-", "_"), v)
                    mlf.log_metric("mean9", m9)
            except Exception as exc:
                print("  [mlflow] log failed:", exc, flush=True)

    # ---- champion baseline (apples): champion codes on FULL query/ref, same harness ----
    cc = np.load(OUT / "champ_codes.npz")
    champ_ref_go_list = [ref_go.get(a, []) for a in raccs]  # full 15000
    champ_cells, champ_m9 = score_codes(cc["q_codes"].astype(np.float32),
                                        cc["r_codes"].astype(np.float32), qaccs, champ_ref_go_list)
    record("champion_L48_apples", champ_cells, champ_m9,
           {"note": "champion head over production L48 base 08234f06, NO z-score; "
                    "pinned knn_confirm=0.2150", "repro_maxdiff": float(cc["repro_maxdiff"])})
    print(f"           (pinned knn_confirm=0.2150, repro_maxdiff={float(cc['repro_maxdiff']):.2e})",
          flush=True)

    # ---- protst_raw ----
    t0 = time.time()
    cells, m9 = score_codes(Q, R, q_have, ref_go_list)
    record("protst_raw", cells, m9, {"runtime_s": round(time.time() - t0, 1)})

    # ---- protst_zscore ----
    t0 = time.time()
    cells, m9 = score_codes(Qz, Rz, q_have, ref_go_list)
    record("protst_zscore", cells, m9, {"runtime_s": round(time.time() - t0, 1)})

    # ---- protst k-WTA arms (champion recipe on z-scored protst pool) ----
    for dict_dim, top_k in [(2048, 128), (4096, 128)]:
        name = f"protst_kwta_d{dict_dim}_k{top_k}"
        t0 = time.time()
        enc, loss, nuniq, npairs = train_head(P100z, clo100, dag, dict_dim, top_k)
        Qc = encode(enc, Qz, top_k)
        Rc = encode(enc, Rz, top_k)
        cells, m9 = score_codes(Qc, Rc, q_have, ref_go_list)
        torch.save({"state_dict": enc.state_dict(),
                    "meta": {"in_dim": 512, "dict_dim": dict_dim, "top_k": top_k,
                             "objective": "hard-neg", "seed": SEED, "reference_n": len(clo100),
                             "source_embedding_config_id": PROTST_CFG,
                             "l2_normalize_input": True, "zscore": True}},
                   OUT / f"{name}.pt")
        np.savez(OUT / f"{name}.scaler.npz", mu=mu, sigma=sigma)
        record(name, cells, m9,
               {"dict_dim": dict_dim, "top_k": top_k, "final_loss": loss,
                "n_unique_train_rows": int(nuniq), "n_pairs": int(npairs),
                "runtime_s": round(time.time() - t0, 1)})
        del enc
        torch.cuda.empty_cache()

    # ---- leaderboard + report ----
    champ = results["champion_L48_apples"]["mean9"]
    raw = results["protst_raw"]["mean9"]
    board = sorted(((r["mean9"], n) for n, r in results.items()), reverse=True)
    best_protst = max((n for n in results if n != "champion_L48_apples"),
                      key=lambda n: results[n]["mean9"])

    def cell_deltas(a, b):
        ca, cb = results[a]["cells"], results[b]["cells"]
        return {c: round(ca.get(c, 0.0) - cb.get(c, 0.0), 5) for c in ORDER}

    report = {
        "experiment": "ProtST representation screen at the KNN-retrieval level (mean9 f_micro_w)",
        "question": "Q1: does a learned k-WTA head on ProtST beat RAW ProtST at kNN (esp. BP)? "
                    "Q2: how does ProtST (raw/zscore/kWTA) compare to the champion 0.2150 as a "
                    "retrieval space, apples-to-apples on the champion knn_confirm harness?",
        "harness": "l10std_train_eval.py verbatim: query=7401, ref=15000, cosine top-30 GO vote, "
                   "cafaeval f_micro_w x 9 cells (prop=fill, norm=cafa, no_orphans, th_step=0.01, "
                   "PK excludes known); mean9 = mean of 9 cells.",
        "base": f"ProtST bank config {PROTST_CFG} (dim 512, single chunk/protein), same accession "
                "lists + GO closures as the champion.",
        "recipe_kwta": "champion hard-neg: Linear(512->dict)+topk_real over l2n(z-scored protst); "
                       "300k random pairs + 2000x30 mined near negatives; target Lin GO-sim; "
                       "Adam lr=1e-3; 150 epochs; bs=32768; seed=42; float32.",
        "coverage": cov,
        "coverage_note": "100% of qaccs/raccs/pool100k have a ProtST vector -> ProtST is a valid "
                         "full-coverage retrieval space here; no ref/pool drop needed.",
        "seed": SEED, "device": DEV,
        "champion_apples_mean9": champ, "champion_pinned_knn_confirm": 0.2150,
        "protst_raw_mean9": raw,
        "best_protst_arm": best_protst, "best_protst_mean9": results[best_protst]["mean9"],
        "leaderboard": [{"arm": n, "mean9": m, "bp_cells": {c: results[n]["cells"].get(c) for c in BP_CELLS},
                         "vs_champion": round(m - champ, 5), "vs_protst_raw": round(m - raw, 5)}
                        for m, n in board],
        "kwta_vs_raw_cell_deltas": cell_deltas("protst_kwta_d2048_k128", "protst_raw"),
        "zscore_vs_raw_cell_deltas": cell_deltas("protst_zscore", "protst_raw"),
        "best_protst_vs_champion_cell_deltas": cell_deltas(best_protst, "champion_L48_apples"),
        "arms": results,
    }
    (OUT / "protst_repr_report.json").write_text(json.dumps(report, indent=2, default=float))

    print("\n=== LEADERBOARD (mean9 f_micro_w, KNN-retrieval) ===", flush=True)
    for m, n in board:
        bp = results[n]["cells"]
        print(f"  {n:24s} mean9={m:.4f}  vs_champ {m-champ:+.4f}  vs_raw {m-raw:+.4f}  "
              f"| BP nk={bp.get('nk-bpo')} lk={bp.get('lk-bpo')} pk={bp.get('pk-bpo')}", flush=True)
    print(f"\nreport -> {OUT/'protst_repr_report.json'}  ({time.time()-t_start:.0f}s total)",
          flush=True)


if __name__ == "__main__":
    main()
