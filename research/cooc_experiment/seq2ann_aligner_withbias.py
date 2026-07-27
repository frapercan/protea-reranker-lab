"""Companion to seq2ann_aligner_ceiling.py: the SAME two-tower seq->term aligner but WITH its trained
per-term bias head (the deployed generator), giving the aligned mechanism its fair shot. Pure-geometry
(no-bias) scored 0.0001 at V=113k; the objection is that dropping the calibration head cripples it. The
p2 heads (V=25130, vocab-aligned bias over GO_V codes) are the actual trained two-tower generator whose
P4 recall gate PASSED. score(prot,term) = mean_seed( temp*(proj(prot).GO_V[term]) + bias[term] ).

Same frame as the ceiling script for apples-to-apples: BP, PK+LK, marginal-over-kNN, literal terms,
IA true/false ratio at V=113k, per-IA-bucket, residual reach. Read-only, no live DB.
"""
import json, collections, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch, glob

t0 = time.time()
W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
FROZEN = Path("/home/frapercan/Thesis2/storage/protea-frozen-v227-2025-09-04")
SC = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier")
OBO = str(T0D / "go-basic.obo"); IA_F = str(T0D / "IA.tsv")
V_MATCH = 113000

par = collections.defaultdict(set); ns = {}; alt = {}
cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]":
        cur = None
    elif line.startswith("id: GO:"):
        cur = line[4:]
    elif cur and line.startswith("namespace: "):
        ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"):
        par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"):
        par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"):
        alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
AC = {}
def anc(t):
    if t in AC:
        return AC[t]
    o, st = set(), [t]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in o:
                o.add(p); st.append(p)
    AC[t] = o; return o
def closure(terms):
    o = set()
    for g in terms:
        o.add(g); o |= anc(g)
    return o
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try:
            IA[p[0]] = float(p[1])
        except ValueError:
            pass
print(f"ontology {len(BP):,} BP  ({time.time()-t0:.0f}s)", flush=True)

def load_gt(fn):
    G = collections.defaultdict(set)
    with (REL / fn).open() as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[2] == "P":
                G[f[0]].add(alt.get(f[1], f[1]))
    return {p: {g for g in closure(ts) if g in BP} for p, ts in G.items()}
gt_pk = load_gt("groundtruth_PK.tsv"); gt_lk = load_gt("groundtruth_LK.tsv")
pk_known_bp = collections.defaultdict(set)
with (REL / "groundtruth_PK_known.tsv").open() as fh:
    next(fh)
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if f[2] == "P":
            pk_known_bp[f[0]] |= {x for x in closure([alt.get(f[1], f[1])]) if x in BP}

esc = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
ecat = np.asarray(esc.column("category").to_pylist()); easp = np.asarray(esc.column("aspect").to_pylist())
def knn_pool(cell):
    m = (ecat == cell) & (easp == "bpo")
    P = np.asarray(esc.column("protein_accession").to_pylist())[m]
    Gr = np.asarray(esc.column("go_term_id").to_pylist())[m]
    d = collections.defaultdict(set)
    for p_, g_ in zip(P, Gr):
        d[p_].add(alt.get(g_, g_))
    return d
knn_pk = knn_pool("pk"); knn_lk = knn_pool("lk")
uc = np.load(W / "union_candidates_scored.npz", allow_pickle=True)
keep = (uc["src"] == "cls") | (uc["src"] == "both")
cls_pk = collections.defaultdict(set)
for p_, g_ in zip(uc["prot"][keep], [alt.get(g, g) for g in uc["term"][keep]]):
    cls_pk[p_].add(g_)
print(f"pools ready  ({time.time()-t0:.0f}s)", flush=True)

# ---- with-bias two-tower over p2 vocab ----
prep = np.load(SC / "p2" / "prep.npz", allow_pickle=True)
vocab_go = [alt.get(g, g) for g in prep["vocab_go"].tolist()]
GO_V = torch.from_numpy(prep["GO_V"].astype(np.float32))           # (25130,1024)
bp_pos = [i for i, g in enumerate(vocab_go) if g in BP]
GO_BP = np.array([vocab_go[i] for i in bp_pos])
GO_BP_V = GO_V[bp_pos]                                              # (n_bp,1024)
heads = []
for f in sorted(glob.glob(str(SC / "p2" / "head_seed*.pt"))):
    d = torch.load(f, map_location="cpu", weights_only=False); sd = d["state_dict"]
    heads.append((sd["net.0.weight"], sd["net.0.bias"], sd["net.3.weight"], sd["net.3.bias"],
                  float(sd["log_temp"].exp()), sd["bias"][bp_pos]))
ec = np.load(SC / "eval_protein_codes.npz", allow_pickle=True)
code_of = {a: ec["codes"][i].astype(np.float32) for i, a in enumerate(ec["accs"].tolist())}
print(f"with-bias aligner ready: {len(GO_BP):,} BP vocab terms, {len(heads)} seeds  ({time.time()-t0:.0f}s)", flush=True)

def score_batch(codes_np):
    x = torch.from_numpy(codes_np); acc = None
    for W0, b0, W3, b3, temp, biasbp in heads:
        proj = torch.nn.functional.gelu(x @ W0.t() + b0) @ W3.t() + b3   # (b,1024)
        s = temp * (proj @ GO_BP_V.t()) + biasbp                         # (b,n_bp)
        acc = s if acc is None else acc + s
    return (acc / len(heads)).numpy()

BUCKETS = [(0.0, 2.0, "IA[0,2) common"), (2.0, 4.0, "IA[2,4)"), (4.0, 6.0, "IA[4,6)"),
           (6.0, 8.0, "IA[6,8)"), (8.0, 1e9, "IA>=8 rare/deep")]
def bkt(w):
    for i, (lo, hi, _) in enumerate(BUCKETS):
        if lo <= w < hi:
            return i
    return len(BUCKETS) - 1

def measure(prots, gt, kbp, knn, cls):
    per = {}
    for p in prots:
        gtc = gt[p]; knnc = closure(knn.get(p, set())) & BP
        bothc = closure(knn.get(p, set()) | cls.get(p, set())) & BP
        novel = gtc - kbp.get(p, set())
        per[p] = (gtc, knnc, bothc, novel)
    res_total = sum(iaw_(per[p][3] - per[p][2]) for p in prots)
    have = [p for p in prots if p in code_of]
    rows = []
    codes = np.stack([code_of[p] for p in have])
    for s in range(0, len(have), 512):
        chunk = have[s:s + 512]; S = score_batch(codes[s:s + 512])
        k = min(500, S.shape[1]); idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        for bi, prot in enumerate(chunk):
            gtc, knnc, bothc, novel = per[prot]
            for j in idx[bi]:
                term = GO_BP[j]
                if term in knnc:
                    continue
                w = IA.get(term, 0.0); is_true = term in gtc
                in_res = is_true and (term in novel) and (term not in bothc)
                rows.append((float(S[bi, j]), w, is_true, in_res, bkt(w)))
    A = np.array(rows, np.float64); A = A[np.argsort(-A[:, 0])]
    V = min(V_MATCH, len(A)); top = A[:V]
    def rat(m):
        w = m[:, 1]; tt = float((w * m[:, 2]).sum()); ff = float((w * (1 - m[:, 2])).sum())
        return {"n": int(len(m)), "true_ia": round(tt, 1), "false_ia": round(ff, 1),
                "ratio": round(tt / ff, 4) if ff else None,
                "true_cnt": int(m[:, 2].sum()), "false_cnt": int((1 - m[:, 2]).sum())}
    out = {"n_prots": len(prots), "cand_universe": int(len(A)), "residual_total_ia": round(res_total, 1),
           "Q_TAX_at_V113k": rat(top),
           "Q_TAIL_buckets_at_V113k": {BUCKETS[i][2]: rat(top[top[:, 4] == i]) if (top[:, 4] == i).any()
                                       else {"n": 0, "ratio": None} for i in range(len(BUCKETS))}}
    rr = float((top[:, 1] * top[:, 3]).sum()); fc = float((top[:, 1] * (1 - top[:, 2])).sum())
    rf = float((A[:, 1] * A[:, 3]).sum())
    out["Q_RESIDUAL"] = {"residual_total_ia": round(res_total, 1),
                         "reached_V113k_ia": round(rr, 1),
                         "reached_V113k_frac": round(rr / res_total, 4) if res_total else None,
                         "false_ia_cost": round(fc, 1),
                         "reached_full_universe_frac": round(rf / res_total, 4) if res_total else None}
    return out
def iaw_(terms):
    return float(sum(IA.get(g, 0.0) for g in terms))

res = {"arm": "two-tower seq->term WITH trained per-term bias (p2 deployed heads, 7-seed ensemble)",
       "frame": "same as seq2ann_aligner_ceiling.py: BP, marginal-over-kNN, literal terms, V=113k",
       "cells": {}}
res["cells"]["PK_BP"] = measure(sorted(set(gt_pk) & set(knn_pk)), gt_pk, pk_known_bp, knn_pk, cls_pk)
print(f"PK done  ({time.time()-t0:.0f}s)", flush=True)
res["cells"]["LK_BP"] = measure(sorted(set(gt_lk) & set(knn_lk)), gt_lk, collections.defaultdict(set),
                                knn_lk, collections.defaultdict(set))
print(f"LK done  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(W / "seq2ann_aligner_withbias.json", "w"), indent=1)
for cell in ("PK_BP", "LK_BP"):
    r = res["cells"][cell]; q = r["Q_TAX_at_V113k"]
    print(f"\n{cell} WITH-BIAS Q_TAX ratio {q['ratio']} ({q['true_ia']:.0f}t/{q['false_ia']:.0f}f IA; "
          f"{q['true_cnt']}/{q['false_cnt']} cand)", flush=True)
    for name, b in r["Q_TAIL_buckets_at_V113k"].items():
        print(f"   {name:22s} ratio {b['ratio']}  ({b.get('true_ia','-')}t/{b.get('false_ia','-')}f)", flush=True)
    rr = r["Q_RESIDUAL"]
    print(f"   Q_RESIDUAL reached {rr['reached_V113k_ia']:.0f} IA ({rr['reached_V113k_frac']:.1%} of "
          f"{rr['residual_total_ia']:.0f}) false-cost {rr['false_ia_cost']:.0f} | full-univ {rr['reached_full_universe_frac']:.1%}", flush=True)
print(f"DONE ({time.time()-t0:.0f}s)", flush=True)
