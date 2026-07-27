"""Phase 3a: separability of the LEARNED k-WTA sparse-overlap score vs the DENSE annotation-RAG.

Reuses the EXACT measure() frame of seq2ann_infonce/evaluate.py (same gt / known / kNN pool /
IA / closure / V=113k / IA-tail buckets), so the Q_TAX and tail-bucket ratios sit directly against
the dense aligner's already-computed numbers. The arm under test = the learned GO codes scored by
sparse overlap of the projected protein code (top-500 BP terms per eval protein, matching the
dense arm's 500/protein universe). Also reports a per-protein tail AUC (true vs false, IA>=4).

Arg1 = variant (default coann_struct_text). Read-only w.r.t. repos.
"""
import json, collections, time, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
OUT = Path("/home/frapercan/Thesis2/storage/kwta_go_encoder")
CW = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
FROZEN = Path("/home/frapercan/Thesis2/storage/protea-frozen-v227-2025-09-04")
OBO = str(T0D / "go-basic.obo"); IA_F = str(T0D / "IA.tsv")
V_MATCH = 113000
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "coann_struct_text"
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- ontology / IA (verbatim from the dense harness) ----
par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"): par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
AC = {}
def anc(t):
    if t in AC: return AC[t]
    o, st = set(), [t]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in o: o.add(p); st.append(p)
    AC[t] = o; return o
def closure(terms):
    o = set()
    for g in terms: o.add(g); o |= anc(g)
    return o
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass
def iaw(terms): return float(sum(IA.get(g, 0.0) for g in terms))

def load_gt(fn):
    G = collections.defaultdict(set)
    with (REL / fn).open() as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[2] == "P": G[f[0]].add(alt.get(f[1], f[1]))
    return {p: {g for g in closure(ts) if g in BP} for p, ts in G.items()}
gt_pk = load_gt("groundtruth_PK.tsv"); gt_lk = load_gt("groundtruth_LK.tsv")
pk_known_bp = collections.defaultdict(set)
with (REL / "groundtruth_PK_known.tsv").open() as fh:
    next(fh)
    for line in fh:
        f = line.rstrip("\n").split("\t"); g = alt.get(f[1], f[1])
        if f[2] == "P": pk_known_bp[f[0]] |= {x for x in closure([g]) if x in BP}
esc = pq.read_table(CW / "rerank_out" / "eval_scores.parquet")
ecat = np.asarray(esc.column("category").to_pylist()); easp = np.asarray(esc.column("aspect").to_pylist())
def knn_pool(cell):
    m = (ecat == cell) & (easp == "bpo")
    P = np.asarray(esc.column("protein_accession").to_pylist())[m]
    Gr = np.asarray(esc.column("go_term_id").to_pylist())[m]
    d = collections.defaultdict(set)
    for p_, g_ in zip(P, Gr): d[p_].add(alt.get(g_, g_))
    return d
knn_pk = knn_pool("pk"); knn_lk = knn_pool("lk")
uc = np.load(CW / "union_candidates_scored.npz", allow_pickle=True)
uc_keep = (uc["src"] == "cls") | (uc["src"] == "both")
uc_prot = uc["prot"][uc_keep]; uc_term = np.array([alt.get(g, g) for g in uc["term"][uc_keep]])
cls_pk = collections.defaultdict(set)
for p_, g_ in zip(uc_prot, uc_term): cls_pk[p_].add(g_)
log(f"gt_pk {len(gt_pk):,} gt_lk {len(gt_lk):,} knn_pk {len(knn_pk):,} knn_lk {len(knn_lk):,}")

# ---- measure() : VERBATIM from seq2ann_infonce/evaluate.py ----
BUCKETS = [(0.0, 2.0, "IA[0,2) common"), (2.0, 4.0, "IA[2,4)"), (4.0, 6.0, "IA[4,6)"),
           (6.0, 8.0, "IA[6,8)"), (8.0, 1e9, "IA>=8 rare/deep")]
def bucket_of(w):
    for i, (lo, hi, _) in enumerate(BUCKETS):
        if lo <= w < hi: return i
    return len(BUCKETS) - 1
def measure(prots, gt, known_bp, knn, cls_extras, cand_iter):
    per = {}
    for p in prots:
        gtc = gt[p]; kbp = known_bp.get(p, set())
        bothc = closure(knn.get(p, set()) | cls_extras.get(p, set())) & BP
        knnc = closure(knn.get(p, set())) & BP
        per[p] = (gtc, knnc, bothc, gtc - kbp)
    residual_total = sum(iaw(per[p][3] - per[p][2]) for p in prots)
    rows = []; seen = set()
    for prot, term, sc in cand_iter:
        if prot not in per or term not in BP: continue
        gtc, knnc, bothc, novel = per[prot]
        if term in knnc: continue
        key = (prot, term)
        if key in seen: continue
        seen.add(key)
        w = IA.get(term, 0.0); is_true = term in gtc
        in_both = term in bothc; in_res = is_true and (term in novel) and (not in_both)
        rows.append((sc, w, is_true, in_both, in_res, bucket_of(w)))
    if not rows: return None
    A = np.array(rows, np.float64); A = A[np.argsort(-A[:, 0])]
    V = min(V_MATCH, len(A)); top = A[:V]
    def ratio(mat):
        w = mat[:, 1]; tt = float((w * mat[:, 2]).sum()); ff = float((w * (1 - mat[:, 2])).sum())
        return {"n": int(len(mat)), "true_ia": round(tt, 1), "false_ia": round(ff, 1),
                "ratio": round(tt / ff, 4) if ff else None,
                "true_cnt": int(mat[:, 2].sum()), "false_cnt": int((1 - mat[:, 2]).sum())}
    out = {"n_prots": len(prots), "cand_universe": int(len(A)), "V_used": int(V),
           "residual_total_ia": round(residual_total, 1), "Q_TAX_at_V": ratio(top)}
    buckets = {}
    for i, (lo, hi, name) in enumerate(BUCKETS):
        m = top[top[:, 5] == i]
        buckets[name] = ratio(m) if len(m) else {"n": 0, "ratio": None, "true_ia": 0.0, "false_ia": 0.0, "true_cnt": 0, "false_cnt": 0}
    out["Q_TAIL_buckets"] = buckets
    res_reached = float((top[:, 1] * top[:, 4]).sum())
    out["Q_RESIDUAL"] = {"residual_total_ia": round(residual_total, 1),
                         "residual_reached_ia": round(res_reached, 1),
                         "residual_reached_frac": round(res_reached / residual_total, 4) if residual_total else None,
                         "false_ia_cost": round(float((top[:, 1] * (1 - top[:, 2])).sum()), 1)}
    return out

# ---- learned-code candidate iterator: top-500 BP terms/protein by sparse overlap ----
gc = np.load(OUT / f"go_codes_{VARIANT}.npz", allow_pickle=True)
vocab = gc["go_ids"]; Gcodes = torch.from_numpy(gc["codes"].astype(np.float32)).to(dev)
proj = np.load(OUT / f"eval_proj_codes_{VARIANT}.npz", allow_pickle=True)
proj_acc = proj["accs"]; Pcodes = torch.from_numpy(proj["codes"].astype(np.float32)).to(dev)
acc2prow = {a: i for i, a in enumerate(proj_acc)}
TOP = 500
def scored_top(prots):
    """dict prot -> (terms[500], scores[500]) by overlap; batched matmul."""
    out = {}
    keep = [p for p in prots if p in acc2prow]
    rows = torch.tensor([acc2prow[p] for p in keep])
    for s in range(0, len(keep), 256):
        S = Pcodes[rows[s:s+256].to(dev)]
        sim = S @ Gcodes.t()                          # (b, Nt) overlap in [0,1]
        v, idx = torch.topk(sim, TOP, dim=1)
        v = v.cpu().numpy(); idx = idx.cpu().numpy()
        for r, p in enumerate(keep[s:s+256]):
            out[p] = (vocab[idx[r]], v[r])
    return out
def code_iter_factory(cache):
    def it(prots):
        for p in prots:
            if p not in cache: continue
            terms, scs = cache[p]
            for t_, sc in zip(terms, scs):
                yield p, alt.get(t_, t_), float(sc)
    return it

# ---- per-protein tail AUC (IA>=4), true vs false, over the method's own top-500 ----
def tail_auc(prots, gt, known_bp, cache):
    aucs = []
    for p in prots:
        if p not in cache: continue
        gtc = gt[p]; kbp = known_bp.get(p, set())
        terms, scs = cache[p]
        yy, ss = [], []
        for t_, sc in zip(terms, scs):
            g = alt.get(t_, t_)
            if g not in BP or IA.get(g, 0.0) < 4.0: continue
            if g in kbp: continue                          # novel-eligible only
            yy.append(1.0 if g in gtc else 0.0); ss.append(sc)
        yy = np.array(yy); ss = np.array(ss)
        if yy.sum() == 0 or yy.sum() == len(yy) or len(yy) < 5: continue
        order = np.argsort(ss); r = np.empty(len(ss)); r[order] = np.arange(len(ss))
        n1 = yy.sum(); n0 = len(yy) - n1
        aucs.append((r[yy == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
    return {"mean_tail_auc": round(float(np.mean(aucs)), 4) if aucs else None, "n_prots_scored": len(aucs)}

# ---- dense annotation-RAG baseline cache (armA, 500/prot) for identical tail-AUC ----
dA = np.load(CW / "seq2ann_infonce" / "eval_scores_armA.npz", allow_pickle=True)
dense_cache = collections.defaultdict(lambda: ([], []))
for p_, t_, s_ in zip(dA["prot"], dA["term"], dA["score"]):
    dense_cache[p_][0].append(t_); dense_cache[p_][1].append(float(s_))
dense_cache = {p: (np.array(v[0]), np.array(v[1])) for p, v in dense_cache.items()}

res = {"variant": VARIANT, "frame": "seq2ann harness verbatim; learned k-WTA sparse-overlap, top500/prot",
       "dense_baseline_ref": "storage/cooc_experiment/seq2ann_infonce/seq2ann_infonce_eval.json (ALIGNER_armA/B)",
       "cells": {}}
pk_prots = sorted(set(gt_pk) & set(knn_pk)); lk_prots = sorted(set(gt_lk) & set(knn_lk))
for cell, prots, gt, kbp, knn, cls in [
        ("PK_BP", pk_prots, gt_pk, pk_known_bp, knn_pk, cls_pk),
        ("LK_BP", lk_prots, gt_lk, collections.defaultdict(set), knn_lk, collections.defaultdict(set))]:
    cache = scored_top(prots)
    log(f"{cell}: scored {len(cache):,} prots")
    it = code_iter_factory(cache)
    m = measure(prots, gt, kbp, knn, cls, it(prots))
    au = tail_auc(prots, gt, kbp, cache)
    au_dense = tail_auc(prots, gt, kbp, dense_cache)
    res["cells"][cell] = {"n_prots": len(prots), "LEARNED_KWTA": m,
                          "tail_auc_learned": au, "tail_auc_dense_baseline": au_dense}
    q = m["Q_TAX_at_V"]; tb = m["Q_TAIL_buckets"]
    log(f"{cell}: Q_TAX ratio={q['ratio']} true_cnt={q['true_cnt']} resid={m['Q_RESIDUAL']['residual_reached_frac']} "
        f"IA[4,6)={tb['IA[4,6)']['ratio']} IA[6,8)={tb['IA[6,8)']['ratio']} "
        f"tail_auc L={au['mean_tail_auc']} vs dense={au_dense['mean_tail_auc']}")
    json.dump(res, open(OUT / f"phase3_separability_{VARIANT}.json", "w"), indent=1)
log(f"DONE separability {VARIANT}")
