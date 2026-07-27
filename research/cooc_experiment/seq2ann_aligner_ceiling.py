"""CEILING for a DIRECT sequence->annotation aligned generator (read-only, no training).

Idea under test: a contrastively-aligned joint space where the QUERY SEQUENCE embedding is used to
retrieve GO terms DIRECTLY over a term-encoder's codes (CLIP/ProtST-style, term vocabulary as the
label side). BYPASSES the protein-kNN hop. This is DISTINCT from annotation_rag_ceiling.py, which
queried from a protein's KNOWN terms.

Best EXISTING on-disk proxy for the proposed aligner = the TWO-TOWER SPARSE CLASSIFIER:
    protein tower: trained proj head net(2048 d8979601 code -> 1024)   [heads/head_seed*.pt]
    term tower   : frozen go_sparse_codes (1024-d, BioBERT-text kWTA + t0 co-annotation PPMI/SVD kWTA)
    score(prot,term) = <proj(prot_code), go_code[term]>   (pure two-tower retrieval geometry)
This is a GENUINE, ALREADY-TRAINED seq->term aligner. We only run inference (matmul). Ensemble the 7
seeds by averaging temp*proj. NO bias (bias is a per-term frequency prior that only exists over the
training vocab and tilts toward COMMON terms; the tail is the whole question, so we score pure geometry
and note bias separately).

ProtST on disk (text_scorer/query_protst.npy) is PROTEIN-SIDE ONLY (512-d protein_feature). There is
NO GO-term embedding in ProtST text space on disk (go_text_emb.npz is BioBERT 768-d, a DIFFERENT
space). So protst DIRECT seq->term retrieval is NOT constructible without running ProtST's text
encoder over GO term text (needs GPU inference we are told not to do). Reported as MISSING.

Discipline: payoff is IA true/false ratio at matched volume, NEVER candidate count. Bar = the full-GO
classifier's 6,332/99,966 = 0.0633 at V=113k (marginal over kNN). The classifier is run through the
SAME pipeline here so the comparison (incl. per-IA-bucket) is apples-to-apples; 0.0633 is the anchor.
BP aspect, PK+LK eval, novel = gt(v230) closure minus v227 known closure.
"""
import json, collections, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch

t0 = time.time()
W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
FROZEN = Path("/home/frapercan/Thesis2/storage/protea-frozen-v227-2025-09-04")
SC = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier")
HEADS = Path("/home/frapercan/Thesis2/storage/two_tower_sparse/heads")
OBO = str(T0D / "go-basic.obo"); IA_F = str(T0D / "IA.tsv")
V_MATCH = 113000
CLS_ANCHOR = (6332.0, 99966.0)   # true_ia, false_ia at V=113k, from prior receipts

# ---------- ontology / IA (verbatim from annotation_rag_ceiling.py) ----------
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
def iaw(terms):
    return float(sum(IA.get(g, 0.0) for g in terms))
print(f"ontology {len(BP):,} BP, IA {len(IA):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---------- ground truth / known / pools (verbatim loaders) ----------
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

pk_known_all = collections.defaultdict(set); pk_known_bp = collections.defaultdict(set)
with (REL / "groundtruth_PK_known.tsv").open() as fh:
    next(fh)
    for line in fh:
        f = line.rstrip("\n").split("\t"); g = alt.get(f[1], f[1])
        pk_known_all[f[0]].add(g)
        if f[2] == "P":
            pk_known_bp[f[0]] |= {x for x in closure([g]) if x in BP}
meta = pq.read_table(FROZEN / "go_term_metadata.parquet")
id2go = {i: g for i, g in zip(meta.column("go_term_id").to_pylist(), meta.column("go_id").to_pylist())}
lk_known_all = collections.defaultdict(set)
ra = pq.read_table(FROZEN / "reference_annotations.parquet", columns=["accession", "go_term_id"])
need = set(gt_lk)
for a_, gi in zip(ra.column("accession").to_pylist(), ra.column("go_term_id").to_pylist()):
    if a_ in need:
        g = id2go.get(gi)
        if g:
            lk_known_all[a_].add(alt.get(g, g))

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
uc_keep = (uc["src"] == "cls") | (uc["src"] == "both")
uc_prot = uc["prot"][uc_keep]; uc_term = np.array([alt.get(g, g) for g in uc["term"][uc_keep]])
uc_score = uc["score"][uc_keep]
cls_pk = collections.defaultdict(set)
for p_, g_ in zip(uc_prot, uc_term):
    cls_pk[p_].add(g_)
print(f"pools: kNN PK {len(knn_pk):,}/LK {len(knn_lk):,}; cls extras PK {len(cls_pk):,}; "
      f"union rows {len(uc_prot):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---------- two-tower aligner: proj heads + term codes ----------
gv = np.load(SC / "go_sparse_codes.npz", allow_pickle=True)
G_go = np.array([alt.get(g, g) for g in gv["go_ids"].tolist()])
G_emb = gv["codes"].astype(np.float32)                      # (39906,1024) as trained (no renorm)
bp_mask = np.array([g in BP for g in G_go])
GO_BP = G_go[bp_mask]
GOE_BP = torch.from_numpy(G_emb[bp_mask])                    # (n_bp,1024)
print(f"term tower: {len(GO_BP):,} BP term codes  ({time.time()-t0:.0f}s)", flush=True)

import glob
heads = []
for f in sorted(glob.glob(str(HEADS / "head_seed*.pt"))):
    d = torch.load(f, map_location="cpu", weights_only=False); sd = d["state_dict"]
    heads.append((sd["net.0.weight"], sd["net.0.bias"], sd["net.3.weight"], sd["net.3.bias"],
                  float(sd["log_temp"].exp())))
print(f"proj heads: {len(heads)} seeds  ({time.time()-t0:.0f}s)", flush=True)

ec = np.load(SC / "eval_protein_codes.npz", allow_pickle=True)
code_of = {a: ec["codes"][i].astype(np.float32) for i, a in enumerate(ec["accs"].tolist())}

def aligner_proj(codes_np):
    """Ensemble combined projection: mean_seed( temp * (GELU(x W0'+b0) W3'+b3) ). Ranking-equivalent
    to averaging the per-seed two-tower logits (term codes are shared)."""
    x = torch.from_numpy(codes_np)
    acc = None
    for W0, b0, W3, b3, temp in heads:
        z = torch.nn.functional.gelu(x @ W0.t() + b0) @ W3.t() + b3
        z = temp * z
        acc = z if acc is None else acc + z
    return acc / len(heads)                                 # (n,1024)

# ---------- shared scoring pipeline (aligner AND classifier through identical code) ----------
# IA buckets by term information accretion (rarity). Low IA = common/shallow, high IA = rare/deep.
BUCKETS = [(0.0, 2.0, "IA[0,2) common"), (2.0, 4.0, "IA[2,4)"), (4.0, 6.0, "IA[4,6)"),
           (6.0, 8.0, "IA[6,8)"), (8.0, 1e9, "IA>=8 rare/deep")]
def bucket_of(w):
    for i, (lo, hi, _) in enumerate(BUCKETS):
        if lo <= w < hi:
            return i
    return len(BUCKETS) - 1

def measure(cell_name, prots, gt, known_bp, knn, cls_extras, cand_iter):
    """cand_iter yields (prot, term, score) candidate rows. We restrict to BP terms, exclude terms
    already in the kNN closure (marginal-over-kNN), take global top-V by score, and report IA
    true/false ratio overall, per bucket, and the residual reach (marginal over kNN UNION cls)."""
    per = {}
    for p in prots:
        gtc = gt[p]
        kbp = known_bp.get(p, set())
        knnc = closure(knn.get(p, set())) & BP
        bothc = closure(knn.get(p, set()) | cls_extras.get(p, set())) & BP
        novel = gtc - kbp
        per[p] = (gtc, knnc, bothc, novel)
    residual_total = 0.0
    for p in prots:
        gtc, knnc, bothc, novel = per[p]
        residual_total += iaw((novel - bothc))
    rows = []  # (score, ia, is_true, in_bothc, in_residual, bucket)
    seen = set()
    for prot, term, sc in cand_iter:
        if prot not in per or term not in BP:
            continue
        gtc, knnc, bothc, novel = per[prot]
        if term in knnc:
            continue                                        # marginal over kNN
        key = (prot, term)
        if key in seen:
            continue
        seen.add(key)
        w = IA.get(term, 0.0)
        is_true = term in gtc
        in_both = term in bothc
        in_res = is_true and (term in novel) and (not in_both)
        rows.append((sc, w, is_true, in_both, in_res, bucket_of(w)))
    if not rows:
        return None
    A = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows], np.float64)
    order = np.argsort(-A[:, 0])
    A = A[order]
    V = min(V_MATCH, len(A))
    top = A[:V]
    def ratio(mat):
        w = mat[:, 1]; tt = float((w * mat[:, 2]).sum()); ff = float((w * (1 - mat[:, 2])).sum())
        return {"n": int(len(mat)), "true_ia": round(tt, 1), "false_ia": round(ff, 1),
                "ratio": round(tt / ff, 4) if ff else None,
                "true_cnt": int(mat[:, 2].sum()), "false_cnt": int((1 - mat[:, 2]).sum())}
    out = {"cell": cell_name, "n_prots": len(prots), "cand_universe": int(len(A)),
           "residual_total_ia": round(residual_total, 1),
           "Q_TAX_at_V113k": ratio(top)}
    # per-bucket at matched volume
    buckets = {}
    for i, (lo, hi, name) in enumerate(BUCKETS):
        m = top[top[:, 5] == i]
        if len(m):
            buckets[name] = ratio(m)
        else:
            buckets[name] = {"n": 0, "true_ia": 0.0, "false_ia": 0.0, "ratio": None,
                             "true_cnt": 0, "false_cnt": 0}
    out["Q_TAIL_buckets_at_V113k"] = buckets
    # Q_RESIDUAL: reach of the (novel & not-in-both) residual at matched volume + its false-IA cost
    res_reached = float((top[:, 1] * top[:, 4]).sum())
    false_cost = float((top[:, 1] * (1 - top[:, 2])).sum())
    out["Q_RESIDUAL_at_V113k"] = {
        "residual_total_ia_16083_analogue": round(residual_total, 1),
        "residual_reached_ia": round(res_reached, 1),
        "residual_reached_frac": round(res_reached / residual_total, 4) if residual_total else None,
        "false_ia_cost": round(false_cost, 1),
        "residual_true_per_false": round(res_reached / false_cost, 5) if false_cost else None}
    # generous: full universe residual reach (no volume cap)
    res_full = float((A[:, 1] * A[:, 4]).sum())
    out["Q_RESIDUAL_full_universe"] = {
        "residual_reached_ia": round(res_full, 1),
        "residual_reached_frac": round(res_full / residual_total, 4) if residual_total else None,
        "cand_universe": int(len(A))}
    return out

# ---- candidate iterators ----
def aligner_iter(prots, topk_per=500):
    """Per-protein top-`topk_per` BP terms by two-tower seq->term score. Superset of any global top-V."""
    have = [p for p in prots if p in code_of]
    miss = len(prots) - len(have)
    codes = np.stack([code_of[p] for p in have]) if have else np.zeros((0, 2048), np.float32)
    for s in range(0, len(have), 512):
        chunk = have[s:s + 512]
        proj = aligner_proj(codes[s:s + 512])               # (b,1024)
        S = (proj @ GOE_BP.t()).numpy()                     # (b,n_bp)
        k = min(topk_per, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        for bi, prot in enumerate(chunk):
            js = idx[bi]
            for j in js:
                yield prot, GO_BP[j], float(S[bi, j])
    aligner_iter.missing = miss

def classifier_iter(prots):
    ps = set(prots)
    for p_, t_, s_ in zip(uc_prot, uc_term, uc_score):
        if p_ in ps:
            yield p_, t_, float(s_)

# ---------- run ----------
pk_prots = sorted(set(gt_pk) & set(knn_pk))
lk_prots = sorted(set(gt_lk) & set(knn_lk))
res = {"scope": "BP; PK+LK eval with truth and a kNN pool. novel = gt(v230) closure - v227 known closure.",
       "aligner": "two-tower seq->term: proj(d8979601 code) . go_sparse_code, 7-seed ensemble, pure "
                  "geometry (no bias). eval codes = eval_protein_codes.npz (d8979601).",
       "protst_direct": "NOT AVAILABLE on disk: ProtST artifacts are protein-side only (query_protst.npy "
                        "512-d protein_feature); no GO-term embedding in ProtST text space. go_text_emb.npz "
                        "is BioBERT 768-d, a different space -> no valid protst seq->term dot product.",
       "classifier_anchor": {"true_ia": CLS_ANCHOR[0], "false_ia": CLS_ANCHOR[1],
                             "ratio": round(CLS_ANCHOR[0] / CLS_ANCHOR[1], 4)},
       "V_matched": V_MATCH, "cells": {}}

for cell, prots, gt, kbp, knn, cls in [
        ("PK_BP", pk_prots, gt_pk, pk_known_bp, knn_pk, cls_pk),
        ("LK_BP", lk_prots, gt_lk, collections.defaultdict(set), knn_lk, collections.defaultdict(set))]:
    print(f"\n### {cell}: {len(prots):,} prots ###", flush=True)
    a = measure(cell, prots, gt, kbp, knn, cls, aligner_iter(prots))
    a["eval_code_coverage_missing"] = int(getattr(aligner_iter, "missing", 0))
    print(f"  aligner done ({time.time()-t0:.0f}s)", flush=True)
    c = measure(cell, prots, gt, kbp, knn, cls, classifier_iter(prots))
    print(f"  classifier done ({time.time()-t0:.0f}s)", flush=True)
    res["cells"][cell] = {"ALIGNER_two_tower": a, "CLASSIFIER_fullgo": c}

json.dump(res, open(W / "seq2ann_aligner_ceiling.json", "w"), indent=1)

# ---------- report ----------
def show(cell, d):
    print(f"\n===== {cell} =====", flush=True)
    for who in ("CLASSIFIER_fullgo", "ALIGNER_two_tower"):
        r = d[who]
        if not r:
            continue
        q = r["Q_TAX_at_V113k"]
        print(f"  {who:20s} Q_TAX V113k ratio {q['ratio']}  ({q['true_ia']:,.0f} true / "
              f"{q['false_ia']:,.0f} false IA; {q['true_cnt']:,} / {q['false_cnt']:,} cand)  "
              f"[universe {r['cand_universe']:,}]", flush=True)
    print(f"  {'BUCKET':22s} {'ALIGNER ratio (t/f IA)':30s} {'CLASSIFIER ratio (t/f IA)':30s}", flush=True)
    ab = d["ALIGNER_two_tower"]["Q_TAIL_buckets_at_V113k"]
    cb = d["CLASSIFIER_fullgo"]["Q_TAIL_buckets_at_V113k"]
    for name in ab:
        a = ab[name]; c = cb[name]
        print(f"  {name:22s} {str(a['ratio']):>7} ({a['true_ia']:>7.0f}/{a['false_ia']:>8.0f})   "
              f"{str(c['ratio']):>7} ({c['true_ia']:>7.0f}/{c['false_ia']:>8.0f})", flush=True)
    for who in ("ALIGNER_two_tower", "CLASSIFIER_fullgo"):
        rr = d[who]["Q_RESIDUAL_at_V113k"]; rf = d[who]["Q_RESIDUAL_full_universe"]
        print(f"  {who:20s} Q_RESIDUAL V113k reached {rr['residual_reached_ia']:,.0f} IA "
              f"({rr['residual_reached_frac']:.1%} of {rr['residual_total_ia_16083_analogue']:,.0f}) "
              f"false-cost {rr['false_ia_cost']:,.0f} | full-universe reach {rf['residual_reached_frac']:.1%}",
              flush=True)

for cell in ("PK_BP", "LK_BP"):
    show(cell, res["cells"][cell])
print(f"\nclassifier anchor ratio {CLS_ANCHOR[0]/CLS_ANCHOR[1]:.4f} (6332/99966, prior receipt)", flush=True)
print(f"DONE ({time.time()-t0:.0f}s)", flush=True)
