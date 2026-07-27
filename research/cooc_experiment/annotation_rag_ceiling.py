"""PRECONDITION / CEILING for an ANNOTATION-SPACE RAG candidate generator (read-only).

Idea: represent a protein by its KNOWN annotations and retrieve additional GO terms from a learned
label space, as a candidate GENERATOR complementary to kNN (protein-embedding retrieval) and the
full-GO classifier extras. Conditions on known terms => PK/LK only.

Discipline (Task #49 lesson): a generator's payoff is IA-marginal-over-the-pool, never candidate
count. We report true-IA / false-IA and a fixed-score-style ratio verdict, not a novelty count.

Q1 HEADROOM: of novel-true IA (gt v230 closure minus v227 known, BP), how much is NOT already in the
   deployed pool closure (kNN candidates UNION classifier extras)? -> residual true-IA (cap).
Q2 ANNOTATION-REACHABILITY: of that residual, how much is reachable from the protein's KNOWN terms via
   label-space nearest neighbors (anc2vec / two-tower GO codes) at several k?
Q3 THE TAX: at matched volume, the annotation-neighborhood generator's true-IA/false-IA ratio vs the
   classifier's 6,332/99,966 = 0.0633 (from TEXT_AS_GENERATOR_IS_SCALE_CONFOUNDED.md, V=113k).

All inputs are on disk. NO live DB. BP aspect only (the campaign's wall + where every artifact lives).
"""
import json, collections, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

t0 = time.time()
W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
FROZEN = Path("/home/frapercan/Thesis2/storage/protea-frozen-v227-2025-09-04")
OBO = str(T0D / "go-basic.obo")
IA_F = str(T0D / "IA.tsv")

# ---- ontology ----
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
    AC[t] = o
    return o
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
print(f"ontology: {len(BP):,} BP terms, IA for {len(IA):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---- ground truth (aspect P), propagated, BP ----
def load_gt(fn):
    G = collections.defaultdict(set)
    with (REL / fn).open() as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[2] == "P":
                G[f[0]].add(alt.get(f[1], f[1]))
    return {p: {g for g in closure(ts) if g in BP} for p, ts in G.items()}
gt_pk = load_gt("groundtruth_PK.tsv")
gt_lk = load_gt("groundtruth_LK.tsv")

# ---- known terms (v227 t0), all aspects for seeds; BP subset for novelty guard ----
# PK: groundtruth_PK_known.tsv (CAFA experimental known set on disk).
pk_known_all = collections.defaultdict(set)     # all aspects, raw (seeds)
pk_known_bp = collections.defaultdict(set)      # BP closure (novelty guard)
with (REL / "groundtruth_PK_known.tsv").open() as fh:
    next(fh)
    for line in fh:
        f = line.rstrip("\n").split("\t")
        g = alt.get(f[1], f[1])
        pk_known_all[f[0]].add(g)
        if f[2] == "P":
            pk_known_bp[f[0]] |= {x for x in closure([g]) if x in BP}
# LK: no experimental BP annotation at t0 by category definition -> novel_bp = full gt closure.
# Seeds for LK come from the v227 frozen reference (all evidence codes, all aspects) via id->GO map.
meta = pq.read_table(FROZEN / "go_term_metadata.parquet")
id2go = {i: g for i, g in zip(meta.column("go_term_id").to_pylist(), meta.column("go_id").to_pylist())}
lk_prots_needed = set(gt_lk)
lk_known_all = collections.defaultdict(set)
ra = pq.read_table(FROZEN / "reference_annotations.parquet", columns=["accession", "go_term_id"])
acc_a = ra.column("accession").to_pylist(); gid_a = ra.column("go_term_id").to_pylist()
for a_, gi in zip(acc_a, gid_a):
    if a_ in lk_prots_needed:
        g = id2go.get(gi)
        if g:
            lk_known_all[a_].add(alt.get(g, g))
print(f"known: PK {len(pk_known_all):,} prots (all-aspect seeds), "
      f"LK {len(lk_known_all):,} prots (frozen v227 ref seeds)  ({time.time()-t0:.0f}s)", flush=True)

# ---- deployed pool: kNN candidates (eval_scores) ----
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

# ---- classifier extras (PK-BP): union_candidates_scored.npz, src in {cls, both} ----
uc = np.load(W / "union_candidates_scored.npz", allow_pickle=True)
uc_src = uc["src"]; keep = (uc_src == "cls") | (uc_src == "both")
cls_pk = collections.defaultdict(set)
for p_, g_ in zip(uc["prot"][keep], uc["term"][keep]):
    cls_pk[p_].add(alt.get(g_, g_))
print(f"pool: kNN PK {len(knn_pk):,}/LK {len(knn_lk):,} prots; classifier extras PK {len(cls_pk):,} prots. "
      f"(LK classifier extras NOT on disk)  ({time.time()-t0:.0f}s)", flush=True)

# ---- label spaces (learned annotation representations, cached) ----
az = np.load("/home/frapercan/Thesis2/repositories/PROTEA/artifacts/anc2vec/anc2vec_2020-10.npz", allow_pickle=True)
A_go = [alt.get(g, g) for g in az["go_ids"].tolist()]
A_emb = az["embeddings"].astype(np.float32)
A_emb /= (np.linalg.norm(A_emb, axis=1, keepdims=True) + 1e-9)
gv = np.load("/home/frapercan/Thesis2/storage/two_tower_sparse/go_sparse_codes.npz", allow_pickle=True)
G_go = [alt.get(g, g) for g in gv["go_ids"].tolist()]
G_emb = gv["codes"].astype(np.float32)
G_emb /= (np.linalg.norm(G_emb, axis=1, keepdims=True) + 1e-9)

class LabelSpace:
    def __init__(self, name, go_ids, emb):
        self.name = name
        self.pos = {g: i for i, g in enumerate(go_ids)}
        self.go = np.array(go_ids)
        self.emb = emb
        self.bp_idx = np.array([i for i, g in enumerate(go_ids) if g in BP])
        self.bp_emb = emb[self.bp_idx]
        self.bp_go = self.go[self.bp_idx]
        self._cache = {}
    def neighbors(self, seed_terms, K=50):
        """For each seed term present in the space, top-K BP neighbors (go, cos). Cached per term."""
        todo = [g for g in seed_terms if g in self.pos and g not in self._cache]
        for s in range(0, len(todo), 512):
            chunk = todo[s:s + 512]
            V = self.emb[[self.pos[g] for g in chunk]]           # (c, d)
            S = V @ self.bp_emb.T                                 # (c, n_bp)
            for r, g in enumerate(chunk):
                row = S[r]
                k = min(K, len(row) - 1)
                idx = np.argpartition(-row, k)[:k + 1]
                idx = idx[np.argsort(-row[idx])]
                self._cache[g] = [(self.bp_go[j], float(row[j])) for j in idx]
        return {g: self._cache.get(g, []) for g in seed_terms}

SPACES = [LabelSpace("anc2vec", A_go, A_emb), LabelSpace("two_tower_go", G_go, G_emb)]
print(f"label spaces: anc2vec BP={len(SPACES[0].bp_idx):,}, two_tower BP={len(SPACES[1].bp_idx):,}"
      f"  ({time.time()-t0:.0f}s)", flush=True)

CLS_BASE = {"true_ia": 6332.0, "false_ia": 99966.0, "ratio": 6332.0 / 99966.0,
            "source": "TEXT_AS_GENERATOR_IS_SCALE_CONFOUNDED.md, marginal over pool, V=113k"}

def run_cell(cell, gt, known_all, known_bp, knn, cls_extras):
    prots = sorted(set(gt) & set(knn))          # eval proteins with truth and a kNN pool
    out = {"n_prots": len(prots)}
    # ---- Q1 headroom ----
    tot_novel_ia = 0.0; res_ia_knn = 0.0; res_ia_both = 0.0
    tot_truth_ia = 0.0; cov_knn_ia = 0.0; cov_both_ia = 0.0
    per_prot = {}                                # prot -> (novel_set, pool_both_closure)
    for p in prots:
        gtc = gt[p]
        kbp = known_bp.get(p, set())
        novel = gtc - kbp
        knnc = closure(knn.get(p, set())) & BP
        both_terms = knn.get(p, set()) | cls_extras.get(p, set())
        bothc = closure(both_terms) & BP
        tot_truth_ia += iaw(gtc)
        cov_knn_ia += iaw(gtc & knnc)
        cov_both_ia += iaw(gtc & bothc)
        nia = iaw(novel)
        tot_novel_ia += nia
        res_ia_knn += iaw(novel - knnc)
        res_ia_both += iaw(novel - bothc)
        per_prot[p] = (novel, bothc, knnc, gtc)
    out["Q1"] = {
        "total_truth_ia_bp": round(tot_truth_ia, 1),
        "covered_by_knn_ia": round(cov_knn_ia, 1),
        "covered_by_knn_frac": round(cov_knn_ia / tot_truth_ia, 4) if tot_truth_ia else None,
        "covered_by_knn_union_cls_ia": round(cov_both_ia, 1),
        "covered_by_knn_union_cls_frac": round(cov_both_ia / tot_truth_ia, 4) if tot_truth_ia else None,
        "total_novel_true_ia": round(tot_novel_ia, 1),
        "residual_novel_ia_vs_knn": round(res_ia_knn, 1),
        "residual_novel_ia_vs_knn_union_cls": round(res_ia_both, 1),
        "residual_frac_vs_knn_union_cls": round(res_ia_both / tot_novel_ia, 4) if tot_novel_ia else None,
        "note": "residual uses pool CLOSURE (prop=fill inherits ancestors). This is the cap on any new generator.",
    }
    # ---- Q2 reachability + Q3 tax, per label space ----
    out["spaces"] = {}
    for sp in SPACES:
        # precompute neighbor lists for all unique seed terms in this cell
        allseeds = set()
        for p in prots:
            allseeds |= known_all.get(p, set())
        sp.neighbors(sorted(allseeds), K=50)
        Ks = [5, 10, 25, 50]
        reach = {k: 0.0 for k in Ks}
        res_total = out["Q1"]["residual_novel_ia_vs_knn_union_cls"]
        # Q3 accumulators at K=50 candidate universe, plus a global-ranked matched-volume pass
        cand_rows = []       # (score, term_ia, is_new_true_vs_both, is_new_false_vs_both, is_new_true_vs_knn, is_new_false_vs_knn)
        seeded_prots = 0
        for p in prots:
            seeds = known_all.get(p, set())
            if not seeds:
                continue
            seeded_prots += 1
            novel, bothc, knnc, gtc = per_prot[p]
            # candidate -> best cosine from any seed (top-50 neighborhoods)
            cand = {}
            for s in seeds:
                for g, c in sp._cache.get(s, []):
                    if c > cand.get(g, -1):
                        cand[g] = c
            # Q2 reachability of residual at each K (rank neighbors per seed, take top-k union)
            for k in Ks:
                nk = set()
                for s in seeds:
                    for g, c in sp._cache.get(s, [])[:k]:
                        nk.add(g)
                reach[k] += iaw((novel - bothc) & nk)
            # Q3: marginal IA of the generator's candidates (not already in pool), vs both closures
            for g, c in cand.items():
                if g in knnc and g in bothc:
                    continue
                w = IA.get(g, 0.0)
                nt_both = (g in gtc) and (g not in bothc)
                nf_both = (g not in gtc) and (g not in bothc)
                nt_knn = (g in gtc) and (g not in knnc)
                nf_knn = (g not in gtc) and (g not in knnc)
                cand_rows.append((c, w, nt_both, nf_both, nt_knn, nf_knn))
        q2 = {f"k{k}": {"residual_reached_ia": round(reach[k], 1),
                        "reachability_frac": round(reach[k] / res_total, 4) if res_total else None}
              for k in Ks}
        # Q3 ratios: full universe (top-50 neighborhoods) and matched-volume top-V
        cr = np.array([(r[1], r[2], r[3], r[4], r[5]) for r in cand_rows], np.float64) if cand_rows else np.zeros((0, 5))
        sc = np.array([r[0] for r in cand_rows], np.float64) if cand_rows else np.zeros(0)
        def ratios(idx):
            if len(idx) == 0:
                return None
            w = cr[idx, 0]
            tt_both = float((w * cr[idx, 1]).sum()); ff_both = float((w * cr[idx, 2]).sum())
            tt_knn = float((w * cr[idx, 3]).sum()); ff_knn = float((w * cr[idx, 4]).sum())
            return {"n_candidates": int(len(idx)),
                    "true_ia_vs_knn": round(tt_knn, 1), "false_ia_vs_knn": round(ff_knn, 1),
                    "ratio_vs_knn": round(tt_knn / ff_knn, 4) if ff_knn else None,
                    "true_ia_vs_both": round(tt_both, 1), "false_ia_vs_both": round(ff_both, 1),
                    "ratio_vs_both": round(tt_both / ff_both, 4) if ff_both else None}
        full = ratios(np.arange(len(sc)))
        matched = {}
        for V in (80000, 113000):
            if len(sc) > V:
                idx = np.argpartition(-sc, V)[:V]
            else:
                idx = np.arange(len(sc))
            matched[f"V{V}"] = ratios(idx)
        out["spaces"][sp.name] = {"seeded_prots": seeded_prots, "Q2_reachability": q2,
                                  "Q3_full_universe": full, "Q3_matched_volume": matched}
    return out

res = {"scope": "BP aspect only; PK and LK eval proteins with truth and a kNN pool.",
       "novelty_guard": "novel = gt(v230) closure minus v227 known closure (BP). PK known from "
                        "groundtruth_PK_known.tsv (experimental); LK has no experimental BP known "
                        "by category => novel = full gt closure.",
       "classifier_baseline": CLS_BASE,
       "artifacts": {
           "gt": str(REL / "groundtruth_{PK,LK}.tsv"),
           "pk_known": str(REL / "groundtruth_PK_known.tsv"),
           "lk_known_seeds": str(FROZEN / "reference_annotations.parquet") + " + go_term_metadata.parquet",
           "knn_pool": str(W / "rerank_out/eval_scores.parquet"),
           "classifier_extras_pk": str(W / "union_candidates_scored.npz"),
           "label_spaces": ["anc2vec_2020-10.npz", "two_tower_sparse/go_sparse_codes.npz"]},
       "cells": {}}
res["cells"]["PK_BP"] = run_cell("pk", gt_pk, pk_known_all, pk_known_bp, knn_pk, cls_pk)
res["cells"]["LK_BP"] = run_cell("lk", gt_lk, lk_known_all, collections.defaultdict(set), knn_lk,
                                 collections.defaultdict(set))
res["cells"]["LK_BP"]["_caveat"] = ("LK deployed pool = kNN only (classifier extras not on disk for LK); "
                                    "residual vs pool may be slightly OVERstated. Seeds from frozen v227 ref "
                                    "(all evidence codes incl IEA), so reachability is a GENEROUS ceiling.")
json.dump(res, open(W / "annotation_rag_ceiling.json", "w"), indent=1)

# ---- report ----
def show(cell, r):
    q = r["Q1"]
    print(f"\n===== {cell}  ({r['n_prots']:,} proteins) =====", flush=True)
    print(f"  Q1 truth covered by kNN            : {q['covered_by_knn_frac']:.1%} "
          f"({q['covered_by_knn_ia']:,.0f} / {q['total_truth_ia_bp']:,.0f} IA)", flush=True)
    print(f"  Q1 truth covered by kNN UNION clf  : {q['covered_by_knn_union_cls_frac']:.1%} "
          f"({q['covered_by_knn_union_cls_ia']:,.0f} IA)", flush=True)
    print(f"  Q1 total novel-true IA             : {q['total_novel_true_ia']:,.0f}", flush=True)
    print(f"  Q1 residual novel IA vs kNN        : {q['residual_novel_ia_vs_knn']:,.0f}", flush=True)
    print(f"  Q1 RESIDUAL vs kNN UNION clf (CAP) : {q['residual_novel_ia_vs_knn_union_cls']:,.0f} "
          f"({q['residual_frac_vs_knn_union_cls']:.1%} of novel)", flush=True)
    for name, s in r["spaces"].items():
        print(f"  --- label space: {name} (seeded {s['seeded_prots']:,} prots) ---", flush=True)
        for k, v in s["Q2_reachability"].items():
            print(f"      Q2 reach {k:>4}: {v['reachability_frac']} of residual "
                  f"({v['residual_reached_ia']:,.0f} IA)", flush=True)
        f = s["Q3_full_universe"]
        if f:
            print(f"      Q3 full universe : {f['n_candidates']:,} cand | vs kNN ratio {f['ratio_vs_knn']} "
                  f"({f['true_ia_vs_knn']:,.0f}/{f['false_ia_vs_knn']:,.0f}) | "
                  f"vs both ratio {f['ratio_vs_both']}", flush=True)
        m = s["Q3_matched_volume"].get("V113000")
        if m:
            print(f"      Q3 matched V=113k: vs kNN ratio {m['ratio_vs_knn']} "
                  f"({m['true_ia_vs_knn']:,.0f}/{m['false_ia_vs_knn']:,.0f})  "
                  f"[classifier baseline {CLS_BASE['ratio']:.4f} = 6332/99966]", flush=True)
show("PK_BP", res["cells"]["PK_BP"])
show("LK_BP", res["cells"]["LK_BP"])
print(f"\nclassifier baseline true/false ratio at V=113k: {CLS_BASE['ratio']:.4f}  (6,332 / 99,966 IA)", flush=True)
print(f"DONE  ({time.time()-t0:.0f}s)", flush=True)
