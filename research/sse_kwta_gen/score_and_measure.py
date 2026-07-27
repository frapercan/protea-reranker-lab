"""Score the FROZEN-rep SSE on the sealed v227->v230 eval frame and MEASURE generalization capacity.

Deliverable: threshold-free RANKING metrics (AUROC, AUPR, protein-centric Fmax) per aspect,
DECOMPOSED into three generalization regimes, plus controls, recall ceiling, and the
containment-semantics check on the frozen champion k-WTA rep.

Regimes (over eval positive (protein,term) pairs; negatives = vocab terms neither true nor known@t0):
  R1  known@t0        annotation already true at t0 (v227) -> excluded by the honest -known frame
                      = memorization / in-distribution recall UPPER REFERENCE
  R2  new / seen-term new in v227->v230, term train-support >= S_HI (well-represented term) -> LK-like
  R3  new / novel-term new in v227->v230, term train-support <  S_HI (rare/deep novel term) -> the wall
                      (out-of-vocab novel terms are UNREACHABLE by accept-all -> the recall-ceiling loss)

Controls: random-order (chance) per regime; K=1 vs K=5-MIN; vs the deployed reranker ordering on the
PK-BPO reachable tail. Footnote: one calibrated f_micro_w for accept-all vs the reproduced deployed
anchor, PK -known frame (confirms accept-all floods; NOT the verdict). Frozen data only; NO live DB.
"""
import json, time, os, collections, sys, tempfile, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import torch as th
from sklearn.metrics import roc_auc_score, average_precision_score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

ROOT = Path("/home/frapercan/Thesis2")
RR   = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
OBO  = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
GTDIR= RR / "lafa_gt"
OUT  = ROOT / "storage/sse_kwta_gen"
RECEIPT = ROOT / "storage/regen_headline"
DEV  = "cuda:0"
ASPECTS = {"bpo": "biological_process", "mfo": "molecular_function", "cco": "cellular_component"}
ASPLET  = {"bpo": "P", "mfo": "F", "cco": "C"}
POOL = {"PK": RR / "percut_rerank/predictions/pk/pk.tsv", "LK": RR / "percut_rerank/predictions/lk/lk.tsv"}
ANCHOR = {"PK-bpo": 0.14351, "PK-mfo": 0.2483, "PK-cco": 0.2677, "LK-bpo": 0.31323}

# ---------------- obo ----------------
par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"):
        par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
NSSET = {a: {t for t, n in ns.items() if n == full} for a, full in ASPECTS.items()}
_anc = {}
def anc(t, aspset):
    t = alt.get(t, t); key = (t, id(aspset))
    if key in _anc: return _anc[key]
    seen, stk = set(), [t]
    while stk:
        x = stk.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): stk.append(p)
    r = frozenset(x for x in seen if x in aspset); _anc[key] = r; return r
IA = {}
for line in open(IA_F):
    p = line.split("\t")
    if len(p) >= 2:
        try: IA[p[0].strip()] = float(p[1])
        except ValueError: pass
def iaw(g): return IA.get(g, 0.0)

# ---------------- eval frozen codes + t0-known labels ----------------
E = np.load(RR / "eval_protein_codes.npz", allow_pickle=True)
eaccs = E["eval_accs" if "eval_accs" in E else "accs"].tolist()
ECODES = E["codes"]                        # (7401,2048) fp16 d8979601
eidx = {a: i for i, a in enumerate(eaccs)}
ESUP = th.tensor((ECODES != 0).astype(np.float32))          # (7401,2048) binary support
lab_big = json.load(open(RR / "clf_labels_big.json"))
log(f"eval codes {ECODES.shape}; t0-known labels loaded")

# cell membership per protein
CELL = {}
for cell in ["LK", "PK"]:
    gt = pd.read_csv(GTDIR / f"groundtruth_{cell}.tsv", sep="\t")
    for p in gt.EntryID.unique(): CELL[p] = cell

def score_aspect(asp, seed_reduce):
    """Return {prot: score_row over vocab}. seed_reduce: 'k1' seed0 soft, 'k5min' min over 5."""
    T = np.load(OUT / f"termcodes_{asp}.npz", allow_pickle=True)
    soft = T["soft"].astype(np.float32)     # (K, n_terms, N)
    return soft

def coverage_rows(prot_ids, soft_seed):
    gt = th.tensor(soft_seed, device=DEV)                 # (T,N)
    denom = gt.sum(1) + 1e-6
    idx = [eidx[p] for p in prot_ids]
    X = ESUP[idx].to(DEV)                                  # (P,N) binary
    outs = []
    for s in range(0, len(idx), 2048):
        outs.append(((X[s:s+2048] @ gt.t()) / denom).cpu().numpy().astype(np.float32))
    return np.concatenate(outs)                           # (P,T)

# ---------------- per-protein ranking metrics ----------------
def perprot_auroc_aupr(S, pos_mask, neg_mask):
    """S (P,T) scores; pos_mask/neg_mask (P,T) bool. Per-protein AUROC/AUPR averaged + micro."""
    aurocs, auprs = [], []
    mic_s, mic_y = [], []
    for i in range(S.shape[0]):
        pm = pos_mask[i]; nm = neg_mask[i]
        npos = int(pm.sum()); nneg = int(nm.sum())
        if npos == 0 or nneg == 0: continue
        sc = np.concatenate([S[i, pm], S[i, nm]])
        y = np.concatenate([np.ones(npos), np.zeros(nneg)])
        aurocs.append(roc_auc_score(y, sc)); auprs.append(average_precision_score(y, sc))
        mic_s.append(sc); mic_y.append(y)
    out = {"n_prot": len(aurocs),
           "auroc_perprot": float(np.mean(aurocs)) if aurocs else None,
           "aupr_perprot": float(np.mean(auprs)) if auprs else None}
    if mic_s:
        ys = np.concatenate(mic_y); ss = np.concatenate(mic_s)
        out["auroc_micro"] = float(roc_auc_score(ys, ss))
        out["aupr_micro"] = float(average_precision_score(ys, ss))
        out["prevalence"] = float(ys.mean())
    return out

def protein_fmax(S, pos_mask, neg_mask, taus=None):
    """Protein-centric threshold-free Fmax over ranked scores; R-positives are the only 'true'."""
    if taus is None: taus = np.linspace(0.0, 1.0, 101)
    P = S.shape[0]
    best = 0.0; best_tau = 0.0; best_rec = 0.0
    cand = pos_mask | neg_mask
    for tau in taus:
        precs, recs = [], []
        for i in range(P):
            if not cand[i].any(): continue
            predm = cand[i] & (S[i] >= tau)
            npred = int(predm.sum())
            if pos_mask[i].sum() == 0:
                rec_i = None
            else:
                tp = int((predm & pos_mask[i]).sum())
                rec_i = tp / int(pos_mask[i].sum())
                if npred > 0: precs.append(tp / npred)
                recs.append(rec_i)
        if recs:
            pr = np.mean(precs) if precs else 0.0; rc = np.mean(recs)
            f = 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0.0
            if f > best: best = f; best_tau = float(tau); best_rec = float(rc)
    return {"fmax": float(best), "tau": best_tau, "recall_at_fmax": best_rec}

# ---------------- main per-aspect measurement ----------------
report = {"model": {
    "name": "single-model accept-all Sparse Semantic-Entailment over the FROZEN champion k-WTA rep (d8979601)",
    "protein_rep": "FROZEN champion learned k-WTA codes d8979601 (2048-d, 128 active); support = non-zero dims",
    "train_codes": "repositories/.../clf_protein_codes_big.npz (554,378 v227 corpus, eval HELD OUT)",
    "eval_codes": "repositories/.../eval_protein_codes.npz (7,401 sealed LAFA v227->v230; SAME d8979601 encoder)",
    "term_tower": "learned embedding -> soft k-WTA per-term cardinality by depth; nf1/nf4 EL axioms, parent gate detached",
    "K": "1 (primary, accept-all by order); K=5 MIN kept as control",
    "temporal_gate": "term codes trained on v227(t0) co-annotations, eval proteins held out; v227->v230 blind"},
    "aspects": {}}

S_HI_used = {}
for asp in ["bpo", "mfo", "cco"]:
    aspset = NSSET[asp]; asplet = ASPLET[asp]
    T = np.load(OUT / f"termcodes_{asp}.npz", allow_pickle=True)
    vocab = T["vocab"].tolist(); tidx = {g: j for j, g in enumerate(vocab)}
    train_support = {g: int(c) for g, c in zip(vocab, T["train_support"])}
    soft = T["soft"].astype(np.float32)     # (K,T,N)
    K = soft.shape[0]; n_terms = len(vocab)
    log(f"===== {asp}: vocab {n_terms}, K {K} =====")

    # eval targets for this aspect (LK+PK), their gt (propagated) and known@t0
    gt_all = {}
    for cell in ["LK", "PK"]:
        g = pd.read_csv(GTDIR / f"groundtruth_{cell}.tsv", sep="\t")
        g = g[g.aspect == asplet]
        for p, sub in g.groupby("EntryID"):
            prop = set()
            for t in sub.term: prop |= anc(t, aspset)
            gt_all.setdefault(p, set()).update(prop)
    targets = [p for p in gt_all if p in eidx]
    targets.sort()
    log(f"  {asp} eval targets: {len(targets)}")

    known = {}   # p -> propagated t0-known set (this aspect)
    for p in targets:
        kk = set()
        for t in lab_big.get(p, ()):
            t = alt.get(t, t)
            if t in aspset: kk |= anc(t, aspset)
        known[p] = kk

    # choose S_HI = median train-support of NEW in-vocab positive terms (data-driven, reported)
    new_supports = []
    for p in targets:
        for t in gt_all[p]:
            if t in known[p]: continue
            if t in tidx: new_supports.append(train_support[t])
    S_HI = int(np.median(new_supports)) if new_supports else 100
    S_HI = max(S_HI, 50)
    S_HI_used[asp] = S_HI
    log(f"  {asp} S_HI (median new in-vocab support) = {S_HI}")

    # build masks over vocab: R1/R2/R3 positives (in-vocab, scoreable) + shared negatives
    P = len(targets)
    r1 = np.zeros((P, n_terms), bool); r2 = np.zeros((P, n_terms), bool); r3 = np.zeros((P, n_terms), bool)
    negm = np.zeros((P, n_terms), bool)
    # IA-mass bookkeeping for recall ceiling
    iamass = {"all": collections.defaultdict(float), "invocab": collections.defaultdict(float),
              "R1": 0.0, "R2": 0.0, "R3_invocab": 0.0, "R3_oov": 0.0}
    oov_terms = collections.Counter()
    for i, p in enumerate(targets):
        truep = gt_all[p]; kn = known[p]
        neg_terms = set(vocab) - truep - kn
        for t in neg_terms:
            negm[i, tidx[t]] = True
        for t in truep:
            w = iaw(t); iamass["all"][asp] += w
            invoc = t in tidx
            if invoc: iamass["invocab"][asp] += w
            if t in kn:
                if invoc: r1[i, tidx[t]] = True; iamass["R1"] += w
            else:
                if not invoc:
                    iamass["R3_oov"] += w; oov_terms[t] += 1
                elif train_support[t] >= S_HI:
                    r2[i, tidx[t]] = True; iamass["R2"] += w
                else:
                    r3[i, tidx[t]] = True; iamass["R3_invocab"] += w

    # scores K1 (seed0) and K5-min
    S_k1 = coverage_rows(targets, soft[0])
    if K > 1:
        per = [coverage_rows(targets, soft[k]) for k in range(K)]
        S_k5 = np.min(np.stack(per), axis=0)
    else:
        S_k5 = S_k1
    rng = np.random.default_rng(0); S_rand = rng.permutation(S_k1.reshape(-1)).reshape(S_k1.shape)

    res = {"n_targets": P, "n_vocab": n_terms, "S_HI": S_HI,
           "counts": {"R1_pos": int(r1.sum()), "R2_pos": int(r2.sum()), "R3_invocab_pos": int(r3.sum()),
                      "R3_oov_terms": int(sum(oov_terms.values())), "neg_mean_per_prot": float(negm.sum(1).mean())},
           "regimes": {}}
    for name, mask in [("R1_known_memorization", r1), ("R2_new_seen_term", r2), ("R3_new_novel_term", r3)]:
        m_k1 = perprot_auroc_aupr(S_k1, mask, negm)
        m_k1["fmax"] = protein_fmax(S_k1, mask, negm)
        m_k5 = perprot_auroc_aupr(S_k5, mask, negm)
        m_rand = perprot_auroc_aupr(S_rand, mask, negm)
        res["regimes"][name] = {"K1": m_k1, "K5min": {k: m_k5.get(k) for k in ("auroc_perprot","aupr_perprot","auroc_micro","aupr_micro")},
                                "random_control": {k: m_rand.get(k) for k in ("auroc_perprot","aupr_perprot")}}
        a1 = m_k1.get("auroc_perprot"); log(f"  {asp} {name}: K1 AUROC {a1} AUPR {m_k1.get('aupr_perprot')} Fmax {m_k1['fmax']['fmax']:.4f} (n {m_k1['n_prot']})")

    # recall ceiling (IA-mass) under accept-all vs pool reachability
    all_mass = iamass["all"][asp]; inv_mass = iamass["invocab"][asp]
    res["recall_ceiling"] = {
        "note": "accept-all recovers every IN-VOCAB true term; the ceiling is a data/reachability property (out-of-vocab novel terms are unreachable), NOT a model choice.",
        "ia_mass_total": all_mass,
        "acceptall_reachable_frac": (inv_mass / all_mass) if all_mass else None,
        "regime_ia_mass": {"R1_known": iamass["R1"], "R2_seen": iamass["R2"],
                           "R3_novel_invocab": iamass["R3_invocab"], "R3_novel_oov_UNREACHABLE": iamass["R3_oov"]}}

    # pool reachability (deployed two-tower+reranker candidate coverage of NEW true IA-mass)
    for cell in ["PK", "LK"]:
        pool = pd.read_csv(POOL[cell], sep="\t", header=None, names=["p", "t", "s"])
        pool["tp"] = pool["t"].map(lambda g: alt.get(g, g))
        pool = pool[pool["tp"].isin(aspset) & (pool["s"] > 0)]
        prop_pool = collections.defaultdict(set)
        for p, tp in zip(pool["p"], pool["tp"]):
            prop_pool[p].add(tp)
        newmass = 0.0; poolmass = 0.0
        for p in targets:
            if CELL.get(p) != cell: continue
            pr = prop_pool.get(p, set())
            for t in gt_all[p]:
                if t in known[p]: continue
                newmass += iaw(t)
                if t in pr: poolmass += iaw(t)
        res["recall_ceiling"][f"pool_reachability_{cell}_new"] = (poolmass / newmass) if newmass else None

    report["aspects"][asp] = res
    json.dump(report, open(OUT / "measure_partial.json", "w"), indent=1, default=float)
    log(f"  {asp} measured + partial written")

# ---------------- vs deployed reranker on PK-BPO reachable tail ----------------
log("===== vs reranker (PK-BPO reachable tail) =====")
asp = "bpo"; aspset = NSSET[asp]
T = np.load(OUT / f"termcodes_{asp}.npz", allow_pickle=True)
vocab = T["vocab"].tolist(); tidx = {g: j for j, g in enumerate(vocab)}
soft = T["soft"].astype(np.float32)
pool = pd.read_csv(POOL["PK"], sep="\t", header=None, names=["p", "t", "s"])
pool["tp"] = pool["t"].map(lambda g: alt.get(g, g))
pool = pool[pool["tp"].isin(aspset)]
g = pd.read_csv(GTDIR / "groundtruth_PK.tsv", sep="\t"); g = g[g.aspect == "P"]
gt_pk = {}
for p, sub in g.groupby("EntryID"):
    s = set()
    for t in sub.term: s |= anc(t, aspset)
    gt_pk[p] = s
known_pk = {}
for p in gt_pk:
    kk = set()
    for t in lab_big.get(p, ()):
        t = alt.get(t, t)
        if t in aspset: kk |= anc(t, aspset)
    known_pk[p] = kk
# SSE K1 scores for PK targets over vocab
pk_targets = sorted([p for p in gt_pk if p in eidx])
S_k1_pk = coverage_rows(pk_targets, soft[0])
pki = {p: i for i, p in enumerate(pk_targets)}
rr_auc, sse_auc = [], []; n_tail = 0
pool_by_p = collections.defaultdict(list)
for p, tp, s in zip(pool["p"], pool["tp"], pool["s"]):
    pool_by_p[p].append((tp, s))
for p in pk_targets:
    rows = pool_by_p.get(p, [])
    if not rows: continue
    # reachable-tail terms = pool BP terms, label = NEW true (exclude known@t0)
    y, sr, gg = [], [], []
    kn = known_pk[p]; truep = gt_pk[p]
    for tp, s in rows:
        if tp in kn: continue
        lab = 1 if tp in truep else 0
        y.append(lab); sr.append(s); gg.append(tp)
    if sum(y) == 0 or sum(y) == len(y): continue
    y = np.array(y)
    # SSE score on the SAME pool terms (in-vocab only; oov -> skip, both models see same set)
    sse_s = []; keep = []
    for k, tp in enumerate(gg):
        if tp in tidx:
            sse_s.append(S_k1_pk[pki[p], tidx[tp]]); keep.append(k)
    if len(set(y[keep].tolist())) < 2: continue
    rr_auc.append(roc_auc_score(y[keep], np.array(sr)[keep]))
    sse_auc.append(roc_auc_score(y[keep], np.array(sse_s)))
    n_tail += 1
report["vs_reranker_PK_bpo_tail"] = {
    "n_proteins": n_tail,
    "reranker_auc_mean": float(np.mean(rr_auc)) if rr_auc else None,
    "sse_k1_auc_mean": float(np.mean(sse_auc)) if sse_auc else None,
    "sse_minus_reranker": float(np.mean(sse_auc) - np.mean(rr_auc)) if rr_auc else None,
    "note": "per-protein AUC on the deployed reachable pool tail (in-vocab, NEW terms, known@t0 excluded); same term set for both."}
log(f"  reranker {report['vs_reranker_PK_bpo_tail']['reranker_auc_mean']} vs SSE-K1 {report['vs_reranker_PK_bpo_tail']['sse_k1_auc_mean']}")

# ---------------- containment semantics on the frozen rep ----------------
log("===== containment semantics check =====")
cont = {}
for asp in ["bpo", "mfo", "cco"]:
    aspset = NSSET[asp]
    T = np.load(OUT / f"termcodes_{asp}.npz", allow_pickle=True)
    vocab = T["vocab"].tolist(); tidx = {g: j for j, g in enumerate(vocab)}
    hard = T["hard"][0]                  # (T,N) bool seed0
    nf1 = T["nf1"]                       # (C,D) child C, parent D: supp(z_D) subset supp(z_C)
    # (a) term-code axiom containment vs random floor
    ratios, full = [], 0
    for C, D in nf1:
        supD = hard[D]; nd = supD.sum()
        if nd == 0: continue
        inter = (supD & hard[C]).sum(); ratios.append(inter / nd); full += int(inter == nd)
    rr = np.random.default_rng(7); perm = rr.permutation(len(nf1)) if len(nf1) else []
    rand = []
    for a_i, (C, D) in enumerate(nf1):
        Dp = nf1[perm[a_i]][1]; supD = hard[Dp]; nd = supD.sum()
        if nd == 0: continue
        rand.append((supD & hard[C]).sum() / nd)
    # (b) frozen-rep hierarchy respect: for true annotations, coverage(parent) >= coverage(child)?
    #     use eval PK+LK gt; coverage from seed0 soft codes
    soft0 = T["soft"][0].astype(np.float32)
    denom = soft0.sum(1) + 1e-6
    mono_ok = 0; mono_tot = 0
    # sample child-parent nf1 pairs where a protein truly has the child
    gtmap = {}
    for cell in ["LK", "PK"]:
        gdf = pd.read_csv(GTDIR / f"groundtruth_{cell}.tsv", sep="\t"); gdf = gdf[gdf.aspect == ASPLET[asp]]
        for p, sub in gdf.groupby("EntryID"):
            if p not in eidx: continue
            s = set()
            for t in sub.term: s |= anc(t, aspset)
            gtmap[p] = s
    samp_p = list(gtmap)[:400]
    Xs = ESUP[[eidx[p] for p in samp_p]].numpy()
    for pi, p in enumerate(samp_p):
        cov = (Xs[pi:pi+1] @ soft0.T / denom).ravel()
        truep = gtmap[p]
        for C, D in nf1[:2000]:
            gc = vocab[C]; gd = vocab[D]
            if gc in truep and gd in truep:   # child C true -> parent D true
                mono_tot += 1
                if cov[D] >= cov[C] - 1e-6: mono_ok += 1
    cont[asp] = {"nf1_pairs": len(ratios),
                 "term_axiom_mean_containment": float(np.mean(ratios)) if ratios else None,
                 "term_axiom_frac_full": full / max(1, len(ratios)),
                 "random_floor": float(np.mean(rand)) if rand else None,
                 "frozen_hierarchy_monotone_frac": (mono_ok / mono_tot) if mono_tot else None,
                 "frozen_hierarchy_n": mono_tot}
    log(f"  {asp} term-axiom containment {cont[asp]['term_axiom_mean_containment']} (floor {cont[asp]['random_floor']}); "
        f"frozen-rep parent>=child coverage {cont[asp]['frozen_hierarchy_monotone_frac']}")
report["containment_semantics"] = cont
report["containment_semantics"]["caveat"] = ("k-WTA codes were learned for DISCRIMINATION, not as a "
    "union-of-required-modules; term-axiom containment sits above the random floor but is not exact "
    "subset-containment. A CO-TRAINED SSE (protein + term towers jointly) is the upper bound; here the "
    "protein rep is frozen so containment is only as good as the discriminative code allows.")
json.dump(report, open(OUT / "measure_partial.json", "w"), indent=1, default=float)

# ---------------- footnote: accept-all f_micro_w vs reproduced deployed anchor (PK -known) ----------------
log("===== anchor footnote (PK -known cafa_eval) =====")
sys.path.insert(0, str(ROOT / "storage/deepgose_faithful"))
foot = {}
if os.environ.get("SKIP_FOOT"):
    foot = {"skipped": True}
try:
    if os.environ.get("SKIP_FOOT"): raise RuntimeError("skip")
    import measure_pk as MPK
    anchor_dir = str(RR / "percut_rerank/predictions/pk")
    cp = MPK.cafa_point(anchor_dir)     # reproduce PK anchors, all aspects
    foot["anchor_reproduced"] = {a: round(v["f_micro_w"], 5) for a, v in cp.items()}
    log(f"  anchor reproduced: {foot['anchor_reproduced']}")
    # build accept-all SSE submission for PK, all aspects, score with -known
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pk"; d.mkdir()
        fh = (d / "pk.tsv").open("w")
        for asp in ["bpo", "mfo", "cco"]:
            aspset = NSSET[asp]
            T = np.load(OUT / f"termcodes_{asp}.npz", allow_pickle=True)
            vocab = T["vocab"].tolist(); soft = T["soft"].astype(np.float32)
            g = pd.read_csv(GTDIR / "groundtruth_PK.tsv", sep="\t"); g = g[g.aspect == ASPLET[asp]]
            tp = sorted([p for p in g.EntryID.unique() if p in eidx])
            Spk = coverage_rows(tp, soft[0])
            # min-max normalise per aspect to [0,1] so accept-all writes all terms with graded score
            lo, hi = Spk.min(), Spk.max(); rng2 = (hi - lo) if hi > lo else 1.0
            for i, p in enumerate(tp):
                row = (Spk[i] - lo) / rng2
                for j, gg in enumerate(vocab):
                    v = row[j]
                    if v > 0.001: fh.write(f"{p}\t{gg}\t{v:.4f}\n")
        fh.close()
        ap = MPK.cafa_point(str(d))
        foot["acceptall_sse_f_micro_w"] = {a: round(v["f_micro_w"], 5) for a, v in ap.items()}
        foot["delta_vs_anchor"] = {a: round(ap[a]["f_micro_w"] - cp[a]["f_micro_w"], 5)
                                   for a in ap if a in cp}
    log(f"  accept-all SSE f_micro_w {foot.get('acceptall_sse_f_micro_w')} delta {foot.get('delta_vs_anchor')}")
except Exception as e:
    foot["error"] = repr(e); log(f"  footnote failed: {e!r}")
foot["note"] = ("accept-all floods the calibrated f_micro_w (max recall, catastrophic precision); "
                "this is a FOOTNOTE confirming the known flood, NOT the generalization verdict.")
report["acceptall_fmicrow_footnote"] = foot

json.dump(report, open(RECEIPT / "SSE_KWTA_GEN.json", "w"), indent=1, default=float)
json.dump(report, open(OUT / "measure_partial.json", "w"), indent=1, default=float)
log("WROTE SSE_KWTA_GEN.json")
print(json.dumps(report["aspects"], indent=1, default=float)[:3000])
