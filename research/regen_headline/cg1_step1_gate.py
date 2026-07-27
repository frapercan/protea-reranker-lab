"""CG.1 STEP 1 -- the GATE.

Does a GENERATOR P(new BP term t | known set K_p), built from FROZEN t0 co-occurrence,
add TRUE BP-tail IA-mass that the deployed sequence-only pool lacks?

This is GENERATION, not rescoring: we propose (protein, BP-term) pairs that are NOT in
eval.parquet's pool for that protein, and measure the true-tail IA-mass they ADD (the ancestors
of a proposed true term inherit TP under prop=fill, which raises f_micro_w structurally).

Frozen data only. No DB, no cafaeval (Step 2 does cafaeval). No live services.

Co-occurrence source: reference_annotations.parquet (frozen v227 = t0, cutoff 2025-09-04). This is
strictly PRE-eval-window (gains are v227->v230), so it cannot leak the target gains. It is the only
full annotation corpus materialised on disk; a strict v<=225 corpus is not on disk. As an
UNSUPERVISED t0 statistic this is the deployable choice (all annotations knowable at t0).

Known set provenance:
  PK: groundtruth_PK_known.tsv, BP terms (board-frozen, clean).  BP->BP within-aspect generation.
  LK: reference_annotations F/C terms for the target protein (frozen v227=t0). MF/CC->BP cross-aspect.
      Leakage self-checks: (i) known set is F/C only, aspect-disjoint from the BP targets;
      (ii) no gained BP term can appear in the known set.  See LEAKAGE section below.
"""
import json, collections, time
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
F = ROOT / "storage/protea-frozen-v227-2025-09-04"
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
EVAL = R / "percut_rerank/eval.parquet"
GTDIR = R / "lafa_gt"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
OUT = ROOT / "storage/regen_headline"
TOPKS = [5, 10, 25, 50]


def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


# ---------- obo: BP ancestors (is_a + part_of), alt_id map, namespace ----------
par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
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
log(f"obo: {len(ns):,} terms, {len(BP):,} BP")

_anc = {}
def anc(t):
    """ancestors-or-self of t, restricted to BP."""
    t = alt.get(t, t)
    if t in _anc: return _anc[t]
    seen, stack = set(), [t]
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): stack.append(p)
    r = frozenset(x for x in seen if x in BP)
    _anc[t] = r
    return r

IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass
def iamass(terms): return float(sum(IA.get(t, 0.0) for t in terms))
log(f"IA: {len(IA):,} terms")

# ---------- frozen reference annotations -> co-occurrence + LK known ----------
meta = pq.read_table(F / "go_term_metadata.parquet").to_pandas()
id2go = dict(zip(meta.go_term_id, meta.go_id))
id2asp = dict(zip(meta.go_term_id, meta.aspect))       # 'P'/'F'/'C'
ref = pq.read_table(F / "reference_annotations.parquet", columns=["accession", "go_term_id"]).to_pandas()
ref["go"] = ref.go_term_id.map(id2go).map(lambda g: alt.get(g, g) if isinstance(g, str) else g)
ref["asp"] = ref.go_term_id.map(id2asp)
ref = ref.dropna(subset=["go"])
log(f"reference_annotations: {len(ref):,} rows, {ref.accession.nunique():,} proteins")

prot_codes, prot_uniq = pd.factorize(ref.accession, sort=False)
term_codes, term_uniq = pd.factorize(ref.go, sort=False)
A = sp.csr_matrix((np.ones(len(ref), np.float32), (prot_codes, term_codes)),
                  shape=(len(prot_uniq), len(term_uniq)))
A.data[:] = 1.0
A = A.tocsc()
termpos = {g: i for i, g in enumerate(term_uniq)}
term_marg = np.asarray(A.sum(0)).ravel()
Ntot = A.shape[0]
log(f"cooc A {A.shape}, nnz {A.nnz:,}")
# candidate BP term universe (present in reference)
BP_cols = np.array([i for i, g in enumerate(term_uniq) if g in BP and term_marg[i] > 0])
BP_terms = [term_uniq[i] for i in BP_cols]
mT = term_marg[BP_cols].astype(np.float64)
log(f"candidate BP terms (in reference): {len(BP_terms):,}")

# LK known = reference F/C terms per protein (frozen v227=t0)
fc = ref[ref.asp.isin(["F", "C"])]
lk_known = fc.groupby("accession").go.apply(set).to_dict()

# PK known = groundtruth_PK_known.tsv, BP terms
pkk = pd.read_csv(GTDIR / "groundtruth_PK_known.tsv", sep="\t")
pkk["term"] = pkk.term.map(lambda g: alt.get(g, g))
pk_known = pkk[pkk.aspect == "P"].groupby("EntryID").term.apply(set).to_dict()


def ppmi_seed_to_BP(seed_terms):
    """PPMI(seed, BP-term) matrix, seeds x |BP_terms|, seeds restricted to those in reference."""
    sidx = np.array([termpos[s] for s in seed_terms if s in termpos])
    kept = [s for s in seed_terms if s in termpos]
    As = A[:, sidx]                      # n_prot x |S|
    At = A[:, BP_cols]                   # n_prot x |T|
    C = (As.T @ At).toarray().astype(np.float64)     # |S| x |T|
    mS = term_marg[sidx].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C / Ntot) / ((mS[:, None] / Ntot) * (mT[None, :] / Ntot) + 1e-12) + 1e-12)
    return np.maximum(0.0, pmi), {s: j for j, s in enumerate(kept)}


def load_pool(target_prots):
    """BP pool terms per protein from eval.parquet (all rows for the protein, BP terms only)."""
    tb = pq.read_table(EVAL, columns=["protein_accession", "go_term_id"]).to_pandas()
    tb = tb[tb.protein_accession.isin(target_prots)]
    tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
    tb = tb[tb.go.isin(BP)]
    return tb.groupby("protein_accession").go.apply(set).to_dict()


results = {}
for cell, cat, gt_fn, known_of in [
    ("LK-BPO", "lk", "groundtruth_LK.tsv", lk_known),
    ("PK-BPO", "pk", "groundtruth_PK.tsv", pk_known),
]:
    log(f"===== {cell} =====")
    gt = pd.read_csv(GTDIR / gt_fn, sep="\t")
    gt = gt[gt.aspect == "P"].copy()
    gt["term"] = gt.term.map(lambda g: alt.get(g, g))
    gt_leaf = gt.groupby("EntryID").term.apply(set).to_dict()
    targets = sorted(gt_leaf)                              # cell = gt BP proteins
    pool = load_pool(set(targets))
    log(f"  {cell}: {len(targets)} target proteins, pool present for {len(set(targets)&set(pool))}")

    # LEAKAGE self-check (LK): known set must be F/C only + disjoint from gained BP
    leak = {}
    if cat == "lk":
        n_with_known = sum(1 for p in targets if known_of.get(p))
        # aspect purity: every known term of an LK target must be F or C
        bad_asp = 0
        for p in targets:
            ks = known_of.get(p, set())
            bad_asp += sum(1 for k in ks if k in BP)                     # BP in "known" => leak
        leak = {"targets_with_frozen_known": n_with_known,
                "mean_known_size": round(np.mean([len(known_of.get(p, set())) for p in targets]), 2),
                "known_terms_that_are_BP_should_be_0": int(bad_asp),
                "known_source": "reference_annotations.parquet F/C (frozen v227=t0)"}
        log(f"  LK leakage self-check: {n_with_known}/{len(targets)} have frozen F/C known; "
            f"BP-in-known={bad_asp} (must be 0)")
    else:
        leak = {"targets_with_frozen_known": sum(1 for p in targets if known_of.get(p)),
                "mean_known_size": round(np.mean([len(known_of.get(p, set())) for p in targets]), 2),
                "known_source": "groundtruth_PK_known.tsv BP (board-frozen)"}

    # ---- propagate gt + pool; ceiling accounting ----
    gt_prop = {p: frozenset().union(*[anc(t) for t in ts]) if ts else frozenset()
               for p, ts in gt_leaf.items()}
    pool_prop = {p: (frozenset().union(*[anc(t) for t in pool[p]]) if pool.get(p) else frozenset())
                 for p in targets}
    tp_pool = sum(iamass(gt_prop[p] & pool_prop[p]) for p in targets)
    fn_total = sum(iamass(gt_prop[p] - pool_prop[p]) for p in targets)
    gt_total = sum(iamass(gt_prop[p]) for p in targets)
    log(f"  gt true IA mass {gt_total:.1f} | pool-captured {tp_pool:.1f} "
        f"({100*tp_pool/gt_total:.1f}%) | FN mass {fn_total:.1f} ({100*fn_total/gt_total:.1f}%)")

    # ---- generator: PPMI(seed->BP), score(p,t)=sum_{s in K_p} PPMI(s,t) ----
    seed_universe = sorted(set().union(*[known_of.get(p, set()) for p in targets]) if targets else set())
    PPMI, spos = ppmi_seed_to_BP(seed_universe)
    log(f"  generator: {len(spos)} seeds in reference, PPMI {PPMI.shape}")
    # rank per protein, take top max(TOPKS) not in pool
    KMAX = max(TOPKS)
    pool_present = {p: pool.get(p, set()) for p in targets}
    # accumulate true/false mass added at each k (micro over proteins)
    added_true = {k: 0.0 for k in TOPKS}
    added_false = {k: 0.0 for k in TOPKS}
    n_prop = {k: 0 for k in TOPKS}
    # ceiling if generator perfect: propose exactly the FN true leaves reachable = fn_total
    for p in targets:
        ks = [spos[s] for s in known_of.get(p, set()) if s in spos]
        if not ks:
            continue
        score = PPMI[ks, :].sum(0)                          # |BP_terms|
        order = np.argsort(-score)
        prd = pool_prop[p]                                  # already-predicted (propagated) set
        gtp = gt_prop[p]
        present = pool_present[p]
        covered = set(prd)                                  # grows as we add proposals
        picks = 0
        kptr = 0
        for j in order:
            if score[j] <= 0:
                break
            g = BP_terms[j]
            if g in present:                                # already in pool -> skip (generation only)
                continue
            picks += 1
            ancg = anc(g)
            new = ancg - covered
            new_true = new & gtp
            new_false = new - gtp
            tm = iamass(new_true); fm = iamass(new_false)
            covered |= new
            for k in TOPKS:
                if picks <= k:
                    added_true[k] += tm; added_false[k] += fm; n_prop[k] += 1
            if picks >= KMAX:
                break

    cell_res = {
        "target_proteins": len(targets),
        "leakage": leak,
        "gt_true_IA_mass": round(gt_total, 2),
        "pool_captured_IA_mass": round(tp_pool, 2),
        "pool_captured_frac": round(tp_pool / gt_total, 4),
        "FN_IA_mass_pool_misses": round(fn_total, 2),
        "FN_frac_of_gt": round(fn_total / gt_total, 4),
        "generator_ceiling_true_mass_reachable": round(fn_total, 2),
        "per_k": {},
    }
    for k in TOPKS:
        at, af = added_true[k], added_false[k]
        cell_res["per_k"][f"top{k}"] = {
            "proposals": n_prop[k],
            "true_IA_mass_added": round(at, 3),
            "false_IA_mass_added": round(af, 3),
            "generation_precision_IA": round(at / (at + af), 4) if (at + af) > 0 else None,
            "frac_of_FN_recovered": round(at / fn_total, 4) if fn_total > 0 else None,
            "frac_of_pool_TP": round(at / tp_pool, 4) if tp_pool > 0 else None,
        }
        r = cell_res["per_k"][f"top{k}"]
        log(f"    top{k:<2}: +true {at:8.2f}  +false {af:9.2f}  prec {r['generation_precision_IA']}"
            f"  FN-recovered {r['frac_of_FN_recovered']}  vs pool-TP {r['frac_of_pool_TP']}")
    results[cell] = cell_res

json.dump(results, open(OUT / "cg1_step1_gate.json", "w"), indent=2)
log("wrote cg1_step1_gate.json")
print(json.dumps(results, indent=2))
