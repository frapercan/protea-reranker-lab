"""CG.3 STEP 1 -- the GATE (network channel).

Does a pre-t0 STRING v12.0 network GENERATE the true BP-tail terms the sequence-only
pool misses, at a precision clearing the ~1% co-occurrence bar / approaching the 11.59%
classifier bar, where four prior channels failed?

Generator: for target protein p with STRING partners q (edges from experimental+coexpression
channels only in the HEADLINE arm; +database in a flagged ABLATION), each partner contributes
its FROZEN t0 BP annotations B(q) (from reference_annotations.parquet = v227 = t0):
    s(t | p) = sum over partners q of  w(p,q) * 1[t in B(q)]
w(p,q) = channel-restricted STRING combined probability (noisy-OR of the retained channels).

Metric machinery (obo BP ancestors, IA, gt/pool propagation, added-true vs added-false IA-mass,
generation precision) is REUSED VERBATIM from cg1_step1_gate.py. Only the proposal source changes:
partners' B(q) instead of the protein's own PPMI(seed->BP).

Leakage: edges STRING v12.0 release 2023-07-26 << t0 (2025-09-04). textmining channel NEVER read
(headline = exp+coexp; database only in the flagged ablation). Partner annotations ONLY from the
frozen v227=t0 corpus. Self / same-accession partners excluded.
"""
import json, collections, time, gzip
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
F = ROOT / "storage/protea-frozen-v227-2025-09-04"
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
EVAL = R / "percut_rerank/eval.parquet"
GTDIR = R / "lafa_gt"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
STR = ROOT / "storage/string_v12"
ACC_TAX = ROOT / "storage/regen_headline/cg3_acc_taxon.tsv"
OUT = ROOT / "storage/regen_headline"
TOPKS = [5, 10, 25, 50]
EDGE_CUT = 0.40           # channel-restricted combined prob >= 0.40 (STRING "medium" == 400)


def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


# ---------- obo: BP ancestors (is_a + part_of), alt_id map, namespace  [VERBATIM CG.1] ----------
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

# ---------- frozen reference annotations -> partner BP sets (frozen v227=t0) ----------
meta = pq.read_table(F / "go_term_metadata.parquet").to_pandas()
id2go = dict(zip(meta.go_term_id, meta.go_id))
id2asp = dict(zip(meta.go_term_id, meta.aspect))
ref = pq.read_table(F / "reference_annotations.parquet", columns=["accession", "go_term_id"]).to_pandas()
ref["go"] = ref.go_term_id.map(id2go).map(lambda g: alt.get(g, g) if isinstance(g, str) else g)
ref["asp"] = ref.go_term_id.map(id2asp)
ref = ref.dropna(subset=["go"])
refP = ref[ref.asp == "P"]
acc2bpraw = refP.groupby("accession").go.apply(lambda s: frozenset(s)).to_dict()
log(f"reference_annotations: {len(ref):,} rows; {len(acc2bpraw):,} proteins with >=1 frozen BP term")

_bpprop = {}
def bp_prop(acc):
    """propagated (ancestors-or-self) frozen t0 BP set of accession acc; memoized."""
    r = _bpprop.get(acc)
    if r is None:
        raw = acc2bpraw.get(acc)
        r = frozenset().union(*[anc(t) for t in raw]) if raw else frozenset()
        _bpprop[acc] = r
    return r

# ---------- pool from eval.parquet  [VERBATIM CG.1] ----------
def load_pool(target_prots):
    tb = pq.read_table(EVAL, columns=["protein_accession", "go_term_id"]).to_pandas()
    tb = tb[tb.protein_accession.isin(target_prots)]
    tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
    tb = tb[tb.go.isin(BP)]
    return tb.groupby("protein_accession").go.apply(set).to_dict()

# ---------- target -> taxon ----------
acc_tax = pd.read_csv(ACC_TAX, sep="\t", header=None, names=["accession", "taxon"], dtype=str)
tax_of = dict(zip(acc_tax.accession, acc_tax.taxon))
DL_TAXA = [p.name.split(".")[0][3:] for p in STR.glob("tax*.protein.links.detailed.v12.0.txt.gz")]
DL_TAXA = sorted(set(DL_TAXA))
log(f"downloaded STRING taxa: {len(DL_TAXA)} -> {DL_TAXA}")


def load_aliases(taxon):
    """string2uni: string_id -> set(UniProt AC); uni2string: UniProt AC -> set(string_id)."""
    fn = STR / f"tax{taxon}.protein.aliases.v12.0.txt.gz"
    s2u = collections.defaultdict(set); u2s = collections.defaultdict(set)
    with gzip.open(fn, "rt") as fh:
        for ln in fh:
            if ln.startswith("#"): continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3 or p[2] != "UniProt_AC": continue
            s2u[p[0]].add(p[1]); u2s[p[1]].add(p[0])
    return s2u, u2s


def collect_edges(taxon, target_sids):
    """Stream links.detailed once; for edges with protein1 in target_sids, keep raw exp/coexp/db.
    Returns dict sid -> list[(partner_sid, coexp, exp, db)]."""
    fn = STR / f"tax{taxon}.protein.links.detailed.v12.0.txt.gz"
    out = collections.defaultdict(list)
    with gzip.open(fn, "rt") as fh:
        fh.readline()  # header
        for ln in fh:
            i = ln.find(" ")
            a = ln[:i]
            if a not in target_sids:
                continue
            # protein1 protein2 neighborhood fusion cooccurence coexpression experimental database textmining combined
            c = ln.split(" ")
            b = c[1]; cx = int(c[5]); ex = int(c[6]); db = int(c[7])
            if ex == 0 and cx == 0 and db == 0:
                continue
            out[a].append((b, cx, ex, db))
    return out


def noisy_or(*probs):
    r = 1.0
    for p in probs:
        r *= (1.0 - p)
    return 1.0 - r


# ============================ per-cell evaluation ============================
ARMS = ["clean", "clean_db"]           # headline = clean (exp+coexp); ablation = clean_db (+database)
results = {}
coverage = {}
strat = {}

for cell, gt_fn in [("LK-BPO", "groundtruth_LK.tsv"), ("PK-BPO", "groundtruth_PK.tsv")]:
    log(f"===== {cell} =====")
    gt = pd.read_csv(GTDIR / gt_fn, sep="\t")
    gt = gt[gt.aspect == "P"].copy()
    gt["term"] = gt.term.map(lambda g: alt.get(g, g))
    gt_leaf = gt.groupby("EntryID").term.apply(set).to_dict()
    targets = sorted(gt_leaf)
    target_set = set(targets)
    pool = load_pool(target_set)
    log(f"  {cell}: {len(targets)} targets; pool present for {len(target_set & set(pool))}")

    # propagate gt + pool; ceiling accounting  [VERBATIM CG.1]
    gt_prop = {p: frozenset().union(*[anc(t) for t in ts]) if ts else frozenset()
               for p, ts in gt_leaf.items()}
    pool_prop = {p: (frozenset().union(*[anc(t) for t in pool[p]]) if pool.get(p) else frozenset())
                 for p in targets}
    tp_pool = sum(iamass(gt_prop[p] & pool_prop[p]) for p in targets)
    fn_total = sum(iamass(gt_prop[p] - pool_prop[p]) for p in targets)
    gt_total = sum(iamass(gt_prop[p]) for p in targets)
    log(f"  gt IA {gt_total:.1f} | pool-captured {tp_pool:.1f} ({100*tp_pool/gt_total:.1f}%) | "
        f"FN {fn_total:.1f} ({100*fn_total/gt_total:.1f}%)")

    # ---- build per-target network scores (both arms) ----
    # score[arm][p] = dict term -> summed edge weight over partners annotated with term
    score = {a: {} for a in ARMS}
    hubdeg = {}          # target -> number of qualifying partners (clean arm)
    tax_targets = collections.defaultdict(list)
    for p in targets:
        tax_targets[tax_of.get(p)].append(p)

    n_cov = {a: 0 for a in ARMS}      # targets with >=1 proposal outside pool
    for taxon in DL_TAXA:
        tps = [p for p in tax_targets.get(taxon, []) if p in target_set]
        if not tps:
            continue
        s2u, u2s = load_aliases(taxon)
        # target accession -> its string ids
        sid_of_target = {p: (u2s.get(p, set())) for p in tps}
        target_sids = set().union(*sid_of_target.values()) if sid_of_target else set()
        if not target_sids:
            continue
        edges = collect_edges(taxon, target_sids)
        # map each target sid back to target accession(s)
        sid2tacc = collections.defaultdict(set)
        for p in tps:
            for sid in sid_of_target[p]:
                sid2tacc[sid].add(p)
        for sid, elist in edges.items():
            taccs = sid2tacc.get(sid)
            if not taccs:
                continue
            # aggregate partner contributions for this target sid
            for tacc in taccs:
                own_sids = sid_of_target[tacc]
                sc = {a: score[a].setdefault(tacc, collections.defaultdict(float)) for a in ARMS}
                deg = 0
                for (b, cx, ex, db) in elist:
                    if b in own_sids:            # self / same-protein edge
                        continue
                    # partner accession(s) -> union frozen BP set (exclude self accession)
                    Bb = frozenset()
                    for pacc in s2u.get(b, ()):
                        if pacc == tacc:
                            continue
                        Bb = Bb | bp_prop(pacc)
                    if not Bb:
                        continue
                    w_clean = noisy_or(ex / 1000.0, cx / 1000.0)
                    w_cd = noisy_or(ex / 1000.0, cx / 1000.0, db / 1000.0)
                    if w_clean >= EDGE_CUT:
                        deg += 1
                        for t in Bb:
                            sc["clean"][t] += w_clean
                    if w_cd >= EDGE_CUT:
                        for t in Bb:
                            sc["clean_db"][t] += w_cd
                hubdeg[tacc] = hubdeg.get(tacc, 0) + deg
    log(f"  network scores built; targets with clean score: {sum(1 for p in targets if score['clean'].get(p))}")

    # ---- added true/false IA-mass accounting  [VERBATIM CG.1 logic] ----
    KMAX = max(TOPKS)
    cell_res = {
        "target_proteins": len(targets),
        "gt_true_IA_mass": round(gt_total, 2),
        "pool_captured_IA_mass": round(tp_pool, 2),
        "pool_captured_frac": round(tp_pool / gt_total, 4),
        "FN_IA_mass_pool_misses": round(fn_total, 2),
        "FN_frac_of_gt": round(fn_total / gt_total, 4),
        "generator_ceiling_true_mass_reachable": round(fn_total, 2),
        "arms": {},
    }
    # per-protein top5 clean precision for stratification
    per_prot_top5 = {}
    for arm in ARMS:
        added_true = {k: 0.0 for k in TOPKS}
        added_false = {k: 0.0 for k in TOPKS}
        n_prop = {k: 0 for k in TOPKS}
        n_targets_scored = 0
        for p in targets:
            sd = score[arm].get(p)
            if not sd:
                continue
            n_targets_scored += 1
            present = pool.get(p, set())
            gtp = gt_prop[p]
            covered = set(pool_prop[p])
            terms_sorted = sorted(sd.items(), key=lambda kv: -kv[1])
            picks = 0
            p_true5 = 0.0; p_false5 = 0.0
            for g, sv in terms_sorted:
                if sv <= 0:
                    break
                if g in present:
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
                if picks <= 5:
                    p_true5 += tm; p_false5 += fm
                if picks >= KMAX:
                    break
            if arm == "clean":
                per_prot_top5[p] = (p_true5, p_false5)
        if arm == "clean":
            for p in targets:
                if score["clean"].get(p):
                    n_cov["clean"] += 1
        arm_res = {"targets_scored": n_targets_scored, "per_k": {}}
        for k in TOPKS:
            at, af = added_true[k], added_false[k]
            arm_res["per_k"][f"top{k}"] = {
                "proposals": n_prop[k],
                "true_IA_mass_added": round(at, 3),
                "false_IA_mass_added": round(af, 3),
                "generation_precision_IA": round(at / (at + af), 4) if (at + af) > 0 else None,
                "frac_of_FN_recovered": round(at / fn_total, 4) if fn_total > 0 else None,
                "frac_of_pool_TP": round(at / tp_pool, 4) if tp_pool > 0 else None,
            }
            r = arm_res["per_k"][f"top{k}"]
            log(f"    [{arm:8}] top{k:<2}: +true {at:8.2f} +false {af:10.2f} "
                f"prec {r['generation_precision_IA']} FNrec {r['frac_of_FN_recovered']}")
        cell_res["arms"][arm] = arm_res

    # ---- stratification (clean arm, top5) ----
    # by taxon
    by_tax = collections.defaultdict(lambda: [0.0, 0.0, 0])
    by_hub = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for p in targets:
        if p not in per_prot_top5:
            continue
        tr, fa = per_prot_top5[p]
        tx = tax_of.get(p, "NA")
        by_tax[tx][0] += tr; by_tax[tx][1] += fa; by_tax[tx][2] += 1
        d = hubdeg.get(p, 0)
        band = "0" if d == 0 else "1-10" if d <= 10 else "11-50" if d <= 50 else "51-200" if d <= 200 else "200+"
        by_hub[band][0] += tr; by_hub[band][1] += fa; by_hub[band][2] += 1
    strat[cell] = {
        "by_taxon_top5_clean": {tx: {"true": round(v[0], 2), "false": round(v[1], 2),
                                     "prec": round(v[0] / (v[0] + v[1]), 4) if (v[0] + v[1]) > 0 else None,
                                     "n": v[2]} for tx, v in sorted(by_tax.items(), key=lambda x: -x[1][2])},
        "by_hubdegree_top5_clean": {b: {"true": round(v[0], 2), "false": round(v[1], 2),
                                        "prec": round(v[0] / (v[0] + v[1]), 4) if (v[0] + v[1]) > 0 else None,
                                        "n": v[2]} for b, v in by_hub.items()},
    }
    coverage[cell] = {
        "targets": len(targets),
        "targets_with_clean_network_proposal": n_cov["clean"],
        "coverage_frac": round(n_cov["clean"] / len(targets), 4),
        "note": "only 18 tier-1 taxa downloaded; tail taxa contribute 0 proposals here",
    }
    results[cell] = cell_res

out = {"slice": "CG.3", "edge_cut_combined_prob": EDGE_CUT, "arms": ARMS,
       "results": results, "coverage": coverage, "stratification": strat,
       "bars": {"cooccurrence_CG1_top5": {"LK-BPO": 0.0161, "PK-BPO": 0.0103},
                "literature_CG2_top5": {"LK-BPO": 0.0188, "PK-BPO": 0.0149},
                "classifier": 0.1159}}
json.dump(out, open(OUT / "cg3_step1_gate.json", "w"), indent=2)
log("wrote cg3_step1_gate.json")
print(json.dumps(out, indent=2))
