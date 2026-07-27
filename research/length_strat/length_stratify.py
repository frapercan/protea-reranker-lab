"""Length-stratified BP failure analysis (CHEAP, frozen data, CPU).

Reuses the pool/anchor/propagation machinery of deepgose_measure.py but strips the
cafaeval driver. For LK-BPO and PK-BPO, buckets proteins by residue length
(<=512, 512-1024, 1024-2048, >2048 TRUNCATED) and reports per bucket:
  1. n proteins + share of true BP IA-mass.
  2. per-protein reachable-tail separability AUC: SE vs deployed reranker.
  3. delivered per-protein IA-weighted-F of the deployed system (at the cell's
     micro-optimal threshold) + generation headroom (unreachable FN-tail IA fraction).
The >2048 truncated slice is isolated and compared vs <=2048, controlling for term count.
"""
import json, collections, time
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
EVAL = R / "percut_rerank/eval.parquet"
PREDDIR = R / "percut_rerank/predictions"
GTDIR = R / "lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
OUT = ROOT / "storage/deepgose_kwta"
LSDIR = ROOT / "storage/length_strat"
LENS = json.load(open(LSDIR / "lengths.json"))
BUCKETS = [("<=512", 0, 512), ("512-1024", 512, 1024),
           ("1024-2048", 1024, 2048), (">2048", 2048, 10**9)]
def bucket_of(p):
    L = LENS.get(p)
    if L is None: return None
    for name, lo, hi in BUCKETS:
        if (L > lo or lo == 0) and L <= hi: return name
    return None

# ---- obo ----
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
BP = {t for t, n in ns.items() if n == "biological_process"}
_anc = {}
def anc(t):
    t = alt.get(t, t)
    if t in _anc: return _anc[t]
    seen, st = set(), [t]
    while st:
        x = st.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): st.append(p)
    r = frozenset(x for x in seen if x in BP); _anc[t] = r; return r
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass
def iamass(ts): return float(sum(IA.get(t, 0.0) for t in ts))

results = {}
for cell, cat, gt_fn in [
    ("LK-BPO", "lk", str(GTDIR / "groundtruth_LK.tsv")),
    ("PK-BPO", "pk", str(GTDIR / "groundtruth_PK.tsv"))]:
    log(f"===== {cell} =====")
    Z = np.load(OUT / f"se_scores_{cell.split('-')[0]}.npz", allow_pickle=True)
    prots = Z["proteins"].tolist(); terms = [alt.get(g, g) for g in Z["terms"].tolist()]
    S = Z["avg"]                          # (P, T) SE score, ~350MB for PK
    tpos = {g: j for j, g in enumerate(terms)}
    pidx = {p: i for i, p in enumerate(prots)}

    gt = pd.read_csv(gt_fn, sep="\t"); gt = gt[gt.aspect == "P"].copy()
    gt["term"] = gt.term.map(lambda g: alt.get(g, g))
    gt_leaf = gt.groupby("EntryID").term.apply(set).to_dict()
    targets = [p for p in sorted(gt_leaf) if p in pidx and p in LENS]
    gt_prop = {p: (frozenset().union(*[anc(t) for t in gt_leaf[p]]) if gt_leaf[p] else frozenset())
               for p in targets}

    # pool BP membership + labels from eval.parquet; reranker score from prediction file
    tb = pq.read_table(EVAL, columns=["protein_accession", "go_term_id", "label", "aspect"]).to_pandas()
    tb = tb[(tb.protein_accession.isin(set(targets))) & (tb.aspect == "bpo")]
    tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
    pool_bp = tb.groupby("protein_accession").go.apply(set).to_dict()
    lab_bp = {(r.protein_accession, r.go): int(r.label) for r in tb.itertuples()}
    dep = pd.read_csv(PREDDIR / cat / f"{cat}.tsv", sep="\t", header=None, names=["prot", "term", "score"])
    dep["term"] = dep.term.map(lambda g: alt.get(g, g))
    rr = {(r.prot, r.term): r.score for r in dep.itertuples()}
    pool_prop = {p: (frozenset().union(*[anc(t) for t in pool_bp[p]]) if pool_bp.get(p) else frozenset())
                 for p in targets}
    log(f"  targets {len(targets)}")

    # ---- propagated deployed scores per protein (prop=fill: ancestor = max descendant score) ----
    prop_scores = {}
    for p in targets:
        ps = {}
        for g in pool_bp.get(p, set()):
            s = rr.get((p, g))
            if s is None: continue
            for a in anc(g):
                if s > ps.get(a, -1e9): ps[a] = s
        prop_scores[p] = ps

    # ---- micro-optimal threshold over the cell (reproduces f_micro_w argmax operating point) ----
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    # precompute per-protein sorted (score, ia) for TP/FP and true ia mass
    tpfp = {}
    true_ia = {}
    for p in targets:
        gtp = gt_prop[p]
        true_ia[p] = iamass(gtp)
        rows = []
        for a, s in prop_scores[p].items():
            rows.append((s, IA.get(a, 0.0), a in gtp))
        tpfp[p] = rows
    best_f, best_tau = -1.0, 0.5
    for tau in grid:
        TP = FP = FN_sum = 0.0
        for p in targets:
            pred_true = pred_all = 0.0
            for s, ia, istrue in tpfp[p]:
                if s >= tau:
                    pred_all += ia
                    if istrue: pred_true += ia
            TP += pred_true; FP += (pred_all - pred_true)
            FN_sum += (true_ia[p] - pred_true)
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN_sum) if (TP + FN_sum) > 0 else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f > best_f: best_f, best_tau = f, tau
    log(f"  micro-optimal tau={best_tau} f_micro_w={best_f:.5f}")

    # ---- per-protein delivered F at best_tau + headroom ----
    per_prot = {}
    for p in targets:
        gtp = gt_prop[p]; ti = true_ia[p]
        pt = pa = 0.0
        for s, ia, istrue in tpfp[p]:
            if s >= best_tau:
                pa += ia
                if istrue: pt += ia
        prec = pt / pa if pa > 0 else 0.0
        rec = pt / ti if ti > 0 else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        # generation headroom: true IA never in the pool (unreachable)
        fn_tail = iamass(gtp - pool_prop.get(p, frozenset()))
        per_prot[p] = {"f": f, "prec": prec, "rec": rec, "true_ia": ti,
                       "fn_tail_ia": fn_tail,
                       "unreach_frac": (fn_tail / ti) if ti > 0 else 0.0,
                       "n_true_terms": len(gtp)}

    # ---- separability AUC per protein (reachable tail) ----
    sep = {}
    for p in targets:
        cands = [g for g in pool_bp.get(p, set()) if (p, g) in lab_bp and g in tpos]
        y = np.array([lab_bp[(p, g)] for g in cands])
        if len(y) < 5 or y.sum() == 0 or y.sum() == len(y): continue
        se = np.array([S[pidx[p], tpos[g]] for g in cands])
        rrv = np.array([rr.get((p, g), 0.0) for g in cands])
        sep[p] = {"se_auc": roc_auc_score(y, se), "rr_auc": roc_auc_score(y, rrv)}

    # ---- aggregate per bucket ----
    total_true_ia = sum(per_prot[p]["true_ia"] for p in targets)
    cell_res = {"n_targets": len(targets), "micro_opt_tau": float(best_tau),
                "micro_f": round(best_f, 5), "total_true_ia": round(total_true_ia, 1),
                "n_sep_proteins": len(sep), "buckets": {}}
    for name, lo, hi in BUCKETS:
        bp_list = [p for p in targets if bucket_of(p) == name]
        if not bp_list:
            cell_res["buckets"][name] = {"n": 0}; continue
        bia = sum(per_prot[p]["true_ia"] for p in bp_list)
        fs = [per_prot[p]["f"] for p in bp_list]
        recs = [per_prot[p]["rec"] for p in bp_list]
        precs = [per_prot[p]["prec"] for p in bp_list]
        unreach = [per_prot[p]["unreach_frac"] for p in bp_list]
        nterms = [per_prot[p]["n_true_terms"] for p in bp_list]
        sb = [p for p in bp_list if p in sep]
        se_a = [sep[p]["se_auc"] for p in sb]; rr_a = [sep[p]["rr_auc"] for p in sb]
        cell_res["buckets"][name] = {
            "n": len(bp_list),
            "n_share_pct": round(100 * len(bp_list) / len(targets), 1),
            "true_ia_mass": round(bia, 1),
            "true_ia_share_pct": round(100 * bia / total_true_ia, 1) if total_true_ia > 0 else None,
            "mean_true_terms": round(float(np.mean(nterms)), 1),
            "delivered_f_mean": round(float(np.mean(fs)), 4),
            "delivered_prec_mean": round(float(np.mean(precs)), 4),
            "delivered_rec_mean": round(float(np.mean(recs)), 4),
            "unreachable_fn_frac_mean": round(float(np.mean(unreach)), 4),
            "n_sep": len(sb),
            "se_auc_mean": round(float(np.mean(se_a)), 4) if se_a else None,
            "rr_auc_mean": round(float(np.mean(rr_a)), 4) if rr_a else None,
            "se_minus_rr_auc": round(float(np.mean(np.array(se_a) - np.array(rr_a))), 4) if se_a else None,
        }
        b = cell_res["buckets"][name]
        log(f"  [{name}] n={b['n']} ia%={b['true_ia_share_pct']} f={b['delivered_f_mean']} "
            f"unreach={b['unreachable_fn_frac_mean']} SE-RR={b['se_minus_rr_auc']} (nsep {b['n_sep']})")

    # ---- isolated >2048 truncated slice vs <=2048, controlling for term count ----
    trunc = [p for p in targets if bucket_of(p) == ">2048"]
    rest = [p for p in targets if bucket_of(p) is not None and bucket_of(p) != ">2048"]
    def agg(ps, key): return float(np.mean([per_prot[p][key] for p in ps])) if ps else None
    def agg_sep(ps, key):
        v = [sep[p][key] for p in ps if p in sep]; return float(np.mean(v)) if v else None
    # term-count-matched control: sample <=2048 proteins matched to the truncated term-count distribution
    rng = np.random.default_rng(0)
    tn = np.array([per_prot[p]["n_true_terms"] for p in trunc])
    rest_n = np.array([per_prot[p]["n_true_terms"] for p in rest])
    # nearest-neighbour match on term count (with replacement) to build a matched control set
    matched = []
    for c in tn:
        cand = np.where(np.abs(rest_n - c) <= max(2, 0.1 * c))[0]
        if len(cand) == 0: cand = np.array([int(np.argmin(np.abs(rest_n - c)))])
        matched.append(rest[int(rng.choice(cand))])
    cell_res["truncated_slice"] = {
        "n_trunc": len(trunc),
        "trunc_mean_true_terms": round(float(np.mean(tn)), 1),
        "rest_mean_true_terms": round(float(np.mean(rest_n)), 1),
        "trunc_delivered_f": round(agg(trunc, "f"), 4),
        "leq2048_delivered_f": round(agg(rest, "f"), 4),
        "termcount_matched_delivered_f": round(float(np.mean([per_prot[p]["f"] for p in matched])), 4),
        "trunc_unreach_frac": round(agg(trunc, "unreach_frac"), 4),
        "leq2048_unreach_frac": round(agg(rest, "unreach_frac"), 4),
        "termcount_matched_unreach_frac": round(float(np.mean([per_prot[p]["unreach_frac"] for p in matched])), 4),
        "trunc_rec": round(agg(trunc, "rec"), 4),
        "leq2048_rec": round(agg(rest, "rec"), 4),
        "trunc_se_auc": round(agg_sep(trunc, "se_auc"), 4) if agg_sep(trunc, "se_auc") else None,
        "trunc_rr_auc": round(agg_sep(trunc, "rr_auc"), 4) if agg_sep(trunc, "rr_auc") else None,
        "leq2048_se_auc": round(agg_sep(rest, "se_auc"), 4) if agg_sep(rest, "se_auc") else None,
        "leq2048_rr_auc": round(agg_sep(rest, "rr_auc"), 4) if agg_sep(rest, "rr_auc") else None,
    }
    ts = cell_res["truncated_slice"]
    log(f"  TRUNC>2048 n={ts['n_trunc']} f={ts['trunc_delivered_f']} vs <=2048 f={ts['leq2048_delivered_f']} "
        f"(termcount-matched {ts['termcount_matched_delivered_f']}); unreach {ts['trunc_unreach_frac']} vs {ts['leq2048_unreach_frac']}")
    results[cell] = cell_res
    del S, Z  # free RAM before next cell
    json.dump(results, open(LSDIR / "length_stratification.json", "w"), indent=1)

json.dump(results, open(LSDIR / "length_stratification.json", "w"), indent=1)
log("wrote length_stratification.json")
print(json.dumps(results, indent=1))
