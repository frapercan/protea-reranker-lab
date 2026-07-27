"""Tune the base<->InterPro2GO noisy-OR blend weight per (category, aspect) on the
v225-v227 VALIDATION window. No test leakage.

Blend: combined = 1 - (1-base_prob)*(1 - w*ip_prob).  w=0 == graft exactly.
Both base_prob and ip_prob are already in [0,1] (LightGBM binary prob / fraction of
domains), so they are scale-compatible; noisy-OR preserves base calibration for
base-only candidates (adds InterPro, never clobbers).

Validation metric = IA-weighted micro-Fmax with a FIXED truth denominator
(propagated base positives, TOI- and namespace-filtered) -> fair graft vs blend.
"""
import os, json, collections
import numpy as np
import pandas as pd
import interpro_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
SC = L.SC
TOI = os.path.join(SC, "lafa_gt", "groundtruth_terms_of_interest.txt")
IA_PATH = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
ASPECTS = ["mfo", "bpo", "cco"]
WEIGHTS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]

ia = {}
for line in open(IA_PATH):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try:
            ia[p[0]] = float(p[1])
        except ValueError:
            pass
toi = set(l.strip() for l in open(TOI) if l.strip())
nm = L.ns_map()
parents = L._parents()

# InterPro graded preds for validation proteins
vprot2ipr = json.load(open(os.path.join(HERE, "valid_protein2ipr.json")))
vip = L.interpro_preds(vprot2ipr)  # acc -> {go: graded}

vb = pd.read_parquet(os.path.join(HERE, "valid_base.parquet"))


def truth_for(df_cat):
    """protein -> set(true terms) via propagate(label==1), TOI+namespace filtered."""
    pos = df_cat[df_cat["label"] == 1]
    truth = collections.defaultdict(set)
    for acc, sub in pos.groupby("protein_accession"):
        t = set()
        for term in sub["go_term_id"]:
            t.add(term)
            t |= L.ancestors(term, parents)
        truth[acc] = {x for x in t if x in toi and x in nm}
    return truth


def fmax_fixed(cand, truth, total_pos_ia, th_step=0.01):
    """cand: list of (score, ia, is_true). Fixed denom = total_pos_ia."""
    if not cand or total_pos_ia <= 0:
        return 0.0, 0.0
    s = np.array([c[0] for c in cand]); w = np.array([c[1] for c in cand])
    y = np.array([c[2] for c in cand], dtype=bool)
    wt, wf = w * y, w * (~y)
    best_f, best_t = 0.0, 0.0
    for tau in np.arange(th_step, 1.0 + 1e-9, th_step):
        sel = s >= tau
        tp = wt[sel].sum(); fp = wf[sel].sum()
        if tp <= 0:
            continue
        pr = tp / (tp + fp); rc = tp / total_pos_ia
        if pr + rc > 0:
            f = 2 * pr * rc / (pr + rc)
            if f > best_f:
                best_f, best_t = f, tau
    return float(best_f), float(best_t)


results = {}
for cat in ["nk", "lk", "pk"]:
    dfc = vb[vb["category"] == cat]
    truth = truth_for(dfc)
    results[cat] = {}
    for asp in ASPECTS:
        # base candidates this aspect
        sub = dfc[dfc["aspect"] == asp]
        base_map = collections.defaultdict(dict)
        for r in sub.itertuples(index=False):
            base_map[r.protein_accession][r.go_term_id] = max(
                base_map[r.protein_accession].get(r.go_term_id, 0.0), float(r.base_score))
        # truth restricted to this namespace
        asp_key = {"mfo": "mfo", "bpo": "bpo", "cco": "cco"}[asp]
        ns_want = {"mfo": "mfo", "bpo": "bpo", "cco": "cco"}[asp]
        truth_ns = {p: {t for t in ts if nm.get(t) == ns_want} for p, ts in truth.items()}
        total_pos_ia = sum(ia.get(t, 0.0) for ts in truth_ns.values() for t in ts)
        proteins = set(base_map) | {p for p in truth_ns if truth_ns[p]}

        cell_scores = {}
        for w in WEIGHTS:
            cand = []
            for p in proteins:
                bm = base_map.get(p, {})
                ipm = {g: s for g, s in vip.get(p, {}).items() if nm.get(g) == ns_want}
                terms = set(bm) | set(ipm)
                tset = truth_ns.get(p, set())
                for t in terms:
                    b = bm.get(t, 0.0); ip = ipm.get(t, 0.0)
                    comb = 1.0 - (1.0 - b) * (1.0 - w * ip)
                    if comb <= 0:
                        continue
                    cand.append((comb, ia.get(t, 0.0), t in tset))
            f, tau = fmax_fixed(cand, truth_ns, total_pos_ia)
            cell_scores[w] = {"fmax": round(f, 5), "tau": round(tau, 3)}
        graft_f = cell_scores[0.0]["fmax"]
        best_w = max(WEIGHTS, key=lambda w: cell_scores[w]["fmax"])
        results[cat][asp] = {
            "graft_valid_fmax": graft_f,
            "best_w": best_w,
            "blend_valid_fmax": cell_scores[best_w]["fmax"],
            "valid_delta": round(cell_scores[best_w]["fmax"] - graft_f, 5),
            "sweep": cell_scores,
            "total_pos_ia": round(total_pos_ia, 2),
        }
        print(f"[{cat}-{asp}] graft {graft_f:.4f} best_w {best_w} blend "
              f"{cell_scores[best_w]['fmax']:.4f} (+{cell_scores[best_w]['fmax']-graft_f:.4f})")

json.dump(results, open(os.path.join(HERE, "valid_tuning.json"), "w"), indent=2)
# compact chosen-weights map
chosen = {cat: {asp: results[cat][asp]["best_w"] for asp in ASPECTS} for cat in results}
json.dump(chosen, open(os.path.join(HERE, "chosen_weights.json"), "w"), indent=2)
print("chosen weights:", json.dumps(chosen))
print("written valid_tuning.json + chosen_weights.json")
