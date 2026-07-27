"""Merge STRING v12.0 network features (mouse+rat) into the BP failure atlas and
stratify headroom + delivered-f by network DEGREE and ANNOTATED-PARTNER count.

Only mouse (10090) and rat (10116) were downloaded, so this axis is measurable for the
~36% of BP targets in those two interactomes. Human (9606) STRING was NOT downloaded; the
network axis says nothing about human targets. Clean channels only (experimental +
coexpression), threshold >0.
"""
import json, numpy as np
from pathlib import Path

W = Path("/home/frapercan/Thesis2/storage/regen_headline")
feat = json.load(open(W / "bp_string_features.json"))
out = {}

for cell in ["lk", "pk"]:
    recs = json.load(open(W / f"bp_atlas_perprotein_{cell}.json"))
    hr = np.array([r["headroom"] for r in recs])
    df = np.array([r["delivered_f"] for r in recs])
    pr = np.array([r["pool_recall_ia"] for r in recs])
    reach = np.array([r["reachable"] for r in recs])
    tax = np.array([r["taxid"] for r in recs], dtype=object)
    deg = np.array([feat.get(r["protein"], {}).get("degree_clean", -1) for r in recs], float)
    annp = np.array([feat.get(r["protein"], {}).get("annotated_partner_count", -1) for r in recs], float)
    covered = (deg >= 0) & np.isin(tax, ["10090", "10116"])
    ncov = int(covered.sum())
    res = {"n_in_mouse_rat": int(np.isin(tax, ["10090", "10116"]).sum()),
           "n_with_string_feature": ncov,
           "note": "mouse+rat only; human STRING not downloaded"}

    def cut(name, v, edges, labels):
        rows = []
        sel = covered & reach
        b = np.digitize(v, edges)
        tot = hr[sel].sum()
        for k, lab in enumerate(labels):
            s = (b == k) & sel
            if s.sum() == 0:
                continue
            rows.append({"band": lab, "n": int(s.sum()),
                         "headroom_per_protein_x1e4": round(float(hr[s].mean()) * 1e4, 3),
                         "headroom_share": round(float(hr[s].sum() / tot), 4) if tot else None,
                         "mean_delivered_f": round(float(df[s].mean()), 4),
                         "mean_pool_recall": round(float(pr[s].mean()), 4)})
        res[name] = rows

    cut("degree_clean", deg, [0, 1, 100, 300, 600, 1000, 2000],
        ["0", "1-99", "100-299", "300-599", "600-999", "1000-1999", ">=2000"])
    cut("annotated_partner_count", annp, [0, 1, 50, 150, 350, 700, 1500],
        ["0", "1-49", "50-149", "150-349", "350-699", "700-1499", ">=1500"])
    # correlations on covered+reachable
    sel = covered & reach
    if sel.sum() > 20:
        from scipy.stats import spearmanr
        res["spearman_degree_vs_delivered_f"] = round(float(spearmanr(deg[sel], df[sel]).statistic), 3)
        res["spearman_annpartner_vs_delivered_f"] = round(float(spearmanr(annp[sel], df[sel]).statistic), 3)
        res["spearman_degree_vs_headroom"] = round(float(spearmanr(deg[sel], hr[sel]).statistic), 3)
        res["frac_zero_annotated_partners"] = round(float((annp[sel] == 0).mean()), 4)
        res["median_degree"] = round(float(np.median(deg[sel])), 1)
    out[cell + "-bpo"] = res

json.dump(out, open(W / "bp_string_cut.json", "w"), indent=1)
print(json.dumps(out, indent=1))
