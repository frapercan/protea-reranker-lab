"""GATE 0 per-evidence separability: AUROC + AP of TRUE-vs-FALSE for EACH single
evidence source ALONE, on the hard PK-BP / LK-BP candidates (and easy/reachable
set for contrast). Honest negatives. No training.
"""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
R = HERE.parent  # results/sparse_classifier
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_TSV = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PK_KNOWN = R / "lafa_gt" / "groundtruth_PK_known.tsv"


def auroc_ap(y, s):
    """AUROC + AP over the rows where score is finite. Returns (auroc, ap, n, npos, cov)."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    m = np.isfinite(s)
    cov = float(m.mean())
    yy, ss = y[m], s[m]
    if yy.sum() == 0 or yy.sum() == len(yy):
        return None, None, int(m.sum()), int(yy.sum()), cov
    return (round(float(roc_auc_score(yy, ss)), 4),
            round(float(average_precision_score(yy, ss)), 4),
            int(m.sum()), int(yy.sum()), round(cov, 4))


def load_obo_ancestors():
    parents = defaultdict(set)
    cur = None
    with open(OBO) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = None
            elif line.startswith("id: GO:"):
                cur = line[4:].strip()
            elif line.startswith("is_a:") and cur:
                parents[cur].add(line.split()[1])
    # depth = longest path to a root (no parents)
    depth = {}

    def d(g):
        if g in depth:
            return depth[g]
        ps = parents.get(g)
        depth[g] = 0 if not ps else 1 + max(d(p) for p in ps)
        return depth[g]

    for g in list(parents.keys()):
        d(g)
    return depth


def build_ppmi(terms_of_interest):
    """Co-annotation PPMI over BP terms from the t0 PK_known corpus."""
    prot_terms = defaultdict(set)
    with open(PK_KNOWN) as f:
        next(f)
        for line in f:
            a, t, asp = line.rstrip("\n").split("\t")
            if asp == "P":
                prot_terms[a].add(t)
    N = len(prot_terms)
    uni = Counter()
    co = defaultdict(Counter)
    for ts in prot_terms.values():
        ts = list(ts)
        for t in ts:
            uni[t] += 1
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i], ts[j]
                co[a][b] += 1
                co[b][a] += 1
    ppmi = defaultdict(dict)
    for a in co:
        pa = uni[a] / N
        for b, c in co[a].items():
            pab = c / N
            pb = uni[b] / N
            val = math.log(pab / (pa * pb + 1e-12) + 1e-12)
            if val > 0:
                ppmi[a][b] = val
    return prot_terms, ppmi


def main():
    hard = pd.read_parquet(HERE / "hard_frame.parquet")
    easy = pd.read_parquet(HERE / "easy_frame.parquet")

    # ---- evidence joins (computed once, applied to both frames) ----
    # literature titles (shipped desc+funcs+titles max-cosine)
    lit = pd.read_parquet(R / "literature_infame" / "text_scores_infame.parquet",
                          columns=["acc", "go", "cos"]).rename(
        columns={"acc": "protein_accession", "go": "go_term_id", "cos": "lit_title"})
    # interpro2go implied GO sets
    p2ipr = json.load(open(R / "interpro2go_test" / "protein2ipr.json"))
    ipr2go = json.load(open(R / "interpro2go_test" / "ipr2go_prop.json"))
    prot_go_count = {}
    for p, iprs in p2ipr.items():
        c = Counter()
        for ipr in iprs:
            for g in ipr2go.get(ipr, []):
                c[g] += 1
        prot_go_count[p] = c
    # IA
    ia = {}
    with open(IA_TSV) as f:
        for line in f:
            g, v = line.rstrip("\n").split("\t")
            ia[g] = float(v)
    depth = load_obo_ancestors()
    # PPMI + term frequency from t0 PK_known
    prot_terms, ppmi = build_ppmi(set())
    uni = Counter()
    for ts in prot_terms.values():
        for t in ts:
            uni[t] += 1
    Nprot = len(prot_terms)

    def annotate(df):
        df = df.merge(lit, on=["protein_accession", "go_term_id"], how="left")
        # interpro membership: count of IPRs implying the candidate GO
        df["interpro_count"] = [
            prot_go_count.get(p, {}).get(g, 0)
            for p, g in zip(df.protein_accession, df.go_term_id)]
        df["interpro_member"] = (df["interpro_count"] > 0).astype(float)
        # IA / depth / frequency priors
        df["ia"] = df.go_term_id.map(ia).astype(float)
        df["depth"] = df.go_term_id.map(depth).astype(float)
        df["freq"] = df.go_term_id.map(lambda g: uni.get(g, 0) / Nprot).astype(float)
        # PPMI: max ppmi(candidate, known term) over the protein's t0 known BP terms
        ppmi_max = []
        for p, g in zip(df.protein_accession, df.go_term_id):
            known = prot_terms.get(p)
            if not known:
                ppmi_max.append(np.nan)
                continue
            row = ppmi.get(g, {})
            best = 0.0
            for k in known:
                v = row.get(k)
                if v and v > best:
                    best = v
            ppmi_max.append(best)
        df["ppmi_max"] = ppmi_max
        # PLM dense cosine = -distance (NaN on clf-only by definition)
        df["plm_cos"] = -df["distance"] if "distance" in df else np.nan
        return df

    hard = annotate(hard)
    easy = annotate(easy)

    EV = {
        "model_reranker": "reranker_score",
        "plm_dense_cosine": "plm_cos",
        "lit_titles": "lit_title",
        "interpro2go_member": "interpro_member",
        "interpro2go_count": "interpro_count",
        "ppmi_coassoc": "ppmi_max",
        "prior_IA": "ia",
        "prior_freq": "freq",
        "prior_depth": "depth",
    }

    def block(df, name):
        out = {}
        for cat in ["pk", "lk"]:
            sub = df[df.category == cat]
            cell = {"rows": int(len(sub)), "pos": int(sub.label.sum()),
                    "pos_rate": round(float(sub.label.mean()), 4),
                    "proteins": int(sub.protein_accession.nunique())}
            for ev, col in EV.items():
                au, ap, n, npos, cov = auroc_ap(sub.label.values, sub[col].values)
                cell[ev] = {"auroc": au, "ap": ap, "cov": cov, "n_scored": n}
            out[cat] = cell
        return out

    sep = {"hard": block(hard, "hard"), "easy": block(easy, "easy")}

    # ---- titles vs abstracts on the abstract subset ----
    abs_df = pd.read_parquet(HERE / "lit_abstract_scores.parquet").rename(
        columns={"acc": "protein_accession", "go": "go_term_id"})
    hlab = hard[["protein_accession", "go_term_id", "label", "category"]].drop_duplicates()
    ab = abs_df.merge(hlab, on=["protein_accession", "go_term_id"], how="inner")
    lit_abs = {}
    for cat in ["pk", "lk", "all"]:
        s = ab if cat == "all" else ab[ab.category == cat]
        if len(s) == 0:
            continue
        at, _, nt, npt, _ = auroc_ap(s.label.values, s.cos_title.values)
        aa, _, na, npa, _ = auroc_ap(s.label.values, s.cos_abstract.values)
        lit_abs[cat] = {
            "n_pairs": int(len(s)), "n_pos": int(s.label.sum()),
            "proteins": int(s.protein_accession.nunique()),
            "auroc_titles": at, "auroc_abstracts": aa,
            "auroc_delta": (round(aa - at, 4) if (aa is not None and at is not None) else None),
        }
    sep["lit_titles_vs_abstracts"] = lit_abs

    json.dump(sep, open(HERE / "separability.json", "w"), indent=2)

    # ---- coverage matrix ----
    cov = {}
    for frame_name, df in [("hard", hard), ("easy", easy)]:
        cov[frame_name] = {}
        for cat in ["pk", "lk"]:
            sub = df[df.category == cat]
            cov[frame_name][cat] = {
                ev: round(float(np.isfinite(sub[col].astype(float)).mean()), 4)
                for ev, col in EV.items()}
            cov[frame_name][cat]["proteins"] = int(sub.protein_accession.nunique())
            cov[frame_name][cat]["proteins_with_ppmi"] = int(
                sub[sub.ppmi_max.notna()].protein_accession.nunique())
            cov[frame_name][cat]["proteins_with_interpro"] = int(
                sub[sub.interpro_count > 0].protein_accession.nunique())
    abm = json.load(open(HERE / "abstracts_fetch_meta.json"))
    asm = json.load(open(HERE / "abstract_score_meta.json"))
    cov["abstracts"] = {**abm, **asm}
    json.dump(cov, open(HERE / "coverage_matrix.json", "w"), indent=2)

    print(json.dumps(sep, indent=2))


if __name__ == "__main__":
    main()
