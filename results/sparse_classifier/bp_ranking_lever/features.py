"""Term-space ranking features, computed in GO-string space.

Given a frame with columns [protein_accession, go_term_id, base_score] (all aspects
present so the per-protein functional profile uses cross-aspect context), produce
BP-only term features:
  profile_fit     : cosine(code_t, score-weighted profile of the protein's candidates)
  topk_coherence  : mean cosine(code_t, codes of protein's top-k base-score terms)
  depth           : DAG depth (longest path to root)
  ia              : information accretion weight
"""
import numpy as np
import pandas as pd


def build_term_features(df, ns_map, code_idx, codes, depth, ia, topk=10):
    """df: protein_accession, go_term_id, base_score. Returns df of BP rows + features."""
    df = df.copy()
    df["row"] = np.arange(len(df))
    df["aspect"] = df["go_term_id"].map(ns_map)
    df["cidx"] = df["go_term_id"].map(code_idx)  # may be NaN if term absent
    has_code = df["cidx"].notna().values
    cidx = np.where(has_code, df["cidx"].fillna(0).astype(int).values, -1)

    n = len(df)
    profile_fit = np.zeros(n, dtype=np.float32)
    topk_coh = np.zeros(n, dtype=np.float32)

    w = np.clip(df["base_score"].values.astype(np.float32), 0.0, None)

    for prot, sub in df.groupby("protein_accession", sort=False):
        rows = sub["row"].values
        ci = cidx[rows]
        valid = ci >= 0
        if valid.sum() == 0:
            continue
        vrows = rows[valid]
        vci = ci[valid]
        vw = w[vrows]
        vcodes = codes[vci]  # (m,1024) unit-normed

        # score-weighted profile over this protein's candidate terms
        sw = vw.sum()
        if sw <= 0:
            prof = vcodes.mean(axis=0)
        else:
            prof = (vcodes * vw[:, None]).sum(axis=0) / sw
        pn = np.linalg.norm(prof)
        if pn > 0:
            prof = prof / pn
        pf = vcodes @ prof  # cosine of each term to profile
        profile_fit[vrows] = pf

        # top-k highest base-score terms (exclude self via leave-one-out)
        order = np.argsort(-vw)
        kk = min(topk, len(order))
        top_idx = order[:kk]
        top_codes = vcodes[top_idx]  # (kk,1024)
        sims = vcodes @ top_codes.T  # (m,kk)
        # exclude self-match: for term that is in top set, drop its own column
        # approximate leave-one-out: subtract self when present
        coh = np.zeros(len(vci), dtype=np.float32)
        for j in range(len(vci)):
            s = sims[j]
            # remove a self (cos==1) if this term is among top set
            if j in top_idx:
                mask = np.ones(kk, dtype=bool)
                # find position of j in top_idx
                pos = np.where(top_idx == j)[0]
                if len(pos):
                    mask[pos[0]] = False
                vals = s[mask]
            else:
                vals = s
            coh[j] = vals.mean() if len(vals) else 0.0
        topk_coh[vrows] = coh

    df["profile_fit"] = profile_fit
    df["topk_coherence"] = topk_coh
    df["depth"] = df["go_term_id"].map(depth).fillna(0).astype(np.float32)
    df["ia"] = df["go_term_id"].map(ia).fillna(0.0).astype(np.float32)
    df["has_code"] = has_code
    bp = df[df["aspect"] == "biological_process"].copy()
    return bp
