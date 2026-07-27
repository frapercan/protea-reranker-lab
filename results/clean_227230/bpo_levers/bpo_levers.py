"""BPO precision levers (pure offline post-processing) on the clf+assoc reranked
scores, scored with the EXACT LAFA cafaeval harness.

We attack the BPO cells (LK-bpo, PK-bpo) where TransFew leads. Every lever is a
deterministic transform of the reranker_score restricted to BPO rows ONLY. MFO/CCO
rows keep the baseline globalmm score verbatim, so MFO/CCO cells are unchanged by
construction (we still re-score them to prove it). Each lever output is mapped into
(0,1] so the cafaeval tau sweep (np.arange(0.01,1,0.01)) is meaningful.

Levers:
  baseline       : globalmm over ALL rows (current board config)
  rank_norm      : per-protein, BPO scores -> rank/(n) in (0,1]
  pminmax        : per-protein soft min-max of BPO scores (+ temperature variants)
  hsmooth        : DAG hierarchical smoothing s' = a*s + (1-a)*mean(ancestor s)
  condprob_mono  : enforce s(child) <= s(parent) (downward min over DAG)
  condprob_mult  : s'(t) = s(t) * mean(parent s)  (consistency multiply, prob space)
  combo_*        : best stacks

Run:  .venv/bin/python results/clean_227230/bpo_levers/bpo_levers.py
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

LAB = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab")
sys.path.insert(0, str(LAB / "results/clean_227230/clfassoc"))
import lafa_harness as H  # noqa: E402

SCORES = LAB / "results/clean_227230/clfassoc/eval_scores.parquet"
PARENTS = LAB / "results/sparse_classifier/go_parents.json"
OUTDIR = LAB / "results/clean_227230/bpo_levers"
WORK = Path("/tmp/claude-1000/-home-frapercan-Thesis2/"
            "afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad/bpo_levers_work")
WORK.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

ASPECTS = ["mfo", "bpo", "cco"]
CATS = ["nk", "lk", "pk"]

CURRENT = {  # current PROTEA-reranked on the board (f_micro_w)
    "nk-mfo": 0.602, "nk-bpo": 0.309, "nk-cco": 0.431,
    "lk-mfo": 0.519, "lk-bpo": 0.348, "lk-cco": 0.419,
    "pk-mfo": 0.235, "pk-bpo": 0.117, "pk-cco": 0.254,
}
LEADER_BPO = {"lk-bpo": ("TransFew", 0.512), "pk-bpo": ("TransFew", 0.294)}


def globalmm(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn)


def to_unit(x):
    """Map any real vector into (0,1] preserving order; min->small>0, max->1."""
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.full_like(x, 0.5)
    u = (x - mn) / (mx - mn)
    # nudge off exact 0 so the lowest score still survives tau=0.01 if desired
    return np.clip(u, 1e-6, 1.0)


# ---------------------------------------------------------------- ancestors
def build_ancestors(parents, terms):
    """Transitive ancestors (within whatever namespace the parents map covers)
    for each term in `terms`. Returns dict term -> set(ancestors)."""
    cache = {}

    def anc(t):
        if t in cache:
            return cache[t]
        cache[t] = set()  # guard cycles
        acc = set()
        for p in parents.get(t, []):
            acc.add(p)
            acc |= anc(p)
        cache[t] = acc
        return acc

    sys.setrecursionlimit(100000)
    return {t: anc(t) for t in terms}


# ---------------------------------------------------------------- levers
def lever_rank_norm(groups, base):
    out = np.empty(len(base))
    for idx in groups.values():
        s = base[idx]
        order = np.argsort(np.argsort(s))  # ranks 0..n-1
        out[idx] = (order + 1) / len(s)
    return out


def lever_pminmax(groups, base, temp=1.0):
    out = np.empty(len(base))
    for idx in groups.values():
        s = base[idx]
        mn, mx = s.min(), s.max()
        if mx - mn < 1e-12:
            out[idx] = 0.5
        else:
            u = (s - mn) / (mx - mn)
            out[idx] = u ** temp if temp != 1.0 else u
    return np.clip(out, 1e-6, 1.0)


def lever_hsmooth(groups, base, ancestors, alpha=0.7):
    """s'(t) = alpha*s(t) + (1-alpha)*mean(s over predicted ancestors of t)."""
    out = base.copy()
    for idx in groups.values():
        sub_terms = go[idx]
        sub_scores = base[idx]
        smap = {sub_terms[i]: sub_scores[i] for i in range(len(idx))}
        new = np.empty(len(idx))
        for i, t in enumerate(sub_terms):
            anc_here = [smap[a] for a in ancestors.get(t, ()) if a in smap]
            if anc_here:
                new[i] = alpha * sub_scores[i] + (1 - alpha) * float(np.mean(anc_here))
            else:
                new[i] = sub_scores[i]
        out[idx] = new
    return out


def lever_condprob_mono(groups, base, ancestors):
    """Enforce s(child) <= s(parent): cap each term at the min over its predicted
    ancestors (downward consistency)."""
    out = base.copy()
    for idx in groups.values():
        sub_terms = go[idx]
        sub_scores = base[idx]
        smap = {sub_terms[i]: sub_scores[i] for i in range(len(idx))}
        new = sub_scores.copy()
        for i, t in enumerate(sub_terms):
            anc_here = [smap[a] for a in ancestors.get(t, ()) if a in smap]
            if anc_here:
                new[i] = min(sub_scores[i], min(anc_here))
        out[idx] = new
    return out


def lever_condprob_mult(groups, base, parents, beta=1.0):
    """s'(t) = s(t) * (mean predicted-parent score)^beta in [0,1] prob space.
    Roots (no predicted parent) keep their score."""
    out = base.copy()
    for idx in groups.values():
        sub_terms = go[idx]
        sub_scores = base[idx]
        smap = {sub_terms[i]: sub_scores[i] for i in range(len(idx))}
        new = sub_scores.copy()
        for i, t in enumerate(sub_terms):
            par_here = [smap[p] for p in parents.get(t, ()) if p in smap]
            if par_here:
                factor = float(np.mean(par_here)) ** beta
                new[i] = sub_scores[i] * factor
        out[idx] = new
    return out


# ---------------------------------------------------------------- driver
def score_bpo_variant(bpo_scores_unit, tag):
    """Build a full 7401 pred file: baseline globalmm for MFO/CCO, the supplied
    unit-scaled scores for BPO. Score NK/LK/PK and return the 9-cell dict."""
    full = base_all.copy()
    full[bpo_mask] = bpo_scores_unit
    keep = in_query
    rows = list(zip(prot[keep], go[keep], full[keep]))
    pred_file = WORK / f"pred_{tag}.tsv"
    H.build_pred_file(rows, pred_file)
    res = H.score_all(pred_file, WORK / f"w_{tag}")
    cells = {}
    for cat in ["NK", "LK", "PK"]:
        for a in ASPECTS:
            cells[f"{cat.lower()}-{a}"] = res[cat].get(a)
    return cells


if __name__ == "__main__":
    t = pq.read_table(SCORES)
    prot = np.asarray(t.column("protein_accession").to_pylist())
    go = np.asarray(t.column("go_term_id").to_pylist())
    aspect = np.asarray(t.column("aspect").to_pylist())
    rr = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)

    queries = set(H.query_ids())
    in_query = np.array([p in queries for p in prot])
    base_all = globalmm(rr)          # current board config, all rows
    bpo_mask = aspect == "bpo"
    print(f"rows={len(rr)} bpo_rows={bpo_mask.sum()} in_query_rows={in_query.sum()}",
          flush=True)

    # per-protein BPO groups (over the rows we will actually emit: in_query & bpo)
    emit_bpo = bpo_mask & in_query
    bpo_idx = np.where(bpo_mask)[0]
    groups = {}
    for i in bpo_idx:
        if in_query[i]:
            groups.setdefault(prot[i], []).append(i)
    groups = {p: np.array(ix) for p, ix in groups.items()}
    print(f"bpo proteins (in query)={len(groups)}", flush=True)

    parents = json.loads(PARENTS.read_text())
    bpo_terms = set(go[bpo_mask].tolist())
    ancestors = build_ancestors(parents, bpo_terms)
    print(f"ancestors built for {len(ancestors)} bpo terms", flush=True)

    base_bpo = base_all[bpo_mask]    # baseline BPO scores (globalmm), order = bpo_idx

    # map a function over the full-length base_all, write back bpo positions
    def full_from_groups(fn):
        return fn(groups, base_all)

    variants = {}
    t0 = time.time()

    variants["baseline"] = base_all[bpo_mask]
    variants["rank_norm"] = full_from_groups(lever_rank_norm)[bpo_mask]
    variants["pminmax"] = full_from_groups(lever_pminmax)[bpo_mask]
    variants["pminmax_t0.5"] = lever_pminmax(groups, base_all, temp=0.5)[bpo_mask]
    variants["pminmax_t2"] = lever_pminmax(groups, base_all, temp=2.0)[bpo_mask]
    variants["hsmooth_a0.7"] = lever_hsmooth(groups, base_all, ancestors, 0.7)[bpo_mask]
    variants["hsmooth_a0.5"] = lever_hsmooth(groups, base_all, ancestors, 0.5)[bpo_mask]
    variants["hsmooth_a0.9"] = lever_hsmooth(groups, base_all, ancestors, 0.9)[bpo_mask]
    variants["condprob_mono"] = lever_condprob_mono(groups, base_all, ancestors)[bpo_mask]
    variants["condprob_mult"] = lever_condprob_mult(groups, base_all, parents, 1.0)[bpo_mask]
    variants["condprob_mult_b0.5"] = lever_condprob_mult(groups, base_all, parents, 0.5)[bpo_mask]

    # combos (stack on top of each other in transform space, then unit-scale)
    # combo A: hierarchical smoothing then per-protein rank-norm
    hs = lever_hsmooth(groups, base_all, ancestors, 0.7)
    variants["combo_hsmooth_rank"] = lever_rank_norm(groups, hs)[bpo_mask]
    # combo B: smoothing then pminmax
    variants["combo_hsmooth_pmm"] = lever_pminmax(groups, hs, 1.0)[bpo_mask]
    # combo C: mono-consistency then rank-norm
    mo = lever_condprob_mono(groups, base_all, ancestors)
    variants["combo_mono_rank"] = lever_rank_norm(groups, mo)[bpo_mask]
    print(f"levers computed in {time.time()-t0:.1f}s", flush=True)

    # score each variant
    results = {}
    for tag, bpo_vals in variants.items():
        unit = to_unit(bpo_vals) if tag != "baseline" else base_all[bpo_mask]
        # baseline keeps raw globalmm; others unit-scaled within bpo
        ts = time.time()
        cells = score_bpo_variant(unit, tag)
        results[tag] = cells
        bp = {k: cells[k] for k in ("nk-bpo", "lk-bpo", "pk-bpo")}
        print(f"[{tag:22s}] bpo={bp} ({time.time()-ts:.0f}s)", flush=True)

    # build comparison
    base = results["baseline"]
    nonbpo = [f"{c}-{a}" for c in CATS for a in ("mfo", "cco")]
    comparison = {}
    for tag, cells in results.items():
        regress = {}
        for cell in nonbpo:
            b = base.get(cell)
            v = cells.get(cell)
            if b is not None and v is not None and v < b - 1e-9:
                regress[cell] = round(v - b, 4)
        comparison[tag] = {
            "lk-bpo": cells.get("lk-bpo"),
            "pk-bpo": cells.get("pk-bpo"),
            "nk-bpo": cells.get("nk-bpo"),
            "d_lk_bpo": round(cells["lk-bpo"] - base["lk-bpo"], 4)
            if cells.get("lk-bpo") is not None else None,
            "d_pk_bpo": round(cells["pk-bpo"] - base["pk-bpo"], 4)
            if cells.get("pk-bpo") is not None else None,
            "d_nk_bpo": round(cells["nk-bpo"] - base["nk-bpo"], 4)
            if cells.get("nk-bpo") is not None else None,
            "nonbpo_regressions": regress,
            "all_cells": cells,
        }

    out = {
        "baseline_cells": base,
        "leaders_bpo": {k: v[1] for k, v in LEADER_BPO.items()},
        "board_bpo": {"lk-bpo": CURRENT["lk-bpo"], "pk-bpo": CURRENT["pk-bpo"],
                      "nk-bpo": CURRENT["nk-bpo"]},
        "levers": comparison,
    }
    (OUTDIR / "comparison.json").write_text(json.dumps(out, indent=1))
    print("\nwrote", OUTDIR / "comparison.json")
