"""Build and LAFA-score the per-aspect reranker variants vs the per-category
baseline, on the 7401 query set. Produces the 9-cell f_micro_w table per variant,
a per-cell BEST-OF routing config, and the leaderboard comparison.

Variants (all restrict to 7401 queries; KNN fallback for the ~54 not in eval):
  baseline   : per-CATEGORY reranker_score, global min-max (the published config)
  per_aspect_globalmm : per-(cat,aspect) routed raw, ONE global min-max
  per_aspect_cellmm   : per-(cat,aspect) routed raw, min-max WITHIN each (cat,aspect)
  per_aspect_cellrank : per-(cat,aspect) routed raw, rank-to-(0,1] within each cell
The cell-local calibrations (cellmm/cellrank) make each (category,namespace)
score range fill (0,1], so cafaeval's pooled per-namespace threshold sweep is
well resolved -- the hypothesised CCO win.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import lafa_harness as H

S = Path("/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad")
PA = S / "per_aspect"
BASE_SCORES = S / "rerank_out" / "eval_scores.parquet"        # per-category
PA_SCORES = PA / "eval_scores_per_aspect.parquet"             # per-(cat,aspect) routed raw
OUTDIR = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/clean_227230/per_aspect")
OUTDIR.mkdir(parents=True, exist_ok=True)

CATS = ["nk", "lk", "pk"]
ASPECTS = ["mfo", "bpo", "cco"]
LEADER = {  # leaderboard #1 to beat (f_micro_w), where known
    "nk-cco": ("FunBind", 0.473), "lk-cco": ("TransFew", 0.434),
    "nk-mfo": ("goa", 0.591), "lk-mfo": ("goa", 0.510),
}
# current published per-category numbers (the headline to not regress)
CURRENT = {
    "nk-mfo": 0.602, "nk-bpo": 0.309, "nk-cco": 0.431,
    "lk-mfo": 0.519, "lk-bpo": 0.348, "lk-cco": 0.419,
    "pk-mfo": 0.235, "pk-bpo": 0.117, "pk-cco": 0.254,
}


def globalmm(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn)


def cellmm(x, cat, asp):
    out = np.empty_like(x)
    for c in CATS:
        for a in ASPECTS:
            m = (cat == c) & (asp == a)
            if not m.any():
                continue
            v = x[m]
            mn, mx = v.min(), v.max()
            out[m] = (v - mn) / (mx - mn) if mx > mn else 0.5
    return out


def cellrank(x, cat, asp):
    out = np.empty_like(x)
    for c in CATS:
        for a in ASPECTS:
            m = (cat == c) & (asp == a)
            if not m.any():
                continue
            v = x[m]
            order = np.argsort(np.argsort(v))          # 0..n-1 ranks
            out[m] = (order + 1) / len(v)              # (0,1]
    return out


def score_variant(name, prot, go, score, queries):
    keep = np.array([p in queries for p in prot])
    rows = zip(prot[keep], go[keep], score[keep])
    work = PA / "_work" / name
    work.mkdir(parents=True, exist_ok=True)
    pred_file = work / "pred.tsv"
    ncov, nfb = H.build_pred_file(rows, pred_file)
    res = H.score_all(pred_file, work)
    # flatten to cell dict
    cells = {}
    for cat in ["NK", "LK", "PK"]:
        for a in ["mfo", "bpo", "cco"]:
            cells[f"{cat.lower()}-{a}"] = res[cat].get(a)
    print(f"[{name}] covered={ncov} fb={nfb} -> {cells}", flush=True)
    return cells


def main():
    queries = set(H.query_ids())

    # baseline (per-category)
    tb = pq.read_table(BASE_SCORES)
    bp = np.asarray(tb.column("protein_accession").to_pylist())
    bg = np.asarray(tb.column("go_term_id").to_pylist())
    brr = tb.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)
    base_cells = score_variant("baseline", bp, bg, globalmm(brr), queries)

    # per-aspect routed raw
    tp = pq.read_table(PA_SCORES)
    pp = np.asarray(tp.column("protein_accession").to_pylist())
    pg = np.asarray(tp.column("go_term_id").to_pylist())
    pcat = np.asarray(tp.column("category").to_pylist())
    pasp = np.asarray(tp.column("aspect").to_pylist())
    praw = tp.column("raw_score").to_numpy(zero_copy_only=False).astype(np.float64)

    variants = {"baseline": base_cells}
    variants["per_aspect_globalmm"] = score_variant(
        "per_aspect_globalmm", pp, pg, globalmm(praw), queries)
    variants["per_aspect_cellmm"] = score_variant(
        "per_aspect_cellmm", pp, pg, cellmm(praw, pcat, pasp), queries)
    variants["per_aspect_cellrank"] = score_variant(
        "per_aspect_cellrank", pp, pg, cellrank(praw, pcat, pasp), queries)

    # per-cell best-of across all variants
    cell_names = [f"{c}-{a}" for c in CATS for a in ASPECTS]
    bestof = {}
    for cell in cell_names:
        best_v, best_s = None, -1.0
        for vname, cells in variants.items():
            s = cells.get(cell)
            if s is not None and s > best_s:
                best_s, best_v = s, vname
        bestof[cell] = {"variant": best_v, "f_micro_w": best_s}

    # leaderboard verdict
    verdict = {}
    for cell in cell_names:
        s = bestof[cell]["f_micro_w"]
        cur = CURRENT.get(cell)
        lead = LEADER.get(cell)
        verdict[cell] = {
            "bestof_f_micro_w": s,
            "bestof_variant": bestof[cell]["variant"],
            "current_published": cur,
            "delta_vs_current": round(s - cur, 4) if cur is not None else None,
            "leader": lead[0] if lead else None,
            "leader_f_micro_w": lead[1] if lead else None,
            "beats_leader": (s > lead[1]) if lead else None,
            "regresses_current": (s < cur - 1e-9) if cur is not None else None,
        }

    out = {"variants": variants, "bestof": bestof, "verdict": verdict,
           "leaderboard": {k: {"who": v[0], "f": v[1]} for k, v in LEADER.items()}}
    (OUTDIR / "comparison_9cell.json").write_text(json.dumps(out, indent=1))
    print("\nWROTE", OUTDIR / "comparison_9cell.json")

    # print table
    print("\n9-CELL f_micro_w (rows=cell):")
    hdr = ["cell", "baseline", "pa_globalmm", "pa_cellmm", "pa_cellrank", "BESTOF", "best_v", "current", "leader"]
    print("\t".join(hdr))
    for cell in cell_names:
        row = [cell,
               f"{variants['baseline'][cell]}",
               f"{variants['per_aspect_globalmm'][cell]}",
               f"{variants['per_aspect_cellmm'][cell]}",
               f"{variants['per_aspect_cellrank'][cell]}",
               f"{bestof[cell]['f_micro_w']}",
               bestof[cell]['variant'].replace('per_aspect_', 'pa_'),
               f"{CURRENT.get(cell)}",
               f"{LEADER[cell][1]}" if cell in LEADER else "-"]
        print("\t".join(row))


if __name__ == "__main__":
    main()
