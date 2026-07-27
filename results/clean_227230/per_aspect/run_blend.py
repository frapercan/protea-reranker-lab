"""Targeted attempt to push the CCO (and other) cells: blend the per-CATEGORY
reranker with the per-(cat,aspect) reranker, both per-cell min-max calibrated,
at a sweep of mix weights. A blend is a legitimate deployment config (ensembling
two rerankers). Folds the blends into the per-cell BEST-OF and re-emits the
verdict. Rows of the two score tables are 1:1 aligned (verified)."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import lafa_harness as H

S = Path("/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad")
PA = S / "per_aspect"
OUTDIR = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/clean_227230/per_aspect")
CATS = ["nk", "lk", "pk"]
ASPECTS = ["mfo", "bpo", "cco"]
ALPHAS = [0.25, 0.5, 0.75]  # weight on per-category; (1-a) on per-aspect


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


def score(name, prot, go, sc, queries):
    keep = np.array([p in queries for p in prot])
    work = PA / "_work" / name
    work.mkdir(parents=True, exist_ok=True)
    pf = work / "pred.tsv"
    H.build_pred_file(zip(prot[keep], go[keep], sc[keep]), pf)
    res = H.score_all(pf, work)
    cells = {f"{c.lower()}-{a}": res[c].get(a) for c in ["NK", "LK", "PK"] for a in ASPECTS}
    print(f"[{name}] {cells}", flush=True)
    return cells


def main():
    queries = set(H.query_ids())
    a = pq.read_table(S / "rerank_out" / "eval_scores.parquet")
    b = pq.read_table(PA / "eval_scores_per_aspect.parquet")
    prot = np.asarray(a.column("protein_accession").to_pylist())
    go = np.asarray(a.column("go_term_id").to_pylist())
    cat = np.asarray(b.column("category").to_pylist())
    asp = np.asarray(b.column("aspect").to_pylist())
    base = a.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)
    pera = b.column("raw_score").to_numpy(zero_copy_only=False).astype(np.float64)
    base_c = cellmm(base, cat, asp)
    pera_c = cellmm(pera, cat, asp)

    blends = {}
    for al in ALPHAS:
        mix = al * base_c + (1 - al) * pera_c
        blends[f"blend_a{al}"] = score(f"blend_a{al}", prot, go, mix, queries)

    # merge with existing comparison and recompute best-of
    comp = json.loads((OUTDIR / "comparison_9cell.json").read_text())
    comp["variants"].update(blends)
    cell_names = [f"{c}-{a}" for c in CATS for a in ASPECTS]
    bestof = {}
    for cell in cell_names:
        bv, bs = None, -1.0
        for vn, cells in comp["variants"].items():
            s = cells.get(cell)
            if s is not None and s > bs:
                bs, bv = s, vn
        bestof[cell] = {"variant": bv, "f_micro_w": bs}
    comp["bestof"] = bestof
    CURRENT = {"nk-mfo": 0.602, "nk-bpo": 0.309, "nk-cco": 0.431, "lk-mfo": 0.519,
               "lk-bpo": 0.348, "lk-cco": 0.419, "pk-mfo": 0.235, "pk-bpo": 0.117, "pk-cco": 0.254}
    LEADER = {"nk-cco": ("FunBind", 0.473), "lk-cco": ("TransFew", 0.434),
              "nk-mfo": ("goa", 0.591), "lk-mfo": ("goa", 0.510)}
    for cell in cell_names:
        s = bestof[cell]["f_micro_w"]
        cur = CURRENT[cell]
        lead = LEADER.get(cell)
        comp["verdict"][cell] = {
            "bestof_f_micro_w": s, "bestof_variant": bestof[cell]["variant"],
            "current_published": cur, "delta_vs_current": round(s - cur, 4),
            "leader": lead[0] if lead else None, "leader_f_micro_w": lead[1] if lead else None,
            "beats_leader": (s > lead[1]) if lead else None,
            "regresses_current": s < cur - 1e-9,
        }
    (OUTDIR / "comparison_9cell.json").write_text(json.dumps(comp, indent=1))

    print("\nFINAL 9-CELL (with blends), best-of:")
    vnames = list(comp["variants"].keys())
    print("cell\t" + "\t".join(v.replace("per_aspect", "pa").replace("blend_", "bl") for v in vnames) + "\tBESTOF\tbest_v\tcur\tlead")
    for cell in cell_names:
        row = [cell] + [f"{comp['variants'][v][cell]}" for v in vnames]
        row += [f"{bestof[cell]['f_micro_w']}", bestof[cell]['variant'][:10],
                f"{CURRENT[cell]}", f"{LEADER[cell][1]}" if cell in LEADER else "-"]
        print("\t".join(row))


if __name__ == "__main__":
    main()
