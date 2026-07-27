"""Targeted blend: apply the text noisy-OR boost ONLY to each protein's top-k
text candidates (reranking-style, not whole-pool flooding). Sweep k and w;
keep board-faithful; report best non-regressive cell."""
import os
import json
import collections
import pandas as pd
from score_blend import load_base, score_cat, GT  # reuse faithful harness

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("TAG", "spubmed")


def build_rows_topk(bmap, text_top, proteins, w):
    rows = []
    for p in proteins:
        bm = bmap.get(p, {})
        tm = text_top.get(p, {})
        terms = set(bm) | set(tm)
        for t in terms:
            b = bm.get(t, 0.0)
            ts = tm.get(t)
            comb = 1.0 - (1.0 - b) * (1.0 - w * ts) if (ts is not None and w > 0) else b
            if comb > 0:
                rows.append((p, t, comb))
    return rows


def main():
    bmap = load_base()
    route = json.load(open(os.path.join(HERE, "bp_route.json")))
    ts = pd.read_parquet(os.path.join(HERE, f"text_scores_bpo_{TAG}_precut.parquet"))
    prots = {c: [p for p, cc in route.items() if cc == c] for c in ("nk", "lk")}

    out = {"method": "noisy-OR text boost restricted to top-k text candidates/protein",
           "results": {}}
    for cat in ("nk", "lk"):
        ps = set(prots[cat])
        sub = ts[ts.acc.isin(ps)]
        base_bp = score_cat(
            [(p, g, bmap[p][g]) for p in prots[cat] for g in bmap.get(p, {})],
            cat)["bpo"]["f_micro_w"]
        best = {"k": 0, "w": 0.0, "bpo": base_bp, "delta": 0.0}
        grid = {}
        for k in (1, 2, 3, 5):
            top = collections.defaultdict(dict)
            for p, g in sub.groupby("acc"):
                gg = g.nlargest(k, "text_score")
                for r in gg.itertuples(index=False):
                    top[p][r.go] = float(r.text_score)
            for w in (0.3, 0.5, 0.7, 1.0):
                cells = score_cat(build_rows_topk(bmap, top, prots[cat], w), cat)
                v = cells["bpo"]["f_micro_w"]
                grid[f"k{k}_w{w}"] = round(v, 5)
                if v > best["bpo"] + 1e-9:
                    best = {"k": k, "w": w, "bpo": round(v, 5),
                            "delta": round(v - base_bp, 5)}
        out["results"][cat] = {"graft_bpo": base_bp, "best": best, "grid": grid}
        print(cat, "graft", round(base_bp, 5), "best", best)
    json.dump(out, open(os.path.join(HERE, f"blend_topk_{TAG}.json"), "w"), indent=2)
    print("written blend_topk_%s.json" % TAG)


if __name__ == "__main__":
    main()
