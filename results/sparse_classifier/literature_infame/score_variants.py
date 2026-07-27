"""Score the WITH/WITHOUT-literature retrained rerankers board-faithful (BP),
via the champion graft (naivemax with InterPro2GO). Compare to:
  - champion (existing reranker tsvs grafted)   [Step 1 anchor]
  - retrained-base (no literature)              [clean A/B baseline]
  - retrained-lit  (literature in-frame)        [the experiment]
  - TransFew BP reference.
"""
import os, json
import pandas as pd
import graft_score as G

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFEW = {"lk": 0.512, "pk": 0.294}
CHAMPION_CANON = {"lk": 0.42807, "pk": 0.21797}


def grafted_cell(cat, rerank_bp):
    u = G.union_naivemax(rerank_bp)
    return G.score_bp(list(zip(u.acc, u.go, u.s_max)), cat)


def main():
    res = {"champion_canonical": CHAMPION_CANON, "transfew_bp": TRANSFEW,
           "cells": {}}
    for cat in ["lk", "pk"]:
        champ = G.champion_bp(cat)  # existing champion tsv grafted
        base = grafted_cell(cat, pd.read_parquet(
            os.path.join(HERE, f"rerank_bp_{cat}_base.parquet")))
        lit = grafted_cell(cat, pd.read_parquet(
            os.path.join(HERE, f"rerank_bp_{cat}_lit.parquet")))
        res["cells"][cat] = {
            "champion_reproduced": champ,
            "retrained_base": base,
            "retrained_lit": lit,
            "transfew": TRANSFEW[cat],
            "delta_lit_minus_base": round(lit["f_micro_w"] - base["f_micro_w"], 5),
            "delta_lit_minus_champion": round(lit["f_micro_w"] - champ["f_micro_w"], 5),
            "delta_base_minus_champion": round(base["f_micro_w"] - champ["f_micro_w"], 5),
        }
        print(cat, "champion", champ["f_micro_w"], "| base", base["f_micro_w"],
              "| lit", lit["f_micro_w"], "| transfew", TRANSFEW[cat], flush=True)
    json.dump(res, open(os.path.join(HERE, "score_variants.json"), "w"), indent=2)
    print("written score_variants.json", flush=True)


if __name__ == "__main__":
    main()
