"""Board-faithful LK-BP scoring of the A/B/C rerankers via the champion graft
(naivemax with InterPro2GO; OBO/IA/TOI, prop=fill, norm=cafa, no_orphans).
Anchors:
  champion        = existing champion LK tsv grafted (Step-1 reproduction anchor)
  retrained_base  = no-text retrain (must reproduce champion 0.42807)
  retrained_titles= desc+func+precut TITLES feature fold
  retrained_abstract = titles + precut ABSTRACT bodies feature fold (the upgrade)
Reuses literature_infame/graft_score.py (shared board-faithful scorer). LK ONLY.
"""
import os
import sys
import json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "literature_infame"))
import graft_score as G  # noqa: E402

TRANSFEW_LK = 0.512
CHAMPION_CANON = 0.42807
TITLES_FOLD_CANON = 0.43957  # prior titles in-frame fold result


def grafted_cell(rerank_bp):
    u = G.union_naivemax(rerank_bp)
    return G.score_bp(list(zip(u.acc, u.go, u.s_max)), "lk")


def main():
    champ = G.champion_bp("lk")
    base = grafted_cell(pd.read_parquet(os.path.join(HERE, "rerank_bp_lk_base.parquet")))
    titles = grafted_cell(pd.read_parquet(os.path.join(HERE, "rerank_bp_lk_titles.parquet")))
    abstract = grafted_cell(pd.read_parquet(os.path.join(HERE, "rerank_bp_lk_abstract.parquet")))

    res = {
        "category": "lk", "aspect": "bpo",
        "board_faithful": ("OBO/IA/TOI, prop=fill, norm=cafa, no_orphans; "
                           "naivemax(reranker_BP, InterPro2GO_BP); GT lafa_gt/groundtruth_LK.tsv"),
        "temporal_honesty": ("text features from publications <= 2025-09 (v227 t0); "
                             "test window v227->v230 (after Sep 2025) -> no test leakage. "
                             "text_score(protein,go) cut-invariant, joined to all rows."),
        "champion_reproduced": champ,
        "champion_canonical": CHAMPION_CANON,
        "titles_fold_canonical": TITLES_FOLD_CANON,
        "transfew_bp": TRANSFEW_LK,
        "cells": {
            "champion": champ,
            "retrained_base": base,
            "retrained_titles": titles,
            "retrained_abstract": abstract,
        },
        "deltas_f_micro_w": {
            "base_minus_champion": round(base["f_micro_w"] - champ["f_micro_w"], 5),
            "titles_minus_base": round(titles["f_micro_w"] - base["f_micro_w"], 5),
            "abstract_minus_base": round(abstract["f_micro_w"] - base["f_micro_w"], 5),
            "abstract_minus_titles": round(abstract["f_micro_w"] - titles["f_micro_w"], 5),
            "abstract_minus_champion": round(abstract["f_micro_w"] - champ["f_micro_w"], 5),
        },
    }
    json.dump(res, open(os.path.join(HERE, "result_lk_bp.json"), "w"), indent=2)
    print("champion", champ["f_micro_w"], "| base", base["f_micro_w"],
          "| titles", titles["f_micro_w"], "| abstract", abstract["f_micro_w"],
          flush=True)
    print("deltas", json.dumps(res["deltas_f_micro_w"]), flush=True)


if __name__ == "__main__":
    main()
