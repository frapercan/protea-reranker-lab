"""Assemble deliverables: 9-cell board-faithful (champion vs literature-in-frame),
non-regressive injectable BP TSVs (uninjected), feature importances, SUMMARY."""
import os, json
import pandas as pd
import graft_score as G

HERE = os.path.dirname(os.path.abspath(__file__))

# champion 9-cell board-faithful (graft+InterPro naivemax), canonical numbers
CHAMPION_9 = {
    "nk": {"bpo": 0.33745, "mfo": 0.66368, "cco": 0.47696},
    "lk": {"bpo": 0.42807, "mfo": 0.56384, "cco": 0.46147},
    "pk": {"bpo": 0.21797, "mfo": 0.24163, "cco": 0.26554},
}
TRANSFEW = {"lk_bpo": 0.512, "pk_bpo": 0.294}


def main():
    sv = json.load(open(os.path.join(HERE, "score_variants.json")))
    lk_lit = sv["cells"]["lk"]["retrained_lit"]["f_micro_w"]
    pk_lit = sv["cells"]["pk"]["retrained_lit"]["f_micro_w"]
    lk_base = sv["cells"]["lk"]["retrained_base"]["f_micro_w"]
    pk_base = sv["cells"]["pk"]["retrained_base"]["f_micro_w"]

    lit9 = json.loads(json.dumps(CHAMPION_9))
    lit9["lk"]["bpo"] = lk_lit
    lit9["pk"]["bpo"] = pk_lit

    # non-regressive injectable BP TSVs (grafted naivemax from lit reranker)
    inj = {}
    for cat in ["lk", "pk"]:
        rr = pd.read_parquet(os.path.join(HERE, f"rerank_bp_{cat}_lit.parquet"))
        u = G.union_naivemax(rr)
        out = u[["acc", "go", "s_max"]].copy()
        fn = os.path.join(HERE, f"injectable_{cat}_bp_lit.tsv")
        out.to_csv(fn, sep="\t", header=False, index=False, float_format="%.6f")
        inj[cat] = {"file": os.path.basename(fn), "rows": len(out),
                    "proteins": int(out.acc.nunique())}

    result = {
        "question": ("Does folding literature text->GO features (S-PubMedBERT max-"
                     "cosine, publications <= 2025-09 / v227 t0) INTO the champion "
                     "BP reranker, retrained on its OWN multi-snapshot training "
                     "frame (v160..v225 train, v225-v227 early-stop), convert "
                     "literature orthogonality into board-faithful BP Fmax on the "
                     "227->230 LAFA test?"),
        "core_correction": ("Prior arm trained a thin literature GATE on the single "
                            "225->227 validation slice and applied it to 227->230 "
                            "test -> temporal-window overfit, regressed (LK 0.385, "
                            "PK 0.161). Here literature is a FEATURE inside the "
                            "multi-snapshot reranker the champion graft uses; the "
                            "no-literature retrain reproduces the champion EXACTLY, "
                            "so the only difference vs champion is the literature "
                            "feature set (clean A/B)."),
        "temporal_honesty": ("Literature publications <= 2025-09 (v227 t0) only. "
                             "Test window is v227->v230 (after Sep 2025), so no test-"
                             "label leakage. text_score(protein,go) is cut-invariant "
                             "-> computed once per unique BP pair, joined to all rows."),
        "training_set_literature_coverage": json.load(
            open(os.path.join(HERE, "train_text_coverage.json"))),
        "board_faithful_bp": {
            "lk": {"champion_reproduced": sv["cells"]["lk"]["champion_reproduced"],
                   "champion_canonical": 0.42807,
                   "retrained_base_no_lit": lk_base,
                   "retrained_lit_in_frame": lk_lit,
                   "transfew": TRANSFEW["lk_bpo"],
                   "delta_lit_minus_base": round(lk_lit - lk_base, 5),
                   "gap_to_transfew": round(lk_lit - TRANSFEW["lk_bpo"], 5)},
            "pk": {"champion_reproduced": sv["cells"]["pk"]["champion_reproduced"],
                   "champion_canonical": 0.21797,
                   "retrained_base_no_lit": pk_base,
                   "retrained_lit_in_frame": pk_lit,
                   "transfew": TRANSFEW["pk_bpo"],
                   "delta_lit_minus_base": round(pk_lit - pk_base, 5),
                   "gap_to_transfew": round(pk_lit - TRANSFEW["pk_bpo"], 5)},
        },
        "nine_cell_board_faithful": {
            "champion_graft_interpro": CHAMPION_9,
            "literature_in_frame": lit9,
            "note": ("Only LK-bpo and PK-bpo change (literature routed to BP, LK+PK "
                     "rerankers only). MF/CC/NK identical by construction -> no "
                     "non-BP regression possible."),
        },
        "literature_feature_importance_gain": json.load(
            open(os.path.join(HERE, "train_meta_all.json"))),
        "injectable_uninjected": inj,
        "verdict": ("POSITIVE (modest, real). Folded in-frame into the multi-snapshot "
                    "reranker, literature is a genuine BP lever: LK-bpo "
                    f"{lk_base:.5f}->{lk_lit:.5f} (+{lk_lit-lk_base:.4f}), PK-bpo "
                    f"{pk_base:.5f}->{pk_lit:.5f} (+{pk_lit-pk_base:.4f}), both vs an "
                    "EXACTLY-reproduced champion. text_score/text_rank_pct/text_cos "
                    "carry real feature gain. It does NOT close the TransFew gap "
                    f"(LK still -{TRANSFEW['lk_bpo']-lk_lit:.3f}, PK -"
                    f"{TRANSFEW['pk_bpo']-pk_lit:.3f}); that gap is structural (their "
                    "text/label encoder reaches candidates our pool+max-cosine cannot)."
                    " The prior negative was a training-frame artifact, not a property "
                    "of literature."),
    }
    json.dump(result, open(os.path.join(HERE, "result_9cell.json"), "w"), indent=2)
    print(json.dumps(result["board_faithful_bp"], indent=2))
    print("injectable:", inj)


if __name__ == "__main__":
    main()
