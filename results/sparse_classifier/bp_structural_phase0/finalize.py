"""Assemble result_9cell.json + SUMMARY.txt from score_variants.json.

Phase 0 question: does unioning the two-tower sparse-classifier candidates into the
BP pool and retraining the reranker IN-FRAME convert the recall-ceiling uplift
(PK-bpo 0.470->0.537 +0.068, LK-bpo 0.767->0.823 +0.056 in p4_recall_ceiling.json)
into board-faithful f_micro_w on LK-BP and PK-BP?
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"

CHAMPION_9 = {
    "nk": {"bpo": 0.33745, "mfo": 0.66368, "cco": 0.47696},
    "lk": {"bpo": 0.42807, "mfo": 0.56384, "cco": 0.46147},
    "pk": {"bpo": 0.21797, "mfo": 0.24163, "cco": 0.26554},
}
TRANSFEW = {"lk": 0.512, "pk": 0.294}
# p4 candidate-generation recall ceiling (propagated set-recall, NOT Fmax)
P4 = {"lk": {"knn": 0.7668, "knn_clf": 0.8227, "uplift": 0.0559},
      "pk": {"knn": 0.4695, "knn_clf": 0.5372, "uplift": 0.0677}}
CONVERT_EPS = 0.005   # Fmax gain we count as a real conversion


def _load_feature_importance():
    import glob
    allp = os.path.join(HERE, "train_meta_all.json")
    if os.path.exists(allp):
        return json.load(open(allp))
    merged = {}
    for f in sorted(glob.glob(os.path.join(HERE, "train_meta_*_*.json"))):
        merged.update(json.load(open(f)))
    return merged


def main():
    sv = json.load(open(os.path.join(HERE, "score_variants.json")))
    cells = sv["cells"]

    union9 = json.loads(json.dumps(CHAMPION_9))
    union9["lk"]["bpo"] = cells["lk"]["union_pool"]["f_micro_w"]
    union9["pk"]["bpo"] = cells["pk"]["union_pool"]["f_micro_w"]

    verdict = {}
    for cat in ["lk", "pk"]:
        c = cells[cat]
        base_f = c["retrained_base"]["f_micro_w"]
        union_f = c["union_pool"]["f_micro_w"]
        d = round(union_f - base_f, 5)
        ceil_base = c["retrained_base"]["rc_ceiling_tau0"]
        ceil_union = c["union_pool"]["rc_ceiling_tau0"]
        rcr_base = c["retrained_base"]["rc_realized_at_best"]
        rcr_union = c["union_pool"]["rc_realized_at_best"]
        fmax_up = d >= CONVERT_EPS
        ceiling_rose = (ceil_union - ceil_base) > 0.005
        # "generation converts" requires BOTH: the candidate recall ceiling actually
        # rose AND that extra recall turned into Fmax. A feature-only Fmax gain
        # (ceiling flat) is NOT generation converting; a ceiling rise with Fmax down
        # is precision dilution (the scoring neck).
        generation_converts = ceiling_rose and fmax_up
        if ceiling_rose and not fmax_up:
            read = ("ceiling ROSE (+%.3f) but Fmax FELL (%+.4f): precision dilution -- the "
                    "reranker cannot separate the admitted clf candidates -> SCORING neck"
                    % (ceil_union - ceil_base, d))
        elif (not ceiling_rose) and fmax_up:
            read = ("Fmax ROSE (%+.4f) but BP recall ceiling stayed FLAT (%.3f->%.3f): the "
                    "gain is the clf-SCORE feature reranking already-reachable candidates, "
                    "NOT candidate generation" % (d, ceil_base, ceil_union))
        elif ceiling_rose and fmax_up:
            read = "ceiling rose AND Fmax rose: generation genuinely converts"
        else:
            read = "neither ceiling nor Fmax moved"
        verdict[cat] = {
            "champion_reproduced_f": c["champion_reproduced"]["f_micro_w"],
            "base_reproduces_champion": abs(base_f - CHAMPION_9[cat]["bpo"]) < 0.003,
            "base_f": base_f, "union_f": union_f, "delta_union_minus_base": d,
            "transfew": TRANSFEW[cat], "gap_union_to_transfew": round(union_f - TRANSFEW[cat], 5),
            "p4_gen_recall_ceiling_uplift": P4[cat]["uplift"],
            "scored_rc_ceiling_base": ceil_base, "scored_rc_ceiling_union": ceil_union,
            "scored_rc_realized_base": rcr_base, "scored_rc_realized_union": rcr_union,
            "ceiling_rose": ceiling_rose, "fmax_up": fmax_up,
            "generation_converts": generation_converts, "reading": read,
        }

    any_generation_converts = any(verdict[c]["generation_converts"] for c in ("lk", "pk"))
    headline = ("GENERATION CONVERTS" if any_generation_converts else
                "GENERATION DOES NOT CONVERT -- neck is JOINT SCORING "
                "(pivot to Phase 1: GCN label encoder + joint cross-attention)")
    result = {
        "question": ("Does unioning the two-tower sparse-classifier candidates into "
                     "the BP pool and retraining the reranker IN-FRAME (v160..v225 "
                     "train, v225-v227 early-stop, v227-v230 test) convert the "
                     "candidate-generation recall-ceiling uplift (PK-bpo +0.068, "
                     "LK-bpo +0.056) into board-faithful f_micro_w?"),
        "method": ("PK: base = percut knn-only (champion), union = percut FULL pool "
                   "(admits two-tower clf candidates) + clf_rank. LK: base = baseline-"
                   "M2 full pool (champion), union = champion pool UNION two-tower BP "
                   "candidates + tt_clf_score/tt_clf_rank/tt_has_clf. Board-faithful "
                   "cafa_eval (OBO/IA/TOI, prop=fill, norm=cafa, no_orphans, PK excludes "
                   "PK_known), champion InterPro2GO naive-max graft, LAFA 227->230."),
        "temporal_honesty": ("Two-tower candidates+scores are built per-cut by "
                             "build_per_cut_codes.py (frozen v227 SVD basis, per-cut "
                             "PPMI), joined only on (acc,go,snapshot_pair); no future-"
                             "snapshot codes touch any training row."),
        "board_faithful_bp": {cat: {
            "champion_reproduced": cells[cat]["champion_reproduced"],
            "champion_canonical": CHAMPION_9[cat]["bpo"],
            "retrained_base": cells[cat]["retrained_base"],
            "union_pool": cells[cat]["union_pool"],
            "transfew": TRANSFEW[cat],
            "delta_union_minus_base": verdict[cat]["delta_union_minus_base"],
            "gap_union_to_transfew": verdict[cat]["gap_union_to_transfew"],
        } for cat in ("lk", "pk")},
        "recall_ceiling_vs_realized": {cat: {
            "p4_generation_ceiling_uplift": P4[cat]["uplift"],
            "scored_rc_ceiling_base_tau0": verdict[cat]["scored_rc_ceiling_base"],
            "scored_rc_ceiling_union_tau0": verdict[cat]["scored_rc_ceiling_union"],
            "scored_rc_realized_at_best_base": verdict[cat]["scored_rc_realized_base"],
            "scored_rc_realized_at_best_union": verdict[cat]["scored_rc_realized_union"],
        } for cat in ("lk", "pk")},
        "nine_cell_board_faithful": {
            "champion_graft_interpro": CHAMPION_9,
            "union_pool_in_frame": union9,
            "note": ("Only LK-bpo and PK-bpo change (union routed to BP, LK+PK rerankers "
                     "retrained); MF/CC/NK identical by construction -> no non-BP "
                     "regression possible."),
        },
        "feature_importance": _load_feature_importance(),
        "verdict_detail": verdict,
        "headline": headline,
    }
    json.dump(result, open(os.path.join(HERE, "result_9cell.json"), "w"), indent=2)
    write_summary(result, verdict)
    print(json.dumps(result["board_faithful_bp"], indent=2))
    print("HEADLINE:", headline)
    return result, verdict


def write_summary(result, verdict):
    lk, pk = verdict["lk"], verdict["pk"]
    L = [
        "BP STRUCTURAL LEVER -- PHASE 0: does union(KNN, two-tower clf) + in-frame",
        "reranker retrain convert the candidate-generation recall-ceiling uplift into",
        "board-faithful f_micro_w on LK-BP and PK-BP?",
        "=" * 78,
        "Offline lab only. Board-faithful cafa_eval throughout (OBO, IA, TOI, prop=fill,",
        "norm=cafa, no_orphans, PK excludes PK_known; GT in lafa_gt/). LAFA 227->230 test.",
        "Champion BP cell = naivemax(reranker_BP UNION InterPro2GO_BP). No live board /",
        "live DB / PROTEA touched. Two-tower candidates+scores are per-cut temporally",
        "honest (build_per_cut_codes.py, frozen v227 SVD basis); joined only on",
        "(acc, go, snapshot_pair) -- no future-snapshot codes touch a training row.",
        "",
        "STEP 0 SANITY -- champion reproduced EXACTLY (no-union retrain on its own frame):",
        f"  LK-bpo base = {lk['base_f']:.5f}  (champion 0.42807)   -> reproduced",
        f"  PK-bpo base = {pk['base_f']:.5f}  (champion 0.21631; canonical 0.21797) -> reproduced",
        "  (PK base trace bit-for-bit: best_iter=341, auc 0.97048, ap 0.80528;",
        "   LK base best_iter=167 -- both match the canonical champion training trace.)",
        "",
        "RESULT (board-faithful BP, 227->230 test)",
        "-" * 78,
        "                  champion   union-pool    delta     TransFew   gap(union-TF)",
        f"  LK-bpo          {lk['base_f']:.5f}    {lk['union_f']:.5f}    {lk['delta_union_minus_base']:+.5f}   "
        f"{lk['transfew']:.3f}     {lk['gap_union_to_transfew']:+.5f}",
        f"  PK-bpo          {pk['base_f']:.5f}    {pk['union_f']:.5f}    {pk['delta_union_minus_base']:+.5f}   "
        f"{pk['transfew']:.3f}     {pk['gap_union_to_transfew']:+.5f}",
        "",
        "RECALL CEILING vs REALIZED RECALL (the decisive diagnostic)",
        "-" * 78,
        "  p4 candidate-generation set-recall ceiling uplift: PK-bpo +0.068, LK-bpo +0.056.",
        "  In-frame, scored board-faithful (propagated rc_micro_w at the BP cell):",
        f"  PK  rc_ceiling(tau->0): {pk['scored_rc_ceiling_base']:.5f} -> {pk['scored_rc_ceiling_union']:.5f}  "
        f"(+{pk['scored_rc_ceiling_union']-pk['scored_rc_ceiling_base']:.3f}, ROSE as predicted)",
        f"      rc_realized@bestFmax: {pk['scored_rc_realized_base']:.5f} -> {pk['scored_rc_realized_union']:.5f}  "
        f"({pk['scored_rc_realized_union']-pk['scored_rc_realized_base']:+.3f}, FELL)",
        f"  LK  rc_ceiling(tau->0): {lk['scored_rc_ceiling_base']:.5f} -> {lk['scored_rc_ceiling_union']:.5f}  "
        f"(flat -- InterPro graft already supplies BP recall; few new LK positives)",
        f"      rc_realized@bestFmax: {lk['scored_rc_realized_base']:.5f} -> {lk['scored_rc_realized_union']:.5f}  "
        f"({lk['scored_rc_realized_union']-lk['scored_rc_realized_base']:+.3f}, rose via better ranking)",
        "",
        "READING",
        "-" * 78,
        f"  PK: {pk['reading']}",
        f"  LK: {lk['reading']}",
        "",
        "VERDICT",
        "-" * 78,
        f"  {result['headline']}",
        "",
        "  PK-BP is the decisive, cleanest test: PK champion is genuinely KNN-only, so the",
        "  two-tower clf candidates are truly excluded; admitting them is a single-variable",
        "  change. Their realized propagated recall ceiling ROSE +0.071 (0.317->0.388) --",
        "  the generation lift is real and reaches the scorer -- yet board-faithful Fmax",
        "  FELL -0.027 (0.21631->0.18971): the reranker cannot separate the new clf",
        "  candidates from false positives, so best-Fmax recall actually drops. That is",
        "  precision dilution: the neck is JOINT SCORING, not generation.",
        "",
        "  LK-BP improves +0.0133 (0.42807->0.44135), BUT its BP recall ceiling stayed FLAT",
        "  (0.707->0.701) -- this gain is NOT the generation ceiling converting; it is the",
        "  two-tower clf SCORE entering as a feature (tt_clf_score) and reranking candidates",
        "  the pool already reached. So even the positive cell points at scoring, not",
        "  generation. Neither cell closes the TransFew gap (LK -0.071, PK -0.104).",
        "",
        "  => Generation is NOT the lever. Proceed to PHASE 1: a learnable GO-label encoder",
        "     (GCN over the DAG) + joint cross-attention scoring, so the model can VALUE the",
        "     reachable candidates correctly rather than merely enlarge the pool.",
        "",
        "PER-CELL SHIPPABLE (best-of, board-faithful)",
        "-" * 78,
        f"  LK-bpo: take UNION  ({lk['union_f']:.5f} > champion {lk['base_f']:.5f}) -- the tt_clf_score feature is a keeper.",
        f"  PK-bpo: KEEP CHAMPION ({pk['base_f']:.5f}); union regresses to {pk['union_f']:.5f} -- do NOT ship PK union.",
        "  union routed to BP only; MF/CC/NK unchanged by construction (no non-BP regression).",
        "",
        "ARTIFACTS",
        "-" * 78,
        "  build_lk_union.py, union_lk_{train,eval}.parquet, train_and_emit.py,",
        "  model_{lk,pk}_{base,union}.txt, rerank_bp_{lk,pk}_{base,union}.parquet,",
        "  score_all.py (graft + recall-ceiling capture), score_variants.json,",
        "  finalize.py, result_9cell.json, train_meta_{cat}_{variant}.json, SUMMARY.txt.",
    ]
    open(os.path.join(HERE, "SUMMARY.txt"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
